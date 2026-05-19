"""InGPO inference strategy: SPO-tree with online Share / Prune triggers.

Subclasses `HybridInferenceStrategy` from SPO and overrides `_construct_tree`
so that for every freshly-expanded child segment we:

  1. Compute K fast logprobs `log pi(y_i | traj(child))` against the
     per-problem answer set Y.
  2. Consult the `TriggerEngine` for SHARE / PRUNE / EXPAND.
  3. Annotate the child node with `ingpo_action`, `ingpo_share_target`,
     `ingpo_avg_lp_K`, `ingpo_avg_lp_m`, `ingpo_tv_m`.
  4. Skip recursion into SHARE / PRUNE children — but keep them in the tree
     because the downstream episode generator still consumes them as edges.

The answer set Y is built once per problem at `depth==1` (i.e. before the
first batch of root children is expanded), in parallel with that expansion.

All other behaviour — branch_factor strategy, segment cap M, reward
function, full_text accounting — is inherited unchanged from SPO so the
codepath remains identical to a faithful SPO-tree run when triggers are
disabled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
import openai

from treetune.common import Lazy
from treetune.inference_strategies.base_inference_strategy import InferenceStrategy
from treetune.inference_strategies.hybrid_inference_strategy import (
    HybridInferenceStrategy,
)
from treetune.inference_strategies.tree_inference import Node
from treetune.logging_utils import get_logger

from ingpo_ext.core.answer_set import (
    DEFAULT_Y_PROMPT_TEMPLATE,
    AnswerSet,
    AnswerSetGenerator,
)
from ingpo_ext.core.budget import BudgetAllocator, BudgetNode
from ingpo_ext.core.logging_helpers import ConstructionEventWriter
from ingpo_ext.core.tb_logger import TensorBoardLogger
from ingpo_ext.core.thresholds import ThresholdConfig
from ingpo_ext.core.triggers import Action, TriggerEngine
from ingpo_ext.core.vllm_scorer import VLLMLogprobClient, make_lp_scorer

logger = get_logger(__name__)


@InferenceStrategy.register("ingpo", exist_ok=True)
class InGPOInferenceStrategy(HybridInferenceStrategy):
    def __init__(
        self,
        # InGPO-specific knobs ------------------------------------------------
        ingpo_K: int = 4,
        ingpo_m: int = 32,
        ingpo_epsilon: float = 0.02,
        ingpo_r_max: float = 1.0,
        ingpo_gamma: float = 0.5,
        ingpo_alpha: float = 0.05,
        ingpo_use_dkw: bool = True,
        ingpo_eta_override: Optional[float] = None,
        ingpo_enable_share: bool = True,
        ingpo_enable_prune: bool = True,
        ingpo_share_target: str = "nearest",
        ingpo_y_prompt_template: str = DEFAULT_Y_PROMPT_TEMPLATE,
        ingpo_y_temperature: float = 0.7,
        ingpo_y_max_tokens: int = 8192,
        ingpo_y_field: str = "answer",  # field on data_instance with gold
        ingpo_score_concurrency: int = 64,
        # Bug-fix / logging / budget knobs -----------------------------------
        ingpo_prune_skip_root: bool = True,
        ingpo_log_construction: bool = True,
        ingpo_log_per_decision: bool = True,
        ingpo_tensorboard_enabled: bool = True,
        ingpo_tensorboard_dir: Optional[str] = None,
        ingpo_construction_log_path: Optional[str] = None,
        ingpo_model_context_size: Optional[int] = None,
        # Inherited ----------------------------------------------------------
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.cfg_thresholds = ThresholdConfig(
            epsilon=ingpo_epsilon,
            r_max=ingpo_r_max,
            gamma=ingpo_gamma,
            alpha=ingpo_alpha,
            K=ingpo_K,
            use_dkw=ingpo_use_dkw,
            eta_override=ingpo_eta_override,
        )
        self.ingpo_m = int(ingpo_m)
        self.ingpo_enable_share = bool(ingpo_enable_share)
        self.ingpo_enable_prune = bool(ingpo_enable_prune)
        self.ingpo_share_target = ingpo_share_target
        self.ingpo_y_prompt_template = ingpo_y_prompt_template
        self.ingpo_y_temperature = float(ingpo_y_temperature)
        self.ingpo_y_max_tokens = int(ingpo_y_max_tokens)
        self.ingpo_y_field = ingpo_y_field
        self.ingpo_score_concurrency = int(ingpo_score_concurrency)
        self.ingpo_prune_skip_root = bool(ingpo_prune_skip_root)
        self.ingpo_log_construction = bool(ingpo_log_construction)
        self.ingpo_log_per_decision = bool(ingpo_log_per_decision)
        self.ingpo_tensorboard_enabled = bool(ingpo_tensorboard_enabled)
        self.ingpo_tensorboard_dir = ingpo_tensorboard_dir
        self.ingpo_construction_log_path = ingpo_construction_log_path
        self.ingpo_model_context_size = ingpo_model_context_size
        self._lp_client: Optional[VLLMLogprobClient] = None
        self._tb_logger: Optional[TensorBoardLogger] = None
        self._event_writer: Optional[ConstructionEventWriter] = None
        self._tree_counter: int = 0

    # ------------------------------------------------------------------
    # vLLM scoring client
    # ------------------------------------------------------------------

    def _ensure_lp_client(self) -> VLLMLogprobClient:
        if self._lp_client is not None:
            return self._lp_client
        guidance_kwargs = self.guidance_llm_lazy._params  # type: ignore[attr-defined]
        api_base = guidance_kwargs.get("api_base")
        if api_base in (None, "none"):
            api_base = os.environ.get("APP_OPENAI_VLLM_API_BASE")
        if api_base is None:
            raise RuntimeError("InGPO needs api_base to call vLLM /completions")
        model = guidance_kwargs.get("model")
        api_key = guidance_kwargs.get("api_key", "EMPTY")
        self._lp_client = VLLMLogprobClient(
            api_base=api_base,
            model=model,
            api_key=api_key,
            max_concurrency=self.ingpo_score_concurrency,
        )
        return self._lp_client

    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("InGPO requires a tokenizer to count suffix tokens")
        return self.tokenizer(text).input_ids

    # ------------------------------------------------------------------
    # Construction-time logging (lazy init so result_dir is resolved)
    # ------------------------------------------------------------------

    def _result_root(self) -> Path:
        rd = getattr(self, "result_dir", None)
        return Path(rd) if rd is not None else Path.cwd()

    def _ensure_tb_logger(self) -> Optional[TensorBoardLogger]:
        if self._tb_logger is not None or not self.ingpo_tensorboard_enabled:
            return self._tb_logger
        logdir = self.ingpo_tensorboard_dir or str(self._result_root() / "tb" / "ingpo")
        self._tb_logger = TensorBoardLogger(logdir=logdir, enabled=True)
        return self._tb_logger

    def _ensure_event_writer(self) -> Optional[ConstructionEventWriter]:
        if self._event_writer is not None or not self.ingpo_log_per_decision:
            return self._event_writer
        path = self.ingpo_construction_log_path or str(
            self._result_root() / "ingpo_demos" / "construction.jsonl"
        )
        self._event_writer = ConstructionEventWriter(path=path, enabled=True)
        return self._event_writer

    def _resolve_context_size(self) -> int:
        if self.ingpo_model_context_size is not None:
            return int(self.ingpo_model_context_size)
        mcs = getattr(self.node_expander, "model_context_size", None)
        if mcs is not None:
            return int(mcs)
        logger.warning(
            "InGPO: model_context_size unknown; falling back to 4096 for budget allocator."
        )
        return 4096

    # ------------------------------------------------------------------
    # Build per-problem Y
    # ------------------------------------------------------------------

    async def _build_answer_set(
        self,
        problem_id: str,
        problem_text: str,
        gold: str,
    ) -> AnswerSet:
        client = openai.AsyncOpenAI(
            api_key=self._lp_client.api_key,
            base_url=self._lp_client.api_base,
        ) if self._lp_client is not None else None

        async def sample_fn(prompt: str, n: int, temperature: float, max_tokens: int):
            if client is None:
                return []
            resp = await client.completions.create(
                model=self._lp_client.model,
                prompt=prompt,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return [c.text for c in resp.choices]

        gen = AnswerSetGenerator(
            sample_fn=sample_fn,
            m=self.ingpo_m,
            temperature=self.ingpo_y_temperature,
            max_tokens=self.ingpo_y_max_tokens,
            prompt_template=self.ingpo_y_prompt_template,
        )
        try:
            return await gen.build(problem_id=problem_id, problem=problem_text, gold=gold)
        except Exception as exc:
            logger.warning(f"Y generation failed for problem {problem_id}: {exc}")
            return AnswerSet(problem_id=problem_id, gold=gold, y=[])

    # ------------------------------------------------------------------
    # Override tree construction
    # ------------------------------------------------------------------

    async def _construct_tree(
        self,
        initial_prompt: str,
        max_depth: int = 2,
        data_instance: Optional[Dict[str, Any]] = None,
    ):
        client = self._ensure_lp_client()
        scorer = make_lp_scorer(client, self._tokenize)

        problem_text = data_instance.get("problem") if data_instance else None
        gold = data_instance.get(self.ingpo_y_field) if data_instance else None
        problem_id = str(data_instance.get("_treetune__idx", uuid.uuid4())) if data_instance else str(uuid.uuid4())

        self._tree_counter += 1
        tree_idx = self._tree_counter

        # Y must exist before the first decide() call. Build it concurrently
        # with the first depth so we don't add latency on the critical path.
        answer_set: AnswerSet = AnswerSet(problem_id=problem_id, gold=gold or "", y=[])
        y_task = asyncio.create_task(
            self._build_answer_set(problem_id, problem_text or "", gold or "")
        )

        engine: Optional[TriggerEngine] = None

        # Budget allocator: one BudgetNode per tree node, root sized at the
        # model's context window. PRUNE/SHARE release leftover to siblings;
        # all-siblings-terminate cascades to aunts/uncles.
        context_size = self._resolve_context_size()
        budgeter = BudgetAllocator(total=context_size, tokenize=self._tokenize)
        root_bn = budgeter.attach_root(initial_prompt)

        tb = self._ensure_tb_logger()
        events = self._ensure_event_writer()

        tree: Node = {
            "text": initial_prompt,
            "depth": 0,
            "full_text": initial_prompt,
            "stop_text": "aaa",
            "_request_object": data_instance,
            "leaf": False,
            "ingpo_action": Action.EXPAND.value,
            "ingpo_segment_id": "root",
            "ingpo_budget_node": root_bn,
        }

        async def _ensure_engine() -> Optional[TriggerEngine]:
            nonlocal engine, answer_set
            if engine is not None:
                return engine
            answer_set = await y_task
            if answer_set.m == 0:
                logger.warning(
                    "InGPO: empty answer set; falling back to vanilla SPO-tree expansion"
                )
                return None
            engine = TriggerEngine(
                answer_set=answer_set,
                scorer=scorer,
                thresholds=self.cfg_thresholds,
                enable_share=self.ingpo_enable_share,
                enable_prune=self.ingpo_enable_prune,
                share_target=self.ingpo_share_target,
                root_segment_id="root",
                prune_skip_root=self.ingpo_prune_skip_root,
            )
            await engine.register_root(initial_prompt)
            return engine

        def _per_child_max_tokens(parent_bn: BudgetNode, n: int, depth: int) -> Optional[int]:
            share = parent_bn.remaining // max(n, 1)
            # Last depth: SPO convention is "unbounded" — but we still cap to
            # the budget share to avoid blowing past the context window.
            if depth == max_depth - 1:
                return share if share > 0 else None
            cap = self.M if self.M else share
            return max(1, min(cap, share)) if share > 0 else max(1, cap or 1)

        async def dfs(node: Node, prefix: str, depth: int) -> None:
            if depth == max_depth:
                node["reward"], _ = self.reward_function(
                    query=prefix,
                    response=node["text"],
                    dataset_instance=data_instance,
                )
                node["leaf"] = True
                return

            parent_bn: BudgetNode = node.get("ingpo_budget_node") or root_bn

            # Choose a uniform max_tokens for this batch derived from the
            # parent's remaining budget (equal split across upcoming children).
            # We don't know n_children before calling expand() — assume the
            # branch_factor strategy returns at most W=branch_factor children.
            branch_factor = getattr(self.node_expander, "branch_factor", None)
            n_planned = int(branch_factor) if branch_factor else 1
            max_tokens = _per_child_max_tokens(parent_bn, n_planned, depth)

            children = await self.node_expander.expand(
                current_node=node,
                prefix=prefix,
                depth=depth,
                max_tokens=max_tokens,
            )
            node["children"] = children

            # Now we know the real fan-out — allocate budgets accordingly.
            child_bns = budgeter.allocate_children(parent_bn, len(children))

            local_engine = await _ensure_engine()

            expansion_tasks: List[asyncio.Task] = []
            pending_decisions: List[Tuple[Dict[str, Any], Node, bool, BudgetNode]] = []
            parent_seg_id = node.get("ingpo_segment_id", "root")

            for ch_idx, (child, child_bn) in enumerate(zip(children, child_bns)):
                child_seg_id = f"{node.get('ingpo_segment_id', 'root')}/{depth}/{ch_idx}"
                child["ingpo_segment_id"] = child_seg_id
                child["ingpo_action"] = Action.EXPAND.value
                child["ingpo_depth"] = depth + 1
                child["ingpo_parent_segment_id"] = parent_seg_id
                child["ingpo_budget_node"] = child_bn

                # Charge this child's tokens against its budget node.
                charged = budgeter.record_used(child_bn, child.get("text") or "")
                if local_engine is not None:
                    local_engine.stats.add_tokens(charged)

                if child["finish_reason"] != "length":
                    child["reward"], _ = self.reward_function(
                        query=prefix,
                        response=child["full_text"],
                        dataset_instance=data_instance,
                    )
                    child["leaf"] = True
                    if local_engine is not None:
                        pending_decisions.append(
                            (
                                {
                                    "segment_id": child_seg_id,
                                    "parent_id": parent_seg_id,
                                    "prefix": child["full_text"],
                                    "is_leaf": True,
                                },
                                child,
                                True,
                                child_bn,
                            )
                        )
                    else:
                        # No engine: still close out the budget node.
                        budgeter.release(child_bn)
                    continue

                child["leaf"] = False
                if local_engine is None:
                    expansion_tasks.append(
                        asyncio.create_task(dfs(child, child["full_text"], depth + 1))
                    )
                    continue

                pending_decisions.append(
                    (
                        {
                            "segment_id": child_seg_id,
                            "parent_id": parent_seg_id,
                            "prefix": child["full_text"],
                            "is_leaf": False,
                        },
                        child,
                        False,
                        child_bn,
                    )
                )

            if local_engine is not None and pending_decisions:
                decision_tasks = [
                    asyncio.create_task(local_engine.decide(**payload))
                    for payload, _, _, _ in pending_decisions
                ]
                decision_results = await asyncio.gather(
                    *decision_tasks, return_exceptions=True
                )
                for (_, child, is_leaf, child_bn), result in zip(
                    pending_decisions, decision_results
                ):
                    if isinstance(result, Exception):
                        if is_leaf:
                            logger.warning(f"InGPO decide() failed (leaf): {result}")
                            budgeter.release(child_bn)
                        else:
                            logger.warning(f"InGPO decide() failed: {result}")
                            expansion_tasks.append(
                                asyncio.create_task(
                                    dfs(child, child["full_text"], depth + 1)
                                )
                            )
                        continue

                    decision = result
                    self._annotate_node(child, decision)
                    local_engine.stats.record_decision(depth + 1, decision.action)

                    # Emit a per-decision event line.
                    self._emit_decision_event(
                        events=events,
                        tree_idx=tree_idx,
                        problem_id=problem_id,
                        depth=depth + 1,
                        child=child,
                        decision=decision,
                        child_bn=child_bn,
                    )

                    if is_leaf:
                        budgeter.release(child_bn)
                        continue

                    if decision.action is Action.EXPAND:
                        expansion_tasks.append(
                            asyncio.create_task(
                                dfs(child, child["full_text"], depth + 1)
                            )
                        )
                    elif decision.action is Action.SHARE:
                        child["leaf"] = True
                        child["reward"] = float("nan")  # sentinel; replaced later
                        budgeter.release(child_bn)
                    else:  # Action.PRUNE
                        child["leaf"] = True
                        child["reward"] = float("nan")  # sentinel; replaced later
                        budgeter.release(child_bn)

            # ---- depth-level live logging --------------------------------
            if local_engine is not None and (
                self.ingpo_log_construction or tb is not None
            ):
                self._emit_depth_log(
                    tb=tb,
                    tree_idx=tree_idx,
                    problem_id=problem_id,
                    depth=depth + 1,
                    max_depth=max_depth,
                    stats=local_engine.stats,
                    budgeter=budgeter,
                )

            if expansion_tasks:
                await asyncio.gather(*expansion_tasks)

            # Bubble any unspent budget for this subtree up to live aunts/uncles.
            budgeter.maybe_bubble_up(parent_bn)

            # Compute reward / reward_std *after* descendants finish.  For
            # SHARE / PRUNE children, we postpone the value until the episode
            # generator can resolve them; for now use the parent's prior
            # average to keep tree-level statistics finite.
            child_rewards = []
            for ch in children:
                r = ch.get("reward")
                if r is None or (isinstance(r, float) and np.isnan(r)):
                    continue
                child_rewards.append(r)
            if child_rewards:
                node["reward"] = float(np.mean(child_rewards))
                node["reward_std"] = float(np.std(child_rewards))
            else:
                node["reward"] = 0.0
                node["reward_std"] = 0.0

        await dfs(tree, initial_prompt, 0)

        # Strip budget nodes from the dict before downstream serialisation —
        # they hold parent references that don't survive json.dumps.
        def _strip_budget_nodes(n: Node) -> None:
            n.pop("ingpo_budget_node", None)
            for c in n.get("children") or []:
                _strip_budget_nodes(c)

        _strip_budget_nodes(tree)

        # Tag the tree with stats and answer-set bookkeeping for downstream.
        if engine is not None:
            stats_dict = engine.stats.as_dict()
            stats_dict.update(budgeter.as_dict())
            tree["ingpo_stats"] = stats_dict
        else:
            tree["ingpo_stats"] = dict(budgeter.as_dict())
        tree["ingpo_answer_set_size"] = answer_set.m
        return tree

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    def _emit_decision_event(
        self,
        *,
        events: Optional[ConstructionEventWriter],
        tree_idx: int,
        problem_id: str,
        depth: int,
        child: Node,
        decision,
        child_bn: BudgetNode,
    ) -> None:
        if events is None:
            return
        events.write(
            {
                "tree_idx": tree_idx,
                "problem_id": problem_id,
                "depth": depth,
                "segment_id": child.get("ingpo_segment_id"),
                "parent_id": child.get("ingpo_parent_segment_id"),
                "action": decision.action.value,
                "avg_lp_K": decision.avg_lp_K,
                "avg_lp_m": decision.avg_lp_m,
                "gap_K": decision.avg_lp_diff_to_pa_K,
                "gap_m": decision.avg_lp_diff_to_pa_m,
                "tv_m": decision.tv_m,
                "eta": decision.eta_used,
                "tau": decision.tau_used,
                "share_target": decision.share_target,
                "budget_initial": child_bn.initial,
                "budget_used": child_bn.used,
                "budget_remaining": child_bn.remaining,
            }
        )

    def _emit_depth_log(
        self,
        *,
        tb: Optional[TensorBoardLogger],
        tree_idx: int,
        problem_id: str,
        depth: int,
        max_depth: int,
        stats,
        budgeter: BudgetAllocator,
    ) -> None:
        bucket = stats.per_depth.get(int(depth))
        depth_total = sum(bucket.values()) if bucket else 0

        if self.ingpo_log_construction:
            logger.info(
                "InGPO tree=%s d=%d/%d n=%d expand=%d share=%d prune=%d "
                "tokens=%d budget_remain=%d max_depth_reached=%d",
                problem_id,
                depth,
                max_depth,
                depth_total,
                (bucket.get("expand", 0) if bucket else 0),
                (bucket.get("share", 0) if bucket else 0),
                (bucket.get("prune", 0) if bucket else 0),
                stats.tokens_generated,
                budgeter.root.remaining,
                stats.max_depth_reached,
            )

        if tb is not None and tb.enabled:
            tb.log_scalars(
                {
                    f"ingpo/tree_{tree_idx}/depth_{depth}/n": depth_total,
                    "ingpo/prune_rate": stats.prune_rate(),
                    "ingpo/tokens_generated": stats.tokens_generated,
                    "ingpo/max_depth_reached": stats.max_depth_reached,
                    "ingpo/budget_used": budgeter._used_total(budgeter.root),
                    "ingpo/budget_redistributed": budgeter.released_total,
                    "ingpo/budget_bubbled_up": budgeter.bubbled_up_total,
                }
            )

    @staticmethod
    def _annotate_node(child: Node, decision) -> None:
        child["ingpo_action"] = decision.action.value
        child["ingpo_share_target"] = decision.share_target
        child["ingpo_avg_lp_K"] = decision.avg_lp_K
        if decision.avg_lp_m is not None:
            child["ingpo_avg_lp_m"] = decision.avg_lp_m
        if decision.tv_m is not None:
            child["ingpo_tv_m"] = decision.tv_m
        if decision.avg_lp_diff_to_pa_m is not None:
            child["ingpo_gap_m"] = decision.avg_lp_diff_to_pa_m
        child["ingpo_eta"] = decision.eta_used
        child["ingpo_tau"] = decision.tau_used
