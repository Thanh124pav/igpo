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
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

import math
import openai

from treetune.common import Lazy
from treetune.inference_strategies.base_inference_strategy import InferenceStrategy
from treetune.inference_strategies.hybrid_inference_strategy import (
    HybridInferenceStrategy,
)
from treetune.inference_strategies.tree_inference import Node
from treetune.logging_utils import get_logger

from treetune.ingpo.answer_set import (
    DEFAULT_Y_PROMPT_TEMPLATE,
    AnswerSet,
    AnswerSetGenerator,
)
from treetune.ingpo.logging_helpers import aggregate_tree_stats
from treetune.ingpo.thresholds import ThresholdConfig, tv_to_value_bound
from treetune.ingpo.budget_allocation import allocate_branch_factors
from treetune.ingpo.budget_scheduler import FlexibleBudgetScheduler
from treetune.ingpo.tv_estimators import ConditionalTVEstimator
from treetune.ingpo.triggers import Action, TriggerEngine
from treetune.ingpo.local_value_share import (
    LocalShareDecision,
    confidence_radius,
    pair_budget,
    sampled_tv_from_logps,
    select_candidate_pairs,
    stable_softmax,
)
from treetune.ingpo.vllm_scorer import VLLMLogprobClient, make_lp_scorer

logger = get_logger(__name__)


@InferenceStrategy.register("ingpo", exist_ok=True)
class InGPOInferenceStrategy(HybridInferenceStrategy):
    def __init__(
        self,
        # InGPO-specific knobs ------------------------------------------------
        ingpo_K: int = 10,
        ingpo_m: int = 100,
        ingpo_epsilon: float = 0.02,
        ingpo_r_max: float = 1.0,
        ingpo_gamma: float = 0.5,
        ingpo_alpha: float = 0.05,
        ingpo_use_dkw: bool = True,
        ingpo_eta_override: Optional[float] = None,
        ingpo_enable_share: bool = True,
        ingpo_enable_prune: bool = True,
        ingpo_share_target: str = "nearest",
        ingpo_local_value_share: bool = True,
        ingpo_share_pair_budget_fraction: float = 0.25,
        ingpo_share_use_confidence: bool = False,
        ingpo_y_prompt_template: str = DEFAULT_Y_PROMPT_TEMPLATE,
        ingpo_y_temperature: float = 0.7,
        ingpo_y_max_tokens: int = 512,
        ingpo_y_field: str = "answer",  # field on data_instance with gold
        ingpo_score_concurrency: int = 64,
        ingpo_algorithm_mode: str = "budget_allocation",
        ingpo_tv_estimator: str = "subnode",
        ingpo_n_tv_estimates: int = 8,
        ingpo_tv_subnode_max_tokens: int = 120,
        ingpo_tv_second_phase_tokens: int = 60,
        ingpo_tv_includes_half_factor: bool = False,
        ingpo_budget_lambda: float = 0.02,
        ingpo_budget_overhead_mode: str = "flexible",
        ingpo_budget_queue_count: int = 2,
        ingpo_budget_queue_timeout_seconds: float = 0.5,
        ingpo_skip_near_leaf_expand: bool = False,
        ingpo_root_allocation: bool = False,
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
        self.ingpo_local_value_share = bool(ingpo_local_value_share)
        self.ingpo_share_pair_budget_fraction = float(ingpo_share_pair_budget_fraction)
        self.ingpo_share_use_confidence = bool(ingpo_share_use_confidence)
        self.ingpo_y_prompt_template = ingpo_y_prompt_template
        self.ingpo_y_temperature = float(ingpo_y_temperature)
        self.ingpo_y_max_tokens = int(ingpo_y_max_tokens)
        self.ingpo_y_field = ingpo_y_field
        self.ingpo_score_concurrency = int(ingpo_score_concurrency)
        if ingpo_algorithm_mode not in {"share_prune", "budget_allocation"}:
            raise ValueError(
                f"Unsupported ingpo_algorithm_mode: {ingpo_algorithm_mode}"
            )
        if ingpo_budget_overhead_mode not in {"flexible", "none"}:
            raise ValueError(
                f"Unsupported ingpo_budget_overhead_mode: {ingpo_budget_overhead_mode}"
            )
        self.ingpo_algorithm_mode = ingpo_algorithm_mode
        self.ingpo_tv_estimator = ingpo_tv_estimator
        self.ingpo_n_tv_estimates = int(ingpo_n_tv_estimates)
        self.ingpo_tv_subnode_max_tokens = int(ingpo_tv_subnode_max_tokens)
        self.ingpo_tv_second_phase_tokens = int(ingpo_tv_second_phase_tokens)
        self.ingpo_tv_includes_half_factor = bool(ingpo_tv_includes_half_factor)
        self.ingpo_budget_lambda = float(ingpo_budget_lambda)
        self.ingpo_budget_overhead_mode = ingpo_budget_overhead_mode
        self.ingpo_budget_queue_count = int(ingpo_budget_queue_count)
        self.ingpo_budget_queue_timeout_seconds = float(
            ingpo_budget_queue_timeout_seconds
        )
        self.ingpo_skip_near_leaf_expand = bool(ingpo_skip_near_leaf_expand)
        self.ingpo_root_allocation = bool(ingpo_root_allocation)
        self._lp_client: Optional[VLLMLogprobClient] = None

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

    async def _prepare_tree_construction_context(
        self,
        dataset,
        question_format_keys: Sequence[str],
    ) -> Dict[str, Any]:
        if (
            not self.ingpo_root_allocation
            or self.ingpo_algorithm_mode != "budget_allocation"
            or len(dataset) == 0
        ):
            return {}

        client = self._ensure_lp_client()
        scorer = make_lp_scorer(client, self._tokenize)
        tv_estimator = ConditionalTVEstimator(
            scorer=scorer,
            node_expander=self.node_expander,
            gamma=self.cfg_thresholds.gamma,
            mode=self.ingpo_tv_estimator,
            n_tv_estimates=self.ingpo_n_tv_estimates,
            first_phase_tokens=self.ingpo_tv_subnode_max_tokens,
            second_phase_tokens=self.ingpo_tv_second_phase_tokens,
            tv_includes_half_factor=self.ingpo_tv_includes_half_factor,
        )

        root_nodes: List[Node] = []
        root_ids: List[Any] = []
        for data_instance in dataset:
            instance_idx = data_instance["_treetune__idx"]
            format_kwargs = {key: data_instance[key] for key in question_format_keys}
            initial_prompt = self.question_template.format(**format_kwargs)
            root_ids.append(instance_idx)
            root_nodes.append(
                {
                    "text": initial_prompt,
                    "depth": 0,
                    "full_text": initial_prompt,
                    "stop_text": "aaa",
                    "_request_object": data_instance,
                    "leaf": False,
                    "ingpo_action": Action.EXPAND.value,
                    "ingpo_segment_id": str(instance_idx),
                    "ingpo_algorithm_mode": "budget_allocation",
                }
            )

        try:
            base_branch_factor = int(
                self.node_expander.branch_factor_strategy({"depth": 0})
            )
        except Exception:
            base_branch_factor = 1
        total_root_budget = base_branch_factor * len(root_nodes)

        t_var = time.time()
        estimate_results = await asyncio.gather(
            *(tv_estimator.estimate_for_parent(node, depth=0) for node in root_nodes)
        )
        variance_seconds = time.time() - t_var

        for node, result in zip(root_nodes, estimate_results):
            node["ingpo_reward_variance"] = result.reward_variance
            node["ingpo_sigma2"] = result.reward_variance
            node["ingpo_sigma4"] = result.reward_variance * result.reward_variance

        t_alloc = time.time()
        if self.ingpo_budget_overhead_mode == "flexible":
            scheduler = FlexibleBudgetScheduler(
                queue_count=self.ingpo_budget_queue_count,
                lambda_=self.ingpo_budget_lambda,
            )
            summaries = scheduler.allocate(
                root_nodes, total_depth_budget=total_root_budget
            )
            allocations: Dict[str, int] = {}
            weights: Dict[str, float] = {}
            for summary in summaries:
                allocations.update(summary.allocations)
                weights.update(summary.weights)
        else:
            summary = allocate_branch_factors(
                root_nodes,
                total_budget=total_root_budget,
                lambda_=self.ingpo_budget_lambda,
            )
            allocations = summary.allocations
            weights = summary.weights
        allocation_seconds = time.time() - t_alloc
        allocated_root_budget = sum(int(value) for value in allocations.values())
        logger.info(
            "InGPO root_allocation allocated depth-0 budget across %d roots: requested=%d allocated=%d",
            len(root_nodes),
            total_root_budget,
            allocated_root_budget,
        )

        root_allocations: Dict[Any, Dict[str, Any]] = {}
        per_root_variance_seconds = variance_seconds / max(len(root_nodes), 1)
        per_root_allocation_seconds = allocation_seconds / max(len(root_nodes), 1)
        for instance_idx, node, result in zip(root_ids, root_nodes, estimate_results):
            node_id = str(instance_idx)
            unique_candidates: List[Node] = []
            seen_candidate_prefixes = set()
            for sample in result.samples:
                prefix_text = sample.first.get("full_text", "")
                if prefix_text in seen_candidate_prefixes:
                    continue
                seen_candidate_prefixes.add(prefix_text)
                unique_candidates.append(sample.first)
            root_allocations[instance_idx] = {
                "allocated_branch_factor": int(allocations.get(node_id, 0)),
                "budget_weight": float(weights.get(node_id, 0.0)),
                "reward_variance": float(result.reward_variance),
                "tv_pair_count": len(result.pair_tvs),
                "tv_support_size": len(result.samples),
                "budget_candidates": unique_candidates,
                "tv_logp_matrix": result.logp_matrix,
                "variance_seconds": per_root_variance_seconds,
                "allocation_seconds": per_root_allocation_seconds,
                "total_root_budget": total_root_budget,
                "allocated_root_budget": allocated_root_budget,
            }

        return {"root_allocations": root_allocations}

    def _get_tree_construction_kwargs(
        self,
        tree_construction_context: Dict[str, Any],
        instance_idx,
        data_instance: Dict[str, Any],
        initial_prompt: str,
    ) -> Dict[str, Any]:
        root_allocations = tree_construction_context.get("root_allocations") or {}
        root_allocation_info = root_allocations.get(instance_idx)
        if root_allocation_info is None:
            return {}
        return {"root_allocation_info": root_allocation_info}

    # ------------------------------------------------------------------
    # Build per-problem Y
    # ------------------------------------------------------------------

    async def _build_answer_set(
        self,
        problem_id: str,
        problem_text: str,
        gold: str,
    ) -> AnswerSet:
        client = (
            openai.AsyncOpenAI(
                api_key=self._lp_client.api_key,
                base_url=self._lp_client.api_base,
            )
            if self._lp_client is not None
            else None
        )

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
            return await gen.build(
                problem_id=problem_id, problem=problem_text, gold=gold
            )
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
        root_allocation_info: Optional[Dict[str, Any]] = None,
    ):
        if self.ingpo_algorithm_mode == "budget_allocation":
            return await self._construct_budget_allocated_tree(
                initial_prompt=initial_prompt,
                max_depth=max_depth,
                data_instance=data_instance,
                root_allocation_info=root_allocation_info,
            )

        t0_tree = time.time()
        client = self._ensure_lp_client()
        scorer = make_lp_scorer(client, self._tokenize)

        problem_text = data_instance.get("problem") if data_instance else None
        gold = data_instance.get(self.ingpo_y_field) if data_instance else None
        problem_id = (
            str(data_instance.get("_treetune__idx", uuid.uuid4()))
            if data_instance
            else str(uuid.uuid4())
        )

        needs_answer_set = self.ingpo_enable_share and not self.ingpo_local_value_share

        # The sibling-local ValueShare / Prune path does not need Y.  We only
        # build Y for the legacy answer-set ValueShare trigger.
        answer_set: AnswerSet = AnswerSet(problem_id=problem_id, gold=gold or "", y=[])
        y_task = (
            asyncio.create_task(
                self._build_answer_set(problem_id, problem_text or "", gold or "")
            )
            if needs_answer_set
            else None
        )

        engine: Optional[TriggerEngine] = None
        local_shared_count = 0
        local_avg_tv_share = 0.0
        local_pruned_count = 0
        local_avg_gap_prune = 0.0
        branch_factor_by_depth: Dict[int, int] = {}

        for d in range(max_depth):
            try:
                branch_factor_by_depth[d] = int(
                    self.node_expander.branch_factor_strategy({"depth": d})
                )
            except Exception:
                pass

        tree: Node = {
            "text": initial_prompt,
            "depth": 0,
            "full_text": initial_prompt,
            "stop_text": "aaa",
            "_request_object": data_instance,
            "leaf": False,
            "ingpo_action": Action.EXPAND.value,
            "ingpo_segment_id": "root",
        }

        async def _ensure_engine() -> Optional[TriggerEngine]:
            nonlocal engine, answer_set
            if not needs_answer_set or y_task is None:
                return None
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
                enable_share=self.ingpo_enable_share
                and not self.ingpo_local_value_share,
                enable_prune=False,
                share_target=self.ingpo_share_target,
                root_segment_id="root",
            )
            await engine.register_root(initial_prompt)
            return engine

        def _share_eta() -> float:
            if self.cfg_thresholds.eta_override is not None:
                return float(self.cfg_thresholds.eta_override)
            return max(
                self.cfg_thresholds.epsilon / max(self.cfg_thresholds.r_max, 1e-8),
                1e-6,
            )

        def _cheap_share_score(child: Node) -> float:
            val = child.get("ingpo_avg_lp_K")
            if val is not None:
                return float(val)
            return float(len(self._tokenize(child.get("text", ""))))

        def _continuation_texts(node: Node) -> List[str]:
            texts: List[str] = []
            seen = set()
            for child in node.get("children", []) or []:
                text = child.get("text", "")
                if not text or text in seen:
                    continue
                seen.add(text)
                texts.append(text)
                if len(texts) >= self.ingpo_m:
                    break
            # Early-trigger path: before probe expansion, a node may not have
            # any children yet. In that case, fall back to using the node text
            # itself as a single continuation anchor so local comparisons can
            # run without waiting for full sibling probe barriers.
            if not texts:
                own = node.get("text", "")
                if own:
                    texts.append(own)
            return texts

        async def _score_local_tv(
            src: Node,
            tgt: Node,
            continuations: Sequence[str],
        ) -> float:
            src_prefix = src["full_text"]
            tgt_prefix = tgt["full_text"]
            src_scores, tgt_scores = await asyncio.gather(
                asyncio.gather(
                    *(scorer.score_one(src_prefix, c) for c in continuations)
                ),
                asyncio.gather(
                    *(scorer.score_one(tgt_prefix, c) for c in continuations)
                ),
            )
            return sampled_tv_from_logps(src_scores, tgt_scores)

        async def _score_sibling_probs(
            parent: Node, siblings: Sequence[Node]
        ) -> Dict[str, float]:
            if not siblings:
                return {}
            # Reuse sum_logprobs stored during generation when available,
            # falling back to a separate vLLM call only for nodes that lack it.
            raw_scores: List[float] = [0.0] * len(siblings)
            pending: List[tuple] = []
            for idx, child in enumerate(siblings):
                cached = child.get("sum_logprobs")
                if cached is not None:
                    raw_scores[idx] = float(cached)
                else:
                    pending.append((idx, child))
            if pending:
                fetched = await asyncio.gather(
                    *(
                        scorer.score_one(parent["full_text"], child.get("text", ""))
                        for _, child in pending
                    )
                )
                for (idx, _), val in zip(pending, fetched):
                    raw_scores[idx] = val
            probs = stable_softmax(raw_scores)
            return {
                child["ingpo_segment_id"]: float(probs[idx])
                for idx, child in enumerate(siblings)
            }

        async def _try_local_value_share_and_prune(
            parent: Node, siblings: Sequence[Node]
        ) -> None:
            nonlocal local_shared_count, local_avg_tv_share
            nonlocal local_pruned_count, local_avg_gap_prune
            candidates = [
                child
                for child in siblings
                if child.get("ingpo_action") == Action.EXPAND.value
                and not child.get("leaf", False)
            ]
            if len(candidates) < 2:
                return

            budget = pair_budget(
                len(candidates),
                fraction=self.ingpo_share_pair_budget_fraction,
            )
            pairs = select_candidate_pairs(
                [c["ingpo_segment_id"] for c in candidates],
                [_cheap_share_score(c) for c in candidates],
                budget=budget,
            )
            eta = _share_eta()
            pair_tvs: Dict[frozenset, float] = {}

            pair_jobs = []
            for i, j in pairs:
                src = candidates[j]
                tgt = candidates[i]

                # Build pair support from both siblings explicitly: take a
                # portion from src children and a portion from tgt children,
                # then deduplicate while preserving the per-side intent.
                src_texts = _continuation_texts(src)
                tgt_texts = _continuation_texts(tgt)
                per_side = max(1, int(self.ingpo_m // 2))
                src_take = src_texts[:per_side]
                tgt_take = tgt_texts[:per_side]

                continuations = list(src_take) + list(tgt_take)

                if not continuations:
                    continue
                pair_jobs.append((src, tgt, continuations[: max(1, int(self.ingpo_m))]))

            async def _score_pair_tv(
                src: Node, tgt: Node, continuations: Sequence[str]
            ):
                try:
                    tv = await _score_local_tv(src, tgt, continuations)
                except Exception as exc:
                    logger.warning(f"InGPO local ValueShare failed: {exc}")
                    return src, tgt, continuations, None
                return src, tgt, continuations, tv

            scored_pairs = await asyncio.gather(
                *(
                    _score_pair_tv(src, tgt, continuations)
                    for src, tgt, continuations in pair_jobs
                )
            )

            for src, tgt, continuations, tv in scored_pairs:
                if tv is None:
                    continue
                pair_tvs[
                    frozenset((src["ingpo_segment_id"], tgt["ingpo_segment_id"]))
                ] = tv
                radius = confidence_radius(
                    len(continuations), self.cfg_thresholds.alpha
                )
                tv_for_bound = tv + radius if self.ingpo_share_use_confidence else tv
                value_bound = tv_to_value_bound(tv_for_bound, self.cfg_thresholds)
                if src.get("ingpo_action") != Action.EXPAND.value:
                    continue
                if tgt.get("ingpo_action") != Action.EXPAND.value:
                    continue
                if value_bound <= self.cfg_thresholds.epsilon:
                    decision = LocalShareDecision(
                        source_id=src["ingpo_segment_id"],
                        target_id=tgt["ingpo_segment_id"],
                        tv=tv,
                        value_bound=value_bound,
                        n_continuations=len(continuations),
                        confidence_radius=radius,
                        eta_used=eta,
                    )
                    src["ingpo_action"] = Action.SHARE.value
                    src["ingpo_share_target"] = decision.target_id
                    src["ingpo_tv_m"] = decision.tv
                    src["ingpo_share_value_bound"] = decision.value_bound
                    src["ingpo_local_support_size"] = decision.n_continuations
                    src["ingpo_confidence_radius"] = decision.confidence_radius
                    src["ingpo_eta"] = decision.eta_used
                    src["ingpo_tau"] = decision.confidence_radius
                    src["ingpo_local_value_share"] = True
                    src["leaf"] = True
                    src["reward"] = float("nan")
                    src.pop("children", None)
                    if engine is not None:
                        if engine.stats.expanded > 0:
                            engine.stats.expanded -= 1
                        engine.stats.update_share(tv)
                    else:
                        n = local_shared_count + 1
                        local_avg_tv_share = (
                            local_avg_tv_share * local_shared_count + tv
                        ) / n
                        local_shared_count = n

            if not self.ingpo_enable_prune:
                return

            try:
                sibling_probs = await _score_sibling_probs(parent, candidates)
            except Exception as exc:
                logger.warning(f"InGPO local Prune probability scoring failed: {exc}")
                return

            for child in candidates:
                if child.get("ingpo_action") != Action.EXPAND.value:
                    continue
                if child.get("leaf", False):
                    continue
                # Never prune depth-1 nodes.  The root's first branching layer
                # is too early for a stable value comparison.
                if int(child.get("ingpo_depth", 0) or 0) <= 1:
                    continue

                weighted_bound = 0.0
                child_id = child["ingpo_segment_id"]
                for other in candidates:
                    other_id = other["ingpo_segment_id"]
                    if other_id == child_id:
                        continue
                    tv = pair_tvs.get(frozenset((child_id, other_id)), 1.0)
                    weighted_bound += (
                        sibling_probs.get(other_id, 0.0)
                        * self.cfg_thresholds.r_max
                        * tv
                    )

                if weighted_bound < self.cfg_thresholds.epsilon:
                    child["ingpo_action"] = Action.PRUNE.value
                    child["ingpo_share_target"] = None
                    child["ingpo_value_parent_gap_bound"] = weighted_bound
                    child["ingpo_prune_value_eps"] = self.cfg_thresholds.epsilon
                    child["ingpo_prune_policy_prob"] = sibling_probs.get(child_id, 0.0)
                    child["leaf"] = True
                    child["reward"] = float("nan")
                    child.pop("children", None)

                    if engine is not None:
                        if engine.stats.expanded > 0:
                            engine.stats.expanded -= 1
                        engine.stats.update_prune(weighted_bound)
                    else:
                        n = local_pruned_count + 1
                        local_avg_gap_prune = (
                            local_avg_gap_prune * local_pruned_count + weighted_bound
                        ) / n
                        local_pruned_count = n

        def _set_reward_summary(node: Node) -> None:
            child_rewards = []
            for ch in node.get("children", []) or []:
                r = ch.get("reward")
                if r is None or (isinstance(r, float) and math.isnan(r)):
                    continue
                child_rewards.append(r)
            if child_rewards:
                node["reward"] = float(sum(child_rewards) / len(child_rewards))
                node["reward_std"] = float(
                    (
                        sum(
                            (x - (sum(child_rewards) / len(child_rewards))) ** 2
                            for x in child_rewards
                        )
                        / len(child_rewards)
                    )
                    ** 0.5
                )
            else:
                node["reward"] = 0.0
                node["reward_std"] = 0.0

        async def dfs(node: Node, prefix: str, depth: int) -> None:
            if depth == max_depth:
                node["reward"], _ = self.reward_function(
                    query=prefix,
                    response=node["text"],
                    dataset_instance=data_instance,
                )
                node["leaf"] = True
                return

            max_tokens = None if depth == max_depth - 1 else self.M

            children = node.get("children")
            if children is None:
                children = await self.node_expander.expand(
                    current_node=node,
                    prefix=prefix,
                    depth=depth,
                    max_tokens=max_tokens,
                )
            node["children"] = children
            branch_factor_by_depth[depth] = max(
                branch_factor_by_depth.get(depth, 0),
                len(children),
            )

            local_engine = await _ensure_engine()

            # Run local sibling triggers early (before probing every child) so
            # share/prune can stop branches without paying extra probe cost.
            if self.ingpo_local_value_share and (
                self.ingpo_enable_share or self.ingpo_enable_prune
            ):
                await _try_local_value_share_and_prune(node, children)

            probe_tasks = []
            for ch_idx, child in enumerate(children):
                child_seg_id = child.get(
                    "ingpo_segment_id",
                    f"{node.get('ingpo_segment_id', 'root')}/{depth}/{ch_idx}",
                )
                child["ingpo_segment_id"] = child_seg_id
                child["ingpo_action"] = Action.EXPAND.value
                child["ingpo_depth"] = depth + 1
                child["ingpo_parent_segment_id"] = node.get("ingpo_segment_id", "root")
                if child["finish_reason"] != "length":
                    child["reward"], _ = self.reward_function(
                        query=prefix,
                        response=child["full_text"],
                        dataset_instance=data_instance,
                    )
                    child["leaf"] = True
                    if local_engine is not None:
                        try:
                            decision = await local_engine.decide(
                                segment_id=child_seg_id,
                                parent_id=node.get("ingpo_segment_id", "root"),
                                prefix=child["full_text"],
                                is_leaf=True,
                            )
                            self._annotate_node(child, decision)
                        except Exception as exc:
                            logger.warning(f"InGPO decide() failed (leaf): {exc}")
                    continue

                child["leaf"] = False
                if local_engine is None:
                    if depth + 1 < max_depth:
                        probe_tasks.append(
                            (
                                child,
                                asyncio.create_task(
                                    self.node_expander.expand(
                                        current_node=child,
                                        prefix=child["full_text"],
                                        depth=depth + 1,
                                        max_tokens=(
                                            None
                                            if depth + 1 == max_depth - 1
                                            else self.M
                                        ),
                                    )
                                ),
                            )
                        )
                    continue

                try:
                    decision = await local_engine.decide(
                        segment_id=child_seg_id,
                        parent_id=node.get("ingpo_segment_id", "root"),
                        prefix=child["full_text"],
                        is_leaf=False,
                    )
                except Exception as exc:
                    logger.warning(f"InGPO decide() failed: {exc}")
                    if depth + 1 < max_depth:
                        probe_tasks.append(
                            (
                                child,
                                asyncio.create_task(
                                    self.node_expander.expand(
                                        current_node=child,
                                        prefix=child["full_text"],
                                        depth=depth + 1,
                                        max_tokens=(
                                            None
                                            if depth + 1 == max_depth - 1
                                            else self.M
                                        ),
                                    )
                                ),
                            )
                        )
                    continue

                self._annotate_node(child, decision)

                if decision.action is Action.EXPAND:
                    if depth + 1 < max_depth:
                        probe_tasks.append(
                            (
                                child,
                                asyncio.create_task(
                                    self.node_expander.expand(
                                        current_node=child,
                                        prefix=child["full_text"],
                                        depth=depth + 1,
                                        max_tokens=(
                                            None
                                            if depth + 1 == max_depth - 1
                                            else self.M
                                        ),
                                    )
                                ),
                            )
                        )
                elif decision.action is Action.SHARE:
                    # Inherit value from share_target (set later in episode
                    # generator) but mark as a leaf so the tree code stops.
                    child["leaf"] = True
                    child["reward"] = float("nan")  # sentinel; replaced later
                else:  # Action.PRUNE
                    child["leaf"] = True
                    child["reward"] = float("nan")  # sentinel; replaced later

            for child, task in probe_tasks:
                try:
                    probe_children = await task
                except Exception as exc:
                    logger.warning(f"InGPO probe expansion failed: {exc}")
                    continue
                for probe_idx, probe in enumerate(probe_children):
                    probe["ingpo_segment_id"] = (
                        f"{child['ingpo_segment_id']}/{depth + 1}/{probe_idx}"
                    )
                    probe["ingpo_action"] = Action.EXPAND.value
                    probe["ingpo_depth"] = depth + 2
                    probe["ingpo_parent_segment_id"] = child["ingpo_segment_id"]
                child["children"] = probe_children

            expansion_tasks = []
            for child in children:
                if child.get("ingpo_action") == Action.EXPAND.value and not child.get(
                    "leaf", False
                ):
                    expansion_tasks.append(
                        asyncio.create_task(dfs(child, child["full_text"], depth + 1))
                    )

            if expansion_tasks:
                await asyncio.gather(*expansion_tasks)

            # Compute reward / reward_std *after* descendants finish.  For
            # SHARE / PRUNE children, we postpone the value until the episode
            # generator can resolve them; for now use the parent's prior
            # average to keep tree-level statistics finite.
            _set_reward_summary(node)

        await dfs(tree, initial_prompt, 0)

        tree["ingpo_max_depth"] = int(max_depth)
        tree["ingpo_branch_factor_by_depth"] = dict(branch_factor_by_depth)

        # Aggregate stats from the final tree once using the full SPO tree as
        # denominator, so PRUNE/SHARE report how much work they skipped.
        stats = aggregate_tree_stats(
            tree,
            max_depth=max_depth,
            branch_factor_by_depth=branch_factor_by_depth,
        )
        if stats:
            stats["ingpo/avg_tv_when_share"] = (
                engine.stats.avg_tv_share if engine is not None else local_avg_tv_share
            )
            stats["ingpo/avg_gap_when_prune"] = (
                engine.stats.avg_gap_prune
                if engine is not None
                else local_avg_gap_prune
            )
        tree["ingpo_stats"] = stats
        tree["ingpo_answer_set_size"] = answer_set.m
        tree["ingpo_tree_construction_seconds"] = time.time() - t0_tree
        tree["ingpo_problem_id"] = problem_id
        return tree

    async def _construct_budget_allocated_tree(
        self,
        initial_prompt: str,
        max_depth: int = 2,
        data_instance: Optional[Dict[str, Any]] = None,
        root_allocation_info: Optional[Dict[str, Any]] = None,
    ):
        """Construct a tree with simulation-lemma budget allocation.

        This path disables the legacy TV SHARE/PRUNE triggers.  TV estimates
        are used only to compute reward variance at each frontier node, then a
        floor-only branch budget is assigned across nodes.  TV candidate
        prefixes are reused as the real child prefixes when selected.
        """

        t0_tree = time.time()
        client = self._ensure_lp_client()
        scorer = make_lp_scorer(client, self._tokenize)
        problem_id = (
            str(data_instance.get("_treetune__idx", uuid.uuid4()))
            if data_instance
            else str(uuid.uuid4())
        )
        logger.info(
            "InGPO budget-allocation mode active: TV share/prune disabled; "
            "TV used only for reward variance. overhead_mode=%s",
            self.ingpo_budget_overhead_mode,
        )

        tree: Node = {
            "text": initial_prompt,
            "depth": 0,
            "full_text": initial_prompt,
            "stop_text": "aaa",
            "_request_object": data_instance,
            "leaf": False,
            "ingpo_action": Action.EXPAND.value,
            "ingpo_segment_id": "root",
            "ingpo_algorithm_mode": "budget_allocation",
        }

        tv_estimator = ConditionalTVEstimator(
            scorer=scorer,
            node_expander=self.node_expander,
            gamma=self.cfg_thresholds.gamma,
            mode=self.ingpo_tv_estimator,
            n_tv_estimates=self.ingpo_n_tv_estimates,
            first_phase_tokens=self.ingpo_tv_subnode_max_tokens,
            second_phase_tokens=self.ingpo_tv_second_phase_tokens,
            tv_includes_half_factor=self.ingpo_tv_includes_half_factor,
        )

        branch_factor_by_depth: Dict[int, int] = {}
        requested_by_depth: Dict[int, int] = {}
        allocated_by_depth: Dict[int, int] = {}
        built_by_depth: Dict[int, int] = {}
        underallocated_by_depth: Dict[int, int] = {}
        variance_seconds_by_depth: Dict[int, float] = {}
        allocation_seconds_by_depth: Dict[int, float] = {}
        expansion_seconds_by_depth: Dict[int, float] = {}

        async def _expand_with_budget(
            *,
            current_node: Node,
            prefix: str,
            depth: int,
            max_tokens: Optional[int],
            branch_factor: int,
        ) -> List[Node]:
            if branch_factor <= 0:
                return []
            try:
                return await self.node_expander.expand(
                    current_node=current_node,
                    prefix=prefix,
                    depth=depth,
                    max_tokens=max_tokens,
                    branch_factor=branch_factor,
                )
            except TypeError:
                logger.warning(
                    "Node expander does not accept branch_factor override; falling back to configured strategy."
                )
                return await self.node_expander.expand(
                    current_node=current_node,
                    prefix=prefix,
                    depth=depth,
                    max_tokens=max_tokens,
                )

        async def _complete_candidate(
            parent: Node, candidate: Node, depth: int, child_idx: int
        ) -> Node:
            child = dict(candidate)
            child["ingpo_segment_id"] = (
                f"{parent.get('ingpo_segment_id', 'root')}/{depth}/{child_idx}"
            )
            child["ingpo_parent_segment_id"] = parent.get("ingpo_segment_id", "root")
            child["ingpo_depth"] = depth + 1
            child["ingpo_action"] = Action.EXPAND.value
            child["depth"] = depth + 1
            child["leaf"] = False

            finish_reason = child.get("finish_reason")
            if finish_reason != "length" or depth + 1 >= max_depth:
                child["reward"], _ = self.reward_function(
                    query=parent.get("full_text", ""),
                    response=child.get("full_text", child.get("text", "")),
                    dataset_instance=data_instance,
                )
                child["leaf"] = True
                return child

            continuation_budget = self.M
            if self.ingpo_tv_subnode_max_tokens > 0:
                continuation_budget = max(
                    int(self.M) - self.ingpo_tv_subnode_max_tokens, 1
                )
            continuations = await _expand_with_budget(
                current_node=child,
                prefix=child.get("full_text", ""),
                depth=depth + 1,
                max_tokens=continuation_budget,
                branch_factor=1,
            )
            if not continuations:
                logger.warning(
                    "Budget candidate continuation produced no nodes; marking %s as leaf",
                    child.get("ingpo_segment_id"),
                )
                child["reward"], _ = self.reward_function(
                    query=parent.get("full_text", ""),
                    response=child.get("full_text", child.get("text", "")),
                    dataset_instance=data_instance,
                )
                child["leaf"] = True
                return child

            cont = continuations[0]
            child["text"] = child.get("text", "") + cont.get("text", "")
            child["full_text"] = cont.get("full_text", child.get("full_text", ""))
            child["finish_reason"] = cont.get(
                "finish_reason", child.get("finish_reason")
            )
            child["stop_text"] = cont.get("stop_text", child.get("stop_text"))
            if (
                child.get("sum_logprobs") is not None
                and cont.get("sum_logprobs") is not None
            ):
                child["sum_logprobs"] = float(child["sum_logprobs"]) + float(
                    cont["sum_logprobs"]
                )
            if (
                child.get("num_tokens") is not None
                and cont.get("num_tokens") is not None
            ):
                child["num_tokens"] = int(child["num_tokens"]) + int(cont["num_tokens"])
            return child

        def _set_reward_summary(node: Node) -> None:
            child_rewards = []
            for ch in node.get("children", []) or []:
                r = ch.get("reward")
                if r is None or (isinstance(r, float) and math.isnan(r)):
                    continue
                child_rewards.append(r)
            if child_rewards:
                node["reward"] = float(sum(child_rewards) / len(child_rewards))
                node["reward_std"] = float(
                    (
                        sum(
                            (x - (sum(child_rewards) / len(child_rewards))) ** 2
                            for x in child_rewards
                        )
                        / len(child_rewards)
                    )
                    ** 0.5
                )
            else:
                node["reward"] = 0.0
                node["reward_std"] = 0.0

        frontier: List[Node] = [tree]
        for depth in range(max_depth):
            expandable = [node for node in frontier if not node.get("leaf", False)]
            if not expandable:
                break

            try:
                base_branch_factor = int(
                    self.node_expander.branch_factor_strategy({"depth": depth})
                )
            except Exception:
                base_branch_factor = 1
            total_depth_budget = base_branch_factor * len(expandable)
            branch_factor_by_depth[depth] = base_branch_factor
            requested_by_depth[depth] = total_depth_budget

            if root_allocation_info is not None and depth == 0:
                variance_seconds_by_depth[depth] = float(
                    root_allocation_info.get("variance_seconds", 0.0)
                )
                allocation_seconds_by_depth[depth] = float(
                    root_allocation_info.get("allocation_seconds", 0.0)
                )
                allocated = int(root_allocation_info.get("allocated_branch_factor", 0))
                allocated_by_depth[depth] = allocated
                underallocated_by_depth[depth] = max(total_depth_budget - allocated, 0)

                tree["ingpo_reward_variance"] = float(
                    root_allocation_info.get("reward_variance", 0.0)
                )
                tree["ingpo_sigma2"] = tree["ingpo_reward_variance"]
                tree["ingpo_sigma4"] = tree["ingpo_sigma2"] * tree["ingpo_sigma2"]
                tree["ingpo_tv_pair_count"] = root_allocation_info.get(
                    "tv_pair_count", 0
                )
                tree["ingpo_tv_support_size"] = root_allocation_info.get(
                    "tv_support_size", 0
                )
                tree["ingpo_budget_candidates"] = list(
                    root_allocation_info.get("budget_candidates", [])
                )
                tree["ingpo_tv_logp_matrix"] = root_allocation_info.get(
                    "tv_logp_matrix", []
                )
                tree["ingpo_root_requested_minibatch_budget"] = int(
                    root_allocation_info.get("total_root_budget", total_depth_budget)
                )
                tree["ingpo_root_allocated_minibatch_budget"] = int(
                    root_allocation_info.get("allocated_root_budget", allocated)
                )
                allocations = {"root": allocated}
                weights = {
                    "root": float(root_allocation_info.get("budget_weight", 0.0))
                }
            elif self.ingpo_skip_near_leaf_expand and depth == max_depth - 1:
                # The final expansion depth is the most likely place to run out
                # of context (e.g. TV second-phase max_tokens being clipped to a
                # tiny value).  Optionally skip TV/budget allocation here and
                # fall back to SPO-style uniform expansion with branch factor B.
                variance_seconds_by_depth[depth] = 0.0
                allocation_seconds_by_depth[depth] = 0.0
                allocated_by_depth[depth] = total_depth_budget
                underallocated_by_depth[depth] = 0

                t_expand = time.time()
                built_total = 0
                for node in expandable:
                    node["ingpo_allocated_branch_factor"] = base_branch_factor
                    node["ingpo_budget_weight"] = 1.0
                    node["ingpo_budget_candidates"] = []
                    children = await _expand_with_budget(
                        current_node=node,
                        prefix=node.get("full_text", ""),
                        depth=depth,
                        max_tokens=None,
                        branch_factor=base_branch_factor,
                    )
                    for child_idx, child in enumerate(children):
                        child["ingpo_segment_id"] = (
                            f"{node.get('ingpo_segment_id', 'root')}/{depth}/{child_idx}"
                        )
                        child["ingpo_parent_segment_id"] = node.get(
                            "ingpo_segment_id", "root"
                        )
                        child["ingpo_depth"] = depth + 1
                        child["ingpo_action"] = Action.EXPAND.value
                        child["depth"] = depth + 1
                        child["reward"], _ = self.reward_function(
                            query=node.get("full_text", ""),
                            response=child.get("full_text", child.get("text", "")),
                            dataset_instance=data_instance,
                        )
                        child["leaf"] = True
                    node["children"] = children
                    node["ingpo_discarded_budget_candidates"] = 0
                    built_total += len(children)
                    _set_reward_summary(node)
                expansion_seconds_by_depth[depth] = time.time() - t_expand
                built_by_depth[depth] = built_total
                frontier = []
                continue

            if root_allocation_info is None or depth != 0:
                t_var = time.time()
                estimate_tasks = [
                    asyncio.create_task(
                        tv_estimator.estimate_for_parent(node, depth=depth)
                    )
                    for node in expandable
                ]
                estimate_results = await asyncio.gather(*estimate_tasks)
                variance_seconds_by_depth[depth] = time.time() - t_var

                for node, result in zip(expandable, estimate_results):
                    node["ingpo_reward_variance"] = result.reward_variance
                    node["ingpo_sigma2"] = result.reward_variance
                    node["ingpo_sigma4"] = (
                        result.reward_variance * result.reward_variance
                    )
                    node["ingpo_tv_pair_count"] = len(result.pair_tvs)
                    node["ingpo_tv_support_size"] = len(result.samples)
                    unique_candidates: List[Node] = []
                    seen_candidate_prefixes = set()
                    for sample in result.samples:
                        prefix_text = sample.first.get("full_text", "")
                        if prefix_text in seen_candidate_prefixes:
                            continue
                        seen_candidate_prefixes.add(prefix_text)
                        unique_candidates.append(sample.first)
                    node["ingpo_budget_candidates"] = unique_candidates
                    # Keep the cached matrix available for debugging without recomputing P(ss_k2 | ss_i1).
                    node["ingpo_tv_logp_matrix"] = result.logp_matrix

                t_alloc = time.time()
                if self.ingpo_budget_overhead_mode == "flexible":
                    scheduler = FlexibleBudgetScheduler(
                        queue_count=self.ingpo_budget_queue_count,
                        lambda_=self.ingpo_budget_lambda,
                    )
                    summaries = scheduler.allocate(
                        expandable, total_depth_budget=total_depth_budget
                    )
                    allocations: Dict[str, int] = {}
                    weights: Dict[str, float] = {}
                    allocated_total = 0
                    underallocated_total = 0
                    for summary in summaries:
                        allocations.update(summary.allocations)
                        weights.update(summary.weights)
                        allocated_total += summary.allocated_budget
                        underallocated_total += summary.underallocated_budget
                else:
                    summary = allocate_branch_factors(
                        expandable,
                        total_budget=total_depth_budget,
                        lambda_=self.ingpo_budget_lambda,
                    )
                    allocations = summary.allocations
                    weights = summary.weights
                    allocated_total = summary.allocated_budget
                    underallocated_total = summary.underallocated_budget
                allocation_seconds_by_depth[depth] = time.time() - t_alloc
                allocated_by_depth[depth] = allocated_total
                underallocated_by_depth[depth] = underallocated_total

            t_expand = time.time()
            next_frontier: List[Node] = []
            built_total = 0
            for node in expandable:
                node_id = str(node.get("ingpo_segment_id", "root"))
                allocated = int(allocations.get(node_id, 0))
                node["ingpo_allocated_branch_factor"] = allocated
                node["ingpo_budget_weight"] = float(weights.get(node_id, 0.0))
                candidates = sorted(
                    node.get("ingpo_budget_candidates", []),
                    key=lambda cand: float(cand.get("sum_logprobs", 0.0) or 0.0),
                    reverse=True,
                )
                selected = candidates[:allocated]
                if len(selected) < allocated:
                    extra_candidates = await _expand_with_budget(
                        current_node=node,
                        prefix=node.get("full_text", ""),
                        depth=depth,
                        max_tokens=self.ingpo_tv_subnode_max_tokens,
                        branch_factor=allocated - len(selected),
                    )
                    selected.extend(extra_candidates)
                completion_tasks = [
                    asyncio.create_task(_complete_candidate(node, cand, depth, idx))
                    for idx, cand in enumerate(selected)
                ]
                children = (
                    await asyncio.gather(*completion_tasks) if completion_tasks else []
                )
                node["children"] = children
                built_total += len(children)
                node["ingpo_discarded_budget_candidates"] = max(
                    len(node.get("ingpo_budget_candidates", [])) - len(selected),
                    0,
                )
                next_frontier.extend(
                    [child for child in children if not child.get("leaf", False)]
                )
                _set_reward_summary(node)
            expansion_seconds_by_depth[depth] = time.time() - t_expand
            built_by_depth[depth] = built_total
            frontier = next_frontier

        tree["ingpo_max_depth"] = int(max_depth)
        tree["ingpo_branch_factor_by_depth"] = dict(branch_factor_by_depth)
        tree["ingpo_requested_node_budget_by_depth"] = dict(requested_by_depth)
        tree["ingpo_allocated_branch_factor_by_depth"] = dict(allocated_by_depth)
        tree["ingpo_built_nodes_by_depth"] = dict(built_by_depth)
        tree["ingpo_underallocated_rollouts_by_depth"] = dict(underallocated_by_depth)
        tree["ingpo_variance_seconds_by_depth"] = dict(variance_seconds_by_depth)
        tree["ingpo_allocation_seconds_by_depth"] = dict(allocation_seconds_by_depth)
        tree["ingpo_expansion_seconds_by_depth"] = dict(expansion_seconds_by_depth)
        tree["ingpo_budget_overhead_mode"] = self.ingpo_budget_overhead_mode
        tree["ingpo_skip_near_leaf_expand"] = self.ingpo_skip_near_leaf_expand
        tree["ingpo_root_allocation"] = self.ingpo_root_allocation
        tree["ingpo_answer_set_size"] = 0
        tree_construction_seconds = time.time() - t0_tree
        tree["tree_construction_seconds"] = tree_construction_seconds
        tree["ingpo_tree_construction_seconds"] = tree_construction_seconds
        tree["ingpo_problem_id"] = problem_id
        tree["ingpo_stats"] = {
            **aggregate_tree_stats(tree),
            "ingpo/budget/underallocated_node_budget": float(
                sum(underallocated_by_depth.values())
            ),
            "ingpo/variance_estimation_seconds": float(
                sum(variance_seconds_by_depth.values())
            ),
            "ingpo/budget_allocation_seconds": float(
                sum(allocation_seconds_by_depth.values())
            ),
            "ingpo/expansion_seconds": float(sum(expansion_seconds_by_depth.values())),
        }
        return tree

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
