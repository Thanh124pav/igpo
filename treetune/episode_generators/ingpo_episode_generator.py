"""InGPO episode generator.

Subclasses `HybridEpisodeGenerator` from SPO and only overrides
`extract_edges_from_tree` so we can:

  * skip pruned children (no edge — segment contributes nothing to PPO),
  * for shared children, inherit `value` / `reward` from the share target
    instead of using the segment's own (NaN) reward,
  * carry through `ingpo_action`, `ingpo_share_target`, `ingpo_tv_m`,
    `ingpo_gap_m` so downstream metric loggers can quote them.

Everything else — replay buffer, _add_logprobs_to_edges, PPO collation —
is inherited unchanged so the trainer behaviour is identical to SPO.

The corresponding `*_strategy` registration lets configs use
`type: 'ingpo_episode_generator'`.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from treetune.episode_generators import EpisodeGenerator
from treetune.episode_generators.hybrid_episode_generator import (
    HybridEpisodeGenerator,
)
from treetune.logging_utils import get_logger

logger = get_logger(__name__)


from treetune.ingpo.logging_helpers import (
    BUDGET_DEMO_COLUMNS,
    DEMO_COLUMNS,
    collect_demo_rows,
    per_depth_action_counts,
    render_md_section,
    to_jsonl_record,
)


@EpisodeGenerator.register("ingpo_episode_generator")
class InGPOEpisodeGenerator(HybridEpisodeGenerator):
    """Tree → edges with online Share/Prune awareness."""

    def __init__(
        self,
        ingpo_zero_advantage_when_pruned: bool = True,
        ingpo_emit_pruned_edges: bool = False,
        ingpo_share_inherit: str = "value_and_reward",  # or "value_only"
        ingpo_demo_examples_per_tree: int = 2,  # how many SHARE / PRUNE demos to log per tree
        ingpo_demos_dir: Optional[
            str
        ] = None,  # absolute path; else exp_root/ingpo_demos
        ingpo_log_demos_to_wandb: bool = False,  # for offline servers, default off
        ingpo_log_reward_variance_nodes: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ingpo_zero_advantage_when_pruned = ingpo_zero_advantage_when_pruned
        self.ingpo_emit_pruned_edges = ingpo_emit_pruned_edges
        self.ingpo_share_inherit = ingpo_share_inherit
        self.ingpo_demo_examples_per_tree = int(ingpo_demo_examples_per_tree)
        self.ingpo_log_demos_to_wandb = bool(ingpo_log_demos_to_wandb)
        self.ingpo_log_reward_variance_nodes = bool(ingpo_log_reward_variance_nodes)
        # Where to dump local-file demos. Resolves on first use because
        # exp_root is only set after super().__init__ on some SPO branches.
        self._ingpo_demos_dir_override = ingpo_demos_dir
        self._ingpo_demos_dir_resolved = None  # type: Optional[Any]
        self._ingpo_jsonl_handle = None
        self._ingpo_md_handle = None
        self._ingpo_reward_variance_jsonl_handle = None
        self._ingpo_reward_variance_csv_handle = None
        self._ingpo_reward_variance_csv_writer = None
        self._tree_seen = 0

    # ------------------------------------------------------------------
    # Override edge extraction
    # ------------------------------------------------------------------

    def extract_edges_from_tree(
        self,
        tree,
        adv_method: str = "rloo",
        only_adv_greater_than_zero: bool = True,
        use_hard_estimation: bool = False,
    ) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        data_instance = tree["_request_object"]
        question_id = data_instance["_treetune__idx"]

        # Index every node so SHARE children can dereference their target.
        index_by_seg_id: Dict[str, Dict[str, Any]] = {}

        def collect(node):
            seg_id = node.get("ingpo_segment_id")
            if seg_id is not None:
                index_by_seg_id[seg_id] = node
            for ch in node.get("children", []) or []:
                collect(ch)

        tree_copy = copy.deepcopy(tree)
        collect(tree_copy)

        ingpo_stats = tree_copy.get("ingpo_stats", {})
        per_depth = self._per_depth_action_counts(tree_copy)
        demo_rows = collect_demo_rows(
            tree_copy,
            index_by_seg_id,
            question_id=question_id,
            n_each=max(self.ingpo_demo_examples_per_tree, 0),
        )
        self._tree_seen += 1

        # ---- Local-file demo dump (works offline, no wandb required) -----
        self._dump_demos_to_disk(
            tree_idx=self._tree_seen,
            question_id=question_id,
            stats=ingpo_stats,
            per_depth=per_depth,
            demo_rows=demo_rows,
            tree_construction_seconds=tree_copy.get(
                "tree_construction_seconds",
                tree_copy.get("ingpo_tree_construction_seconds"),
            ),
        )

        # ---- Optional wandb scalar+table logging --------------------------
        if ingpo_stats or per_depth:
            reward_variance_summary = self._summarize_reward_variance_nodes(tree_copy)
            log_entry = {
                **ingpo_stats,
                "ingpo/tree_idx": self._tree_seen,
                **reward_variance_summary,
                **per_depth,
                **(
                    {"ingpo/n_budget_demos_in_tree": len(demo_rows.get("budget", []))}
                    if tree_copy.get("ingpo_algorithm_mode") == "budget_allocation"
                    else {
                        "ingpo/n_share_demos_in_tree": len(demo_rows.get("share", [])),
                        "ingpo/n_prune_demos_in_tree": len(demo_rows.get("prune", [])),
                    }
                ),
            }
            if self.ingpo_log_demos_to_wandb:
                table = self._maybe_build_wandb_table(demo_rows)
                if table is not None:
                    log_entry["ingpo/demos"] = table
            self._cloud_log(log_entry)

        self._dump_reward_variance_nodes_to_disk(
            tree=tree_copy,
            tree_idx=self._tree_seen,
            question_id=question_id,
        )

        def BoK(value, bok=4):
            return 1 - (1 - value) ** bok

        def resolve_reward(node, fallback_parent_reward: float) -> Optional[float]:
            r = node.get("reward")
            if r is None:
                return None
            if isinstance(r, float) and np.isnan(r):
                action = node.get("ingpo_action")
                if action == "share":
                    target_id = node.get("ingpo_share_target")
                    if target_id is not None and target_id in index_by_seg_id:
                        target_r = index_by_seg_id[target_id].get("reward")
                        if target_r is not None and not (
                            isinstance(target_r, float) and np.isnan(target_r)
                        ):
                            return float(target_r)
                    return float(fallback_parent_reward)
                if action == "prune":
                    eps = node.get("ingpo_prune_value_eps")
                    if eps is not None:
                        key = str(node.get("ingpo_segment_id", node.get("text", "")))
                        digest = hashlib.sha256(key.encode("utf-8")).digest()
                        u = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
                        return float(fallback_parent_reward) + (2.0 * u - 1.0) * float(
                            eps
                        )
                    return float(fallback_parent_reward)
                return None
            return float(r)

        def dfs(node, parent=None):
            if parent is not None:
                query_text = parent["full_text"]
                response_text = node["text"]
                parent_reward = parent.get("reward", 0.0) or 0.0
                parent_reward_std = parent.get("reward_std", 0.0) or 0.0

                child_reward = resolve_reward(node, parent_reward)
                if child_reward is None:
                    # Unable to resolve; skip edge to avoid corrupting PPO.
                    pass
                else:
                    leaf = node.get("leaf", False)
                    ingpo_action = node.get("ingpo_action", "expand")
                    is_pruned = ingpo_action == "prune"
                    is_shared = ingpo_action == "share"

                    if is_pruned and not self.ingpo_emit_pruned_edges:
                        # Drop the edge entirely - PPO does not see it.
                        pass
                    else:
                        if adv_method == "rloo":
                            advantage = child_reward - parent_reward
                        elif adv_method == "grpo":
                            advantage = (child_reward - parent_reward) / (
                                parent_reward_std + 1e-8
                            )
                        else:
                            raise ValueError(
                                f"adv_method {adv_method} is not supported"
                            )

                        if is_pruned and self.ingpo_zero_advantage_when_pruned:
                            advantage = 0.0

                        prover_advantage = BoK(child_reward) - BoK(parent_reward)
                        pav_advantage = advantage + prover_advantage

                        keep = True
                        if (
                            only_adv_greater_than_zero
                            and pav_advantage == 0
                            and not is_pruned
                        ):
                            keep = False

                        if keep and len(response_text) > 0:
                            edges.append(
                                {
                                    "question_id": question_id,
                                    "instance": data_instance,
                                    "query_text": query_text,
                                    "response_text": response_text,
                                    "advantage": advantage,
                                    "prover_advantage": prover_advantage,
                                    "value": child_reward,
                                    "leaf": leaf,
                                    "reward": child_reward,
                                    "ingpo_action": ingpo_action,
                                    "ingpo_share_target": node.get(
                                        "ingpo_share_target"
                                    ),
                                    "ingpo_tv_m": node.get("ingpo_tv_m"),
                                    "ingpo_gap_m": node.get("ingpo_gap_m"),
                                }
                            )

            for child in node.get("children", []) or []:
                dfs(child, node)
            node.pop("children", None)

        dfs(tree_copy)
        # Round-trip through json so downstream Datasets.from_list works
        # (matches SPO behaviour).
        edges = json.loads(json.dumps(edges, default=lambda o: None))
        return edges

    # ------------------------------------------------------------------
    # Logging helpers (thin wrappers around module-level pure helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _per_depth_action_counts(tree) -> Dict[str, float]:
        return per_depth_action_counts(tree)

    def _maybe_build_wandb_table(self, demo_rows):
        budget_rows = demo_rows.get("budget", [])
        share_prune_rows = demo_rows.get("share", []) + demo_rows.get("prune", [])
        if not (budget_rows or share_prune_rows):
            return None
        try:
            import wandb  # type: ignore
        except ImportError:
            return None
        if budget_rows:
            table = wandb.Table(columns=BUDGET_DEMO_COLUMNS)
            for r in budget_rows:
                table.add_data(*r)
            return table
        table = wandb.Table(columns=DEMO_COLUMNS)
        for r in share_prune_rows:
            table.add_data(*r)
        return table

    # ------------------------------------------------------------------
    # Offline-friendly file dump
    # ------------------------------------------------------------------

    def _resolve_demos_dir(self):
        if self._ingpo_demos_dir_resolved is not None:
            return self._ingpo_demos_dir_resolved

        from pathlib import Path

        if self._ingpo_demos_dir_override:
            base = Path(self._ingpo_demos_dir_override)
        elif getattr(self, "exp_root", None) is not None:
            base = Path(self.exp_root) / "ingpo_demos"
        else:
            base = Path.cwd() / "ingpo_demos"

        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"InGPO: could not create demos dir {base}: {exc}")
            self._ingpo_demos_dir_resolved = False
            return False
        self._ingpo_demos_dir_resolved = base
        return base

    def _open_demo_handles(self):
        base = self._resolve_demos_dir()
        if base is False:
            return None, None
        if self._ingpo_jsonl_handle is None:
            self._ingpo_jsonl_handle = (base / "demos.jsonl").open("a", buffering=1)
        if self._ingpo_md_handle is None:
            self._ingpo_md_handle = (base / "demos.md").open("a", buffering=1)
            if self._ingpo_md_handle.tell() == 0:
                self._ingpo_md_handle.write(
                    "# InGPO SHARE / PRUNE demos\n\n"
                    "One section per tree. `tail -F demos.md` to follow live.\n\n"
                )
        return self._ingpo_jsonl_handle, self._ingpo_md_handle

    def _open_reward_variance_handles(self):
        base = self._resolve_demos_dir()
        if base is False:
            return None, None
        if self._ingpo_reward_variance_jsonl_handle is None:
            self._ingpo_reward_variance_jsonl_handle = (
                base / "reward_variance_nodes.jsonl"
            ).open("a", buffering=1)
        if self._ingpo_reward_variance_csv_handle is None:
            csv_path = base / "reward_variance_nodes.csv"
            needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
            self._ingpo_reward_variance_csv_handle = csv_path.open(
                "a", buffering=1, newline=""
            )
            self._ingpo_reward_variance_csv_writer = csv.DictWriter(
                self._ingpo_reward_variance_csv_handle,
                fieldnames=self._reward_variance_fieldnames(),
                extrasaction="ignore",
            )
            if needs_header:
                self._ingpo_reward_variance_csv_writer.writeheader()
        return (
            self._ingpo_reward_variance_jsonl_handle,
            self._ingpo_reward_variance_csv_writer,
        )

    @staticmethod
    def _reward_variance_fieldnames() -> List[str]:
        return [
            "tree_idx",
            "question_id",
            "depth",
            "seg_id",
            "parent_seg_id",
            "action",
            "reward",
            "reward_std",
            "empirical_child_reward_variance",
            "ingpo_reward_variance",
            "ingpo_sigma2",
            "ingpo_sigma4",
            "ingpo_tv_pair_count",
            "ingpo_tv_support_size",
            "ingpo_allocated_branch_factor",
            "ingpo_budget_weight",
            "ingpo_discarded_budget_candidates",
            "n_children",
        ]

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _iter_reward_variance_rows(self, tree, tree_idx: int, question_id):
        stack = [tree]
        while stack:
            node = stack.pop()
            children = node.get("children") or []
            child_rewards = [
                self._safe_float(ch.get("reward"))
                for ch in children
                if self._safe_float(ch.get("reward")) is not None
            ]
            empirical_var = None
            if child_rewards:
                mean = sum(child_rewards) / len(child_rewards)
                empirical_var = sum((r - mean) ** 2 for r in child_rewards) / len(
                    child_rewards
                )
            row = {
                "tree_idx": tree_idx,
                "question_id": question_id,
                "depth": node.get("ingpo_depth", node.get("depth")),
                "seg_id": node.get("ingpo_segment_id"),
                "parent_seg_id": node.get("ingpo_parent_segment_id"),
                "action": node.get("ingpo_action"),
                "reward": self._safe_float(node.get("reward")),
                "reward_std": self._safe_float(node.get("reward_std")),
                "empirical_child_reward_variance": empirical_var,
                "ingpo_reward_variance": self._safe_float(
                    node.get("ingpo_reward_variance")
                ),
                "ingpo_sigma2": self._safe_float(node.get("ingpo_sigma2")),
                "ingpo_sigma4": self._safe_float(node.get("ingpo_sigma4")),
                "ingpo_tv_pair_count": node.get("ingpo_tv_pair_count"),
                "ingpo_tv_support_size": node.get("ingpo_tv_support_size"),
                "ingpo_allocated_branch_factor": node.get(
                    "ingpo_allocated_branch_factor"
                ),
                "ingpo_budget_weight": self._safe_float(
                    node.get("ingpo_budget_weight")
                ),
                "ingpo_discarded_budget_candidates": node.get(
                    "ingpo_discarded_budget_candidates"
                ),
                "n_children": len(children),
            }
            if row["ingpo_reward_variance"] is not None or row["reward"] is not None:
                yield row
            stack.extend(reversed(children))

    def _summarize_reward_variance_nodes(self, tree) -> Dict[str, float]:
        rows = list(
            self._iter_reward_variance_rows(
                tree=tree,
                tree_idx=self._tree_seen,
                question_id="",
            )
        )
        variances = [
            r["ingpo_reward_variance"]
            for r in rows
            if r["ingpo_reward_variance"] is not None
        ]
        rewards = [r["reward"] for r in rows if r["reward"] is not None]
        out: Dict[str, float] = {
            "ingpo/reward_variance_nodes/n": float(len(rows)),
            "ingpo/reward_variance_nodes/n_with_variance": float(len(variances)),
        }
        if variances:
            out["ingpo/reward_variance_nodes/mean_sigma2"] = float(
                sum(variances) / len(variances)
            )
            out["ingpo/reward_variance_nodes/max_sigma2"] = float(max(variances))
        if rewards:
            out["ingpo/reward_variance_nodes/mean_reward"] = float(
                sum(rewards) / len(rewards)
            )
        return out

    def _dump_reward_variance_nodes_to_disk(
        self,
        tree,
        tree_idx: int,
        question_id,
    ) -> None:
        if not self.ingpo_log_reward_variance_nodes:
            return

        jsonl, csv_writer = self._open_reward_variance_handles()
        if jsonl is None or csv_writer is None:
            return

        for row in self._iter_reward_variance_rows(tree, tree_idx, question_id):
            try:
                jsonl.write(json.dumps(row, default=str) + "\n")
                csv_writer.writerow(row)
            except Exception as exc:
                logger.warning(
                    f"InGPO: failed to append reward variance node record: {exc}"
                )
                return

    def _dump_demos_to_disk(
        self,
        tree_idx: int,
        question_id,
        stats: Dict[str, Any],
        per_depth: Dict[str, float],
        demo_rows: Dict[str, List[List[Any]]],
        tree_construction_seconds: Optional[float] = None,
    ) -> None:
        if max(self.ingpo_demo_examples_per_tree, 0) == 0 and not stats:
            return

        jsonl, md = self._open_demo_handles()
        if jsonl is None:
            return  # silent: never block training because logging failed

        # JSONL row: machine-readable. One line per tree.
        record = to_jsonl_record(
            tree_idx=tree_idx,
            question_id=question_id,
            stats=stats,
            per_depth=per_depth,
            demo_rows=demo_rows,
            tree_construction_seconds=tree_construction_seconds,
        )
        try:
            jsonl.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.warning(f"InGPO: failed to append demos.jsonl: {exc}")

        # Markdown: human-readable. One section per tree, only if there's
        # something interesting (rates or demos) to show.
        if not (
            demo_rows.get("share") or demo_rows.get("prune") or demo_rows.get("budget")
        ):
            return
        try:
            md.write(render_md_section(tree_idx, question_id, stats, demo_rows))
        except Exception as exc:
            logger.warning(f"InGPO: failed to append demos.md: {exc}")

    def __del__(self):
        for h in (
            self._ingpo_jsonl_handle,
            self._ingpo_md_handle,
            self._ingpo_reward_variance_jsonl_handle,
            self._ingpo_reward_variance_csv_handle,
        ):
            try:
                if h is not None:
                    h.close()
            except Exception:
                pass
