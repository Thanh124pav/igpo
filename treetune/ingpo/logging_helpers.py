"""Pure helpers used by the InGPO episode generator's wandb logging.

Lives in `core/` (no SPO deps) so it can be unit-tested without installing
the full SPO Python stack.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


_DEMO_TEXT_TRUNC = None


DEMO_COLUMNS = [
    "question_id", "action", "depth", "seg_id",
    "parent_text", "child_text", "target_text", "target_seg_id",
    "avg_lp_K", "tv_m", "gap_m", "eta", "tau",
]


def truncate(s: Optional[str], n: Optional[int] = _DEMO_TEXT_TRUNC) -> str:
    if not s:
        return ""
    s = s.replace("\n", " \\n ")
    if n is None:
        return s
    return s if len(s) <= n else s[: n - 3] + "..."


def _coerce_depth_map(value: Optional[Dict[Any, Any]]) -> Dict[int, int]:
    if not value:
        return {}
    out: Dict[int, int] = {}
    for k, v in value.items():
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _infer_max_depth(tree) -> int:
    max_depth = 0
    stack = [tree]
    while stack:
        n = stack.pop()
        d = n.get("ingpo_depth")
        if isinstance(d, int) and d > max_depth:
            max_depth = d
        stack.extend(n.get("children") or [])
    return max_depth


def _infer_branch_factors(tree) -> Dict[int, int]:
    widths: Dict[int, int] = {}
    stack = [tree]
    while stack:
        n = stack.pop()
        d = n.get("ingpo_depth")
        children = n.get("children") or []
        if isinstance(d, int) and children:
            widths[d] = max(widths.get(d, 0), len(children))
        stack.extend(children)
    return widths


def _branch_factor_at(branch_factors: Dict[int, int], depth: int) -> int:
    if depth in branch_factors:
        return max(int(branch_factors[depth]), 0)
    if not branch_factors:
        return 0
    earlier = [d for d in branch_factors if d <= depth]
    if earlier:
        return max(int(branch_factors[max(earlier)]), 0)
    return max(int(branch_factors[min(branch_factors)]), 0)


def _subtree_size_from_depth(depth: int, max_depth: int, branch_factors: Dict[int, int]) -> int:
    if depth <= 0 or depth > max_depth:
        return 0
    total = 1
    frontier = 1
    for d in range(depth, max_depth):
        w = _branch_factor_at(branch_factors, d)
        if w <= 0:
            break
        frontier *= w
        total += frontier
    return total


def aggregate_tree_stats(
    tree,
    max_depth: Optional[int] = None,
    branch_factor_by_depth: Optional[Dict[Any, Any]] = None,
) -> Dict[str, float]:
    """Aggregate InGPO action counts and SPO-counterfactual prune rates.

    ``spo_node_count`` uses the constructed tree for factual nodes, then adds
    virtual descendants skipped by PRUNE/SHARE. This keeps EOS-short branches
    from being inflated into a full W^D tree. ``prune_rate`` counts the full
    SPO subtree removed by PRUNE, including the pruned node itself.
    ``share_prune_rate`` counts only descendants skipped by SHARE, because the
    shared node is still emitted as an edge.
    """

    counts: Counter = Counter()
    prune_spo_count = 0
    share_prune_spo_count = 0
    virtual_pruned_spo_count = 0
    max_depth = int(max_depth if max_depth is not None else tree.get("ingpo_max_depth") or _infer_max_depth(tree))
    branch_factors = _coerce_depth_map(branch_factor_by_depth or tree.get("ingpo_branch_factor_by_depth"))
    inferred = _infer_branch_factors(tree)
    for d, w in inferred.items():
        branch_factors[d] = max(branch_factors.get(d, 0), w)

    stack = [tree]
    while stack:
        n = stack.pop()
        a = n.get("ingpo_action")
        depth = n.get("ingpo_depth")
        if a is not None and isinstance(depth, int) and depth > 0:
            counts[a] += 1
            subtree_size = _subtree_size_from_depth(depth, max_depth, branch_factors)
            virtual_descendants = max(subtree_size - 1, 0)
            if a == "prune":
                prune_spo_count += subtree_size
                virtual_pruned_spo_count += virtual_descendants
            elif a == "share":
                share_prune_spo_count += virtual_descendants
                virtual_pruned_spo_count += virtual_descendants
        stack.extend(n.get("children") or [])

    expanded = counts.get("expand", 0)
    shared = counts.get("share", 0)
    pruned = counts.get("prune", 0)
    total_visible = expanded + shared + pruned
    total_spo = total_visible + virtual_pruned_spo_count
    if total_visible == 0 and total_spo == 0:
        return {}
    total_pruned_spo = prune_spo_count + share_prune_spo_count
    return {
        "ingpo/expanded_count": expanded,
        "ingpo/shared_count": shared,
        "ingpo/pruned_count": pruned,
        "ingpo/factual_node_count": total_visible,
        "ingpo/virtual_pruned_spo_count": virtual_pruned_spo_count,
        "ingpo/spo_node_count": total_spo,
        "ingpo/pruned_spo_count": prune_spo_count,
        "ingpo/share_pruned_spo_count": share_prune_spo_count,
        "ingpo/total_pruned_spo_count": total_pruned_spo,
        "ingpo/prune_rate": prune_spo_count / max(total_spo, 1),
        "ingpo/share_prune_rate": share_prune_spo_count / max(total_spo, 1),
        "ingpo/total_prune_rate": total_pruned_spo / max(total_spo, 1),
    }


def per_depth_action_counts(tree) -> Dict[str, float]:
    """Walk the tree and return per-depth share/prune/expand counts and
    rates as a flat dict of wandb-friendly metric names.
    """

    per_depth_count: Dict[int, Counter] = {}
    stack = [tree]
    while stack:
        n = stack.pop()
        d = n.get("ingpo_depth")
        a = n.get("ingpo_action")
        if d is not None and a is not None:
            per_depth_count.setdefault(d, Counter())[a] += 1
        stack.extend(n.get("children") or [])

    out: Dict[str, float] = {}
    for d, c in sorted(per_depth_count.items()):
        total = sum(c.values())
        if total == 0:
            continue
        out[f"ingpo/depth_{d}/n"] = total
        out[f"ingpo/depth_{d}/expand_count"] = c.get("expand", 0)
        out[f"ingpo/depth_{d}/share_count"] = c.get("share", 0)
        out[f"ingpo/depth_{d}/prune_count"] = c.get("prune", 0)
    return out


def collect_demo_rows(
    tree,
    index_by_seg_id: Dict[str, Dict[str, Any]],
    question_id,
    n_each: int,
) -> Dict[str, List[List[Any]]]:
    """Return up to `n_each` SHARE demos and `n_each` PRUNE demos, each as a
    list of column-aligned cells matching `DEMO_COLUMNS`.
    """

    prune_rows: List[List[Any]] = []
    share_rows: List[List[Any]] = []

    stack = [tree]
    while stack:
        n = stack.pop()
        stack.extend(n.get("children") or [])
        action = n.get("ingpo_action")
        if action not in ("share", "prune"):
            continue

        seg_id = n.get("ingpo_segment_id", "")
        depth = n.get("ingpo_depth", "")
        parent_id = n.get("ingpo_parent_segment_id", "")
        parent = index_by_seg_id.get(parent_id, {})
        target_id = n.get("ingpo_share_target")
        target = index_by_seg_id.get(target_id, {}) if target_id else {}

        row = [
            str(question_id),
            action,
            int(depth) if isinstance(depth, int) else depth,
            str(seg_id),
            truncate(parent.get("text") or parent.get("full_text")),
            truncate(n.get("text") or n.get("full_text")),
            truncate(target.get("text") or target.get("full_text")) if target_id else "",
            str(target_id or ""),
            float(n.get("ingpo_avg_lp_K") or 0.0),
            float(n.get("ingpo_tv_m") or 0.0) if n.get("ingpo_tv_m") is not None else None,
            float(n.get("ingpo_gap_m") or 0.0) if n.get("ingpo_gap_m") is not None else None,
            float(n.get("ingpo_eta") or 0.0),
            float(n.get("ingpo_tau") or 0.0),
        ]
        (prune_rows if action == "prune" else share_rows).append(row)

    return {
        "share": share_rows[:n_each],
        "prune": prune_rows[:n_each],
    }


def row_to_dict(row: List[Any]) -> Dict[str, Any]:
    return dict(zip(DEMO_COLUMNS, row))


def render_md_section(
    tree_idx: int,
    question_id,
    stats: Dict[str, Any],
    demo_rows: Dict[str, List[List[Any]]],
) -> str:
    """One Markdown section for one tree, ready to append to demos.md."""

    out = [f"## Tree #{tree_idx}  (question_id={question_id})\n"]
    if stats:
        if stats.get("tree_construction_seconds") is not None:
            out.append(
                f"- tree_construction_seconds: "
                f"**{float(stats.get('tree_construction_seconds')):.3f}**\n"
            )
        prune_rate = float(stats.get("ingpo/prune_rate", 0.0) or 0.0)
        share_prune_rate = float(stats.get("ingpo/share_prune_rate", 0.0) or 0.0)
        total_prune_rate = float(stats.get("ingpo/total_prune_rate", 0.0) or 0.0)
        out.append(
            f"- prune_rate: **{prune_rate:.3f}**, "
            f"share_prune_rate: **{share_prune_rate:.3f}**, "
            f"total_prune_rate: **{total_prune_rate:.3f}**, "
            f"#shared={stats.get('ingpo/shared_count', 0)}, "
            f"#pruned={stats.get('ingpo/pruned_count', 0)}, "
            f"#expanded={stats.get('ingpo/expanded_count', 0)}\n"
        )

    for label, rows in (("SHARE", demo_rows["share"]), ("PRUNE", demo_rows["prune"])):
        if not rows:
            continue
        out.append(f"### {label} demos\n")
        for row in rows:
            d = row_to_dict(row)
            line = (
                f"- depth={d['depth']}  seg={d['seg_id']}  "
                f"AvgLP_K={float(d['avg_lp_K']):.3f}  "
            )
            if d.get("tv_m") is not None:
                line += f"TV_m={float(d['tv_m']):.3f}  "
            if d.get("gap_m") is not None:
                line += f"gap_m={float(d['gap_m']):.3f}  "
            line += f"(eta={float(d['eta']):.3f}, tau={float(d['tau']):.3f})\n"
            out.append(line)
            out.append(f"  - parent : `{d['parent_text']}`\n")
            out.append(f"  - child  : `{d['child_text']}`\n")
            if d.get("target_text"):
                out.append(
                    f"  - shared->`{d['target_seg_id']}` : `{d['target_text']}`\n"
                )
        out.append("\n")
    out.append("\n")
    return "".join(out)


def to_jsonl_record(
    tree_idx: int,
    question_id,
    answer_set_size: int,
    stats: Dict[str, Any],
    per_depth: Dict[str, float],
    demo_rows: Dict[str, List[List[Any]]],
    tree_construction_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Pack one tree's metrics + demos into a JSONL-ready dict."""

    record: Dict[str, Any] = {
        "tree_idx": tree_idx,
        "question_id": question_id,
        "answer_set_size": answer_set_size,
        "stats": stats,
        "per_depth": per_depth,
        "demos": {
            "share": [row_to_dict(r) for r in demo_rows["share"]],
            "prune": [row_to_dict(r) for r in demo_rows["prune"]],
        },
    }
    if tree_construction_seconds is not None:
        record["tree_construction_seconds"] = float(tree_construction_seconds)
    return record
