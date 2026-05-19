"""Recompute/justify InGPO prune-rate metrics from ingpo_demos.

Usage:
    python scripts/justify_prune_rate.py <exp_dir>/ingpo_demos
    python scripts/justify_prune_rate.py <exp_dir>/ingpo_demos/demos.jsonl

Newer demos.jsonl records already contain exact SPO-counterfactual metrics in
``record["stats"]``. For older logs that only contain capped demo rows, this
script can compute a lower-bound estimate from the visible demos if you provide
``--width`` and ``--max-depth``. Exact recomputation from capped demos is not
possible because non-demo PRUNE/SHARE decisions and factual EOS-short branches
were not written to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def resolve_jsonl(path: Path) -> Path:
    if path.is_dir():
        path = path / "demos.jsonl"
    return path


def full_tree_nodes(width: int, max_depth: int) -> int:
    total = 0
    frontier = 1
    for _ in range(1, max_depth + 1):
        frontier *= width
        total += frontier
    return total


def subtree_size(width: int, depth: int, max_depth: int) -> int:
    if depth <= 0 or depth > max_depth:
        return 0
    total = 1
    frontier = 1
    for _ in range(depth, max_depth):
        frontier *= width
        total += frontier
    return total


def estimate_from_demo_rows(
    rec: Dict[str, Any],
    *,
    width: int,
    max_depth: int,
) -> Dict[str, float]:
    demos = rec.get("demos") or {}
    prune_rows: List[Dict[str, Any]] = demos.get("prune") or []
    share_rows: List[Dict[str, Any]] = demos.get("share") or []
    prune_count = 0
    share_prune_count = 0
    for row in prune_rows:
        prune_count += subtree_size(width, int(row.get("depth") or 0), max_depth)
    for row in share_rows:
        share_prune_count += max(subtree_size(width, int(row.get("depth") or 0), max_depth) - 1, 0)
    denom = full_tree_nodes(width, max_depth)
    total = prune_count + share_prune_count
    return {
        "spo_node_count": denom,
        "factual_node_count": 0,
        "virtual_pruned_spo_count": 0,
        "pruned_spo_count": prune_count,
        "share_pruned_spo_count": share_prune_count,
        "total_pruned_spo_count": total,
        "prune_rate": prune_count / max(denom, 1),
        "share_prune_rate": share_prune_count / max(denom, 1),
        "total_prune_rate": total / max(denom, 1),
    }


def get_stats(rec: Dict[str, Any]) -> Dict[str, Any]:
    stats = rec.get("stats") or {}
    required = (
        "ingpo/spo_node_count",
        "ingpo/pruned_spo_count",
        "ingpo/share_pruned_spo_count",
        "ingpo/total_pruned_spo_count",
        "ingpo/prune_rate",
        "ingpo/share_prune_rate",
        "ingpo/total_prune_rate",
    )
    return stats if all(k in stats for k in required) else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="ingpo_demos directory or demos.jsonl")
    ap.add_argument("--width", type=int, default=None, help="SPO branch factor for old/capped logs")
    ap.add_argument("--max-depth", type=int, default=None, help="SPO max depth for old/capped logs")
    ap.add_argument("--per-tree", action="store_true", help="print one line per tree")
    args = ap.parse_args()

    jsonl = resolve_jsonl(Path(args.path))
    if not jsonl.exists():
        print(f"error: {jsonl} does not exist", file=sys.stderr)
        return 2

    n = 0
    exact = 0
    sums = {
        "spo_node_count": 0.0,
        "factual_node_count": 0.0,
        "virtual_pruned_spo_count": 0.0,
        "pruned_spo_count": 0.0,
        "share_pruned_spo_count": 0.0,
        "total_pruned_spo_count": 0.0,
    }
    for rec in iter_records(jsonl):
        n += 1
        stats = get_stats(rec)
        if stats:
            exact += 1
            row = {
                "spo_node_count": float(stats["ingpo/spo_node_count"]),
                "factual_node_count": float(stats.get("ingpo/factual_node_count", 0.0)),
                "virtual_pruned_spo_count": float(stats.get("ingpo/virtual_pruned_spo_count", 0.0)),
                "pruned_spo_count": float(stats["ingpo/pruned_spo_count"]),
                "share_pruned_spo_count": float(stats["ingpo/share_pruned_spo_count"]),
                "total_pruned_spo_count": float(stats["ingpo/total_pruned_spo_count"]),
                "prune_rate": float(stats["ingpo/prune_rate"]),
                "share_prune_rate": float(stats["ingpo/share_prune_rate"]),
                "total_prune_rate": float(stats["ingpo/total_prune_rate"]),
            }
        else:
            if args.width is None or args.max_depth is None:
                continue
            row = estimate_from_demo_rows(rec, width=args.width, max_depth=args.max_depth)

        for k in sums:
            sums[k] += float(row[k])
        if args.per_tree:
            print(
                f"tree={rec.get('tree_idx')} q={rec.get('question_id')} "
                f"prune={row['prune_rate']:.4f} "
                f"share_prune={row['share_prune_rate']:.4f} "
                f"total={row['total_prune_rate']:.4f}"
            )

    if n == 0:
        print("error: no records found", file=sys.stderr)
        return 1
    if sums["spo_node_count"] <= 0:
        print(
            "error: no exact prune-rate stats found. For old logs, rerun with "
            "--width W --max-depth D to compute a lower-bound estimate from capped demo rows.",
            file=sys.stderr,
        )
        return 1

    denom = max(sums["spo_node_count"], 1.0)
    print(f"records: {n}")
    print(f"exact_records: {exact}")
    if exact < n:
        print("note: non-exact records were estimated from visible demo rows only.")
    print(f"spo_node_count: {int(sums['spo_node_count'])}")
    print(f"factual_node_count: {int(sums['factual_node_count'])}")
    print(f"virtual_pruned_spo_count: {int(sums['virtual_pruned_spo_count'])}")
    print(f"pruned_spo_count: {int(sums['pruned_spo_count'])}")
    print(f"share_pruned_spo_count: {int(sums['share_pruned_spo_count'])}")
    print(f"total_pruned_spo_count: {int(sums['total_pruned_spo_count'])}")
    print(f"prune_rate: {sums['pruned_spo_count'] / denom:.6f}")
    print(f"share_prune_rate: {sums['share_pruned_spo_count'] / denom:.6f}")
    print(f"total_prune_rate: {sums['total_pruned_spo_count'] / denom:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
