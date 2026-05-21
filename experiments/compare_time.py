#!/usr/bin/env python3
"""Compare runtime metrics across experiment directories.

Usage:
    python experiments/compare_time.py exp-a exp-b
    python experiments/compare_time.py exp-a exp-b --root experiments --output-dir experiments/time_compare

The script reads:
  - <exp>/training_timing.jsonl
  - <exp>/ingpo_demos/demos.jsonl

and writes Markdown/CSV comparison tables for episode generation, training
step, train total, eval, cumulative train/eval/wall, and tree construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ITER_METRICS = {
    "episode_generation": (
        "timing/iter/episode_generation_seconds",
        "timing/total/episode_generation",
        "episode_generation_seconds",
    ),
    "training_step": (
        "timing/iter/training_step_seconds",
        "timing/total/training_step",
        "training_step_seconds",
    ),
    "train_total": (
        "timing/iter/train_total_seconds",
        "train_total_seconds",
    ),
    "eval": (
        "timing/iter/eval_seconds",
        "eval_seconds",
    ),
}

CUMULATIVE_METRICS = {
    "train": ("timing/cumulative/train_seconds", "cumulative_train_seconds"),
    "eval": ("timing/cumulative/eval_seconds", "cumulative_eval_seconds"),
    "wall": ("timing/cumulative/wall_seconds", "cumulative_wall_seconds"),
}

SUMMARY_METRICS = [
    "episode_generation_total_s",
    "training_step_total_s",
    "train_total_s",
    "eval_total_s",
    "train_cumulative_s",
    "eval_cumulative_s",
    "wall_s",
    "tree_construction_total_s",
]


@dataclass
class ExperimentStats:
    name: str
    path: Path
    algorithm: str = ""
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    tree_construction_seconds: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def total(self, metric: str) -> Optional[float]:
        values = [as_float(row.get(metric)) for row in self.iterations]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return sum(values)

    def average(self, metric: str) -> Optional[float]:
        values = [as_float(row.get(metric)) for row in self.iterations]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return sum(values) / len(values)

    def last(self, metric: str) -> Optional[float]:
        for row in reversed(self.iterations):
            value = as_float(row.get(metric))
            if value is not None:
                return value
        return None


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] Skip invalid JSONL line {path}:{line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                yield obj


def first_number(record: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = as_float(record.get(key))
        if value is not None:
            return value
    return None


def infer_algorithm(exp_path: Path, name: str) -> str:
    config_path = exp_path / "config.json"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
            trainer_type = nested_get(config, ("trainer", "type"))
            generator_type = nested_get(config, ("episode_generator", "type"))
            inference_type = nested_get(config, ("episode_generator", "inference_strategy", "type"))
            parts = [part for part in (trainer_type, generator_type, inference_type) if part]
            if parts:
                return " / ".join(str(part) for part in parts)
        except Exception:
            pass

    lower_name = name.lower()
    for label in ("ingpo", "spo_tree", "spo_chain", "ppo", "grpo", "rloo", "restem", "dpo", "vineppo"):
        if label in lower_name:
            return label
    return ""


def nested_get(obj: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def resolve_experiment(name_or_path: str, root: Path) -> Path:
    direct = Path(name_or_path).expanduser()
    if direct.exists():
        return direct
    return root / name_or_path


def load_training_timing(stats: ExperimentStats) -> None:
    timing_path = stats.path / "training_timing.jsonl"
    if not timing_path.exists():
        stats.warnings.append(f"missing {timing_path}")
        return

    for idx, record in enumerate(read_jsonl(timing_path)):
        iteration = record.get("iteration", idx)
        row: Dict[str, Any] = {
            "experiment": stats.name,
            "iteration": iteration,
            "global_step": record.get("global_step"),
        }

        for metric, keys in ITER_METRICS.items():
            row[metric] = first_number(record, keys)

        if row.get("train_total") is None:
            ep = as_float(row.get("episode_generation")) or 0.0
            step = as_float(row.get("training_step")) or 0.0
            if row.get("episode_generation") is not None or row.get("training_step") is not None:
                row["train_total"] = ep + step

        for metric, keys in CUMULATIVE_METRICS.items():
            row[f"{metric}_cumulative"] = first_number(record, keys)

        stats.iterations.append(row)


def load_tree_construction(stats: ExperimentStats) -> None:
    tree_values = []
    tree_paths = list(stats.path.glob("iteration_*/**/trees/*.json"))
    tree_paths.extend(stats.path.glob("iteration_*/episodes/**/*.json"))
    for tree_path in sorted(set(tree_paths)):
        try:
            with tree_path.open("r", encoding="utf-8") as fh:
                tree = json.load(fh)
            if isinstance(tree, str):
                tree = json.loads(tree)
        except Exception:
            continue
        if isinstance(tree, dict):
            value = first_number(
                tree,
                (
                    "tree_construction_seconds",
                    "ingpo_tree_construction_seconds",
                ),
            )
            if value is not None:
                tree_values.append(value)

    if tree_values:
        stats.tree_construction_seconds.extend(tree_values)
        return

    demo_paths = [
        stats.path / "ingpo_demos" / "demos.jsonl",
        stats.path / "demos" / "ingpo" / "demos.jsonl",
        stats.path / "demos" / "spo" / "samples.jsonl",
        stats.path / "demos" / "spo" / "sample.jsonl",
        stats.path / "demos" / "grpo" / "samples.jsonl",
        stats.path / "demos" / "grpo" / "sample.jsonl",
    ]
    demo_paths.extend(sorted((stats.path / "demos").glob("*/*.jsonl")))

    seen_paths = set()
    for demos_path in demo_paths:
        if demos_path in seen_paths or not demos_path.exists():
            continue
        seen_paths.add(demos_path)
        for record in read_jsonl(demos_path):
            value = first_number(
                record,
                (
                    "tree_construction_seconds",
                    "ingpo_tree_construction_seconds",
                    "timing/tree_construction_seconds",
                ),
            )
            if value is not None:
                stats.tree_construction_seconds.append(value)


def load_experiment(name_or_path: str, root: Path) -> ExperimentStats:
    exp_path = resolve_experiment(name_or_path, root)
    stats = ExperimentStats(name=exp_path.name, path=exp_path)
    stats.algorithm = infer_algorithm(exp_path, stats.name)

    if not exp_path.exists():
        stats.warnings.append(f"experiment path not found: {exp_path}")
        return stats

    load_training_timing(stats)
    load_tree_construction(stats)
    return stats


def round_or_blank(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def summary_row(stats: ExperimentStats) -> Dict[str, Any]:
    tree_values = stats.tree_construction_seconds
    tree_total = sum(tree_values) if tree_values else None
    tree_avg = tree_total / len(tree_values) if tree_values else None

    return {
        "experiment": stats.name,
        "algorithm": stats.algorithm,
        "n_iterations": len(stats.iterations),
        "n_trees": len(tree_values),
        "episode_generation_total_s": stats.total("episode_generation"),
        "training_step_total_s": stats.total("training_step"),
        "train_total_s": stats.total("train_total"),
        "eval_total_s": stats.total("eval"),
        "train_cumulative_s": stats.last("train_cumulative"),
        "eval_cumulative_s": stats.last("eval_cumulative"),
        "wall_s": stats.last("wall_cumulative"),
        "tree_construction_total_s": tree_total,
        "tree_construction_avg_s": tree_avg,
    }


def average_row(stats: ExperimentStats) -> Dict[str, Any]:
    tree_values = stats.tree_construction_seconds
    return {
        "experiment": stats.name,
        "algorithm": stats.algorithm,
        "n_iterations": len(stats.iterations),
        "episode_generation_avg_s": stats.average("episode_generation"),
        "training_step_avg_s": stats.average("training_step"),
        "train_total_avg_s": stats.average("train_total"),
        "eval_avg_s": stats.average("eval"),
        "tree_construction_avg_s": (sum(tree_values) / len(tree_values)) if tree_values else None,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "_No data._\n"
    table = ["| " + " | ".join(columns) + " |"]
    table.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(round_or_blank(value))
            else:
                cells.append(str(value) if value is not None else "")
        table.append("| " + " | ".join(cells) + " |")
    return "\n".join(table) + "\n"


def metric_matrix(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for metric in SUMMARY_METRICS:
        row = {"metric": metric}
        for summary in summary_rows:
            row[summary["experiment"]] = summary.get(metric)
        out.append(row)
    return out


def natural_sort_key(value: Any) -> List[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def iteration_rows(stats_list: List[ExperimentStats]) -> List[Dict[str, Any]]:
    rows = []
    for stats in stats_list:
        for row in stats.iterations:
            rows.append(
                {
                    "experiment": stats.name,
                    "iteration": row.get("iteration"),
                    "global_step": row.get("global_step"),
                    "episode_generation_s": row.get("episode_generation"),
                    "training_step_s": row.get("training_step"),
                    "train_total_s": row.get("train_total"),
                    "eval_s": row.get("eval"),
                    "train_cumulative_s": row.get("train_cumulative"),
                    "eval_cumulative_s": row.get("eval_cumulative"),
                    "wall_s": row.get("wall_cumulative"),
                }
            )
    rows.sort(key=lambda r: (r["experiment"], natural_sort_key(r["iteration"])))
    return rows


def build_markdown(
    stats_list: List[ExperimentStats],
    totals: List[Dict[str, Any]],
    averages: List[Dict[str, Any]],
    matrix: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    lines = [
        "# Time Comparison",
        "",
        f"Output directory: `{output_dir}`",
        "",
        "## Totals",
        markdown_table(
            totals,
            [
                "experiment",
                "algorithm",
                "n_iterations",
                "episode_generation_total_s",
                "training_step_total_s",
                "train_total_s",
                "eval_total_s",
                "train_cumulative_s",
                "eval_cumulative_s",
                "wall_s",
                "tree_construction_total_s",
            ],
        ),
        "## Averages",
        markdown_table(
            averages,
            [
                "experiment",
                "algorithm",
                "n_iterations",
                "episode_generation_avg_s",
                "training_step_avg_s",
                "train_total_avg_s",
                "eval_avg_s",
                "tree_construction_avg_s",
            ],
        ),
        "## Metric Matrix",
        markdown_table(matrix, ["metric"] + [row["experiment"] for row in totals]),
    ]

    warnings = [(stats.name, warning) for stats in stats_list for warning in stats.warnings]
    if warnings:
        lines.extend(["## Warnings", markdown_table([{"experiment": n, "warning": w} for n, w in warnings], ["experiment", "warning"])])

    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare runtime tables for multiple experiment names."
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Experiment names under --root, or explicit experiment directory paths.",
    )
    parser.add_argument(
        "--root",
        default="experiments",
        help="Root directory containing experiment folders. Default: experiments",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output tables. Default: <root>/time_comparison",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Write files only; do not print Markdown tables to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else root / "time_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_list = [load_experiment(name, root) for name in args.experiments]
    totals = [summary_row(stats) for stats in stats_list]
    averages = [average_row(stats) for stats in stats_list]
    matrix = metric_matrix(totals)
    per_iteration = iteration_rows(stats_list)

    write_csv(
        output_dir / "compare_time_totals.csv",
        totals,
        [
            "experiment",
            "algorithm",
            "n_iterations",
            "n_trees",
            "episode_generation_total_s",
            "training_step_total_s",
            "train_total_s",
            "eval_total_s",
            "train_cumulative_s",
            "eval_cumulative_s",
            "wall_s",
            "tree_construction_total_s",
            "tree_construction_avg_s",
        ],
    )
    write_csv(
        output_dir / "compare_time_averages.csv",
        averages,
        [
            "experiment",
            "algorithm",
            "n_iterations",
            "episode_generation_avg_s",
            "training_step_avg_s",
            "train_total_avg_s",
            "eval_avg_s",
            "tree_construction_avg_s",
        ],
    )
    write_csv(output_dir / "compare_time_metrics.csv", matrix, ["metric"] + [row["experiment"] for row in totals])
    write_csv(
        output_dir / "compare_time_iterations.csv",
        per_iteration,
        [
            "experiment",
            "iteration",
            "global_step",
            "episode_generation_s",
            "training_step_s",
            "train_total_s",
            "eval_s",
            "train_cumulative_s",
            "eval_cumulative_s",
            "wall_s",
        ],
    )

    markdown = build_markdown(stats_list, totals, averages, matrix, output_dir)
    (output_dir / "compare_time_summary.md").write_text(markdown, encoding="utf-8")

    if not args.no_print:
        print(markdown)

    return 1 if any(stats.warnings for stats in stats_list) else 0


if __name__ == "__main__":
    raise SystemExit(main())
