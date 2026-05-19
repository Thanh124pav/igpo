"""Small offline demo logger for comparing GRPO/SPO/InGPO samples.

The logger writes full-text examples to local files only.  It intentionally
keeps the schema compact so demo files stay readable during long runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class DemoFileLogger:
    def __init__(
        self,
        *,
        algorithm: str,
        exp_root: Optional[Path],
        demos_dir: Optional[str] = None,
    ) -> None:
        self.algorithm = algorithm
        self.exp_root = Path(exp_root) if exp_root is not None else None
        self.demos_dir_override = demos_dir
        self._base: Optional[Path] = None
        self._jsonl = None
        self._md = None
        self._seen = 0

    def _resolve_base(self) -> Path:
        if self._base is not None:
            return self._base
        if self.demos_dir_override:
            base = Path(self.demos_dir_override) / self.algorithm
        elif self.exp_root is not None:
            base = self.exp_root / "demos" / self.algorithm
        else:
            base = Path.cwd() / "demos" / self.algorithm
        base.mkdir(parents=True, exist_ok=True)
        self._base = base
        return base

    def _open(self):
        base = self._resolve_base()
        if self._jsonl is None:
            self._jsonl = (base / "samples.jsonl").open("a", encoding="utf-8", buffering=1)
        if self._md is None:
            self._md = (base / "samples.md").open("a", encoding="utf-8", buffering=1)
            if self._md.tell() == 0:
                self._md.write(f"# {self.algorithm.upper()} demos\n\n")
        return self._jsonl, self._md

    @staticmethod
    def _compact_record(sample: Dict[str, Any]) -> Dict[str, Any]:
        keep = [
            "algorithm",
            "iteration",
            "sample_idx",
            "question_id",
            "score",
            "reward",
            "advantage",
            "value",
            "leaf",
            "tree_construction_seconds",
            "query_text",
            "response_text",
        ]
        return {k: sample.get(k) for k in keep if k in sample}

    def log_samples(
        self,
        samples: Iterable[Dict[str, Any]],
        *,
        iteration: Optional[int],
        limit: int,
    ) -> None:
        if limit <= 0:
            return
        jsonl, md = self._open()
        count = 0
        for sample in samples:
            if count >= limit:
                break
            self._seen += 1
            row = self._compact_record(
                {
                    "algorithm": self.algorithm,
                    "iteration": iteration,
                    "sample_idx": self._seen,
                    **sample,
                }
            )
            jsonl.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self._write_md(md, row)
            count += 1

    def _write_md(self, md, row: Dict[str, Any]) -> None:
        md.write(
            f"## Sample #{row.get('sample_idx')} "
            f"(iteration={row.get('iteration')}, question_id={row.get('question_id')})\n\n"
        )
        for key in ("score", "reward", "advantage", "value", "leaf", "tree_construction_seconds"):
            if key in row and row.get(key) is not None:
                md.write(f"- {key}: `{row.get(key)}`\n")
        md.write("\n### Query\n\n")
        md.write("```text\n")
        md.write(str(row.get("query_text", "")))
        md.write("\n```\n\n")
        md.write("### Response\n\n")
        md.write("```text\n")
        md.write(str(row.get("response_text", "")))
        md.write("\n```\n\n")

    def close(self) -> None:
        for handle in (self._jsonl, self._md):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass

    def __del__(self):
        self.close()
