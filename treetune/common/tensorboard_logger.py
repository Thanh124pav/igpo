"""Minimal TensorBoard-compatible logger facade.

The runtime and trainers historically receive a W&B Run-like object and call
``log``, ``save``, ``define_metric``, and ``summary`` on it.  This module keeps
that small surface area while allowing experiments to run on machines where W&B
is blocked or unavailable.
"""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Optional


def _as_scalar(value: Any) -> Optional[float]:
    """Return a TensorBoard scalar value, or ``None`` for unsupported objects."""

    if isinstance(value, bool):
        return float(int(value))
    if not isinstance(value, numbers.Number):
        return None

    scalar = float(value)
    if math.isnan(scalar) or math.isinf(scalar):
        return None
    return scalar


class TensorBoardLogger:
    """Small W&B Run-compatible wrapper around ``SummaryWriter``.

    Only scalar metrics are emitted to TensorBoard.  Rich W&B-only payloads such
    as tables, lists, or nested dicts are skipped instead of failing the run.
    """

    def __init__(
        self,
        log_dir: str | Path,
        *,
        writer: Any = None,
        flush_secs: int = 30,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.summary: Dict[str, Any] = {}
        self._next_step = 0

        if writer is None:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)
        self.writer = writer

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        """Accept W&B metric declarations; TensorBoard does not need them."""

        return None

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Accept W&B artifact-save calls; metadata is already written locally."""

        return None

    def log(
        self,
        data: Optional[MutableMapping[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Write scalar metrics to TensorBoard.

        If no explicit ``step`` is supplied, common global-step keys are reused;
        otherwise an internal monotonically increasing step is used.
        """

        if data is None:
            data = {}
        if not isinstance(data, MutableMapping):
            return None

        step = kwargs.get("step")
        if step is None:
            step = self._infer_step(data)
        if step is None:
            step = self._next_step

        step = int(step)
        self._next_step = max(self._next_step, step + 1)

        for key, value in data.items():
            scalar = _as_scalar(value)
            if scalar is None:
                continue
            self.writer.add_scalar(str(key), scalar, step)

        if hasattr(self.writer, "flush"):
            self.writer.flush()
        return None

    @staticmethod
    def _infer_step(data: MutableMapping[str, Any]) -> Optional[int]:
        for key in (
            "train/global_step",
            "train/global_iteration",
            "global_step",
            "iteration",
        ):
            if key not in data:
                continue
            scalar = _as_scalar(data[key])
            if scalar is not None:
                return int(scalar)
        return None

    def finish(self) -> None:
        if hasattr(self.writer, "close"):
            self.writer.close()


class CompositeSummary(dict):
    """Dict-like summary that mirrors updates to multiple backend summaries."""

    def __init__(self, summaries: Iterable[MutableMapping[str, Any]]) -> None:
        super().__init__()
        self._summaries = list(summaries)

    def __setitem__(self, key: str, value: Any) -> None:
        for summary in self._summaries:
            summary[key] = value
        super().__setitem__(key, value)

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        updates = dict(*args, **kwargs)
        for key, value in updates.items():
            self[key] = value


class CompositeLogger:
    """Fan out W&B-like logger calls to several logging backends."""

    def __init__(self, loggers: Iterable[Any]) -> None:
        self.loggers = [logger for logger in loggers if logger is not None]
        self.summary = CompositeSummary(
            getattr(logger, "summary", {}) for logger in self.loggers
        )

    def log(self, *args: Any, **kwargs: Any) -> None:
        for logger in self.loggers:
            if hasattr(logger, "log"):
                logger.log(*args, **kwargs)

    def save(self, *args: Any, **kwargs: Any) -> None:
        for logger in self.loggers:
            if hasattr(logger, "save"):
                logger.save(*args, **kwargs)

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        for logger in self.loggers:
            if hasattr(logger, "define_metric"):
                logger.define_metric(*args, **kwargs)

    def finish(self) -> None:
        for logger in self.loggers:
            if hasattr(logger, "finish"):
                logger.finish()
