"""Thin wrapper around `torch.utils.tensorboard.SummaryWriter`.

TensorBoard is a soft dependency: if `torch` is not importable we silently
disable the writer so trainers without TB are not broken.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    def __init__(self, logdir: str, enabled: bool = True):
        self.logdir = logdir
        self.enabled = bool(enabled)
        self._step = 0
        self.writer = None
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception as exc:
            logger.warning(
                "TensorBoard not available (%s); disabling InGPO TB logger.", exc
            )
            self.enabled = False
            return
        try:
            self.writer = SummaryWriter(log_dir=logdir)
        except Exception as exc:
            logger.warning("Failed to open TensorBoard writer at %s: %s", logdir, exc)
            self.enabled = False
            self.writer = None

    def log_scalars(
        self, scalars: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if not self.enabled or self.writer is None:
            return
        s = self._step if step is None else int(step)
        for k, v in scalars.items():
            try:
                self.writer.add_scalar(k, float(v), s)
            except Exception:
                continue
        self._step = max(self._step, s) + 1

    def flush(self) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
            self.writer = None
