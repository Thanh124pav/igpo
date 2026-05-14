"""Log-probability matrix used by InGPO triggers.

Implements PLAN.md Def 2.1 / 2.2:

    LP[i][s] = log pi_theta(y_i | traj(s))
    delta_s  = log( 1 - sum_i exp(LP[i][s]) )

`LP` is stored only for the K fast indices first; the remaining m-K columns
are filled lazily when a Share or Prune trigger needs the full vector.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SegmentLP:
    """Per-segment log-prob row."""

    segment_id: str
    K: int
    m: int
    fast: np.ndarray
    full: Optional[np.ndarray] = None
    prefix: Optional[str] = None  # traj(s) used to score this row

    @property
    def avg_lp_K(self) -> float:
        return float(np.mean(self.fast))

    @property
    def has_full(self) -> bool:
        return self.full is not None

    @property
    def avg_lp_m(self) -> float:
        if self.full is None:
            raise RuntimeError(f"Full LP vector not computed for {self.segment_id}")
        return float(np.mean(self.full))

    def delta(self) -> float:
        """log( 1 - sum_i exp(LP[i]) ) using the full vector if available.

        Numerically stable via the two-regime log1mexp identity:
            log1mexp(x) = log(-expm1(x))   if x > -log(2)   (cancellation-safe)
                        = log1p(-exp(x))   otherwise         (underflow-safe)
        """

        vec = self.full if self.full is not None else self.fast
        log_sum = _logsumexp(vec)
        # sum_i exp(LP[i]) <= 1 by construction; clip strictly below 0 to avoid
        # log(0) when LP places ~all mass on Y.
        return float(_log1mexp(min(log_sum, -1e-300)))


def _logsumexp(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return -math.inf
    m = float(np.max(arr))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(arr - m))))


_LOG2 = math.log(2.0)


def _log1mexp(x: float) -> float:
    """log(1 - exp(x)) for x <= 0, numerically stable.

    Two-regime split (Mächler 2012): for x near 0 use log(-expm1(x)) to avoid
    catastrophic cancellation in (1 - exp(x)); for x deeply negative use
    log1p(-exp(x)) to avoid underflow when exp(x) is tiny.
    """

    if x >= 0.0:
        # Caller responsibility, but be defensive: return a finite floor.
        return -math.inf if x == 0.0 else float("nan")
    if x > -_LOG2:
        return math.log(-math.expm1(x))
    return math.log1p(-math.exp(x))


def log1mexp_array(x: np.ndarray) -> np.ndarray:
    """Vectorised log(1 - exp(x)) for x <= 0; same two-regime split as above."""

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    near_zero = x > -_LOG2
    out[near_zero] = np.log(-np.expm1(x[near_zero]))
    out[~near_zero] = np.log1p(-np.exp(x[~near_zero]))
    return out


class LogProbMatrix:
    """Thread-safe registry of SegmentLP rows.

    Designed as a global per-problem store: the inference strategy creates one
    instance for each tree and discards it when the tree is finished.
    """

    def __init__(self, K: int, m: int):
        if not (0 < K <= m):
            raise ValueError(f"Require 0 < K <= m, got K={K} m={m}")
        self.K = K
        self.m = m
        self._rows: Dict[str, SegmentLP] = {}
        self._lock = threading.Lock()

    def add_row(
        self,
        segment_id: str,
        fast: Sequence[float],
        prefix: Optional[str] = None,
    ) -> SegmentLP:
        if len(fast) != self.K:
            raise ValueError(f"Expected fast vector of length K={self.K}, got {len(fast)}")
        row = SegmentLP(
            segment_id=segment_id,
            K=self.K,
            m=self.m,
            fast=np.asarray(fast, dtype=np.float64),
            prefix=prefix,
        )
        with self._lock:
            self._rows[segment_id] = row
        return row

    def get(self, segment_id: str) -> SegmentLP:
        with self._lock:
            return self._rows[segment_id]

    def has(self, segment_id: str) -> bool:
        with self._lock:
            return segment_id in self._rows

    def fill_full(self, segment_id: str, tail: Sequence[float]) -> SegmentLP:
        """Append the K+1 .. m logprobs into the row's full vector.

        `tail` must have length m - K and represent indices K..m-1.
        Idempotent: if the row already has a full vector, do nothing.
        """

        expected = self.m - self.K
        if len(tail) != expected:
            raise ValueError(f"Expected tail length {expected}, got {len(tail)}")
        with self._lock:
            row = self._rows[segment_id]
            if row.full is None:
                row.full = np.concatenate([row.fast, np.asarray(tail, dtype=np.float64)])
        return row

    def avg_delta(self) -> float:
        """exp-mean of delta over all rows that have a full vector.

        Computed in log-space: returns exp( logsumexp(deltas) - log(N) ). This
        never materialises an intermediate exp(-200) (which underflows to 0)
        and therefore keeps `eta = epsilon/R_max - delta_avg` meaningful even
        when LP rows are deeply negative.
        """

        with self._lock:
            deltas = [r.delta() for r in self._rows.values() if r.has_full]
        if not deltas:
            return 0.0
        arr = np.asarray(deltas, dtype=np.float64)
        log_mean = _logsumexp(arr) - math.log(arr.size)
        # log_mean <= 0 since each delta <= 0; clip to a safe regime.
        return float(math.exp(min(log_mean, 0.0)))

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)
