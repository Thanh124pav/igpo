"""TV distance and AvgLP utilities.

Implements PLAN.md Def 2.3:

    AvgLP_K(s) = (1/K) * sum_{i=1..K} LP[i][s]
    TV_m(a,b)  = 0.5 * sum_{i=1..m} | exp(LP[i][a]) - exp(LP[i][b]) |
                 + 0.5 * ( exp(delta_a) + exp(delta_b) )

The +0.5*(exp(delta_a)+exp(delta_b)) tail accounts for the residual mass
outside the answer set Y; it bounds the true total variation between the two
conditional distributions over completions.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from ingpo_ext.core.log_prob_matrix import SegmentLP, log1mexp_array


def avg_lp_K(row: SegmentLP) -> float:
    return row.avg_lp_K


def avg_lp_m(row: SegmentLP) -> float:
    return row.avg_lp_m


def tv_m(row_a: SegmentLP, row_b: SegmentLP) -> float:
    """0.5 * sum_i |exp(LP_a[i]) - exp(LP_b[i])| + 0.5 * (exp(delta_a) + exp(delta_b)).

    Body computed in log-space to stay finite when LP values are deeply
    negative:
        |exp(a) - exp(b)| = exp(max(a,b)) * (1 - exp(-|a-b|))
                          = exp( max(a,b) + log1mexp(-|a-b|) )
    The tail uses delta() (also log-space) and exponentiates only the final
    scalar.
    """

    if row_a.full is None or row_b.full is None:
        raise RuntimeError(
            "tv_m requires both rows to have their full m-length LP vector. "
            "Call LogProbMatrix.fill_full(...) first."
        )
    if row_a.m != row_b.m:
        raise ValueError(f"Row m mismatch: {row_a.m} vs {row_b.m}")

    a = np.asarray(row_a.full, dtype=np.float64)
    b = np.asarray(row_b.full, dtype=np.float64)
    diff = np.abs(a - b)
    # Where diff == 0, |exp(a) - exp(b)| = 0 exactly. log1mexp(0) is -inf;
    # mask those entries so they contribute 0 instead of NaN.
    nonzero = diff > 0.0
    log_abs = np.full_like(diff, fill_value=-np.inf)
    if np.any(nonzero):
        upper = np.maximum(a[nonzero], b[nonzero])
        log_abs[nonzero] = upper + log1mexp_array(-diff[nonzero])
    # sum_i exp(log_abs[i]) via stable accumulation
    finite = log_abs[np.isfinite(log_abs)]
    if finite.size == 0:
        body = 0.0
    else:
        m = float(np.max(finite))
        body = 0.5 * math.exp(m) * float(np.sum(np.exp(finite - m)))

    tail = 0.5 * (math.exp(min(row_a.delta(), 0.0)) + math.exp(min(row_b.delta(), 0.0)))
    return body + tail


def avg_lp_diff_K(row_a: SegmentLP, row_b: SegmentLP) -> float:
    return abs(row_a.avg_lp_K - row_b.avg_lp_K)


def conditional_ig_lower_bound(
    row_s: SegmentLP, row_pa: SegmentLP
) -> Tuple[float, float]:
    """Pinsker-style lower bound on I(A*; Y_s | Y_pa) via TV.

    Returns (lower_bound, raw_tv). Used in PLAN Lemma 2.4 sketch:
        I(A*; Y_s | Y_pa) >= 2 * TV(s, pa)^2 - O(exp(delta_s) + exp(delta_pa))
    """

    tv = tv_m(row_s, row_pa)
    correction = math.exp(row_s.delta()) + math.exp(row_pa.delta())
    return 2.0 * tv * tv - correction, tv
