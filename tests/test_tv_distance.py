import math

import pytest

from ingpo_ext.core.log_prob_matrix import LogProbMatrix
from ingpo_ext.core.tv_distance import (
    avg_lp_diff_K,
    conditional_ig_lower_bound,
    tv_m,
)


def make_row(M: LogProbMatrix, sid: str, full):
    M.add_row(sid, full[: M.K])
    M.fill_full(sid, full[M.K :])
    return M.get(sid)


def test_tv_m_identical_rows_zero():
    M = LogProbMatrix(K=2, m=4)
    r = [math.log(0.3), math.log(0.3), math.log(0.2), math.log(0.1)]
    a = make_row(M, "a", r)
    b = make_row(M, "b", r)
    assert tv_m(a, b) == pytest.approx(2 * 0.5 * math.exp(a.delta()), abs=1e-9)


def test_tv_m_disjoint_rows_close_to_one():
    M = LogProbMatrix(K=2, m=4)
    a = make_row(M, "a", [math.log(0.95), math.log(0.04), -50.0, -50.0])
    b = make_row(M, "b", [-50.0, -50.0, math.log(0.95), math.log(0.04)])
    assert tv_m(a, b) > 0.85


def test_avg_lp_diff_K():
    M = LogProbMatrix(K=2, m=2)
    a = M.add_row("a", [-1.0, -2.0])
    b = M.add_row("b", [-1.5, -1.5])
    assert avg_lp_diff_K(a, b) == pytest.approx(0.0)


def test_ig_lower_bound_nonnegative_when_tv_large():
    M = LogProbMatrix(K=2, m=4)
    a = make_row(M, "a", [math.log(0.9), math.log(0.04), math.log(0.03), math.log(0.02)])
    b = make_row(M, "b", [math.log(0.02), math.log(0.03), math.log(0.04), math.log(0.9)])
    bound, tv = conditional_ig_lower_bound(a, b)
    assert tv > 0.5
    assert bound > 0


def test_tv_m_finite_for_deeply_negative_LP():
    """Naive np.exp(-300) underflows to 0, making the old |pa-pb| sum 0/NaN.
    The log-space rewrite must return a finite, non-NaN tv_m."""
    M = LogProbMatrix(K=2, m=4)
    a = make_row(M, "a", [-300.0, -300.0, -300.0, -300.0])
    b = make_row(M, "b", [-300.0, -300.0, -300.0, -300.0])
    tv = tv_m(a, b)
    assert math.isfinite(tv)
    # Identical rows ⇒ body ≈ 0; tv collapses to tail = 0.5*(exp(delta_a)+exp(delta_b)) ≈ 1.0.
    assert tv == pytest.approx(1.0, abs=1e-6)


def test_tv_m_log_space_matches_direct_exp_for_moderate_LP():
    """For moderate LP magnitudes the log-space rewrite must match the
    obvious direct formulation to high precision."""
    import numpy as np
    M = LogProbMatrix(K=2, m=4)
    a = make_row(M, "a", [math.log(0.40), math.log(0.30), math.log(0.20), math.log(0.05)])
    b = make_row(M, "b", [math.log(0.10), math.log(0.20), math.log(0.30), math.log(0.35)])
    expected_body = 0.5 * float(np.sum(np.abs(np.exp(a.full) - np.exp(b.full))))
    expected_tail = 0.5 * (math.exp(a.delta()) + math.exp(b.delta()))
    assert tv_m(a, b) == pytest.approx(expected_body + expected_tail, abs=1e-9)
