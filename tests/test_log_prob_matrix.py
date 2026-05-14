import math

import numpy as np
import pytest

from ingpo_ext.core.log_prob_matrix import LogProbMatrix


def test_add_and_full_fill():
    M = LogProbMatrix(K=3, m=5)
    fast = [-1.0, -2.0, -3.0]
    row = M.add_row("s0", fast, prefix="prefix")
    assert row.avg_lp_K == pytest.approx(np.mean(fast))

    M.fill_full("s0", [-4.0, -5.0])
    row = M.get("s0")
    assert row.full is not None
    assert row.avg_lp_m == pytest.approx(np.mean([-1, -2, -3, -4, -5]))


def test_delta_residual():
    M = LogProbMatrix(K=2, m=2)
    # Heavy mass on first answer, residual = small.
    M.add_row("s0", [math.log(0.6), math.log(0.3)])
    row = M.get("s0")
    # delta = log(1 - (0.6 + 0.3)) = log(0.1)
    assert row.delta() == pytest.approx(math.log(0.1), abs=1e-6)


def test_validates_dimensions():
    M = LogProbMatrix(K=2, m=4)
    with pytest.raises(ValueError):
        M.add_row("s0", [-1.0, -2.0, -3.0])
    M.add_row("s0", [-1.0, -2.0])
    with pytest.raises(ValueError):
        M.fill_full("s0", [-3.0])


def test_delta_stable_when_mass_concentrated_on_Y():
    """When sum_i exp(LP[i]) is extremely close to 1, the naive
    `log(1 - exp(log_sum))` cancels to log(1e-12). The log1mexp rewrite
    keeps the residual accurate to many more decimals."""
    M = LogProbMatrix(K=1, m=1)
    # log_sum = log(1 - 1e-15). delta should equal log(1e-15) ≈ -34.54
    # (within the precision of representing 1e-15 in float64).
    M.add_row("s0", [math.log(1.0 - 1e-15)])
    row = M.get("s0")
    # Within ~ulp(log(1e-15)) — the input itself is only good to ~5e-16.
    assert row.delta() == pytest.approx(math.log(1e-15), abs=1e-2)
    # Sanity: should not be floored at log(1e-12) anymore.
    assert row.delta() < math.log(1e-12)


def test_delta_finite_for_deeply_negative_LP():
    """LP rows of -300 must not produce NaN/inf in delta()."""
    M = LogProbMatrix(K=4, m=4)
    M.add_row("s0", [-300.0, -300.0, -300.0, -300.0])
    row = M.get("s0")
    d = row.delta()
    assert math.isfinite(d)
    # sum exp(-300) ≈ 0 ⇒ delta ≈ log(1) = 0.
    assert d == pytest.approx(0.0, abs=1e-6)


def test_avg_delta_finite_with_deep_LP_rows():
    """avg_delta uses log-space averaging now; should stay > 0 even when
    every individual exp(delta) underflows in float64."""
    M = LogProbMatrix(K=2, m=2)
    for i in range(5):
        M.add_row(f"s{i}", [-2.0, -2.0])
        M.fill_full(f"s{i}", [])  # K == m, fill is a noop but marks has_full
    avg = M.avg_delta()
    assert math.isfinite(avg)
    assert avg > 0.0
