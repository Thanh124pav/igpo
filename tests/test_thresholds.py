import math

import pytest

from treetune.ingpo.thresholds import (
    ThresholdConfig,
    compute_eta,
    compute_tau,
    tv_to_value_bound,
)


def test_eta_from_lemma_2_4():
    cfg = ThresholdConfig(epsilon=0.02, r_max=1.0, K=10)
    assert compute_eta(cfg, delta_avg=0.0) == pytest.approx(0.02)
    # Higher delta_avg should shrink eta.
    assert compute_eta(cfg, delta_avg=0.005) == pytest.approx(0.015)


def test_eta_override_bypasses_formula():
    cfg = ThresholdConfig(epsilon=0.02, K=10, eta_override=0.07)
    assert compute_eta(cfg) == 0.07


def test_tau_dkw_band():
    cfg = ThresholdConfig(K=10, alpha=0.05)
    eta = 0.02
    expected_band = math.sqrt(math.log(2 / 0.05) / (2 * 10))
    assert compute_tau(cfg, eta) == pytest.approx(eta + expected_band)


def test_tau_no_dkw():
    cfg = ThresholdConfig(K=10, use_dkw=False)
    assert compute_tau(cfg, 0.02) == 0.02


def test_tv_to_value_bound_uses_epsilon_t_formula():
    cfg = ThresholdConfig(r_max=2.0, gamma=0.5)
    tv = 0.1
    expected = (1.0 / (1.0 - cfg.gamma)) - (
        1.0 / (1.0 - cfg.gamma * (1.0 - tv))
    )
    assert tv_to_value_bound(tv, cfg) == pytest.approx(expected)


def test_tv_to_value_bound_is_zero_for_matching_distributions():
    cfg = ThresholdConfig(gamma=0.5)
    assert tv_to_value_bound(0.0, cfg) == pytest.approx(0.0)
