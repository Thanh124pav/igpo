import math

import pytest

from treetune.ingpo.budget_allocation import (
    allocate_branch_factors,
    reward_variance_from_pair_tvs,
    simulation_lemma_gap,
)


def test_simulation_lemma_gap_formula():
    tv = 0.2
    gamma = 0.5
    expected = gamma * tv / ((1 - gamma) * (1 - gamma + tv))
    assert simulation_lemma_gap(tv, gamma) == pytest.approx(expected)


def test_reward_variance_uses_ordered_pair_normalization():
    pair_tvs = {(0, 1): 0.2, (0, 2): 0.4, (1, 2): 0.1}
    gamma = 0.5
    expected = sum(2 * simulation_lemma_gap(tv, gamma) ** 2 for tv in pair_tvs.values()) / (
        2 * 3 * 2
    )
    assert reward_variance_from_pair_tvs(pair_tvs, n=3, gamma=gamma) == pytest.approx(expected)


def test_allocate_branch_factors_keeps_floor_underallocation():
    nodes = [
        {"ingpo_segment_id": "a", "ingpo_reward_variance": 0.0},
        {"ingpo_segment_id": "b", "ingpo_reward_variance": 1.0},
    ]
    summary = allocate_branch_factors(nodes, total_budget=5, lambda_=0.02)
    assert summary.allocated_budget <= 5
    assert summary.underallocated_budget == 5 - summary.allocated_budget
    assert summary.allocations["b"] >= summary.allocations["a"]


def test_allocate_branch_factors_handles_zero_budget_and_fallback_ids():
    nodes = [
        {"ingpo_reward_variance": 0.1},
        {"id": "explicit", "ingpo_reward_variance": 0.2},
    ]

    summary = allocate_branch_factors(nodes, total_budget=-3, lambda_=0.02)

    assert summary.requested_budget == 0
    assert summary.allocated_budget == 0
    assert summary.underallocated_budget == 0
    assert summary.allocations == {"node_0": 0, "explicit": 0}


def test_reward_variance_and_gap_clamp_degenerate_inputs():
    assert reward_variance_from_pair_tvs({}, n=1, gamma=0.5) == 0.0
    assert simulation_lemma_gap(tv=-1.0, gamma=0.5) == 0.0
    assert math.isfinite(simulation_lemma_gap(tv=0.2, gamma=1.0))
