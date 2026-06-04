"""Simulation-lemma budget allocation utilities for InGPO.

This module is intentionally independent from the legacy SHARE/PRUNE triggers.
It converts pairwise TV estimates into per-node reward variance and then
allocates a depth budget across nodes with the floor-only rule requested for
budget-allocation runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


PairKey = Tuple[int, int]


@dataclass(frozen=True)
class AllocationSummary:
    allocations: Dict[str, int]
    weights: Dict[str, float]
    requested_budget: int
    allocated_budget: int
    underallocated_budget: int


def simulation_lemma_gap(tv: float, gamma: float) -> float:
    """Return gamma*TV / ((1-gamma)*(1-gamma+TV))."""

    gamma = min(max(float(gamma), 0.0), 1.0 - 1e-8)
    tv = max(float(tv), 0.0)
    denom = (1.0 - gamma) * (1.0 - gamma + tv)
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return gamma * tv / denom


def reward_variance_from_pair_tvs(
    pair_tvs: Mapping[PairKey, float],
    *,
    n: int,
    gamma: float,
) -> float:
    """Compute Var(P) from unordered pairwise TV estimates.

    The requested formula is

        1 / (2*n*(n-1)) * sum_{i,j} gap(TV_ij)^2.

    ``pair_tvs`` normally stores unordered pairs with ``i < j``.  We multiply
    the unordered-pair contribution by two so the normalization matches the
    ordered ``sum_{i,j}``; diagonal terms are zero and omitted.
    """

    if n <= 1:
        return 0.0
    total = 0.0
    for tv in pair_tvs.values():
        gap = simulation_lemma_gap(tv, gamma)
        total += 2.0 * gap * gap
    return total / (2.0 * float(n) * float(n - 1))


def _node_id(node: Mapping[str, Any], fallback: int) -> str:
    return str(
        node.get("ingpo_segment_id")
        or node.get("segment_id")
        or node.get("id")
        or f"node_{fallback}"
    )


def allocate_branch_factors(
    nodes: Sequence[Mapping[str, Any]],
    *,
    total_budget: int,
    lambda_: float = 0.02,
) -> AllocationSummary:
    """Allocate branch factors with strict floor rounding.

    ``sigma_i^2`` is stored as ``ingpo_reward_variance``.  Therefore
    ``sigma_i^4 = Var(P_i)^2`` and the weight is
    ``(sigma_i^4 + lambda_) ** 0.25``.

    Leftover budget from floor rounding is intentionally not redistributed.
    """

    total_budget = max(int(total_budget), 0)
    lambda_ = max(float(lambda_), 0.0)
    weights: Dict[str, float] = {}
    for idx, node in enumerate(nodes):
        sigma2 = max(float(node.get("ingpo_reward_variance", 0.0) or 0.0), 0.0)
        sigma4 = sigma2 * sigma2
        weights[_node_id(node, idx)] = (sigma4 + lambda_) ** 0.25

    weight_sum = sum(weights.values())
    allocations: Dict[str, int] = {}
    if total_budget <= 0 or weight_sum <= 0.0:
        allocations = {node_id: 0 for node_id in weights}
    else:
        for node_id, weight in weights.items():
            allocations[node_id] = int(math.floor(total_budget * weight / weight_sum))

    allocated = sum(allocations.values())
    return AllocationSummary(
        allocations=allocations,
        weights=weights,
        requested_budget=total_budget,
        allocated_budget=allocated,
        underallocated_budget=max(total_budget - allocated, 0),
    )
