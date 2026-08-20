"""Budget allocation helpers for InGPO.

The allocation objective is to reduce the variance of a mean-like estimator
under a fixed rollout budget.  If ``v_i`` is a variance proxy, the natural
Neyman-style weight is ``sqrt(v_i)`` (or ``sqrt(max(v_i-lambda, 0))`` when a
variance threshold is explicitly requested), not ``sqrt(v_i**2-lambda)``.

The allocator is deliberately budget conserving: after minimum allocations it
uses largest-remainder apportionment, and if all adaptive weights are zero it
falls back to uniform allocation instead of silently dropping rollout budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple


PairKey = Tuple[int, int]


@dataclass(frozen=True)
class AllocationSummary:
    allocations: Dict[str, int]
    weights: Dict[str, float]
    requested_budget: int
    allocated_budget: int
    underallocated_budget: int


def simulation_lemma_gap(tv: float, gamma: float) -> float:
    """Return the simulation-lemma value-gap proxy for standard TV distance."""

    gamma = min(max(float(gamma), 0.0), 1.0 - 1e-8)
    tv = min(max(float(tv), 0.0), 1.0)
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
    """Compute a pairwise squared value-gap proxy.

    The result is a variance *proxy*, not an unbiased estimate of PPO gradient
    variance.  Its units are treated as reward-variance units by the allocator.
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


def _largest_remainder_allocate(
    ids: Sequence[str], weights: Mapping[str, float], budget: int
) -> Dict[str, int]:
    """Apportion ``budget`` units using Hamilton/largest-remainder rounding."""

    budget = max(int(budget), 0)
    if not ids or budget == 0:
        return {node_id: 0 for node_id in ids}
    weight_sum = sum(max(float(weights.get(node_id, 0.0)), 0.0) for node_id in ids)
    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
        return {
            node_id: (budget // len(ids)) + (1 if idx < budget % len(ids) else 0)
            for idx, node_id in enumerate(ids)
        }

    quotas = {
        node_id: budget * max(float(weights.get(node_id, 0.0)), 0.0) / weight_sum
        for node_id in ids
    }
    allocations = {node_id: int(math.floor(quota)) for node_id, quota in quotas.items()}
    remainder = budget - sum(allocations.values())
    ranked = sorted(
        ids,
        key=lambda node_id: (quotas[node_id] - allocations[node_id], -ids.index(node_id)),
        reverse=True,
    )
    for node_id in ranked[:remainder]:
        allocations[node_id] += 1
    return allocations


def allocate_branch_factors(
    nodes: Sequence[Mapping[str, Any]],
    *,
    total_budget: int,
    lambda_: float = 0.0,
    n_min: int = 1,
    allocation_weight_mode: str = "std",
) -> AllocationSummary:
    """Allocate a fixed branch budget across frontier nodes.

    ``ingpo_reward_variance`` is interpreted as ``v_i``.  The default
    ``allocation_weight_mode='std'`` uses the Neyman-style weight
    ``sqrt(max(v_i - lambda_, 0))``.  ``allocation_weight_mode='variance'`` is
    retained as an explicit ablation and uses ``max(v_i-lambda_, 0)``.

    Every node receives up to ``n_min`` branches while budget remains.  The
    remainder is apportioned with largest-remainder rounding.  If all adaptive
    weights are zero, the remainder is distributed uniformly so that budget is
    not lost merely because the proxy is uninformative.
    """

    if allocation_weight_mode not in {"std", "variance"}:
        raise ValueError(
            "allocation_weight_mode must be either 'std' or 'variance', "
            f"got {allocation_weight_mode!r}"
        )

    total_budget = max(int(total_budget), 0)
    lambda_ = max(float(lambda_), 0.0)
    n_min = max(int(n_min), 0)
    node_ids = [_node_id(node, idx) for idx, node in enumerate(nodes)]
    if not node_ids:
        return AllocationSummary({}, {}, total_budget, 0, total_budget)

    weights: Dict[str, float] = {}
    for node_id, node in zip(node_ids, nodes):
        variance = max(float(node.get("ingpo_reward_variance", 0.0) or 0.0), 0.0)
        margin = max(variance - lambda_, 0.0)
        if allocation_weight_mode == "std":
            weights[node_id] = math.sqrt(margin)
        else:
            weights[node_id] = margin

    # Reserve the minimum fairly, but never exceed the requested budget.
    allocations = {node_id: 0 for node_id in node_ids}
    remaining = total_budget
    if n_min > 0:
        for node_id in node_ids:
            if remaining <= 0:
                break
            grant = min(n_min, remaining)
            allocations[node_id] = grant
            remaining -= grant

    # Adaptive allocation goes only to nodes with positive adaptive weight.
    eligible_ids = [node_id for node_id in node_ids if weights[node_id] > 0.0]
    adaptive = _largest_remainder_allocate(eligible_ids, weights, remaining)
    for node_id, grant in adaptive.items():
        allocations[node_id] += grant

    # If the threshold made every weight zero, preserve the budget uniformly.
    if remaining > 0 and not eligible_ids:
        uniform = _largest_remainder_allocate(node_ids, {}, remaining)
        for node_id, grant in uniform.items():
            allocations[node_id] += grant

    allocated = sum(allocations.values())
    return AllocationSummary(
        allocations=allocations,
        weights=weights,
        requested_budget=total_budget,
        allocated_budget=allocated,
        underallocated_budget=max(total_budget - allocated, 0),
    )
