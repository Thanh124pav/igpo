from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence

from treetune.ingpo.budget_allocation import AllocationSummary, allocate_branch_factors


@dataclass
class BudgetQueue:
    queue_id: int
    nodes: List[Any] = field(default_factory=list)
    active: bool = False


class FlexibleBudgetScheduler:
    """Attach queue metadata without fragmenting the depth budget.

    Earlier versions allocated a floored sub-budget independently inside each
    queue.  That introduced avoidable under-allocation before the actual
    variance-based allocator ran.  Queue assignment is now metadata only; all
    ready nodes share one budget-conserving apportionment call.
    """

    def __init__(
        self,
        *,
        queue_count: int = 2,
        lambda_: float = 0.0,
        n_min: int = 1,
        allocation_weight_mode: str = "std",
    ):
        self.queues = [
            BudgetQueue(queue_id=i) for i in range(max(int(queue_count), 1))
        ]
        self.lambda_ = float(lambda_)
        self.n_min = max(int(n_min), 0)
        self.allocation_weight_mode = allocation_weight_mode

    def allocate(
        self,
        nodes: Sequence[Any],
        *,
        total_depth_budget: int,
    ) -> List[AllocationSummary]:
        for queue in self.queues:
            queue.nodes.clear()
            queue.active = False
        if not nodes:
            return []

        # Queue ids remain useful for logs and future async execution, but do
        # not affect the mathematical budget split.
        for idx, node in enumerate(nodes):
            queue = self.queues[idx % len(self.queues)]
            queue.nodes.append(node)
            if isinstance(node, dict):
                node["ingpo_budget_queue_id"] = queue.queue_id

        summary = allocate_branch_factors(
            nodes,
            total_budget=total_depth_budget,
            lambda_=self.lambda_,
            n_min=self.n_min,
            allocation_weight_mode=self.allocation_weight_mode,
        )
        return [summary]
