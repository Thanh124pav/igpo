"""Token-budget allocator for InGPO tree construction.

Each tree starts with a total budget equal to the model's context size. The
budget is split equally across children at every depth. When a child is
PRUNED or SHARED, its unused share is returned to its still-live siblings;
when *all* siblings of a node terminate, the parent's leftover bubbles up
to the parent's siblings (i.e. the pruning current node's aunts/uncles).

The allocator is purely bookkeeping — the inference strategy is responsible
for honouring the returned `max_tokens` per child when it calls
`node_expander.expand(...)`. We do not try to truncate already-generated
text; we only cap the next round's generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class BudgetNode:
    initial: int = 0          # tokens the subtree rooted here may still spend
    used: int = 0             # tokens already spent inside this subtree (incl. self)
    released: int = 0         # leftover routed up from terminated descendants
    parent: Optional["BudgetNode"] = None
    children: List["BudgetNode"] = field(default_factory=list)
    closed: bool = False      # true once this node will spend no more tokens

    @property
    def remaining(self) -> int:
        return max(self.initial - self.used + self.released, 0)


class BudgetAllocator:
    """Tree-wide accountant of token budget for InGPO tree construction."""

    def __init__(self, total: int, tokenize: Callable[[str], List[int]]):
        if total <= 0:
            raise ValueError(f"total budget must be positive, got {total}")
        self.total = int(total)
        self.tokenize = tokenize
        self.root = BudgetNode(initial=self.total)
        self.released_total = 0
        self.bubbled_up_total = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach_root(self, prefix: str) -> BudgetNode:
        n = self._count(prefix)
        self.root.used = min(n, self.root.initial)
        return self.root

    def allocate_children(
        self, parent: BudgetNode, n_children: int
    ) -> List[BudgetNode]:
        """Equal split of parent's remaining budget across n_children."""

        if n_children <= 0:
            return []
        share = parent.remaining // n_children
        children = [BudgetNode(initial=share, parent=parent) for _ in range(n_children)]
        parent.children = children
        return children

    def record_used(self, node: BudgetNode, text: str) -> int:
        """Charge the tokens of `text` against this node's used budget.

        Returns the number of tokens charged (after clamping to the node's
        remaining budget so used never exceeds initial+released).
        """

        n = self._count(text)
        capacity = node.initial + node.released - node.used
        charged = max(0, min(n, capacity))
        node.used += charged
        return charged

    def release(self, node: BudgetNode) -> int:
        """Terminate `node` (PRUNE / SHARE / leaf) and return its leftover.

        Leftover is distributed equally across still-live siblings. If no
        live siblings remain, the leftover bubbles up to the grandparent's
        live children.
        """

        if node.closed:
            return 0
        node.closed = True
        leftover = max(node.initial - node.used + node.released, 0)
        node.released = 0
        # Mark as fully spent so subsequent accounting is idempotent.
        node.initial = node.used
        if leftover <= 0 or node.parent is None:
            return leftover
        self.released_total += leftover
        self._distribute_to_live_siblings(node.parent, exclude=node, amount=leftover)
        return leftover

    def maybe_bubble_up(self, node: BudgetNode) -> int:
        """Called after all of `node`'s children finished. If they left any
        budget on the table and node itself is closed (cannot use it), route
        it up to node's siblings."""

        if node.parent is None:
            return 0
        # Sum of leftover sitting on this node:
        leftover = node.released
        if node.closed:
            leftover += max(node.initial - node.used, 0)
            node.initial = node.used
        node.released = 0
        if leftover <= 0:
            return 0
        self.bubbled_up_total += leftover
        self._distribute_to_live_siblings(node.parent, exclude=node, amount=leftover)
        return leftover

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _distribute_to_live_siblings(
        self, parent: BudgetNode, exclude: BudgetNode, amount: int
    ) -> None:
        live = [c for c in parent.children if c is not exclude and not c.closed]
        if live:
            bonus = amount // len(live)
            for c in live:
                self._inject(c, bonus)
            # Remainder stays with the parent for a future round.
            parent.released += amount - bonus * len(live)
            return
        # No live siblings - bubble further up the tree.
        parent.released += amount
        if parent.parent is not None:
            self.maybe_bubble_up(parent)

    def _inject(self, node: BudgetNode, amount: int) -> None:
        """Inject `amount` of fresh budget into `node`.

        If the node has already been expanded (has children), the budget is
        flowed down to its still-live children so descendants that are
        already running can actually consume it. Otherwise it sits on the
        node and will be split when `allocate_children` is eventually called.
        """

        if amount <= 0 or node.closed:
            return
        live = [c for c in node.children if not c.closed]
        if not live:
            node.initial += amount
            return
        share = amount // len(live)
        for c in live:
            self._inject(c, share)
        node.released += amount - share * len(live)

    def _count(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenize(text))
        except Exception:
            # Fall back to a rough character/4 estimate if tokenizer fails.
            return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict[str, float]:
        return {
            "ingpo/budget_total": self.total,
            "ingpo/budget_used": self._used_total(self.root),
            "ingpo/budget_leftover": self._leftover_total(self.root),
            "ingpo/budget_redistributed": self.released_total,
            "ingpo/budget_bubbled_up": self.bubbled_up_total,
        }

    def _used_total(self, node: BudgetNode) -> int:
        return node.used + sum(self._used_total(c) for c in node.children)

    def _leftover_total(self, node: BudgetNode) -> int:
        own = max(node.initial - node.used, 0) + node.released
        return own + sum(self._leftover_total(c) for c in node.children)
