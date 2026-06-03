"""BST-like index over segments keyed by AvgLP_K.

PLAN.md line 33: `Insert(BST, key=AvgLP_K, value=s)`. `FindNearest` returns the
segment with the closest key to a query AvgLP_K value.  The implementation uses
Python's built-in ``bisect`` module so tests and lightweight environments do not
need the optional ``sortedcontainers`` package.
"""

from __future__ import annotations

import bisect
import threading
from typing import List, Optional, Tuple


class SegmentBST:
    """Sorted list of (key, segment_id) pairs.

    Ties on key are allowed; FindNearest returns the closest by absolute key
    difference, breaking ties by insertion order.
    """

    def __init__(self):
        self._items: List[Tuple[float, str]] = []
        self._lock = threading.Lock()

    def insert(self, key: float, segment_id: str) -> None:
        with self._lock:
            bisect.insort(self._items, (float(key), segment_id))

    def find_nearest(self, key: float) -> Optional[Tuple[float, str]]:
        """Return (key, segment_id) of the entry closest to `key`, or None if empty."""

        with self._lock:
            if len(self._items) == 0:
                return None
            target = (float(key), "")
            idx = bisect.bisect_left(self._items, target)
            candidates = []
            if idx < len(self._items):
                candidates.append(self._items[idx])
            if idx - 1 >= 0:
                candidates.append(self._items[idx - 1])
            best = min(candidates, key=lambda kv: abs(kv[0] - float(key)))
            return best

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
