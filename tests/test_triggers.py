"""Triggers integration test using a fake LP scorer.

We craft a synthetic AnswerSet of m=6 strings, then feed each segment a
hand-picked LP vector via a stub scorer.  Verifies that:
  1. Identical segments fire SHARE.
  2. Segments with much lower AvgLP than parent fire PRUNE.
  3. Otherwise they EXPAND.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List

import pytest

from ingpo_ext.core.answer_set import AnswerSet
from ingpo_ext.core.lp_scorer import LPScorer
from ingpo_ext.core.thresholds import ThresholdConfig
from ingpo_ext.core.triggers import Action, TriggerEngine


@dataclass
class FakeScorer:
    """Returns a fixed LP vector indexed by (segment_id, idx)."""

    table: Dict[str, List[float]]
    cache: Dict = None

    def __post_init__(self):
        self.cache = {}

    async def score_batch(self, segment_id: str, prefix, y, indices):
        vec = self.table[segment_id]
        return [vec[i] for i in indices]


def make_engine(table, *, share=True, prune=True):
    Y = AnswerSet(problem_id="p", gold="g", y=["a", "b", "c", "d", "e", "f"])
    cfg = ThresholdConfig(K=3, epsilon=0.05, r_max=1.0, alpha=0.05)
    eng = TriggerEngine(
        answer_set=Y,
        scorer=FakeScorer(table=table),  # type: ignore[arg-type]
        thresholds=cfg,
        enable_share=share,
        enable_prune=prune,
        share_target="nearest",
    )
    return eng


def test_share_fires_for_identical_lp(monkeypatch):
    table = {
        "root": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "s1":   [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "s2":   [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    }
    eng = make_engine(table)

    async def go():
        await eng.register_root("root prefix")
        d1 = await eng.decide(segment_id="s1", parent_id="root", prefix="p1", is_leaf=False)
        d2 = await eng.decide(segment_id="s2", parent_id="root", prefix="p2", is_leaf=False)
        return d1, d2

    d1, d2 = asyncio.run(go())
    assert d1.action is Action.EXPAND  # nothing to share with yet
    assert d2.action is Action.SHARE
    assert d2.share_target in ("root", "s1")


def test_prune_fires_when_avg_lp_drops():
    """PRUNE still fires when parent is a real depth>=1 segment (not root)."""
    table = {
        "root":  [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "mid":   [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "child": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
    }
    eng = make_engine(table, share=False, prune=True)

    async def go():
        await eng.register_root("root prefix")
        # First register a non-root parent so the child can be compared
        # against it under the depth-1-skip rule.
        await eng.decide(segment_id="mid", parent_id="root", prefix="pmid", is_leaf=False)
        return await eng.decide(
            segment_id="child", parent_id="mid", prefix="p", is_leaf=False
        )

    d = asyncio.run(go())
    assert d.action is Action.PRUNE


def test_prune_skip_root_blocks_depth1_prune():
    """With prune_skip_root=True, a depth-1 child (parent==root) never PRUNEs
    even if its AvgLP is far below root's."""
    table = {
        "root":  [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "child": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
    }
    eng = make_engine(table, share=False, prune=True)
    assert eng.prune_skip_root is True  # default

    async def go():
        await eng.register_root("root prefix")
        return await eng.decide(
            segment_id="child", parent_id="root", prefix="p", is_leaf=False
        )

    d = asyncio.run(go())
    assert d.action is Action.EXPAND


def test_prune_skip_root_false_restores_legacy_behaviour():
    """Setting prune_skip_root=False brings back the original PRUNE-at-root
    behaviour (useful for ablations)."""
    table = {
        "root":  [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "child": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
    }
    eng = make_engine(table, share=False, prune=True)
    eng.prune_skip_root = False

    async def go():
        await eng.register_root("root prefix")
        return await eng.decide(
            segment_id="child", parent_id="root", prefix="p", is_leaf=False
        )

    d = asyncio.run(go())
    assert d.action is Action.PRUNE


def test_expand_when_neither():
    table = {
        "root": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "child":[-1.4, -1.4, -1.4, -1.4, -1.4, -1.4],
    }
    eng = make_engine(table, share=True, prune=True)

    async def go():
        await eng.register_root("root prefix")
        return await eng.decide(segment_id="child", parent_id="root", prefix="p", is_leaf=False)

    d = asyncio.run(go())
    # With prune_skip_root=True PRUNE cannot fire at depth=1; SHARE also
    # doesn't fire (constant -0.4 offset gives nontrivial TV_m) so the
    # decision must be EXPAND.
    assert d.action is Action.EXPAND
