"""Unit tests for the BudgetAllocator module.

Covers the four key behaviours required by the spec:
  1. Equal split of remaining budget across siblings.
  2. Single PRUNE/SHARE redistributes leftover to surviving siblings.
  3. When all siblings of a node terminate, leftover bubbles up to aunts.
  4. The cascade walks across multiple levels until it finds live cousins.
"""

from __future__ import annotations

from ingpo_ext.core.budget import BudgetAllocator


def _tok(text: str):
    """Stand-in tokenizer: one token per character. Deterministic & fast."""
    return list(text)


def test_equal_split_of_remaining_budget():
    alloc = BudgetAllocator(total=1000, tokenize=_tok)
    root_bn = alloc.attach_root("rootprompt")  # 10 "tokens"
    children = alloc.allocate_children(root_bn, n_children=4)
    # remaining = 1000 - 10 = 990; share = 247
    assert [c.initial for c in children] == [247, 247, 247, 247]
    assert root_bn.children == children


def test_release_redistributes_to_live_siblings():
    alloc = BudgetAllocator(total=1000, tokenize=_tok)
    root_bn = alloc.attach_root("x" * 10)
    c0, c1, c2 = alloc.allocate_children(root_bn, 3)
    # c0 spends 100 then is pruned. leftover = 330 - 100 = 230, split 115/115.
    alloc.record_used(c0, "x" * 100)
    alloc.release(c0)
    assert c1.initial == 330 + 115
    assert c2.initial == 330 + 115
    assert alloc.released_total == 230


def test_release_then_release_cascades_to_aunts():
    """All children of a parent terminate ⇒ leftover should auto-cascade up
    to live siblings of the parent (the aunts of the terminated children)."""
    alloc = BudgetAllocator(total=1200, tokenize=_tok)
    root_bn = alloc.attach_root("")
    aunt, parent = alloc.allocate_children(root_bn, 2)
    parent.used = 50
    g0, g1 = alloc.allocate_children(parent, 2)  # share = (600-50)//2 = 275
    # g0 spends 50 then is pruned ⇒ leftover 225 flows to g1.
    alloc.record_used(g0, "x" * 50)
    alloc.release(g0)
    assert g1.initial == 275 + 225

    # g1 also gets pruned. With g0 closed there is no live sibling, so the
    # leftover bubbles automatically up to live aunts.
    alloc.record_used(g1, "x" * 50)
    alloc.release(g1)
    assert aunt.initial > 600
    assert alloc.bubbled_up_total > 0


def test_cascade_walks_multiple_levels():
    """Cascade should keep climbing when every sibling at a level closes and
    then flow down into the live cousin subtree."""
    alloc = BudgetAllocator(total=400, tokenize=_tok)
    root_bn = alloc.attach_root("")
    # root --> [A, B] ; A --> [A0, A1] ; B --> [B0]
    a, b = alloc.allocate_children(root_bn, 2)         # share 200 each
    a0, a1 = alloc.allocate_children(a, 2)             # share 100 each
    (b0,) = alloc.allocate_children(b, 1)              # share 200
    # Burn a tiny bit on the grandchildren, then prune all of them.
    for g in (a0, a1):
        alloc.record_used(g, "x" * 10)
        alloc.release(g)
    # Subtree A still holds budget. Closing A bubbles it up to B's children
    # (since B has already been expanded, the bonus flows down to live B0).
    a.closed = True
    alloc.maybe_bubble_up(a)
    assert b0.initial > 200
    assert alloc.bubbled_up_total > 0


def test_as_dict_reports_totals():
    alloc = BudgetAllocator(total=100, tokenize=_tok)
    root_bn = alloc.attach_root("abc")  # 3 tokens used by root
    c0, c1 = alloc.allocate_children(root_bn, 2)
    alloc.record_used(c0, "xxxxx")  # 5
    alloc.release(c0)
    d = alloc.as_dict()
    assert d["ingpo/budget_total"] == 100
    assert d["ingpo/budget_used"] >= 3 + 5
    assert d["ingpo/budget_redistributed"] > 0
