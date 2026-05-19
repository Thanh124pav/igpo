"""Tests for SPO-counterfactual InGPO prune-rate accounting."""

from __future__ import annotations

import pytest

from treetune.ingpo.logging_helpers import (
    aggregate_tree_stats,
    per_depth_action_counts,
)


def _full_children(parent_id: str, depth: int, width: int):
    return [
        {
            "ingpo_segment_id": f"{parent_id}/{depth - 1}/{i}",
            "ingpo_action": "expand",
            "ingpo_depth": depth,
            "children": [],
        }
        for i in range(width)
    ]


def _root_with_depth1(actions, width=4, max_depth=2):
    children = []
    for i, action in enumerate(actions):
        node = {
            "ingpo_segment_id": f"root/0/{i}",
            "ingpo_action": action,
            "ingpo_depth": 1,
        }
        if action == "expand":
            node["children"] = _full_children(node["ingpo_segment_id"], 2, width)
        children.append(node)
    return {
        "ingpo_action": "expand",
        "ingpo_segment_id": "root",
        "ingpo_max_depth": max_depth,
        "ingpo_branch_factor_by_depth": {0: width, 1: width},
        "children": children,
    }


def test_prune_rate_counts_entire_pruned_subtree():
    # W=4, D=2. One depth-1 PRUNE removes itself + 4 SPO children = 5 nodes.
    # Full SPO tree has 4 + 16 = 20 non-root nodes, so prune_rate = 25%.
    tree = _root_with_depth1(["prune", "expand", "expand", "expand"])

    stats = aggregate_tree_stats(tree)

    assert stats["ingpo/spo_node_count"] == 20
    assert stats["ingpo/factual_node_count"] == 16
    assert stats["ingpo/virtual_pruned_spo_count"] == 4
    assert stats["ingpo/pruned_spo_count"] == 5
    assert stats["ingpo/prune_rate"] == pytest.approx(0.25)
    assert stats["ingpo/share_prune_rate"] == pytest.approx(0.0)
    assert stats["ingpo/total_prune_rate"] == pytest.approx(0.25)
    assert "ingpo/share_rate" not in stats


def test_share_prune_rate_counts_only_descendants():
    # W=4, D=2. One depth-1 SHARE is still emitted, but its 4 SPO children are
    # skipped, so share_prune_rate = 4 / 20 = 20%.
    tree = _root_with_depth1(["share", "expand", "expand", "expand"])

    stats = aggregate_tree_stats(tree)

    assert stats["ingpo/spo_node_count"] == 20
    assert stats["ingpo/factual_node_count"] == 16
    assert stats["ingpo/virtual_pruned_spo_count"] == 4
    assert stats["ingpo/pruned_spo_count"] == 0
    assert stats["ingpo/share_pruned_spo_count"] == 4
    assert stats["ingpo/prune_rate"] == pytest.approx(0.0)
    assert stats["ingpo/share_prune_rate"] == pytest.approx(0.20)
    assert stats["ingpo/total_prune_rate"] == pytest.approx(0.20)


def test_total_prune_rate_combines_prune_and_share_prune():
    # W=4, D=2. One depth-1 PRUNE removes 5 nodes; one depth-1 SHARE skips 4
    # descendants. Total prune rate = (5 + 4) / 20 = 45%.
    tree = _root_with_depth1(["prune", "share", "expand", "expand"])

    stats = aggregate_tree_stats(tree)

    assert stats["ingpo/pruned_spo_count"] == 5
    assert stats["ingpo/share_pruned_spo_count"] == 4
    assert stats["ingpo/total_pruned_spo_count"] == 9
    assert stats["ingpo/total_prune_rate"] == pytest.approx(0.45)


def test_spo_node_count_uses_factual_nodes_when_branches_end_early():
    # W=4, D=2, but expanded branches terminate early and only produce two
    # factual children each. Denominator = 4 depth-1 factual nodes + 4 factual
    # depth-2 nodes + 8 virtual descendants under PRUNE/SHARE = 16, not 20.
    tree = _root_with_depth1(["prune", "expand", "expand", "share"])
    tree["children"][1]["children"] = tree["children"][1]["children"][:2]
    tree["children"][2]["children"] = tree["children"][2]["children"][:2]

    stats = aggregate_tree_stats(tree)

    assert stats["ingpo/spo_node_count"] == 16
    assert stats["ingpo/factual_node_count"] == 8
    assert stats["ingpo/virtual_pruned_spo_count"] == 8
    assert stats["ingpo/pruned_spo_count"] == 5
    assert stats["ingpo/share_pruned_spo_count"] == 4
    assert stats["ingpo/total_prune_rate"] == pytest.approx(9 / 16)


def test_aggregate_counts_exclude_root_but_per_depth_keeps_raw_breakdown():
    tree = _root_with_depth1(["prune", "share", "expand", "expand"])
    stats = aggregate_tree_stats(tree)
    per_depth = per_depth_action_counts(tree)

    assert stats["ingpo/expanded_count"] == 10  # two depth-1 + eight depth-2
    assert stats["ingpo/shared_count"] == 1
    assert stats["ingpo/pruned_count"] == 1
    assert per_depth["ingpo/depth_1/share_count"] == 1
    assert per_depth["ingpo/depth_1/prune_count"] == 1
    assert "ingpo/depth_1/share_rate" not in per_depth
    assert "ingpo/depth_1/prune_rate" not in per_depth


def test_empty_tree_returns_empty_dict():
    assert aggregate_tree_stats({}) == {}
    assert aggregate_tree_stats({"children": [{}]}) == {}
