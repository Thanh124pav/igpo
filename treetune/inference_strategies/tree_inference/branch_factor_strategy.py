from typing import Any, Dict, List, Optional, Sequence

from treetune.common import Registrable

Node = Dict[str, Any]
from treetune.logging_utils import get_logger

logger = get_logger(__name__)


class BranchFactorStrategy(Registrable):
    def decide_branch_factor(self, node: Node) -> int:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.decide_branch_factor(*args, **kwargs)


@BranchFactorStrategy.register("constant", exist_ok=True)
class ConstantBranchFactor(BranchFactorStrategy):
    def __init__(self, constant: int):
        super().__init__()
        self.constant = constant

    def decide_branch_factor(self, node: Node) -> int:
        return self.constant


@BranchFactorStrategy.register("list", exist_ok=True)
class ListBranchFactor(BranchFactorStrategy):
    def __init__(
        self,
        branch_factors: Optional[List[Dict[str, int]]] = None,
        tree_shape: Optional[Sequence[int]] = None,
    ):
        """
        New format: ``tree_shape`` is the cumulative number of nodes at each
        depth.  For example, an old 6-6-6 tree is represented as
        ``tree_shape=[6, 36, 216]`` and is converted to branch factors
        ``[6, 6, 6]``.

        Legacy format: ``branch_factors`` is a list of ``{"depth": d,
        "branch_factor": b}`` entries.  It is still supported and logs a
        warning so older configs keep running.
        """
        super().__init__()
        if tree_shape is not None:
            self.branch_factors = self._from_tree_shape(tree_shape)
            return

        if branch_factors is None:
            raise ValueError(
                "ListBranchFactor requires either tree_shape or legacy branch_factors"
            )
        logger.warning(
            "Using legacy branch_factors config; prefer tree_shape cumulative node counts."
        )
        assert len(branch_factors) > 0, "branch_factors should not be empty, use at least {'depth': 0, 'branch_factor': x}}"
        self.branch_factors = sorted(branch_factors, key=lambda x: x["depth"])
        if self.branch_factors[0]['depth'] != 0:
            raise ValueError("The first depth must be 0")

    def _from_tree_shape(self, tree_shape: Sequence[int]) -> List[Dict[str, int]]:
        if not tree_shape:
            raise ValueError("tree_shape should not be empty")
        cumulative = [int(x) for x in tree_shape]
        if any(x <= 0 for x in cumulative):
            raise ValueError(f"tree_shape values must be positive: {tree_shape}")
        branch_factors: List[Dict[str, int]] = []
        prev = 1
        for depth, count in enumerate(cumulative):
            if count % prev != 0:
                logger.warning(
                    "tree_shape[%s]=%s is not divisible by previous depth node count %s; using floor division.",
                    depth,
                    count,
                    prev,
                )
            branch_factor = max(count // prev, 1)
            branch_factors.append({"depth": depth, "branch_factor": branch_factor})
            prev = count
        return branch_factors

    def decide_branch_factor(self, node: Node) -> int:
        depth = node["depth"]
        if len(self.branch_factors) == 1:
            return self.branch_factors[0]['branch_factor']

        for i in range(len(self.branch_factors)):
            if depth < self.branch_factors[i]['depth']:
                return self.branch_factors[i-1]['branch_factor']

        return self.branch_factors[-1]['branch_factor']


@BranchFactorStrategy.register("fish_bone", exist_ok=True)
class FishBoneBranchFactor(BranchFactorStrategy):
    def __init__(self, fish_bone_samples: int):
        super().__init__()
        self.fish_bone_samples = fish_bone_samples

    def decide_branch_factor(self, node: Node) -> int:
        if 'is_in_spine' in node and node['is_in_spine'] is True:
            return self.fish_bone_samples
        else:
            return 1
