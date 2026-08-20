"""TV estimators for simulation-lemma budget allocation.

The estimators build and cache a conditional log-probability matrix
``log P(ss_k2 | ss_i1)``.  Pairwise TV values are computed from this matrix,
so each unique (prefix, continuation) scoring request is made at most once per
estimator instance.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


Node = Dict[str, Any]
from treetune.ingpo.budget_allocation import reward_variance_from_pair_tvs
from treetune.ingpo.local_value_share import stable_softmax


PairKey = Tuple[int, int]


@dataclass
class TVSample:
    first: Node
    second: Node


@dataclass
class TVEstimateResult:
    samples: List[TVSample]
    logp_matrix: List[List[float]]
    prob_matrix: List[List[float]]
    pair_tvs: Dict[PairKey, float]
    reward_variance: float
    candidates: List[Node] = field(default_factory=list)


class ConditionalTVEstimator:
    """Generate TV samples and compute pairwise TV from a cached matrix."""

    def __init__(
        self,
        *,
        scorer: Any,
        node_expander: Any,
        gamma: float,
        mode: str = "subnode",
        n_tv_estimates: int = 8,
        first_phase_tokens: int = 120,
        second_phase_tokens: int = 60,
        tv_includes_half_factor: bool = True,
    ):
        if mode not in {"subnode", "hierachical", "hierarchical"}:
            raise ValueError(f"Unsupported TV estimator mode: {mode}")
        self.scorer = scorer
        self.node_expander = node_expander
        self.gamma = float(gamma)
        self.mode = "hierachical" if mode == "hierarchical" else mode
        self.n_tv_estimates = max(int(n_tv_estimates), 2)
        self.first_phase_tokens = max(int(first_phase_tokens), 1)
        self.second_phase_tokens = max(int(second_phase_tokens), 1)
        self.tv_includes_half_factor = bool(tv_includes_half_factor)
        self._score_cache: Dict[Tuple[str, str], float] = {}

    async def estimate_for_parent(self, parent: Node, *, depth: int) -> TVEstimateResult:
        samples, candidates = await self._generate_samples(parent, depth=depth)
        result = await self.estimate_from_samples(samples)
        result.candidates = candidates
        return result

    async def estimate_from_samples(self, samples: Sequence[TVSample]) -> TVEstimateResult:
        samples = list(samples)
        n = len(samples)
        if n < 2:
            return TVEstimateResult(
                samples=samples,
                logp_matrix=[],
                prob_matrix=[],
                pair_tvs={},
                reward_variance=0.0,
            )

        prefixes = [sample.first.get("full_text", "") for sample in samples]
        support = [sample.second.get("text", "") for sample in samples]
        logp_matrix = await self._score_matrix(prefixes, support)
        prob_matrix = [list(stable_softmax(row)) for row in logp_matrix]
        pair_tvs = self._pair_tvs(prob_matrix)
        variance = reward_variance_from_pair_tvs(pair_tvs, n=n, gamma=self.gamma)
        return TVEstimateResult(
            samples=samples,
            logp_matrix=logp_matrix,
            prob_matrix=prob_matrix,
            pair_tvs=pair_tvs,
            reward_variance=variance,
        )

    async def _generate_samples(
        self, parent: Node, *, depth: int
    ) -> Tuple[List[TVSample], List[Node]]:
        if self.mode == "hierachical":
            first_count = max(2, int(math.ceil(math.sqrt(self.n_tv_estimates))))
            second_per_first = max(1, int(math.ceil(self.n_tv_estimates / first_count)))
        else:
            first_count = self.n_tv_estimates
            second_per_first = 1

        first_nodes = await self._expand(
            current_node=parent,
            prefix=parent.get("full_text", ""),
            depth=depth,
            max_tokens=self.first_phase_tokens,
            branch_factor=first_count,
        )

        # A non-`length` finish reason means the model reached a terminal
        # response during phase one.  Keep it as a reusable budget candidate,
        # but do not ask the model to continue from an already-finished prefix.
        continuable_first_nodes = [
            first for first in first_nodes if first.get("finish_reason") == "length"
        ]

        samples: List[TVSample] = []
        second_tasks = [
            asyncio.create_task(
                self._expand(
                    current_node=first,
                    prefix=first.get("full_text", ""),
                    depth=depth + 1,
                    max_tokens=self.second_phase_tokens,
                    branch_factor=second_per_first,
                )
            )
            for first in continuable_first_nodes
        ]
        second_batches = await asyncio.gather(*second_tasks) if second_tasks else []
        for first, seconds in zip(continuable_first_nodes, second_batches):
            for second in seconds:
                samples.append(TVSample(first=first, second=second))
                if len(samples) >= self.n_tv_estimates:
                    return samples, first_nodes
        return samples, first_nodes

    async def _expand(
        self,
        *,
        current_node: Node,
        prefix: str,
        depth: int,
        max_tokens: int,
        branch_factor: int,
    ) -> List[Node]:
        try:
            return await self.node_expander.expand(
                current_node=current_node,
                prefix=prefix,
                depth=depth,
                max_tokens=max_tokens,
                branch_factor=branch_factor,
            )
        except TypeError:
            return await self.node_expander.expand(
                current_node=current_node,
                prefix=prefix,
                depth=depth,
                max_tokens=max_tokens,
            )

    async def _score_matrix(
        self,
        prefixes: Sequence[str],
        support: Sequence[str],
    ) -> List[List[float]]:
        matrix: List[List[float]] = [[0.0 for _ in support] for _ in prefixes]
        pending: List[Tuple[int, int, asyncio.Task]] = []
        for i, prefix in enumerate(prefixes):
            for k, continuation in enumerate(support):
                key = (prefix, continuation)
                if key in self._score_cache:
                    matrix[i][k] = self._score_cache[key]
                else:
                    pending.append((i, k, asyncio.create_task(self.scorer.score_one(prefix, continuation))))
        if pending:
            values = await asyncio.gather(*(task for _, _, task in pending))
            for (i, k, _), value in zip(pending, values):
                value = float(value)
                self._score_cache[(prefixes[i], support[k])] = value
                matrix[i][k] = value
        return matrix

    def _pair_tvs(self, prob_matrix: Sequence[Sequence[float]]) -> Dict[PairKey, float]:
        pair_tvs: Dict[PairKey, float] = {}
        for i in range(len(prob_matrix)):
            pi = prob_matrix[i]
            for j in range(i + 1, len(prob_matrix)):
                pj = prob_matrix[j]
                # Total Variation is 0.5 * L1 distance.  Keep the switch only
                # for an explicit legacy ablation; the production default is
                # the mathematically standard definition.
                tv = sum(abs(a - b) for a, b in zip(pi, pj))
                if self.tv_includes_half_factor:
                    tv *= 0.5
                pair_tvs[(i, j)] = tv
        return pair_tvs


def rank_samples_by_score(samples: Sequence[TVSample]) -> List[TVSample]:
    """Return samples sorted by descending first-phase generation score."""

    def score(sample: TVSample) -> float:
        first = sample.first
        if first.get("sum_logprobs") is not None:
            return float(first["sum_logprobs"])
        return 0.0

    return sorted(samples, key=score, reverse=True)
