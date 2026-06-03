import asyncio

import pytest

from treetune.ingpo.tv_estimators import ConditionalTVEstimator, TVSample


class FakeScorer:
    def __init__(self):
        self.calls = []

    async def score_one(self, prefix, continuation):
        self.calls.append((prefix, continuation))
        return float(len(prefix) - len(continuation))


class FakeExpander:
    async def expand(self, *args, **kwargs):
        raise AssertionError("not used")


def test_conditional_tv_estimator_caches_logp_matrix_scores():
    scorer = FakeScorer()
    estimator = ConditionalTVEstimator(
        scorer=scorer,
        node_expander=FakeExpander(),
        gamma=0.5,
        n_tv_estimates=2,
    )
    samples = [
        TVSample(first={"full_text": "p1"}, second={"text": "a"}),
        TVSample(first={"full_text": "p2"}, second={"text": "bb"}),
    ]

    async def go():
        first = await estimator.estimate_from_samples(samples)
        second = await estimator.estimate_from_samples(samples)
        return first, second

    first, second = asyncio.run(go())
    assert len(scorer.calls) == 4  # 2 prefixes x 2 support continuations, only once.
    assert first.logp_matrix == second.logp_matrix
    assert set(first.pair_tvs) == {(0, 1)}
