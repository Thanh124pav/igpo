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


class RecordingExpander:
    def __init__(self):
        self.calls = []

    async def expand(self, *args, **kwargs):
        self.calls.append(kwargs)
        branch_factor = kwargs["branch_factor"]
        prefix = kwargs["prefix"]
        depth = kwargs["depth"]
        return [
            {
                "text": f" c{i}",
                "full_text": f"{prefix} c{i}",
                "sum_logprobs": float(i),
                "finish_reason": "length",
                "depth": depth + 1,
            }
            for i in range(branch_factor)
        ]


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


def test_pair_tvs_can_use_half_factor():
    estimator = ConditionalTVEstimator(
        scorer=FakeScorer(),
        node_expander=FakeExpander(),
        gamma=0.5,
        n_tv_estimates=2,
        tv_includes_half_factor=True,
    )

    pair_tvs = estimator._pair_tvs([[1.0, 0.0], [0.0, 1.0]])

    assert pair_tvs[(0, 1)] == pytest.approx(1.0)


def test_estimate_for_parent_generates_subnode_samples_with_budgeted_expansion():
    scorer = FakeScorer()
    expander = RecordingExpander()
    estimator = ConditionalTVEstimator(
        scorer=scorer,
        node_expander=expander,
        gamma=0.5,
        n_tv_estimates=3,
        first_phase_tokens=11,
        second_phase_tokens=7,
    )

    result = asyncio.run(estimator.estimate_for_parent({"full_text": "root"}, depth=0))

    assert len(result.samples) == 3
    assert expander.calls[0]["branch_factor"] == 3
    assert expander.calls[0]["max_tokens"] == 11
    assert all(call["branch_factor"] == 1 for call in expander.calls[1:])
    assert all(call["max_tokens"] == 7 for call in expander.calls[1:])
    assert result.pair_tvs


class MixedFinishExpander:
    def __init__(self):
        self.calls = []

    async def expand(self, *args, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            prefix = kwargs["prefix"]
            return [
                {
                    "text": " done",
                    "full_text": f"{prefix} done",
                    "finish_reason": "stop",
                },
                {
                    "text": " partial",
                    "full_text": f"{prefix} partial",
                    "finish_reason": "length",
                },
            ]
        prefix = kwargs["prefix"]
        return [
            {
                "text": " continuation",
                "full_text": f"{prefix} continuation",
                "finish_reason": "stop",
            }
        ]


def test_estimate_does_not_continue_terminal_first_phase_nodes():
    expander = MixedFinishExpander()
    estimator = ConditionalTVEstimator(
        scorer=FakeScorer(),
        node_expander=expander,
        gamma=0.5,
        n_tv_estimates=2,
    )

    result = asyncio.run(estimator.estimate_for_parent({"full_text": "root"}, depth=0))

    assert len(expander.calls) == 2
    assert expander.calls[1]["prefix"] == "root partial"
    assert [candidate["finish_reason"] for candidate in result.candidates] == [
        "stop",
        "length",
    ]
    assert len(result.samples) == 1
