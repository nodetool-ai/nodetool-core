import pytest

from nodetool.agents.agent_evaluator import (
    AgentEvaluator,
    SingleAgentRunResult,
    Usage,
)


class _FixedChoiceRng:
    def __init__(self, index: int) -> None:
        self._index = index

    def choice(self, seq):
        return seq[self._index]


@pytest.mark.asyncio
async def test_run_single_agent_random_problem_reports_success():
    problems = [(2, 4), (3, 6)]
    calls: list[tuple[str, str, int, list | None]] = []

    async def runner(*, provider_key, model, problem, tools=None):
        calls.append((provider_key, model, problem, tools))
        # Provide usage metadata to ensure it is preserved
        return SingleAgentRunResult(result=problem * 2, usage=Usage(input_tokens=11, output_tokens=7))

    evaluator = AgentEvaluator(
        models=[("provider", "model-a")],
        problems=problems,
        result_checker=lambda result, expected: result == expected,
        single_agent_runner=runner,
    )

    report = await evaluator.run_single_agent_random_problem(
        provider_key="provider",
        model="model-a",
        rng=_FixedChoiceRng(index=0),
    )

    assert len(calls) == 1
    assert calls[0][0] == "provider"
    assert calls[0][1] == "model-a"
    assert calls[0][2] == problems[0][0]
    assert calls[0][3] is None
    assert report.problem == problems[0][0]
    assert report.expected == problems[0][1]
    assert report.log.result == problems[0][1]
    assert report.log.correct is True
    assert report.usage.input_tokens == 11
    assert report.usage.output_tokens == 7
    assert report.log.runtime_seconds >= 0.0
