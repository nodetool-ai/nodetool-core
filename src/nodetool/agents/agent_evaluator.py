"""Generic agent evaluation framework with subprocess batch mode.

This module provides a reusable, provider-agnostic evaluator that invokes
agents via a standardized CLI runner in isolated subprocesses for batch runs.
It measures token usage and runtime, and computes correctness via a pluggable
result checker. For quick spot-checks it also exposes a helper that can execute
a single (model, problem) pair in-process without going through the CLI runner.

Key components:
- AgentEvaluator: Orchestrates parallel, subprocess-based evaluations, and
  provides an in-process helper for ad-hoc single runs.
- ResultCheckerFn: Callback to determine correctness of a result relative to an
  expected value.

All (model, problem) pairs are executed in separate subprocesses with stdout
and stderr redirected to per-run log files under /tmp and results passed back
through a JSON file written by the CLI runner.
"""

import asyncio
import asyncio.subprocess as asp
import json
import os
import random
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

ProviderKey = str
ModelName = str


@dataclass
class ModelStats:
    """Aggregated metrics collected per model across all problems."""

    finished: int = 0
    correct: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_runtime_seconds: float = 0.0


@dataclass
class Usage:
    """Minimal usage summary for a single agent run."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LogEntry:
    """One row for the log table covering a single (model, problem) execution."""

    provider_key: str
    model: str
    problem: str
    result: Any | None
    correct: bool | None
    runtime_seconds: float


@dataclass
class EvaluationResult:
    """Full evaluation output returned by AgentEvaluator.evaluate()."""

    stats: dict[str, ModelStats]
    logs: list[LogEntry]


@dataclass
class SingleAgentRunResult:
    """Outcome returned by a single in-process agent execution."""

    result: Any
    usage: Usage | dict[str, Any] | None = None


@dataclass
class SingleRunReport:
    """Structured report produced by run_single_agent_random_problem."""

    log: LogEntry
    usage: Usage
    problem: Any
    expected: Any


class ChoiceGenerator(Protocol):
    """Protocol for objects providing a choice() helper."""

    def choice(self, seq: Sequence[Any]) -> Any: ...


class SingleAgentRunner(Protocol):
    """Callable that executes a single agent in-process."""

    def __call__(
        self,
        *,
        provider_key: ProviderKey,
        model: ModelName,
        problem: Any,
        tools: Sequence[Any] | None = None,
    ) -> Awaitable[SingleAgentRunResult]: ...


ResultCheckerFn = Callable[[Any, Any], bool]


# Batch execution continues to use subprocesses. The helper method
# run_single_agent_random_problem provides an opt-in in-process path.


class AgentEvaluator:
    """Parallel, subprocess-based evaluator for arbitrary agents.

    Parameters
    - models: sequence of (provider_key, model_name) pairs
    - problems: iterable of problems. Each item can be either a problem payload or
      a tuple/list (problem_payload, expected_value)
    - result_checker: callback that returns True/False for (result, expected)
    - single_agent_runner: optional in-process executor used by
      run_single_agent_random_problem
    - concurrency: max concurrent subprocesses
    """

    def __init__(
        self,
        models: Sequence[tuple[ProviderKey, ModelName]],
        problems: Iterable[Any],
        result_checker: ResultCheckerFn,
        tools: Sequence[Any] | None = None,
        concurrency: int = 8,
        on_update: Callable[[dict[str, ModelStats], list[LogEntry]], None] | None = None,
        subprocess_runner_path: str | None = None,
        subprocess_agent: str | None = None,
        single_agent_runner: SingleAgentRunner | None = None,
    ) -> None:
        self.models = list(models)
        self.problems = list(problems)
        self.result_checker = result_checker
        self.tools = list(tools) if tools is not None else []
        self.concurrency = int(concurrency)
        self.on_update = on_update
        self.subprocess_runner_path = subprocess_runner_path
        self.subprocess_agent = subprocess_agent
        self.single_agent_runner = single_agent_runner

    async def run_single_agent_random_problem(
        self,
        provider_key: ProviderKey,
        model: ModelName,
        *,
        run_agent_fn: SingleAgentRunner | None = None,
        rng: ChoiceGenerator | None = None,
    ) -> SingleRunReport:
        """Execute a single agent for a random problem in-process.

        Args:
            provider_key: Provider identifier for the agent.
            model: Model name to use with the provider.
            run_agent_fn: Optional override callable that executes the agent.
                If omitted, the evaluator must have been constructed with
                single_agent_runner.
            rng: Optional random generator to control problem selection.

        Returns:
            SingleRunReport with log entry, usage, original problem payload,
            and expected value (if any).
        """
        if not self.problems:
            raise ValueError("No problems available to evaluate.")

        chooser = cast("ChoiceGenerator", rng or random)
        problem_entry = chooser.choice(self.problems)
        if isinstance(problem_entry, tuple | list) and len(problem_entry) >= 2:
            problem_payload, expected = problem_entry[0], problem_entry[1]
        else:
            problem_payload, expected = problem_entry, None

        runner = run_agent_fn or self.single_agent_runner
        if runner is None:
            raise RuntimeError("Provide run_agent_fn or configure single_agent_runner to execute agents in-process.")

        start_time = time.perf_counter()
        outcome = await runner(
            provider_key=provider_key,
            model=model,
            problem=problem_payload,
            tools=self.tools if self.tools else None,
        )
        elapsed_seconds = time.perf_counter() - start_time

        if not isinstance(outcome, SingleAgentRunResult):
            raise TypeError("Single agent runner must return a SingleAgentRunResult instance.")

        def _coerce_usage(raw_usage: Usage | dict[str, Any] | None) -> Usage:
            if raw_usage is None:
                return Usage()
            if isinstance(raw_usage, Usage):
                return raw_usage
            if isinstance(raw_usage, dict):
                return Usage(
                    input_tokens=int(raw_usage.get("input_tokens", 0)),
                    output_tokens=int(raw_usage.get("output_tokens", 0)),
                )

        usage = _coerce_usage(outcome.usage)
        is_correct: bool | None = None
        if expected is not None:
            is_correct = bool(self.result_checker(outcome.result, expected))

        problem_text = str(
            problem_payload[0]
            if isinstance(problem_payload, tuple | list) and len(problem_payload) >= 1
            else problem_payload
        )
        log_entry = LogEntry(
            provider_key=provider_key,
            model=model,
            problem=problem_text,
            result=outcome.result,
            correct=(True if is_correct else (False if expected is not None else None)),
            runtime_seconds=float(elapsed_seconds),
        )
        return SingleRunReport(log=log_entry, usage=usage, problem=problem_payload, expected=expected)

    async def evaluate(self) -> EvaluationResult:
        """Run the evaluation across all (model, problem) pairs.

        Returns EvaluationResult containing per-model aggregated stats and per-run logs.
        """
        stats: dict[str, ModelStats] = {model: ModelStats() for _, model in self.models}
        logs: list[LogEntry] = []
        lock = asyncio.Lock()
        if not (self.subprocess_runner_path and self.subprocess_agent):
            raise RuntimeError("AgentEvaluator.evaluate requires subprocess_runner_path and subprocess_agent.")
        sem = asyncio.Semaphore(self.concurrency)

        async def worker(
            provider_key: str,
            model: str,
            problem: Any,
            expected: Any,
        ) -> None:
            async with sem:

                def _sanitize_for_filename(value: str) -> str:
                    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in value)[:64]

                timestamp_ms = int(time.time() * 1000)
                pid = os.getpid()
                safe_provider = _sanitize_for_filename(str(provider_key))
                safe_model = _sanitize_for_filename(str(model))
                stdout_path = f"/tmp/agent_eval_{timestamp_ms}_{pid}_{safe_provider}_{safe_model}.stdout.log"
                stderr_path = f"/tmp/agent_eval_{timestamp_ms}_{pid}_{safe_provider}_{safe_model}.stderr.log"
                output_json_path = f"/tmp/agent_eval_{timestamp_ms}_{pid}_{safe_provider}_{safe_model}.result.json"
                cmd = [
                    sys.executable,
                    self.subprocess_runner_path,  # type: ignore[arg-type]
                    "--agent",
                    str(self.subprocess_agent),  # type: ignore[arg-type]
                    "--provider",
                    str(provider_key),
                    "--model",
                    str(model),
                    "--problem-json",
                    json.dumps(problem),
                    "--output-json",
                    output_json_path,
                ]
                start_time = time.perf_counter()
                stdout_path_obj = Path(stdout_path)
                stderr_path_obj = Path(stderr_path)
                output_json_path_obj = Path(output_json_path)
                stdout_path_obj.parent.mkdir(parents=True, exist_ok=True)
                with (
                    stdout_path_obj.open("a", encoding="utf-8") as stdout_file,
                    stderr_path_obj.open("a", encoding="utf-8") as stderr_file,
                ):
                    proc = await asp.create_subprocess_exec(
                        *cmd,
                        stdout=stdout_file,
                        stderr=stderr_file,
                    )
                    await proc.wait()
                elapsed_seconds = time.perf_counter() - start_time
                # Read JSON output
                result: Any | None = None
                input_toks = 0
                output_toks = 0
                if output_json_path_obj.exists():
                    with output_json_path_obj.open(encoding="utf-8") as f:
                        payload = json.load(f)
                    result = payload.get("result")
                    input_toks = int(payload.get("input_tokens", 0))
                    output_toks = int(payload.get("output_tokens", 0))
                is_correct = bool(self.result_checker(result, expected))
            async with lock:
                s = stats[model]
                s.input_tokens += int(input_toks)
                s.output_tokens += int(output_toks)
                s.finished += 1
                if is_correct:
                    s.correct += 1
                s.total_runtime_seconds += float(elapsed_seconds)
                problem_text = str(problem[0] if isinstance(problem, tuple | list) and len(problem) >= 1 else problem)
                logs.append(
                    LogEntry(
                        provider_key=provider_key,
                        model=model,
                        problem=problem_text,
                        result=(None if result is None else result),
                        correct=(True if is_correct else (False if result is not None else None)),
                        runtime_seconds=float(elapsed_seconds),
                    )
                )
                if self.on_update is not None:
                    self.on_update(stats, logs)

        try:
            tasks: list[asyncio.Task[None]] = []
            for problem in self.problems:
                if isinstance(problem, tuple | list) and len(problem) >= 2:
                    problem_payload, expected = problem[0], problem[1]
                else:
                    problem_payload, expected = problem, None
                for provider_key, model in self.models:
                    tasks.append(
                        asyncio.create_task(
                            worker(
                                provider_key=provider_key,
                                model=model,
                                problem=problem_payload,
                                expected=expected,
                            )
                        )
                    )
            if tasks:
                await asyncio.gather(*tasks)
        finally:
            pass

        return EvaluationResult(stats=stats, logs=logs)
