"""Generic agent evaluation framework (subprocess-only).

This module provides a reusable, provider-agnostic evaluator that invokes
agents via a standardized CLI runner in isolated subprocesses. It measures
token usage and runtime, and computes correctness via a pluggable result
checker.

Key components:
- AgentEvaluator: Orchestrates parallel, subprocess-based evaluations.
- BuildAgentFn: Kept for API compatibility with existing call sites; not used
  by the evaluator directly when running in subprocess mode.
- ResultCheckerFn: Callback to determine correctness of a result relative to an
  expected value.

All (model, problem) pairs are executed in separate subprocesses with stdout
and stderr redirected to per-run log files under /tmp and results passed back
through a JSON file written by the CLI runner.
"""

import asyncio
import asyncio.subprocess as asp
import os
import time
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple


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
    logs: List[LogEntry]


BuildAgentFn = Callable[..., Any]
ResultCheckerFn = Callable[[Any, Any], bool]


# Note: The in-process execution helpers have been removed. All execution goes
# through the standardized CLI runner invoked as a subprocess.


class AgentEvaluator:
    """Parallel, subprocess-based evaluator for arbitrary agents.

    Parameters
    - models: sequence of (provider_key, model_name) pairs
    - problems: iterable of problems. Each item can be either a problem payload or
      a tuple/list (problem_payload, expected_value)
    - build_agent_fn: flexible callback to build the agent. Kept for API compatibility, not used directly by the evaluator
    - result_checker: callback that returns True/False for (result, expected)
    - concurrency: max concurrent subprocesses
    """

    def __init__(
        self,
        models: Sequence[Tuple[ProviderKey, ModelName]],
        problems: Iterable[Any],
        result_checker: ResultCheckerFn,
        tools: Optional[Sequence[Any]] = None,
        concurrency: int = 8,
        on_update: Optional[
            Callable[[dict[str, ModelStats], List[LogEntry]], None]
        ] = None,
        subprocess_runner_path: Optional[str] = None,
        subprocess_agent: Optional[str] = None,
    ) -> None:
        self.models = list(models)
        self.problems = list(problems)
        self.result_checker = result_checker
        self.concurrency = int(concurrency)
        self.on_update = on_update
        self.subprocess_runner_path = subprocess_runner_path
        self.subprocess_agent = subprocess_agent

    async def evaluate(self) -> EvaluationResult:
        """Run the evaluation across all (model, problem) pairs.

        Returns EvaluationResult containing per-model aggregated stats and per-run logs.
        """
        stats: dict[str, ModelStats] = {model: ModelStats() for _, model in self.models}
        logs: List[LogEntry] = []
        lock = asyncio.Lock()
        if not (self.subprocess_runner_path and self.subprocess_agent):
            raise RuntimeError(
                "AgentEvaluator is subprocess-only. Provide subprocess_runner_path and subprocess_agent."
            )
        sem = asyncio.Semaphore(self.concurrency)

        async def worker(
            provider_key: str,
            model: str,
            problem: Any,
            expected: Any,
        ) -> None:
            async with sem:

                def _sanitize_for_filename(value: str) -> str:
                    return "".join(
                        ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_"
                        for ch in value
                    )[:64]

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
                os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
                with (
                    open(stdout_path, "a", encoding="utf-8") as stdout_file,
                    open(stderr_path, "a", encoding="utf-8") as stderr_file,
                ):
                    proc = await asp.create_subprocess_exec(
                        *cmd,
                        stdout=stdout_file,
                        stderr=stderr_file,
                    )
                    await proc.wait()
                elapsed_seconds = time.perf_counter() - start_time
                # Read JSON output
                result = None
                input_toks = 0
                output_toks = 0
                try:
                    if os.path.exists(output_json_path):
                        with open(output_json_path, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                        result = payload.get("result")
                        input_toks = int(payload.get("input_tokens", 0))
                        output_toks = int(payload.get("output_tokens", 0))
                except Exception:
                    result = None
                    input_toks = 0
                    output_toks = 0
            is_correct: Optional[bool] = None
            if result is not None:
                try:
                    is_correct = bool(self.result_checker(result, expected))
                except Exception:
                    is_correct = False
            async with lock:
                s = stats[model]
                s.input_tokens += int(input_toks)
                s.output_tokens += int(output_toks)
                s.finished += 1
                if is_correct:
                    s.correct += 1
                s.total_runtime_seconds += float(elapsed_seconds)
                problem_text = str(
                    problem[0]
                    if isinstance(problem, (tuple, list)) and len(problem) >= 1
                    else problem
                )
                logs.append(
                    LogEntry(
                        provider_key=provider_key,
                        model=model,
                        problem=problem_text,
                        result=(None if result is None else result),
                        correct=(
                            True
                            if is_correct
                            else (False if result is not None else None)
                        ),
                        runtime_seconds=float(elapsed_seconds),
                    )
                )
                if self.on_update is not None:
                    self.on_update(stats, logs)

        try:
            tasks: List[asyncio.Task[None]] = []
            for problem in self.problems:
                if isinstance(problem, (tuple, list)) and len(problem) >= 2:
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
