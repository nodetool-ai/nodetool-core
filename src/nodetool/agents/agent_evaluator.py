"""Generic agent evaluation framework.

This module provides a reusable, provider-agnostic evaluator that can execute an
arbitrary agent implementation across a set of models and problems, measure
token usage and runtime, and compute correctness via a pluggable result checker.

Key components:
- AgentEvaluator: Orchestrates parallel, process-isolated evaluations.
- BuildAgentFn: Flexible factory callback to construct the agent. Can accept any signature but typically receives (provider, model, tools, problem).
- ResultCheckerFn: Callback to determine correctness of a result relative to an expected value.

The evaluator is intentionally minimal and opinionated about isolation: each
(model, problem) pair is executed inside a separate process using a
ProcessPoolExecutor to prevent stdout/stderr bleed and ensure provider state is
not shared across runs.
"""

import asyncio
import contextlib
import logging
import io
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.chat.providers.gemini_provider import GeminiProvider
from nodetool.chat.providers.anthropic_provider import AnthropicProvider
from nodetool.chat.providers.huggingface_provider import HuggingFaceProvider
from nodetool.workflows.processing_context import ProcessingContext


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
ProviderFactoryFn = Callable[[ProviderKey], ChatProvider]


def default_provider_factory(provider_key: ProviderKey) -> ChatProvider:
    """Create a provider instance from a short provider key.

    Supported keys:
    - "openai"
    - "gemini"
    - "anthropic"
    - "huggingface:<inference_provider>"
    """
    if provider_key == "openai":
        return OpenAIProvider()
    if provider_key == "gemini":
        return GeminiProvider()
    if provider_key == "anthropic":
        return AnthropicProvider()
    if provider_key.startswith("huggingface:"):
        _, inference_provider = provider_key.split(":", 1)
        return HuggingFaceProvider(inference_provider=inference_provider)  # type: ignore[arg-type]
    raise ValueError(f"Unknown provider key: {provider_key}")


async def _run_agent_and_collect(
    agent: Any,
    processing_context: ProcessingContext,
) -> Tuple[Any, Usage]:
    """Execute the agent and collect results and token usage."""
    usage: Usage = Usage()
    try:
        async for _ in agent.execute(processing_context):
            pass
        usage = Usage(
            input_tokens=int(getattr(agent.subtask_context, "input_tokens_total", 0)),
            output_tokens=int(getattr(agent.subtask_context, "output_tokens_total", 0)),
        )
        return agent.get_results(), usage
    except Exception:
        try:
            if getattr(agent, "subtask_context", None) is not None:
                usage = Usage(
                    input_tokens=int(
                        getattr(agent.subtask_context, "input_tokens_total", 0)
                    ),
                    output_tokens=int(
                        getattr(agent.subtask_context, "output_tokens_total", 0)
                    ),
                )
        except Exception:
            pass
        return None, usage


def _execute_single_problem(
    provider_key: ProviderKey,
    model: str,
    problem: Any,
    tools: Sequence[Any],
    build_agent_fn: BuildAgentFn,
    provider_factory: ProviderFactoryFn,
) -> Tuple[Any | None, int, int, float]:
    """Run one (model, problem) evaluation in an isolated process.

    Returns tuple of (result, input_tokens, output_tokens, elapsed_seconds).
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            start_time = time.perf_counter()
            provider = provider_factory(provider_key)
            context = ProcessingContext()
            agent = build_agent_fn(provider, model, tools, problem)
            result, usage = asyncio.run(_run_agent_and_collect(agent, context))
            elapsed_seconds = time.perf_counter() - start_time
    safe_result = None if result is None else result
    input_tokens = int(getattr(usage, "input_tokens", 0))
    output_tokens = int(getattr(usage, "output_tokens", 0))
    return safe_result, input_tokens, output_tokens, float(elapsed_seconds)


class AgentEvaluator:
    """Parallel, process-isolated evaluator for arbitrary agents.

    Parameters
    - models: sequence of (provider_key, model_name) pairs
    - problems: iterable of problems. Each item can be either a problem payload or
      a tuple/list (problem_payload, expected_value)
    - build_agent_fn: flexible callback to build the agent. Typically called with (provider, model, tools, problem) but can accept any signature
    - result_checker: callback that returns True/False for (result, expected)
    - tools: optional sequence passed to the agent factory
    - concurrency: max worker processes
    - provider_factory: maps provider_key to provider instance
    """

    def __init__(
        self,
        models: Sequence[Tuple[ProviderKey, ModelName]],
        problems: Iterable[Any],
        build_agent_fn: BuildAgentFn,
        result_checker: ResultCheckerFn,
        tools: Optional[Sequence[Any]] = None,
        concurrency: int = 8,
        provider_factory: ProviderFactoryFn = default_provider_factory,
        on_update: Optional[
            Callable[[dict[str, ModelStats], List[LogEntry]], None]
        ] = None,
    ) -> None:
        self.models = list(models)
        self.problems = list(problems)
        self.build_agent_fn = build_agent_fn
        self.result_checker = result_checker
        self.tools = list(tools or [])
        self.concurrency = int(concurrency)
        self.provider_factory = provider_factory
        self.on_update = on_update

    async def evaluate(self) -> EvaluationResult:
        """Run the evaluation across all (model, problem) pairs.

        Returns EvaluationResult containing per-model aggregated stats and per-run logs.
        """
        stats: dict[str, ModelStats] = {model: ModelStats() for _, model in self.models}
        logs: List[LogEntry] = []
        lock = asyncio.Lock()
        executor = ProcessPoolExecutor(max_workers=self.concurrency)

        async def worker(
            provider_key: str,
            model: str,
            problem: Any,
            expected: Any,
        ) -> None:
            loop = asyncio.get_running_loop()
            result, input_toks, output_toks, elapsed_seconds = (
                await loop.run_in_executor(
                    executor,
                    _execute_single_problem,
                    provider_key,
                    model,
                    problem,
                    self.tools,
                    self.build_agent_fn,
                    self.provider_factory,
                )
            )
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
            executor.shutdown(wait=True)

        return EvaluationResult(stats=stats, logs=logs)
