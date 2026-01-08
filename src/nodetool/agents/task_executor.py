"""Execute discover → process → aggregate task plans.

Process-mode steps automatically fan out over list inputs produced by the
preceding discover step. Each item is rendered into the parent step's
natural-language `content` template (``"fetch {url}"``) and executed as an
ephemeral step. Results are aggregated into a simple list that downstream
aggregate steps consume.
"""

import asyncio
import hashlib
import json
import os
import time
from typing import Any, AsyncGenerator, List, Sequence, Union

from nodetool.agents.step_executor import (
    StepExecutor,
    TaskUpdate,
)
from nodetool.agents.tools.base import Tool
from nodetool.agents.wrap_generators_parallel import (
    wrap_generators_parallel,
)
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Step, Task, ToolCall
from nodetool.providers import BaseProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_types import Chunk, StepResult, TaskUpdateEvent

log = get_logger(__name__)

DEFAULT_PROCESS_CONCURRENCY = 4


def _short_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:12]


class TaskExecutor:
    """Execute an entire task plan, including process-mode fan-out steps."""

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        task: Task,
        system_prompt: str | None = None,
        inputs: dict[str, Any] | None = None,
        max_steps: int = 50,
        max_step_iterations: int = 10,
        max_token_limit: int = 100000,
        parallel_execution: bool = False,
        display_manager: Any | None = None,
    ):
        """
        Initialize the TaskExecutor.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (str): The model to use with the provider
            processing_context (ProcessingContext): The processing context
            tools (List[Tool]): List of tools available for task execution
            task (Task): The task to execute
            system_prompt (str, optional): Custom system prompt
            inputs (dict[str, Any], optional): Inputs to use for the task
            max_steps (int, optional): Maximum execution steps to prevent infinite loops
            max_step_iterations (int, optional): Maximum iterations allowed per step
            max_token_limit (int, optional): Maximum token limit before summarization
            parallel_execution (bool, optional): Whether to execute steps in parallel (True) or sequentially (False)
            display_manager (Any, optional): The display manager for reporting progress
        """
        self.provider = provider
        self.model = model
        self.tools = tools
        self.task = task
        self.processing_context = processing_context
        self.inputs = inputs or {}
        self.max_token_limit = max_token_limit
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_step_iterations = max_step_iterations
        self.output_files = []
        self.parallel_execution = parallel_execution
        self.display_manager = display_manager
        self._finish_step_id: str | None = None

        # Lock for thread-safe step list access in parallel mode
        self._step_lock = asyncio.Lock()

    async def execute_tasks(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[TaskUpdate | Chunk | ToolCall, None]:
        """
        🎭 The Conductor - Orchestrates task execution with dependencies

        This method can operate in two modes:
        - Parallel: Executes all eligible steps concurrently (default)
        - Sequential: Executes eligible steps one after another

        In both modes, the method:
        1. Identifies steps that can be executed (dependencies satisfied)
        2. Creates execution contexts for each executable step
        3. Launches contexts based on execution mode
        4. Monitors completion and updates the task plan
        5. Repeats until all tasks are complete or max steps reached

        Args:
            print_usage (bool): Whether to print token usage statistics

        Yields:
            Union[TaskUpdate, Chunk, ToolCall]: Live updates during execution
        """
        steps_taken = 0

        for key, value in self.inputs.items():
            context.set(key, value)

        # Remember which step is the designated finisher so dynamic additions can slot before it.
        if self.task.steps:
            self._finish_step_id = self.task.steps[-1].id
        else:
            self._finish_step_id = None

        # Continue until all tasks are complete or we reach max steps
        while not self._all_tasks_complete() and steps_taken < self.max_steps:
            steps_taken += 1

            # Find all executable tasks
            executable_tasks = self._get_all_executable_tasks()
            executable_tasks = self._maybe_defer_finish_step(executable_tasks)

            if not executable_tasks:
                # If no tasks are executable but we're not done, there might be a dependency issue
                if not self._all_tasks_complete():
                    yield Chunk(
                        content="\nNo executable tasks but not all complete. Possible file dependency issues.\n",
                        done=False,
                    )
                break

            # Create a list to store all step execution generators
            step_generators = []

            # Create execution contexts for all executable steps
            for step in executable_tasks:
                step_executor = StepExecutor(
                    task=self.task,
                    step=step,
                    processing_context=context,
                    system_prompt=self.system_prompt,
                    tools=list(self.tools).copy(),
                    model=self.model,
                    provider=self.provider,
                    max_token_limit=self.max_token_limit,
                    use_finish_task=self._is_finish_step(step),
                    display_manager=self.display_manager,
                )
                step_generators.append(step_executor.execute())

            if not step_generators:
                continue

            [s.id for s in executable_tasks]
            if self.parallel_execution:
                # Execute all steps concurrently using wrap_generators_parallel
                async for message in wrap_generators_parallel(*step_generators):
                    if isinstance(message, StepResult):
                        log.debug(
                            f"TaskExecutor: Yielding StepResult from parallel for step {message.step.id}. is_task_result={message.is_task_result}"
                        )
                    yield message
            else:
                # Execute steps sequentially, one at a time
                for generator in step_generators:
                    async for message in generator:
                        if isinstance(message, StepResult):
                            log.debug(
                                f"TaskExecutor: Yielding StepResult from sequential for step {message.step.id}. is_task_result={message.is_task_result}"
                            )
                        yield message

    def _get_all_executable_tasks(self) -> List[Step]:
        """
        Get all executable tasks from the task list, respecting file dependencies.

        A step is considered executable when:
        1. It has not been completed yet
        2. It is not currently running
        3. All its file dependencies (if any) exist in the workspace

        Returns:
            List[Step]: All executable steps
        """
        executable_tasks = []

        for step in self.task.steps:
            if not step.completed and not step.is_running():
                # Check if all task dependencies are completed
                all_task_dependencies_met = self._check_depends_on(
                    step.depends_on, self.processing_context.workspace_dir
                )

                if all_task_dependencies_met:
                    executable_tasks.append(step)

        return executable_tasks

    def _check_depends_on(self, depends_on: List[str], workspace_dir: str) -> bool:
        """
        Check if all file dependencies exist in the workspace.

        Args:
            input_files: List of file paths to check

        Returns:
            bool: True if all dependencies exist, False otherwise
        """

        def find_task_id(task_id: str) -> Step | None:
            for step in self.task.steps:
                if step.id == task_id:
                    return step
            return None

        for task_id in depends_on:
            step = find_task_id(task_id)
            if not step or not step.completed:
                return False

        return True

    def _all_tasks_complete(self) -> bool:
        """
        Check if all tasks are marked as complete.

        Returns:
            bool: True if all tasks are complete, False otherwise
        """
        return all(step.completed for step in self.task.steps)

    def _is_finish_step(self, step: Step) -> bool:
        if self._finish_step_id:
            return step.id == self._finish_step_id
        return bool(self.task.steps) and step == self.task.steps[-1]

    def _maybe_defer_finish_step(self, executable_tasks: List[Step]) -> List[Step]:
        if not self._finish_step_id:
            return executable_tasks

        finish_ready = any(task.id == self._finish_step_id for task in executable_tasks)
        if not finish_ready:
            return executable_tasks

        other_pending = any(not step.completed and step.id != self._finish_step_id for step in self.task.steps)
        if not other_pending:
            return executable_tasks

        log.debug(
            "Deferring finish step %s until all other steps complete",
            self._finish_step_id,
        )
        return [task for task in executable_tasks if task.id != self._finish_step_id]
