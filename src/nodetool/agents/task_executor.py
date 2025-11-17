"""Execute discover → process → aggregate task plans.

Process-mode subtasks automatically fan out over list inputs produced by the
preceding discover step. Each item is rendered into the parent subtask's
natural-language `content` template (``"fetch {url}"``) and executed as an
ephemeral subtask. Results are aggregated into a simple list that downstream
aggregate subtasks consume.
"""

import asyncio
import hashlib
import json
import os
import time
from typing import Any, AsyncGenerator, List, Sequence, Union

from nodetool.agents.sub_task_context import (
    SubTaskContext,
    TaskUpdate,
)
from nodetool.agents.tools.base import Tool
from nodetool.agents.wrap_generators_parallel import (
    wrap_generators_parallel,
)
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import SubTask, Task, ToolCall
from nodetool.providers import BaseProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, SubTaskResult, TaskUpdateEvent

log = get_logger(__name__)

DEFAULT_PROCESS_CONCURRENCY = 4


def _short_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:12]


class TaskExecutor:
    """Execute an entire task plan, including process-mode fan-out subtasks."""

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
        max_subtask_iterations: int = 10,
        max_token_limit: int = 100000,
        parallel_execution: bool = False,
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
            max_subtask_iterations (int, optional): Maximum iterations allowed per subtask
            max_token_limit (int, optional): Maximum token limit before summarization
            parallel_execution (bool, optional): Whether to execute subtasks in parallel (True) or sequentially (False)
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
        self.max_subtask_iterations = max_subtask_iterations
        self.output_files = []
        self.parallel_execution = parallel_execution
        self._finish_subtask_id: str | None = None

        # Lock for thread-safe subtask list access in parallel mode
        self._subtask_lock = asyncio.Lock()

    async def execute_tasks(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[TaskUpdate | Chunk | ToolCall, None]:
        """
        🎭 The Conductor - Orchestrates task execution with dependencies

        This method can operate in two modes:
        - Parallel: Executes all eligible subtasks concurrently (default)
        - Sequential: Executes eligible subtasks one after another

        In both modes, the method:
        1. Identifies subtasks that can be executed (dependencies satisfied)
        2. Creates execution contexts for each executable subtask
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

        # Remember which subtask is the designated finisher so dynamic additions can slot before it.
        if self.task.subtasks:
            finish_subtask_id = context.get("__finish_subtask_id")
            if not finish_subtask_id:
                context.set("__finish_subtask_id", self.task.subtasks[-1].id)
            self._finish_subtask_id = context.get("__finish_subtask_id")
        else:
            self._finish_subtask_id = None

        # Continue until all tasks are complete or we reach max steps
        while not self._all_tasks_complete() and steps_taken < self.max_steps:
            steps_taken += 1

            # Find all executable tasks
            executable_tasks = self._get_all_executable_tasks()
            executable_tasks = self._maybe_defer_finish_subtask(executable_tasks)

            if not executable_tasks:
                # If no tasks are executable but we're not done, there might be a dependency issue
                if not self._all_tasks_complete():
                    yield Chunk(
                        content="\nNo executable tasks but not all complete. Possible file dependency issues.\n",
                        done=False,
                    )
                break

            # Create a list to store all subtask execution generators
            subtask_generators = []

            # Create execution contexts for all executable subtasks
            for subtask in executable_tasks:
                if subtask.mode == "process":
                    subtask_generators.append(
                        self._execute_process_mode_subtask(subtask, context)
                    )
                else:
                    enhanced_tools = list(self.tools).copy()
                    subtask_context = SubTaskContext(
                        task=self.task,
                        subtask=subtask,
                        processing_context=context,
                        system_prompt=self.system_prompt,
                        tools=enhanced_tools,
                        model=self.model,
                        provider=self.provider,
                        max_token_limit=self.max_token_limit,
                        use_finish_task=self._is_finish_subtask(subtask),
                    )
                    subtask_generators.append(subtask_context.execute())

            if not subtask_generators:
                continue

            if self.parallel_execution:
                # Execute all subtasks concurrently using wrap_generators_parallel
                async for message in wrap_generators_parallel(*subtask_generators):
                    yield message
            else:
                # Execute subtasks sequentially, one at a time
                for generator in subtask_generators:
                    async for message in generator:
                        yield message

    def _get_all_executable_tasks(self) -> List[SubTask]:
        """
        Get all executable tasks from the task list, respecting file dependencies.

        A subtask is considered executable when:
        1. It has not been completed yet
        2. It is not currently running
        3. All its file dependencies (if any) exist in the workspace

        Returns:
            List[SubTask]: All executable subtasks
        """
        executable_tasks = []

        for subtask in self.task.subtasks:
            if not subtask.completed and not subtask.is_running():
                # Check if all task and file dependencies exist
                all_task_dependencies_met = self._check_input_tasks(
                    subtask.input_tasks, self.processing_context.workspace_dir
                )
                all_file_dependencies_met = self._check_input_files(
                    subtask.input_files, self.processing_context.workspace_dir
                )

                if all_task_dependencies_met and all_file_dependencies_met:
                    executable_tasks.append(subtask)

        return executable_tasks

    def _check_input_tasks(self, input_tasks: List[str], workspace_dir: str) -> bool:
        """
        Check if all file dependencies exist in the workspace.

        Args:
            input_files: List of file paths to check

        Returns:
            bool: True if all dependencies exist, False otherwise
        """

        def find_task_id(task_id: str) -> SubTask | None:
            for subtask in self.task.subtasks:
                if subtask.id == task_id:
                    return subtask
            return None

        for task_id in input_tasks:
            subtask = find_task_id(task_id)
            if not subtask or not subtask.completed:
                return False

        return True

    def _check_input_files(self, input_files: List[str], workspace_dir: str) -> bool:
        """
        Check if all file dependencies exist in the workspace.

        Args:
            input_files: List of file paths to check
            workspace_dir: The workspace directory to check files in

        Returns:
            bool: True if all files exist, False otherwise
        """
        for file_path in input_files:
            full_path = os.path.join(workspace_dir, file_path)
            if not os.path.exists(full_path):
                return False
        return True

    def _all_tasks_complete(self) -> bool:
        """
        Check if all tasks are marked as complete.

        Returns:
            bool: True if all tasks are complete, False otherwise
        """
        return all(subtask.completed for subtask in self.task.subtasks)

    def _is_finish_subtask(self, subtask: SubTask) -> bool:
        if self._finish_subtask_id:
            return subtask.id == self._finish_subtask_id
        return bool(self.task.subtasks) and subtask == self.task.subtasks[-1]

    def _maybe_defer_finish_subtask(
        self, executable_tasks: List[SubTask]
    ) -> List[SubTask]:
        if not self._finish_subtask_id:
            return executable_tasks

        finish_ready = any(
            task.id == self._finish_subtask_id for task in executable_tasks
        )
        if not finish_ready:
            return executable_tasks

        other_pending = any(
            not subtask.completed and subtask.id != self._finish_subtask_id
            for subtask in self.task.subtasks
        )
        if not other_pending:
            return executable_tasks

        log.debug(
            "Deferring finish subtask %s until all other subtasks complete",
            self._finish_subtask_id,
        )
        return [task for task in executable_tasks if task.id != self._finish_subtask_id]

    async def _run_process_mode(
        self, subtask: SubTask, context: ProcessingContext
    ) -> list:
        """Iterate over the upstream list and run templated subtasks per item."""

        if not subtask.input_tasks:
            raise ValueError(
                f"Process subtask '{subtask.id}' must depend on a discover-mode subtask."
            )

        upstream_id = subtask.input_tasks[0]
        upstream_value = context.load_subtask_result(upstream_id, default=None)
        if upstream_value is None:
            raise ValueError(
                f"Process subtask '{subtask.id}' could not load upstream result '{upstream_id}'."
            )
        if not isinstance(upstream_value, list):
            raise ValueError(
                f"Process subtask '{subtask.id}' expected upstream '{upstream_id}' to be a list but received {type(upstream_value).__name__}."
            )

        if not upstream_value:
            return []

        concurrency = min(DEFAULT_PROCESS_CONCURRENCY, len(upstream_value))
        semaphore = asyncio.Semaphore(concurrency)
        results: list[Any | None] = [None] * len(upstream_value)

        async def run_item(index: int, item: Any) -> None:
            async with semaphore:
                try:
                    results[index] = await self._run_process_item_subtask(
                        parent_subtask=subtask,
                        item=item,
                        index=index,
                        context=context,
                    )
                except Exception as exc:
                    log.error(
                        "Process subtask '%s' item %d failed: %s",
                        subtask.id,
                        index,
                        exc,
                    )

        tasks = [
            asyncio.create_task(run_item(index, item))
            for index, item in enumerate(upstream_value)
        ]
        await asyncio.gather(*tasks, return_exceptions=False)

        return [result for result in results if result is not None]
    def _format_process_content(
        self, subtask: SubTask, item: Any, index: int
    ) -> str:
        """Render the process subtask's content template against an item."""
        template = (subtask.item_template or subtask.content or "").strip()
        if not template:
            raise ValueError(
                f"Process subtask '{subtask.id}' must define an item_template describing per-item instructions."
            )
        context_vars: dict[str, Any] = {
            "item": item,
            "index": index,
            "item_index": index,
        }
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(key, str):
                    context_vars.setdefault(key, value)
        try:
            return template.format(**context_vars)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"Process subtask '{subtask.id}' per-item template references missing field '{missing}'. "
                "Ensure the discovery output provides this value."
            ) from exc

    def _item_output_schema(self, process_subtask: SubTask) -> str:
        """Return the JSON schema string for an individual item."""
        item_schema_str = (process_subtask.item_output_schema or "").strip()
        if item_schema_str:
            try:
                json.loads(item_schema_str)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Process subtask '{process_subtask.id}' item_output_schema is not valid JSON: {exc}"
                ) from exc
            return item_schema_str

        schema_str = getattr(process_subtask, "output_schema", "") or ""
        if not schema_str:
            raise ValueError(
                f"Process subtask '{process_subtask.id}' must declare an output_schema."
            )
        try:
            schema_obj = json.loads(schema_str)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Process subtask '{process_subtask.id}' output_schema is not valid JSON: {exc}"
            ) from exc
        if schema_obj.get("type") != "array":
            raise ValueError(
                f"Process subtask '{process_subtask.id}' output_schema must declare type 'array'."
            )
        item_schema = schema_obj.get("items")
        if not item_schema:
            raise ValueError(
                f"Process subtask '{process_subtask.id}' output_schema must include an 'items' schema or set item_output_schema."
            )
        return json.dumps(item_schema)

    def _select_tools_for_subtask(self, subtask: SubTask) -> list[Tool]:
        """Filter the global tool list to match the subtask's declared tools."""
        if not subtask.tools:
            return list(self.tools)
        allowed = set(subtask.tools)
        return [tool for tool in self.tools if tool.name in allowed]

    def _make_ephemeral_subtask(
        self,
        parent_subtask: SubTask,
        content: str,
        index: int,
        item_hash: str,
        item_schema: str,
    ) -> SubTask:
        """Create a minimal SubTask instance for a single process item."""
        ephemeral = SubTask(
            id=f"{parent_subtask.id}__{index}_{item_hash}",
            content=content,
            input_tasks=[],
            output_schema=item_schema,
            tools=parent_subtask.tools or [],
        )
        return ephemeral

    async def _run_process_item_subtask(
        self,
        parent_subtask: SubTask,
        item: Any,
        index: int,
        context: ProcessingContext,
    ) -> Any | None:
        """Execute a single item as a regular subtask and return its result."""
        rendered_content = self._format_process_content(parent_subtask, item, index)
        item_hash = _short_hash({"index": index, "item": item})
        item_schema = self._item_output_schema(parent_subtask)
        ephemeral_subtask = self._make_ephemeral_subtask(
            parent_subtask=parent_subtask,
            content=rendered_content,
            index=index,
            item_hash=item_hash,
            item_schema=item_schema,
        )
        tools = self._select_tools_for_subtask(parent_subtask)
        subtask_context = SubTaskContext(
            task=self.task,
            subtask=ephemeral_subtask,
            processing_context=context,
            system_prompt=self.system_prompt,
            tools=tools,
            model=self.model,
            provider=self.provider,
            max_token_limit=self.max_token_limit,
            use_finish_task=False,
        )
        result_payload: Any | None = None
        async for message in subtask_context.execute():
            if isinstance(message, SubTaskResult):
                result_payload = message.result
        if result_payload is None:
            log.warning(
                "Process subtask '%s' item %d completed without a result payload.",
                parent_subtask.id,
                index,
            )
        return result_payload

    async def _execute_process_mode_subtask(
        self,
        subtask: SubTask,
        context: ProcessingContext,
    ) -> AsyncGenerator[TaskUpdate | Chunk | ToolCall | SubTaskResult, None]:
        """Run a process-mode subtask end-to-end and emit standard task events."""

        subtask.start_time = int(time.time())
        yield TaskUpdate(
            task=self.task,
            subtask=subtask,
            event=TaskUpdateEvent.SUBTASK_STARTED,
        )

        log.debug("Running process-mode subtask %s", subtask.id)
        result_payload = await self._run_process_mode(subtask, context)

        subtask.completed = True
        subtask.end_time = int(time.time())
        context.store_subtask_result(subtask.id, result_payload)

        yield TaskUpdate(
            task=self.task,
            subtask=subtask,
            event=TaskUpdateEvent.SUBTASK_COMPLETED,
        )
        yield SubTaskResult(subtask=subtask, result=result_payload, is_task_result=False)
