from nodetool.agents.wrap_generators_parallel import (
    wrap_generators_parallel,
)
from nodetool.providers import BaseProvider
from nodetool.agents.sub_task_context import (
    SubTaskContext,
    TaskUpdate,
)
from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.task_tools import AddSubtaskTool, ListSubtasksTool
from nodetool.metadata.types import (
    SubTask,
    Task,
    ToolCall,
)
import os
import asyncio
from typing import AsyncGenerator, List, Sequence, Union, Any

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class TaskExecutor:
    """
    🚀 The Parallel Orchestrator - Runs multiple subtasks concurrently

    This class manages the execution of an entire task plan, strategically
    scheduling subtasks to maximize parallelism while respecting dependencies.
    It's like a project manager who knows exactly which tasks can run in parallel
    and which ones need to wait for others to finish.

    Features:
    - Parallel or sequential execution of independent subtasks
    - Dependency tracking and enforcement
    - Dynamic subtask addition during execution (via AddSubtaskTool)
    - Progress persistence through file updates
    - Result collection and reporting
    - Workspace file management
    - Thread-safe subtask list management for parallel execution
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        task: Task,
        system_prompt: str | None = None,
        inputs: dict[str, Any] = {},
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
        self.inputs = inputs
        self.max_token_limit = max_token_limit
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_subtask_iterations = max_subtask_iterations
        self.output_files = []
        self.parallel_execution = parallel_execution

        # Lock for thread-safe subtask list access in parallel mode
        self._subtask_lock = asyncio.Lock()
        # Track initial subtask count to detect dynamic additions
        self._initial_subtask_count = len(task.subtasks)

    async def execute_tasks(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Union[TaskUpdate, Chunk, ToolCall], None]:
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

        # Continue until all tasks are complete or we reach max steps
        while not self._all_tasks_complete() and steps_taken < self.max_steps:
            steps_taken += 1

            # Check for dynamically added subtasks and log them
            current_subtask_count = len(self.task.subtasks)
            if current_subtask_count > self._initial_subtask_count:
                new_count = current_subtask_count - self._initial_subtask_count
                log.info(
                    f"Detected {new_count} dynamically added subtask(s). Total subtasks: {current_subtask_count}"
                )
                # Update the initial count for next iteration
                self._initial_subtask_count = current_subtask_count

            # Find all executable tasks
            executable_tasks = self._get_all_executable_tasks()

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
                # Create enhanced tools list with task management capabilities
                enhanced_tools = list(self.tools).copy()

                # Add task management tools to allow dynamic subtask creation
                enhanced_tools.extend(
                    [
                        AddSubtaskTool(task=self.task),
                        ListSubtasksTool(task=self.task),
                    ]
                )

                # Create subtask context
                subtask_context = SubTaskContext(
                    task=self.task,
                    subtask=subtask,
                    processing_context=context,
                    system_prompt=self.system_prompt,
                    tools=enhanced_tools,
                    model=self.model,
                    provider=self.provider,
                    max_token_limit=self.max_token_limit,
                    use_finish_task=(subtask == self.task.subtasks[-1]),
                )

                # Start the subtask execution and add it to our generators
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
            if not subtask:
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
        for subtask in self.task.subtasks:
            if not subtask.completed:
                return False
        return True
