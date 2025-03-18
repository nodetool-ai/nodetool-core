from nodetool.chat.wrap_generators_parallel import (
    wrap_generators_parallel,
)
from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.sub_task_context import SubTaskContext
from nodetool.chat.tools import Tool
from nodetool.metadata.types import (
    FunctionModel,
    Message,
    SubTask,
    Task,
    TaskPlan,
    ToolCall,
)


import datetime
import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Tuple, Union
import time

DEFAULT_EXECUTION_SYSTEM_PROMPT = """
You are a task execution agent that completes subtasks efficiently with minimal steps.

EFFICIENCY REQUIREMENTS:
- Complete each subtask in as few steps as possible (ideally 1-3 tool calls)
- Batch operations whenever possible instead of making sequential calls
- Use the most direct approach to solve the problem
- Avoid unnecessary reasoning - be concise and focused

WORKSPACE CONSTRAINTS:
- All file operations must use the /workspace directory as root
- Never attempt to access files outside the /workspace directory
- All file paths must start with /workspace/ for proper access
- Make sure to save artifacts in the /workspace directory

FILE DEPENDENCY CONSTRAINTS:
- Check that all required file dependencies exist before proceeding
- Use the same file naming conventions defined in the task plan
- Read input files when needed but avoid unnecessary reads
- Always save your output to the specified output_file path

EXECUTION CONTEXT:
- You are executing a single subtask from a larger task plan
- You have access to specific tools defined in the subtask
- Dependency files from previous subtasks are available in the /workspace directory
- Each subtask must be completed with a call to finish_subtask

EXECUTION PROTOCOL:
1. Focus exclusively on the current subtask - ignore other subtasks
2. Always call finish_subtask with the result and the specified output_file
3. Aim to make only 1-3 tool calls per subtask - be efficient!

RESULT REQUIREMENTS:
1. Results can be a string or a JSON object
2. Pick the output type defined in the subtask
3. Include metadata with these fields:
   - title: Descriptive title of the result
   - description: Detailed description of what was accomplished
   - source: Origin of the data (e.g., URL, calculation, file)
4. Call finish_subtask with a result object, output_file, and metadata
"""


class TaskExecutor:
    """
    🚀 The Parallel Orchestrator - Runs multiple subtasks concurrently

    This class manages the execution of an entire task plan, strategically
    scheduling subtasks to maximize parallelism while respecting dependencies.
    It's like a project manager who knows exactly which tasks can run in parallel
    and which ones need to wait for others to finish.

    Features:
    - Parallel execution of independent subtasks
    - Dependency tracking and enforcement
    - Progress persistence through file updates
    - Result collection and reporting
    - Workspace file management
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: str,
        workspace_dir: str,
        tools: List[Tool],
        task: Task,
        system_prompt: str | None = None,
        max_steps: int = 50,
        max_subtask_iterations: int = 5,
        max_token_limit: int = 20000,
    ):
        """
        Initialize the IsolatedTaskExecutor.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use with the provider
            workspace_dir (str): Directory for workspace files
            tools (List[Tool]): List of tools available for task execution
            tasks (List[Task]): List of tasks to execute
            system_prompt (str, optional): Custom system prompt
            max_steps (int, optional): Maximum execution steps to prevent infinite loops
            max_subtask_iterations (int, optional): Maximum iterations allowed per subtask
            max_token_limit (int, optional): Maximum token limit before summarization
        """
        self.provider = provider
        self.model = model
        self.workspace_dir = workspace_dir
        self.tools = tools
        self.task = task
        self.max_token_limit = max_token_limit

        # Prepare system prompt
        prefix = f"""
        Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
        Your workspace directory is: {self.workspace_dir}
        """
        self.system_prompt = prefix + (
            system_prompt if system_prompt else DEFAULT_EXECUTION_SYSTEM_PROMPT
        )

        self.max_steps = max_steps
        self.max_subtask_iterations = max_subtask_iterations
        self.output_files = []

        # Check if tasks.json exists in the workspace directory
        self.tasks_file_path = Path(workspace_dir) / "tasks.json"

    async def execute_tasks(
        self,
        print_usage: bool = True,
    ) -> AsyncGenerator[Union[Message, Chunk, ToolCall], None]:
        """
        🎭 The Conductor - Orchestrates parallel task execution with dependencies

        This method is the heart of the agent's execution capabilities, implementing
        a sophisticated parallel execution strategy:

        1. Identifies all subtasks that can be executed (dependencies satisfied)
        2. Creates execution contexts for each executable subtask
        3. Launches all contexts in parallel using wrap_generators_parallel
        4. Monitors completion and updates the task plan
        5. Repeats until all tasks are complete or max steps reached

        When execution completes, provides a summary of all workspace files created.

        Args:
            print_usage (bool): Whether to print token usage statistics

        Yields:
            Union[Message, Chunk, ToolCall]: Live updates during execution
        """
        steps_taken = 0

        # Continue until all tasks are complete or we reach max steps
        while not self._all_tasks_complete() and steps_taken < self.max_steps:
            steps_taken += 1

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
                # Create subtask context
                context = SubTaskContext(
                    task=self.task,
                    subtask=subtask,
                    system_prompt=self.system_prompt,
                    tools=self.tools,
                    model=self.model if subtask.model == "" else subtask.model,
                    provider=self.provider,
                    workspace_dir=self.workspace_dir,
                    print_usage=print_usage,
                    max_token_limit=self.max_token_limit,
                )

                # Create the task prompt
                task_prompt = self._create_task_prompt(self.task, subtask)

                # Start the subtask execution and add it to our generators
                subtask_generators.append(context.execute(task_prompt))

            if not subtask_generators:
                continue

            # Execute all subtasks in parallel
            print(f"Executing {len(subtask_generators)} subtasks in parallel")

            # Use wrap_generators_parallel to execute all subtasks concurrently
            async for message in wrap_generators_parallel(*subtask_generators):
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
                # Check if all file dependencies exist
                all_dependencies_met = self._check_file_dependencies(
                    subtask.file_dependencies
                )

                if all_dependencies_met:
                    executable_tasks.append(subtask)

        return executable_tasks

    def _check_file_dependencies(self, file_dependencies: List[str]) -> bool:
        """
        Check if all file dependencies exist in the workspace.

        Args:
            file_dependencies: List of file paths to check

        Returns:
            bool: True if all dependencies exist, False otherwise
        """
        for file_path in file_dependencies:
            # Handle /workspace prefixed paths
            if file_path.startswith("/workspace/"):
                relative_path = file_path[len("/workspace/") :]
                full_path = os.path.join(self.workspace_dir, relative_path)
            else:
                full_path = os.path.join(self.workspace_dir, file_path)

            if not os.path.exists(full_path):
                return False

        return True

    def _create_task_prompt(self, task: Task, subtask: SubTask) -> str:
        """
        Create a specific prompt for this task.

        Args:
            task: The high-level task
            subtask: The specific subtask to execute

        Returns:
            str: The task-specific prompt
        """
        # Get file dependencies
        prompt = f"""
        Context: {task.title} - {task.description if task.description else task.title}
        YOUR GOAL FOR THIS SUBTASK: {subtask.content}
        TASK TYPE: {subtask.task_type}
        
        Read these files for context:
        {json.dumps(subtask.file_dependencies)}

        Generate this output file: {subtask.output_file}

        Execute this sub task to accomplish the high level task:
        IMPORTANT: Call the `finish_subtask` tool with the final result and specify the output_file as "{subtask.output_file}"
        """

        return prompt

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

    def get_results(self) -> Dict[str, Any]:
        """
        Get all subtask results from the global result store.
        Dynamically reads output files for completed subtasks if they aren't already in the result store.

        Returns:
            Dict[str, Any]: Dictionary of results indexed by subtask ID
        """
        # Update result store with any new completed subtasks
        results = {}
        for subtask in self.task.subtasks:
            if subtask.completed and subtask.output_file:
                results[subtask.id] = subtask.output_file

        return results
