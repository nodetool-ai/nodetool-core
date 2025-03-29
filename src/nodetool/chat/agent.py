"""
Chain of Thought (CoT) Agent implementation with tool calling capabilities.

This module implements a Chain of Thought reasoning agent that can use large language
models (LLMs) from various providers (OpenAI, Anthropic, Ollama) to solve problems
step by step. The agent can leverage external tools to perform actions like mathematical
calculations, web browsing, file operations, and shell command execution.

The implementation provides:
1. A TaskPlanner class that creates a task list with dependencies
2. A TaskExecutor class that executes tasks in the correct order
3. An Agent class that combines planning and execution
4. Integration with the existing provider and tool system
5. Support for streaming results during reasoning
"""

import os
import json
import yaml
from typing import AsyncGenerator, List, Sequence, Union, Any
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

from nodetool.workflows.types import TaskUpdate, TaskUpdateEvent
from nodetool.chat.task_executor import TaskExecutor
from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.task_planner import TaskPlanner
from nodetool.chat.tools import Tool
from nodetool.chat.tools.base import resolve_workspace_path
from nodetool.metadata.types import (
    Message,
    ToolCall,
    Task,
)
from nodetool.workflows.processing_context import ProcessingContext


class Agent:
    """
    🤖 Agent Base Class - Foundation for specialized agents

    This class provides the core functionality for all specialized agents,
    establishing a common interface and execution flow that can be customized
    by specific agent types (retrieval, summarization, etc.).

    Think of it as a base employee template that defines standard duties,
    while specialized roles add their unique skills and responsibilities.

    Features:
    - Task execution capabilities
    - Tool integration
    - Result tracking
    - System prompt customization
    """

    def __init__(
        self,
        name: str,
        objective: str,
        provider: ChatProvider,
        model: str,
        tools: Sequence[Tool],
        description: str = "",
        input_files: List[str] = [],
        system_prompt: str | None = None,
        max_subtasks: int = 10,
        max_steps: int = 50,
        max_subtask_iterations: int = 5,
        max_token_limit: int = 20000,
        output_schema: dict | None = None,
        output_type: str | None = None,
    ):
        """
        Initialize the base agent.

        Args:
            name (str): The name of the agent
            objective (str): The objective of the agent
            description (str): The description of the agent
            provider (ChatProvider): An LLM provider instance
            model (str): The model to use with the provider
            tools (List[Tool]): List of tools available for this agent
            input_files (List[str]): List of input files to use for the agent
            system_prompt (str, optional): Custom system prompt
            max_steps (int, optional): Maximum reasoning steps
            max_subtask_iterations (int, optional): Maximum iterations per subtask
            max_token_limit (int, optional): Maximum token limit before summarization
            max_subtasks (int, optional): Maximum number of subtasks to be created
            output_schema (dict, optional): JSON schema for the final task output
            output_type (str, optional): Type of the final task output
        """
        self.name = name
        self.objective = objective
        self.description = description
        self.provider = provider
        self.model = model
        self.max_steps = max_steps
        self.max_subtask_iterations = max_subtask_iterations
        self.max_token_limit = max_token_limit
        self.tools = tools
        self.input_files = input_files
        self.max_subtasks = max_subtasks
        self.system_prompt = system_prompt or ""
        self.results: Any = None
        self.console = Console()
        self.output_schema = output_schema
        self.output_type = output_type

    def _create_subtasks_table(self, task: Task) -> Table:
        """Create a rich table for displaying subtasks."""
        table = Table(title=f"Task:\n{self.objective}", title_justify="left")
        table.add_column("Status", style="cyan", no_wrap=True, ratio=1)
        table.add_column("Content", style="green", ratio=10)  # 50% of remaining space
        table.add_column("Output", style="yellow", ratio=3)
        table.add_column("Dependencies", style="blue", ratio=2)

        for subtask in task.subtasks:
            status = "✓" if subtask.completed else "▶" if subtask.is_running() else "⏳"
            status_style = (
                "green"
                if subtask.completed
                else "yellow" if subtask.is_running() else "white"
            )

            deps = ", ".join(subtask.input_files) if subtask.input_files else "none"

            table.add_row(
                f"[{status_style}]{status}[/]",
                subtask.content,
                subtask.output_file,
                deps,
            )

        return table

    async def execute(
        self,
        processing_context: ProcessingContext,
    ) -> AsyncGenerator[Union[TaskUpdate, Chunk, ToolCall], None]:
        """
        Execute the agent using the task plan.

        Args:
            task (Task): The task to execute
            processing_context (ProcessingContext): The processing context
            input_files (List[str]): List of input files to use for the task

        Yields:
            Union[Message, Chunk, ToolCall]: Execution progress
        """
        tools = list(self.tools)

        task_planner = TaskPlanner(
            provider=self.provider,
            model=self.model,
            objective=self.objective,
            workspace_dir=processing_context.workspace_dir,
            tools=tools,
            input_files=self.input_files,
            output_schema=self.output_schema,
        )

        task = await task_planner.create_task(self.objective)

        yield TaskUpdate(
            task=task,
            event=TaskUpdateEvent.TASK_CREATED,
        )

        if self.output_type:
            task.subtasks[-1].output_type = self.output_type

        if self.output_schema:
            task.subtasks[-1].output_schema = self.output_schema

        with Live(self._create_subtasks_table(task), refresh_per_second=4) as live:
            executor = TaskExecutor(
                provider=self.provider,
                model=self.model,
                processing_context=processing_context,
                tools=tools,
                task=task,
                system_prompt=self.system_prompt,
                input_files=self.input_files,
                max_steps=self.max_steps,
                max_subtask_iterations=self.max_subtask_iterations,
                max_token_limit=self.max_token_limit,
            )

            # Execute all subtasks within this task and yield results
            async for item in executor.execute_tasks(processing_context):
                yield item
                self._save_task(task, processing_context.workspace_dir)
                live.update(self._create_subtasks_table(task))
                if isinstance(item, ToolCall):
                    if item.name == "finish_task":
                        self.results = item.args["result"]

        if not self.output_schema and not self.output_type:
            self.results = executor.get_output_files()

        print(self.provider.usage)

    def get_results(self) -> List[Any]:
        """
        Get the results produced by this agent.
        If a final result exists from finish_task, return that.
        Otherwise, return all collected results.

        Returns:
            List[Any]: Results with priority given to finish_task output
        """
        return self.results

    def _save_task(self, task: Task, workspace_dir: str) -> None:
        """
        Save the current task plan to the tasks file.
        """
        task_dict = task.model_dump()
        import yaml

        def sanitize_file_path(file_path: str) -> str:
            return file_path.replace(" ", "_").replace("/", "_").replace("\\", "_")

        with open(
            os.path.join(workspace_dir, sanitize_file_path(self.name) + "_tasks.yaml"),
            "w",
        ) as f:
            yaml.dump(task_dict, f, indent=2, sort_keys=False)
