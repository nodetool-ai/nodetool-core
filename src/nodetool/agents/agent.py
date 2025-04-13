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

import datetime
import json
import os
import shutil
from typing import AsyncGenerator, List, Sequence, Union, Any
from rich.console import Console
from rich.table import Table
from rich.live import Live

from nodetool.agents.tools.browser import GoogleSearchTool
from nodetool.agents.tools.google import GoogleGroundedSearchTool
from nodetool.agents.tools.openai import OpenAIWebSearchTool
from nodetool.chat.providers.ollama_provider import OllamaProvider
from nodetool.common.environment import Environment
from nodetool.common.settings import get_log_path
from nodetool.workflows.types import Chunk, TaskUpdate, TaskUpdateEvent
from nodetool.agents.task_executor import TaskExecutor
from nodetool.chat.providers import ChatProvider
from nodetool.agents.task_planner import TaskPlanner
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import (
    Task,
    ToolCall,
)
from nodetool.workflows.processing_context import ProcessingContext


def sanitize_file_path(file_path: str) -> str:
    """
    Sanitize a file path by replacing spaces and slashes with underscores.

    Args:
        file_path (str): The file path to sanitize.

    Returns:
        str: The sanitized file path.
    """
    return file_path.replace(" ", "_").replace("/", "_").replace("\\", "_")


class Agent:
    """
    🤖 Orchestrates AI-driven task execution using Language Models and Tools.

    The Agent class acts as a high-level controller that takes a complex objective,
    breaks it down into a step-by-step plan, and then executes that plan using
    a specified Language Model (LLM) and a set of available tools.

    Think of it as an intelligent assistant that can understand your goal, figure out
    the necessary actions (like searching the web, reading files, performing calculations,
    or running code), and carry them out autonomously to achieve the objective.

    Key Capabilities:
    - **Planning:** Decomposes complex objectives into manageable subtasks.
    - **Execution:** Runs the subtasks in the correct order, handling dependencies.
    - **Tool Integration:** Leverages specialized tools to interact with external
      systems or perform specific actions (e.g., file operations, web browsing,
      code execution).
    - **LLM Agnostic:** Works with different LLM providers (OpenAI, Anthropic, Ollama).
    - **Progress Tracking:** Can stream updates as the task progresses.
    - **Input/Output Management:** Handles input files and collects final results.

    Use this class to automate workflows that require reasoning, planning, and
    interaction with various data sources or tools.
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
        enable_analysis_phase: bool = True,
        enable_data_contracts_phase: bool = True,
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
            enable_analysis_phase (bool, optional): Whether to run the analysis phase (PHASE 1)
            enable_data_contracts_phase (bool, optional): Whether to run the data contracts phase (PHASE 2)
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
        self.enable_analysis_phase = enable_analysis_phase
        self.enable_data_contracts_phase = enable_data_contracts_phase

    def _create_subtasks_table(self, task: Task, tool_calls: List[ToolCall]) -> Table:
        """Create a rich table for displaying subtasks and their tool calls."""
        table = Table(title=f"Task:\n{self.objective}", title_justify="left")
        table.add_column("Status", style="cyan", no_wrap=True, ratio=1)
        table.add_column("Content", style="green", ratio=6)
        table.add_column("Tools", style="magenta", ratio=3)
        table.add_column("Output", style="yellow", ratio=3)
        table.add_column("Dependencies", style="blue", ratio=2)

        for subtask in task.subtasks:
            status = "✓" if subtask.completed else "▶" if subtask.is_running() else "⏳"
            status_style = (
                "green"
                if subtask.completed
                else "yellow" if subtask.is_running() else "white"
            )
            subtask_tool_calls = [
                call for call in tool_calls if call.subtask_id == subtask.content
            ]
            tool_calls_str = ", ".join([call.name for call in subtask_tool_calls])

            deps = ", ".join(subtask.input_files) if subtask.input_files else "none"

            table.add_row(
                f"[{status_style}]{status}[/]",
                subtask.content,
                tool_calls_str,
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
            processing_context (ProcessingContext): The processing context

        Yields:
            Union[Message, Chunk, ToolCall]: Execution progress
        """
        # Copy input files to the workspace directory if they are not already there
        input_files = []
        os.makedirs(
            os.path.join(processing_context.workspace_dir, "input_files"), exist_ok=True
        )
        for file_path in self.input_files:
            destination_path = os.path.join(
                processing_context.workspace_dir,
                "input_files",
                os.path.basename(file_path),
            )
            shutil.copy(file_path, destination_path)
            input_files.append(os.path.join("input_files", os.path.basename(file_path)))

        tools = list(self.tools)

        self.provider.log_file = str(
            get_log_path(
                sanitize_file_path(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{self.name}__planner.jsonl"
                )
            )
        )

        retrieval_tools = []

        if Environment.get("GEMINI_API_KEY"):
            retrieval_tools.append(
                GoogleGroundedSearchTool(
                    workspace_dir=processing_context.workspace_dir
                ),
            )

        if Environment.get("BRIGHTDATA_API_KEY"):
            retrieval_tools.append(
                GoogleSearchTool(workspace_dir=processing_context.workspace_dir),
            )

        if Environment.get("OPENAI_API_KEY"):
            retrieval_tools.append(
                OpenAIWebSearchTool(workspace_dir=processing_context.workspace_dir),
            )

        task_planner = TaskPlanner(
            provider=self.provider,
            model=self.model,
            objective=self.objective,
            workspace_dir=processing_context.workspace_dir,
            execution_tools=tools,
            retrieval_tools=retrieval_tools,
            input_files=input_files,
            output_schema=self.output_schema,
            enable_analysis_phase=self.enable_analysis_phase,
            enable_data_contracts_phase=self.enable_data_contracts_phase,
            # use_structured_output=isinstance(self.provider, OllamaProvider),
            use_structured_output=True,
        )

        task = await task_planner.create_task(processing_context, self.objective)

        yield TaskUpdate(
            task=task,
            event=TaskUpdateEvent.TASK_CREATED,
        )

        if self.output_type and len(task.subtasks) > 0:
            task.subtasks[-1].output_type = self.output_type

        if self.output_schema and len(task.subtasks) > 0:
            task.subtasks[-1].output_schema = json.dumps(self.output_schema)

        tool_calls = []

        with Live(
            self._create_subtasks_table(task, tool_calls), refresh_per_second=4
        ) as live:
            executor = TaskExecutor(
                provider=self.provider,
                model=self.model,
                processing_context=processing_context,
                tools=tools,
                task=task,
                system_prompt=self.system_prompt,
                input_files=input_files,
                max_steps=self.max_steps,
                max_subtask_iterations=self.max_subtask_iterations,
                max_token_limit=self.max_token_limit,
            )

            # Execute all subtasks within this task and yield results
            async for item in executor.execute_tasks(processing_context):
                # self._save_task(task, processing_context.workspace_dir)
                live.update(self._create_subtasks_table(task, tool_calls))
                if isinstance(item, ToolCall):
                    tool_calls.append(item)
                    if item.name == "finish_task":
                        self.results = item.args["result"]
                        yield TaskUpdate(
                            task=task,
                            event=TaskUpdateEvent.TASK_COMPLETED,
                        )
                if isinstance(item, TaskUpdate):
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    subtask_title = item.subtask.content if item.subtask else ""
                    subtask_title = subtask_title[:20].translate(
                        str.maketrans(
                            {
                                " ": "_",
                                "\n": "_",
                                ".": "_",
                                "/": "_",
                                "\\": "_",
                                ":": "_",
                                "-": "_",
                                "=": "_",
                                "+": "_",
                                "*": "_",
                                "?": "_",
                                "!": "_",
                            }
                        )
                    )
                    self.provider.log_file = str(
                        get_log_path(
                            sanitize_file_path(
                                f"{timestamp}__{self.name}__{item.task.title}__{subtask_title}.jsonl"
                            )
                        )
                    )

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

        with open(
            os.path.join(workspace_dir, sanitize_file_path(self.name) + "_tasks.yaml"),
            "w",
        ) as f:
            yaml.dump(task_dict, f, indent=2, sort_keys=False)
