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
from typing import AsyncGenerator, List, Sequence, Union, Any, Optional

from nodetool.common.settings import get_log_path
from nodetool.workflows.types import (
    Chunk,
    PlanningUpdate,
    SubTaskResult,
    TaskUpdate,
    TaskUpdateEvent,
)
from nodetool.agents.task_executor import TaskExecutor
from nodetool.chat.providers import ChatProvider
from nodetool.agents.task_planner import TaskPlanner
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import (
    Task,
    ToolCall,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.ui.console import AgentConsole
from nodetool.agents.base_agent import BaseAgent


def sanitize_file_path(file_path: str) -> str:
    """
    Sanitize a file path by replacing spaces and slashes with underscores.

    Args:
        file_path (str): The file path to sanitize.

    Returns:
        str: The sanitized file path.
    """
    return file_path.replace(" ", "_").replace("/", "_").replace("\\", "_")


class Agent(BaseAgent):
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
        planning_model: str | None = None,
        reasoning_model: str | None = None,
        tools: Sequence[Tool] = [],
        description: str = "",
        input_files: List[str] = [],
        system_prompt: str | None = None,
        max_subtasks: int = 10,
        max_steps: int = 50,
        max_subtask_iterations: int = 5,
        max_token_limit: int | None = None,
        output_schema: dict | None = None,
        output_type: str | None = None,
        enable_analysis_phase: bool = True,
        enable_data_contracts_phase: bool = True,
        task: Task | None = None,  # Add optional task parameter
        verbose: bool = True,  # Add verbose flag
    ):
        """
        Initialize the base agent.

        Args:
            name (str): The name of the agent
            objective (str): The objective of the agent
            description (str): The description of the agent
            provider (ChatProvider): An LLM provider instance
            model (str): The model to use with the provider
            reasoning_model (str, optional): The model to use for reasoning, defaults to the same as the provider model
            planning_model (str, optional): The model to use for planning, defaults to the same as the provider model
            tools (List[Tool]): List of tools available for this agent
            input_files (List[str]): List of input files to use for the agent
            system_prompt (str, optional): Custom system prompt
            max_steps (int, optional): Maximum reasoning steps
            max_subtask_iterations (int, optional): Maximum iterations per subtask
            max_token_limit (int, optional): Maximum token limit before summarization
            max_subtasks (int, optional): Maximum number of subtasks to be created
            output_schema (dict, optional): JSON schema for the final task output
            output_type (str, optional): Type of the final task output
            enable_analysis_phase (bool, optional): Whether to run the analysis phase (PHASE 2)
            enable_data_contracts_phase (bool, optional): Whether to run the data contracts phase (PHASE 3)
            task (Task, optional): Pre-defined task to execute, skipping planning
            verbose (bool, optional): Enable/disable console output (default: True)
        """
        super().__init__(
            name=name,
            objective=objective,
            provider=provider,
            model=model,
            tools=tools,
            input_files=input_files,
            system_prompt=system_prompt,
            max_token_limit=max_token_limit,
        )
        self.description = description
        self.planning_model = planning_model or model
        self.reasoning_model = reasoning_model or model
        self.max_steps = max_steps
        self.max_subtask_iterations = max_subtask_iterations
        self.max_subtasks = max_subtasks
        self.output_schema = output_schema
        self.output_type = output_type
        self.enable_analysis_phase = enable_analysis_phase
        self.enable_data_contracts_phase = enable_data_contracts_phase
        self.initial_task = task
        if self.initial_task:
            self.task = self.initial_task
        self.verbose = verbose
        self.display_manager = AgentConsole(verbose=self.verbose)

    async def execute(
        self,
        processing_context: ProcessingContext,
    ) -> AsyncGenerator[
        Union[TaskUpdate, Chunk, ToolCall, PlanningUpdate, SubTaskResult], None
    ]:
        """
        Execute the agent using the task plan.

        Args:
            processing_context (ProcessingContext): The processing context

        Yields:
            Union[Message, Chunk, ToolCall]: Execution progress
        """
        # Copy input files to the workspace directory if they are not already there
        input_files = []
        for file_path in self.input_files:
            destination_path = os.path.join(
                processing_context.workspace_dir,
                os.path.basename(file_path),
            )
            shutil.copy(file_path, destination_path)
            input_files.append(os.path.basename(file_path))

        tools = list(self.tools)
        task_planner_instance: Optional[TaskPlanner] = (
            None  # Keep track of planner instance
        )

        if self.task:  # If self.task is already set (e.g. by initial_task in __init__)
            if self.initial_task:
                for subtask in self.task.subtasks:
                    if subtask.output_file and not os.path.isabs(subtask.output_file):
                        subtask.output_file = os.path.join(
                            processing_context.workspace_dir, subtask.output_file
                        )
            # If self.task was set by initial_task, we skip planning.
            # We need to ensure it passes the None check for subsequent operations.
            pass
        else:
            self.display_manager.print(
                f"Agent '{self.name}' planning task for objective: {self.objective}"
            )
            self.provider.log_file = str(
                get_log_path(
                    sanitize_file_path(
                        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{self.name}__planner.jsonl"
                    )
                )
            )

            task_planner_instance = TaskPlanner(
                provider=self.provider,
                model=self.planning_model,
                reasoning_model=self.reasoning_model,
                objective=self.objective,
                workspace_dir=processing_context.workspace_dir,
                execution_tools=tools,
                input_files=input_files,
                output_schema=self.output_schema,
                enable_analysis_phase=self.enable_analysis_phase,
                enable_data_contracts_phase=self.enable_data_contracts_phase,
                use_structured_output=True,
                verbose=self.verbose,
            )

            async for chunk in task_planner_instance.create_task(
                processing_context, self.objective
            ):
                yield chunk

            if (
                task_planner_instance.task_plan
                and task_planner_instance.task_plan.tasks
            ):
                self.task = task_planner_instance.task_plan.tasks[0]

            assert (
                self.task is not None
            ), "Task was not created by planner and was not provided initially."

            yield TaskUpdate(
                task=self.task,
                event=TaskUpdateEvent.TASK_CREATED,
            )

        # At this point, self.task should be non-None if execution is to proceed.
        if not self.task:
            # This case should ideally be caught by the assertion above or if initial_task was None
            # and planning failed to produce a task.
            # However, as a safeguard:
            raise RuntimeError("Agent execution cannot proceed: Task is not defined.")

        if self.output_type and len(self.task.subtasks) > 0:
            self.task.subtasks[-1].output_type = self.output_type

        if self.output_schema and len(self.task.subtasks) > 0:
            self.task.subtasks[-1].output_schema = json.dumps(self.output_schema)

        tool_calls: List[ToolCall] = []

        # Start live display managed by AgentConsole
        self.display_manager.start_live(
            self.display_manager.create_execution_table(
                title=self.name, task=self.task, tool_calls=tool_calls
            )
        )

        if task_planner_instance:  # Only save if planner was used
            await task_planner_instance.save_task_plan()

        try:
            executor = TaskExecutor(
                provider=self.provider,
                model=self.model,
                processing_context=processing_context,
                tools=list(self.tools),  # Ensure it's a list of Tool
                task=self.task,
                system_prompt=self.system_prompt,
                input_files=input_files,
                max_steps=self.max_steps,
                max_subtask_iterations=self.max_subtask_iterations,
                max_token_limit=self.max_token_limit,
            )

            # Execute all subtasks within this task and yield results
            async for item in executor.execute_tasks(processing_context):
                # Update tool_calls list if item is a ToolCall
                if isinstance(item, ToolCall):
                    tool_calls.append(item)

                # Create the updated table and update the live display
                new_table = self.display_manager.create_execution_table(
                    title=f"Task:\\n{self.objective}",
                    task=self.task,
                    tool_calls=tool_calls,
                )
                self.display_manager.update_live(new_table)

                # Yield the item
                if isinstance(item, ToolCall) and item.name == "finish_task":
                    self.results = item.args["result"]
                    yield TaskUpdate(
                        task=self.task,
                        event=TaskUpdateEvent.TASK_COMPLETED,
                    )
                elif isinstance(item, ToolCall) and (
                    item.name == "finish_subtask" or item.name == "finish_task"
                ):
                    for subtask in self.task.subtasks:
                        if (
                            subtask.id == item.subtask_id
                            and not subtask.is_intermediate_result
                            and "result" in item.args
                        ):
                            yield SubTaskResult(
                                subtask=subtask,
                                result=item.args["result"],
                            )
                elif isinstance(item, TaskUpdate):
                    yield item
                    # Update provider log file when a subtask starts/completes
                    if item.event in [
                        TaskUpdateEvent.SUBTASK_STARTED,
                        TaskUpdateEvent.SUBTASK_COMPLETED,
                    ]:
                        timestamp = datetime.datetime.now().strftime(
                            "%Y-%m-%d_%H-%M-%S"
                        )
                        assert item.subtask is not None
                        self.provider.log_file = str(
                            get_log_path(
                                sanitize_file_path(
                                    f"{timestamp}__{self.name}__{item.subtask.id}.jsonl"
                                )
                            )
                        )
                elif isinstance(
                    item, (Chunk, ToolCall)
                ):  # Yield chunks and other tool calls too
                    yield item

        finally:
            # Ensure live display is stopped
            self.display_manager.stop_live()

        self.display_manager.print(self.provider.usage)

    def get_results(self) -> List[Any]:
        """
        Get the results produced by this agent.
        If a final result exists from finish_task, return that.
        Otherwise, return all collected results.

        Returns:
            List[Any]: Results with priority given to finish_task output
        """
        return self.results
