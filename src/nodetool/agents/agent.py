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
import uuid
from jinja2 import BaseLoader, Environment as JinjaEnvironment
import asyncio
import traceback

from nodetool.common.settings import get_log_path
from nodetool.workflows.types import Chunk, PlanningUpdate, TaskUpdate, TaskUpdateEvent
from nodetool.agents.task_executor import TaskExecutor
from nodetool.chat.providers import ChatProvider
from nodetool.agents.task_planner import TaskPlanner, clean_and_validate_path
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import (
    Message,
    SubTask,
    Task,
    ToolCall,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.sub_task_context import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKEN_LIMIT,
    SubTaskContext,
)
from nodetool.ui.console import AgentConsole


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
        planning_model: str | None = None,
        reasoning_model: str | None = None,
        tools: Sequence[Tool] = [],
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
        self.name = name
        self.objective = objective
        self.description = description
        self.provider = provider
        self.model = model
        self.planning_model = planning_model or model
        self.reasoning_model = reasoning_model or model
        self.max_steps = max_steps
        self.max_subtask_iterations = max_subtask_iterations
        self.max_token_limit = max_token_limit
        self.tools = tools
        self.input_files = input_files
        self.max_subtasks = max_subtasks
        self.system_prompt = system_prompt or ""
        self.results: Any = None
        self.output_schema = output_schema
        self.output_type = output_type
        self.enable_analysis_phase = enable_analysis_phase
        self.enable_data_contracts_phase = enable_data_contracts_phase
        self.initial_task = task  # Store the initial task
        self.verbose = verbose  # Store verbose flag
        self.display_manager = AgentConsole(
            verbose=self.verbose
        )  # Initialize display manager

    async def execute(
        self,
        processing_context: ProcessingContext,
    ) -> AsyncGenerator[Union[TaskUpdate, Chunk, ToolCall, PlanningUpdate], None]:
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
        # Ensure output_files directory exists as well
        os.makedirs(
            os.path.join(processing_context.workspace_dir, "output_files"),
            exist_ok=True,
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
        task: Optional[Task] = None  # Initialize task

        if self.initial_task:
            task = self.initial_task
            for subtask in task.subtasks:
                if subtask.output_file and not os.path.isabs(subtask.output_file):
                    subtask.output_file = os.path.join(
                        processing_context.workspace_dir, subtask.output_file
                    )

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

            task_planner = TaskPlanner(
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
                verbose=self.verbose,  # Pass verbose flag
            )

            async for chunk in task_planner.create_task(
                processing_context, self.objective
            ):
                yield chunk

            task = task_planner.task
            assert task is not None, "Task was not created by planner"

            yield TaskUpdate(
                task=task,
                event=TaskUpdateEvent.TASK_CREATED,
            )

        if self.output_type and len(task.subtasks) > 0:
            task.subtasks[-1].output_type = self.output_type

        if self.output_schema and len(task.subtasks) > 0:
            task.subtasks[-1].output_schema = json.dumps(self.output_schema)

        tool_calls: List[ToolCall] = []

        # Start live display managed by AgentConsole
        self.display_manager.start_live(
            self.display_manager.create_execution_table(
                title=f"Task:\n{self.objective}", task=task, tool_calls=tool_calls
            )
        )

        try:
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
                self._save_task(task, processing_context.workspace_dir)
                # Update tool_calls list if item is a ToolCall
                if isinstance(item, ToolCall):
                    tool_calls.append(item)

                # Create the updated table and update the live display
                new_table = self.display_manager.create_execution_table(
                    title=f"Task:\n{self.objective}", task=task, tool_calls=tool_calls
                )
                self.display_manager.update_live(new_table)

                # Yield the item
                if isinstance(item, ToolCall) and item.name == "finish_task":
                    self.results = item.args["result"]
                    yield TaskUpdate(
                        task=task,
                        event=TaskUpdateEvent.TASK_COMPLETED,
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

    def _save_task(self, task: Task, workspace_dir: str) -> None:
        """
        Save the current task plan to the tasks file.
        """
        task_dict = task.model_dump()
        import json

        with open(
            os.path.join(workspace_dir, "tasks.json"),
            "w",
        ) as f:
            json.dump(task_dict, f, indent=2)


# Schema for the LLM to generate a single subtask definition
SINGLE_SUBTASK_DEFINITION_SCHEMA = {
    "type": "object",
    "description": "Definition for a single subtask to achieve a specific objective.",
    "properties": {
        "content": {
            "type": "string",
            "description": "High-level natural language instructions for the agent executing this subtask.",
        },
    },
    "additionalProperties": False,
    "required": [
        "content",
    ],
}


class SingleTaskAgent:
    """
    🎯 Plans and executes a single task based on an objective.

    This agent takes a high-level objective and an output filename. It performs
    a lightweight planning step to define the necessary instructions, output type,
    and schema for a *single* subtask required to meet the objective. It then
    executes this subtask using a SubTaskContext.
    """

    def __init__(
        self,
        name: str,
        objective: str,
        provider: ChatProvider,
        model: str,
        tools: Sequence[Tool],
        output_type: str,
        output_schema: dict[str, Any],
        input_files: List[str] = [],  # Add initial input files
        system_prompt: str | None = None,  # System prompt for execution phase
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_token_limit: int = DEFAULT_MAX_TOKEN_LIMIT,
    ):
        """
        Initialize the SingleTaskAgent.

        Args:
            name (str): The name of the agent.
            objective (str): The high-level goal for the agent to achieve.
            provider (ChatProvider): An LLM provider instance.
            model (str): The model to use with the provider.
            tools (Sequence[Tool]): List of tools potentially usable by the subtask executor.
            input_files (List[str], optional): List of initial input files available.
            system_prompt (str, optional): Custom system prompt for the subtask execution phase.
            output_type (str): The type of the output file.
            output_schema (dict): The schema of the output file.
            max_iterations (int, optional): Maximum iterations for the subtask execution.
            max_token_limit (int, optional): Maximum token limit for the subtask context.
        """
        self.name = name
        self.objective = objective
        self.provider = provider
        self.model = model
        self.tools = tools  # Tools for the *execution* context
        self.input_files = input_files  # Store initial inputs
        self.execution_system_prompt = system_prompt  # Renamed for clarity
        self.max_iterations = max_iterations
        self.max_token_limit = max_token_limit
        self.output_type = output_type
        self.output_schema = output_schema

        self.task: Task | None = None
        self.subtask: SubTask | None = None
        self.results: Any = None  # To store the final result if needed
        self.jinja_env = JinjaEnvironment(loader=BaseLoader())  # For prompt rendering

    def _get_execution_tools_info(self) -> str:
        """Helper to format execution tool info for prompts."""
        if not self.tools:
            return "No execution tools available."
        info = []
        for tool in self.tools:
            # Basic info, could be expanded like in TaskPlanner
            info.append(f"- {tool.name}: {tool.description}")
        return "\n".join(info)

    async def _plan_single_subtask(
        self, context: ProcessingContext, max_retries: int = 3
    ):
        """
        Uses the LLM to define the properties (content, schema, type) for the single subtask.
        """
        if self.task and self.subtask:
            return  # Already planned

        # --- Prepare Planning Prompt ---
        template_string = """
You are the {{ name }} agent.
Your goal is to define a *single* subtask to achieve the given objective.
You must define the subtask's execution instructions (`content`) and specify the output type (`output_type`).

Objective: {{ objective }}
Initial Input Files Available:
{%- if input_files_list %}
{{ input_files_list | join('\\n') }}
{%- else %}
None
{%- endif %}

Available Execution Tools (Agent might use these during execution):
{{ execution_tools_info }}

Generate a JSON object conforming EXACTLY to the 'SingleSubtaskDefinition' schema, describing the single subtask required to fulfill the objective.
"""
        variables = {
            "objective": self.objective,
            "input_files_list": self.input_files,
            "execution_tools_info": self._get_execution_tools_info(),
        }
        planning_prompt = self.jinja_env.from_string(template_string).render(variables)

        # --- LLM Interaction with Retry ---
        current_retry = 0
        last_error = None
        while current_retry < max_retries:
            attempt = current_retry + 1
            messages = [
                # No system prompt needed here usually, the user prompt is specific
                Message(role="user", content=planning_prompt)
            ]
            if last_error:  # Add error feedback for retry attempts
                messages.insert(
                    0,
                    Message(
                        role="system",
                        content=f"Previous attempt failed: {last_error}. Please correct the output.",
                    ),
                )

            try:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "SingleSubtaskDefinition",
                        "schema": SINGLE_SUBTASK_DEFINITION_SCHEMA,
                        "strict": True,
                    },
                }
                message = await self.provider.generate_message(
                    messages=messages,
                    model=self.model,  # Use the agent's model
                    response_format=response_format,
                )

                if not isinstance(message.content, str) or not message.content.strip():
                    raise ValueError(
                        "LLM returned empty content for subtask definition."
                    )

                # Parse and Validate JSON
                try:
                    subtask_data = json.loads(message.content)
                except json.JSONDecodeError as json_err:
                    raise ValueError(
                        f"Failed to decode JSON: {json_err}. Response: {message.content}"
                    )

                # --- Validate Subtask Data ---
                validation_errors = []

                # Check for other required fields implicitly handled by schema, but double-check content
                if not subtask_data.get("content"):
                    validation_errors.append(
                        "Validation Error: 'content' field is missing or empty."
                    )

                if validation_errors:
                    raise ValueError(
                        "Subtask definition failed validation:\n"
                        + "\n".join(validation_errors)
                    )

                # --- Create Task and Subtask Objects ---
                # Generate a unique output filename within the 'output_files' directory
                output_dir_rel = "output_files"
                output_filename = f"{self.name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.txt"  # Default extension
                # Determine extension based on output_type if possible
                ext_map = {
                    "json": ".json",
                    "csv": ".csv",
                    "markdown": ".md",
                    "text": ".txt",
                    "html": ".html",
                    "yaml": ".yaml",
                    # Add other relevant types if needed
                }
                output_filename = f"{self.name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}{ext_map.get(self.output_type, '.txt')}"
                output_file_rel = os.path.join(output_dir_rel, output_filename)

                # Ensure all fields expected by SubTask are present or defaulted
                subtask_args = {
                    "content": subtask_data["content"],
                    "input_files": self.input_files,
                    "output_type": self.output_type,
                    "output_schema": json.dumps(self.output_schema),
                    "output_file": output_file_rel,  # Use generated relative path
                }
                self.subtask = SubTask(**subtask_args)
                self.task = Task(title=self.objective, subtasks=[self.subtask])

                print(
                    f"Single Task Agent: Successfully planned subtask for objective: '{self.objective}'"
                )
                return

            except Exception as e:
                last_error = e
                print(
                    f"Single Task Agent: Planning attempt {attempt}/{max_retries} failed: {e}"
                )
                current_retry += 1
                if current_retry >= max_retries:
                    print(
                        f"Single Task Agent: Planning failed after {max_retries} attempts."
                    )
                    raise ValueError(
                        f"Failed to plan single subtask after {max_retries} attempts."
                    ) from e
                await asyncio.sleep(1)  # Small delay before retry

        # Should not be reached if max_retries > 0
        raise RuntimeError("Planning loop exited unexpectedly.")

    async def execute(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Union[TaskUpdate, Chunk, ToolCall], None]:
        """
        Plans (if needed) and executes the single subtask for the agent's objective.

        Yields:
            Union[TaskUpdate, Chunk, ToolCall]: Execution progress from the SubTaskContext.
        """
        # --- Plan Phase (if not already done) ---
        if not self.task or not self.subtask:
            try:
                await self._plan_single_subtask(context)
            except Exception as planning_error:
                # Yield an error status or re-raise? Let's re-raise for now.
                # Alternatively, could yield a specific error TaskUpdate.
                print(
                    f"Single Task Agent: Fatal planning error: {planning_error}\n{traceback.format_exc()}"
                )
                raise RuntimeError(
                    f"Failed to plan the required subtask: {planning_error}"
                ) from planning_error

        # Ensure planning was successful
        if not self.task or not self.subtask:
            raise RuntimeError("Task and Subtask are not defined after planning phase.")

        # --- Execution Phase ---
        print(
            f"Single Task Agent: Executing planned subtask for objective: '{self.objective}'"
        )
        subtask_context = SubTaskContext(
            task=self.task,
            subtask=self.subtask,
            processing_context=context,
            system_prompt=self.execution_system_prompt,  # Use the execution prompt
            tools=self.tools,  # Provide tools for execution
            model=self.model,
            provider=self.provider,
            max_token_limit=self.provider.get_max_token_limit(self.model),
            max_iterations=self.max_iterations,
            save_output_to_file=False,
        )

        # Execute the subtask and yield all its updates
        final_result_from_tool = None
        async for item in subtask_context.execute():
            if isinstance(item, ToolCall) and item.name in [
                "finish_subtask",
                "finish_task",
            ]:
                if isinstance(item.args, dict):
                    # Capture the direct result from the finish tool call args
                    final_result_from_tool = item.args.get("result")
            yield item  # Continue yielding progress

        # --- Post-Execution Result Processing ---
        output_file_path_abs = None
        if self.subtask and self.subtask.output_file:
            # Construct absolute path from relative path stored in subtask
            output_file_path_abs = os.path.join(
                context.workspace_dir, self.subtask.output_file
            )
            # Ensure the output directory exists (it should, but double-check)
            os.makedirs(os.path.dirname(output_file_path_abs), exist_ok=True)

        processed_result = (
            final_result_from_tool  # Start with the result from the tool call
        )

        # Scenario 1: Tool returned a path (or we expect output in the designated file)
        # Check if the result looks like a path structure OR if an output file exists
        result_is_path_dict = (
            isinstance(processed_result, dict)
            and "path" in processed_result
            and isinstance(processed_result.get("path"), str)
        )
        output_file_to_read = None

        if result_is_path_dict:
            # Prefer path from result dict if available and seems valid
            # Ensure processed_result is actually a dict before accessing "path"
            if isinstance(processed_result, dict):  # <-- Added check
                maybe_path = os.path.join(
                    context.workspace_dir, processed_result["path"]
                )
                if os.path.exists(maybe_path):
                    output_file_to_read = maybe_path
                elif output_file_path_abs and os.path.exists(
                    output_file_path_abs
                ):  # Fallback to subtask's path
                    output_file_to_read = output_file_path_abs
            # else: If not a dict, can't have a path key, do nothing here
        elif output_file_path_abs and os.path.exists(output_file_path_abs):
            # If result wasn't a path dict, but the expected output file exists, read it
            output_file_to_read = output_file_path_abs

        if output_file_to_read:
            try:
                with open(output_file_to_read, "r") as f:
                    file_content = f.read()
                # Try parsing if output type suggests structured data (e.g., JSON)
                if self.output_type == "json":
                    try:
                        processed_result = json.loads(file_content)
                    except json.JSONDecodeError:
                        processed_result = file_content  # Fallback to raw content
                else:
                    processed_result = (
                        file_content  # Return raw content for non-JSON types
                    )
            except Exception as read_err:
                print(
                    f"Single Task Agent: Error reading output file {output_file_to_read}: {read_err}"
                )
                # Keep the original `processed_result` (which might be the path dict or None)

        # Update self.results with the potentially processed content
        self.results = processed_result

    def get_results(self) -> Any:
        """
        Returns the result captured from the finish_subtask tool call, if any.
        """
        return self.results
