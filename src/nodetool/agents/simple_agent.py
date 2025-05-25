"""
Chain of Thought (CoT) Agent implementation with tool calling capabilities.

"""

import asyncio
import json
import traceback
from typing import AsyncGenerator, List, Sequence, Union, Any

from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.workflows.types import Chunk, TaskUpdate
from nodetool.chat.providers import ChatProvider
from nodetool.agents.task_planner import clean_and_validate_path
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import (
    Message,
    SubTask,
    Task,
    ToolCall,
)
from nodetool.workflows.processing_context import ProcessingContext
from jinja2 import Environment as JinjaEnvironment, BaseLoader
from nodetool.agents.base_agent import BaseAgent

# Schema for the LLM to generate a single subtask definition
SINGLE_SUBTASK_DEFINITION_SCHEMA = {
    "type": "object",
    "description": "Definition for a single subtask to achieve a specific objective.",
    "properties": {
        "content": {
            "type": "string",
            "description": "High-level natural language instructions for the agent executing this subtask.",
        },
        "output_schema": {
            "type": "string",
            "description": 'Output schema for the subtask as a JSON string. Use \'{"type": "string"}\' for unstructured output types.',
        },
        "output_type": {
            "type": "string",
            "description": "The file format of the output of the subtask, e.g. 'json', 'markdown', 'csv', 'html'",
        },
    },
    "additionalProperties": False,
    "required": [
        "content",
        "output_type",
        "output_schema",
    ],
}


class SimpleAgent(BaseAgent):
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
        input_files: List[str] = [],
        system_prompt: str | None = None,  # System prompt for execution phase
        max_iterations: int = 20,
        max_token_limit: int | None = None,
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
        self.execution_system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.output_type = output_type
        self.output_schema = output_schema

        self.subtask: SubTask | None = None
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

## Goal
Your goal is to define a *single* subtask to achieve the given objective.
You must define the subtask's execution instructions (`content`).

Objective: {{ objective }}
Initial Input Files Available:
{%- if input_files_list %}
{{ input_files_list | join('\\n') }}
{%- else %}
None
{%- endif %}

Available Execution Tools (Agent might use these during execution):
{{ execution_tools_info }}

Define the output file's type (`output_type`) and schema (`output_schema`).

Generate a JSON object conforming EXACTLY to the 'SingleSubtaskDefinition' schema, describing the single subtask required to fulfill the objective.
"""
        variables = {
            "objective": self.objective,
            "input_files_list": self.input_files,
            "execution_tools_info": self._get_execution_tools_info(),
            "output_type": self.output_type,
            "output_schema": self.output_schema,
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

                # 2. Validate paths (output, inputs, artifacts)
                try:
                    if "artifacts" in subtask_data:
                        cleaned_artifacts = []
                        for i, f in enumerate(subtask_data.get("artifacts", [])):
                            cleaned_artifacts.append(
                                clean_and_validate_path(
                                    context.workspace_dir, f, f"artifacts[{i}]"
                                )
                            )
                        subtask_data["artifacts"] = cleaned_artifacts

                except ValueError as path_err:
                    validation_errors.append(f"Path Validation Error: {path_err}")

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

                output_filename = f"output.{self.output_type}"

                # Ensure all fields expected by SubTask are present or defaulted
                subtask_args = {
                    "content": subtask_data["content"],
                    "artifacts": subtask_data.get("artifacts", []),
                    "input_files": self.input_files,
                    "output_type": self.output_type,
                    "output_schema": json.dumps(self.output_schema),
                    "output_file": output_filename,
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
            system_prompt=self.execution_system_prompt or self.system_prompt,
            tools=self.tools,
            model=self.model,
            provider=self.provider,
            max_token_limit=self.max_token_limit,
            max_iterations=self.max_iterations,
        )

        # Execute the subtask and yield all its updates
        async for item in subtask_context.execute():
            if isinstance(item, ToolCall) and item.name in [
                "finish_subtask",
            ]:
                self.results = item.args.get("result")
            yield item

        # Execution finished (successfully or not, handled by SubTaskContext)
        print(
            f"Single Task Agent: Finished execution for objective: '{self.objective}'"
        )

    def get_results(self) -> Any:
        """
        Returns the result captured from the finish_subtask tool call, if any.
        """
        return self.results
