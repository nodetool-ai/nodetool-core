"""
Chain of Thought (CoT) Agent implementation with tool calling capabilities.

"""

import json
from typing import Any, AsyncGenerator, Sequence

from jinja2 import BaseLoader
from jinja2 import Environment as JinjaEnvironment

from nodetool.agents.base_agent import BaseAgent
from nodetool.agents.step_executor import StepExecutor
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import (
    Step,
    Task,
    ToolCall,
)
from nodetool.providers import BaseProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import StepResult


class SimpleAgent(BaseAgent):
    """
    🎯 Plans and executes a single task based on an objective.

    This agent takes a high-level objective and an output schema. It performs
    a lightweight planning step to define the necessary instructions, output type,
    and schema for a *single* step required to meet the objective. It then
    executes this step using a StepExecutor.
    """

    def __init__(
        self,
        name: str,
        objective: str,
        provider: BaseProvider,
        model: str,
        tools: Sequence[Tool],
        output_schema: dict[str, Any],
        inputs: dict[str, Any] | None = None,
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
            tools (Sequence[Tool]): List of tools potentially usable by the step executor.
            inputs (dict[str, Any], optional): Dictionary of initial input files available.
            system_prompt (str, optional): Custom system prompt for the step execution phase.
            output_schema (dict): The schema of the output file.
            max_iterations (int, optional): Maximum iterations for the step execution.
            max_token_limit (int, optional): Maximum token limit for the step context.
        """
        super().__init__(
            name=name,
            objective=objective,
            provider=provider,
            model=model,
            tools=tools,
            inputs=inputs,
            system_prompt=system_prompt,
            max_token_limit=max_token_limit,
        )
        self.execution_system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.output_schema = output_schema

        self.step: Step | None = None
        self.jinja_env = JinjaEnvironment(loader=BaseLoader())  # For prompt rendering
        self.step_executor: StepExecutor | None = None

    def _get_execution_tools_info(self) -> str:
        """Helper to format execution tool info for prompts."""
        if not self.tools:
            return "No execution tools available."
        info = []
        for tool in self.tools:
            # Basic info, could be expanded like in TaskPlanner
            info.append(f"- {tool.name}: {tool.description}")
        return "\n".join(info)

    async def execute(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Any, None]:
        """
        Plans (if needed) and executes the single step for the agent's objective.

        Yields:
            Union[TaskUpdate, Chunk, ToolCall, StepResult]: Execution progress from the StepExecutor.
        """
        self.step = Step(
            instructions=self.objective,
            output_schema=json.dumps(self.output_schema),
        )
        self.task = Task(title=self.objective, steps=[self.step])

        self.step_executor = StepExecutor(
            task=self.task,
            step=self.step,
            processing_context=context,
            system_prompt=self.execution_system_prompt or self.system_prompt,
            tools=self.tools,
            model=self.model,
            provider=self.provider,
            max_token_limit=self.max_token_limit,
            max_iterations=self.max_iterations,
        )

        # Execute the step and yield all its updates
        async for item in self.step_executor.execute():
            if isinstance(item, StepResult):
                self.results = item.result
            yield item

    def get_results(self) -> Any:
        """
        Returns the result captured from the completion JSON, if any.
        """
        return self.results
