from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Sequence, Union, Any, Optional

from nodetool.chat.providers import ChatProvider
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import Task, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, PlanningUpdate, TaskUpdate


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Defines the common interface for planning and execution.
    """

    def __init__(
        self,
        name: str,
        objective: str,
        provider: ChatProvider,
        model: str,
        tools: Optional[Sequence[Tool]] = None,
        inputs: dict[str, Any] | None = None,
        system_prompt: Optional[str] = None,
        max_token_limit: Optional[int] = None,
    ):
        self.name = name
        self.objective = objective
        self.provider = provider
        self.model = model
        self.tools = tools or []
        self.inputs = inputs or {}
        self.system_prompt = system_prompt or ""  # Ensure system_prompt is a string
        self.max_token_limit = max_token_limit or provider.get_max_token_limit(model)
        self.results: Any = None  # To store results, consistent with both agent types
        self.task: Optional[Task] = None  # Common attribute to store the task

    @abstractmethod
    async def execute(
        self,
        processing_context: ProcessingContext,
    ) -> AsyncGenerator[Union[TaskUpdate, Chunk, ToolCall, PlanningUpdate], None]:
        """
        Execute the agent's objective.

        This method should be implemented by subclasses to define the specific
        planning and execution logic.

        Args:
            processing_context (ProcessingContext): The processing context.

        Yields:
            Union[TaskUpdate, Chunk, ToolCall, PlanningUpdate]: Execution progress.
        """
        pass

    @abstractmethod
    def get_results(self) -> Any:
        """
        Retrieve the results of the agent's execution.

        This method should be implemented by subclasses to return the
        final output or product of the agent's work.

        Returns:
            Any: The results of the execution.
        """
        pass

    # Potentially add other common utility methods here if identified
    # e.g., _validate_inputs, _setup_workspace, etc.
