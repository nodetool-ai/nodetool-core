from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, Any

from nodetool.agents.tools.base import Tool
from nodetool.providers import BaseProvider
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    from nodetool.metadata.types import Task


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Defines the common interface for planning and execution.
    """

    def __init__(
        self,
        name: str,
        objective: str,
        provider: BaseProvider,
        model: str,
        tools: Sequence[Tool] | None = None,
        inputs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        max_token_limit: int | None = None,
    ):
        self.name = name
        self.objective = objective
        self.provider = provider
        self.model = model
        self.tools = tools or []
        self.inputs = inputs or {}
        self.system_prompt = system_prompt or ""  # Ensure system_prompt is a string
        self.max_token_limit = max_token_limit or provider.get_context_length(model)
        self.results: Any = None  # To store results, consistent with both agent types
        self.task: Task | None = None  # Common attribute to store the task

    @abstractmethod
    async def execute(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Any, None]:
        """
        Execute the agent's objective.

        This method should be implemented by subclasses to define the specific
        planning and execution logic.

        Args:
            context (ProcessingContext): The processing context.

        Yields:
            Any: Execution progress updates.
        """
        if False:
            yield None
        raise NotImplementedError

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
