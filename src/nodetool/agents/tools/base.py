"""
Base module providing the Tool class and common utility functions.

This module includes the fundamental Tool class that all tools inherit from,
and utility functions used by multiple tools.
"""

from typing import Any, Dict

from nodetool.config.logging_config import get_logger
from nodetool.workflows.processing_context import ProcessingContext

logger = get_logger(__name__)


class Tool:
    """Base class that all tools inherit from."""

    # Class attributes expected to be defined by subclasses
    name: str = "base_tool"  # Provide a default or make abstract
    description: str = "Base tool description"
    input_schema: Dict[str, Any] = {}  # Default schema
    example: str = ""

    def user_message(self, params: Dict[str, Any]) -> str:
        """
        Returns a user message for the tool.
        """
        return f"Running {self.name}"

    def tool_param(self) -> Dict[str, Any]:
        """
        Returns the tool's definition in a format suitable for LLM function calling.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        """
        Process the tool's action. Subclasses MUST override this method.

        Args:
            context: The processing context containing shared state.
            params: A dictionary of parameters matching the tool's input_schema.

        Returns:
            The result of the tool's execution. The type depends on the tool.
        """
        logger.warning(f"Process method not implemented for tool: {self.name}")
        # Default implementation returns params, but subclasses should override
        return params

    def get_container_env(self, context: ProcessingContext) -> Dict[str, str]:
        """Return environment variables needed when running inside Docker."""
        return {}
