"""
Base module providing the Tool class and common utility functions.

This module includes the fundamental Tool class that all tools inherit from,
and utility functions used by multiple tools.
"""

from typing import Any, Optional, Type, Dict
import logging
from nodetool.workflows.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


def sanitize_node_name(node_name: str) -> str:
    """
    Convert node type to tool name format.
    
    Converts from node type format (e.g., "namespace.TestNode") to tool name format
    (e.g., "node_test"). Handles CamelCase to snake_case conversion and adds "node_" prefix.

    Args:
        node_name: The node type string.

    Returns:
        The sanitized tool name.
    """
    if not isinstance(node_name, str):
        logger.warning(
            f"Invalid node_name type: {type(node_name)}. Returning empty string."
        )
        return ""

    # Extract the class name from the full node type (e.g., "test.TestNode" -> "TestNode")
    class_name = node_name.split(".")[-1]
    
    # Remove "Node" suffix if present
    if class_name.endswith("Node"):
        class_name = class_name[:-4]
    
    # Convert CamelCase to snake_case
    import re
    snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', class_name).lower()
    
    # Add "node_" prefix
    result = f"node_{snake_case}"
    
    # Truncate if necessary (adjust max length as needed)
    max_length = 64  # Example max length
    if len(result) > max_length:
        # Simple truncation, could be smarter (e.g., keep start/end)
        return result[:max_length]
    else:
        return result


# Tool registry to keep track of all tool subclasses
_tool_registry: Dict[str, Type["Tool"]] = {}


def get_registered_tools() -> Dict[str, Type["Tool"]]:
    """
    Get all registered tool classes.

    Returns:
        A dictionary mapping tool class names to tool classes.
    """
    return _tool_registry.copy()


def get_tool_by_name(name: str) -> Optional[Type["Tool"]]:
    """
    Get a tool class by its registered name.

    Args:
        name: The name of the tool class.

    Returns:
        The tool class if found, None otherwise.
    """
    return _tool_registry.get(name)


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
        return f"Running {self.name} with the following parameters: {params}"

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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Automatically register all valid Tool subclasses in the registry.
        This method is called when a subclass of Tool is defined.
        """
        super().__init_subclass__(**kwargs)
        # Register only if name is defined and not the base default
        if hasattr(cls, "name") and cls.name and cls.name != "base_tool":
            if cls.name in _tool_registry:
                logger.warning(
                    f"Tool name '{cls.name}' from class {cls.__name__} conflicts with existing tool {_tool_registry[cls.name].__name__}. Overwriting."
                )
            _tool_registry[cls.name] = cls
            # logger.debug(f"Registered tool: {cls.name} ({cls.__name__})")
        # else:
        #      if cls.__name__ != "Tool": # Don't warn for the base class itself
        #          logger.debug(f"Skipping registration for class {cls.__name__} (missing or default name).")

    def get_container_env(self) -> Dict[str, str]:
        """Return environment variables needed when running inside Docker."""
        return {}
