"""
Base module providing the Tool class and common utility functions.

This module includes the fundamental Tool class that all tools inherit from,
and utility functions used by multiple tools.
"""

from typing import Any, Dict, Optional, Type, Dict, cast
import os
import logging
from nodetool.workflows.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


def resolve_workspace_path(workspace_dir: str, path: str) -> str:
    """
    Resolve a path relative to the workspace directory.
    Handles paths starting with '/workspace/', 'workspace/', or absolute paths
    by interpreting them relative to the `workspace_dir`.

    Args:
        workspace_dir: The absolute path to the workspace directory.
        path: The path to resolve, which can be:
            - Prefixed with '/workspace/' (e.g., '/workspace/output/file.txt')
            - Prefixed with 'workspace/' (e.g., 'workspace/output/file.txt')
            - An absolute path (e.g., '/input/data.csv') - treated relative to workspace root
            - A relative path (e.g., 'output/file.txt')

    Returns:
        The absolute path in the actual filesystem.

    Raises:
        ValueError: If workspace_dir is not provided or empty.
    """
    if not workspace_dir:
        raise ValueError("Workspace directory is required")

    relative_path: str
    # Normalize path separators for consistent checks
    normalized_path = path.replace("\\", "/")

    # Handle paths with /workspace/ prefix
    if normalized_path.startswith("/workspace/"):
        relative_path = normalized_path[len("/workspace/") :]
    # Handle paths with workspace/ prefix (without leading slash)
    elif normalized_path.startswith("workspace/"):
        relative_path = normalized_path[len("workspace/") :]
    # Handle absolute paths by stripping leading slash and treating as relative to workspace
    elif os.path.isabs(normalized_path):
        # On Windows, isabs('/') is False. Check explicitly.
        if normalized_path.startswith("/"):
            relative_path = normalized_path[1:]
        else:
            # For Windows absolute paths (e.g., C:\...), we still want to join them relative to workspace?
            # This behaviour might need clarification. Assuming here they are treated as relative for consistency.
            # If absolute paths outside workspace should be allowed, this needs change.
            logger.warning(
                f"Treating absolute path '{path}' as relative to workspace root '{workspace_dir}'."
            )
            # Attempt to get path relative to drive root
            drive, path_part = os.path.splitdrive(normalized_path)
            relative_path = path_part.lstrip(
                "\\/"
            )  # Strip leading slashes from the part after drive
    # Handle relative paths
    else:
        relative_path = normalized_path

    # Prevent path traversal attempts (e.g., ../../etc/passwd)
    # Join the workspace directory with the potentially cleaned relative path
    abs_path = os.path.abspath(os.path.join(workspace_dir, relative_path))

    # Final check: ensure the resolved path is still within the workspace directory
    # Use commonprefix for robustness across OS
    common_prefix = os.path.commonprefix([os.path.abspath(workspace_dir), abs_path])
    if os.path.abspath(workspace_dir) != common_prefix:
        logger.error(
            f"Resolved path '{abs_path}' is outside the workspace directory '{workspace_dir}'. Original path: '{path}'"
        )
        # Option 1: Raise an error
        raise ValueError(
            f"Resolved path '{abs_path}' is outside the workspace directory."
        )
        # Option 2: Return a default safe path or the workspace root (less ideal)
        # return workspace_dir

    return abs_path


def sanitize_node_name(node_name: str) -> str:
    """
    Sanitize a node name, typically replacing '.' with '__' and potentially truncating.

    Args:
        node_name: The node name.

    Returns:
        The sanitized node name.
    """
    if not isinstance(node_name, str):
        logger.warning(
            f"Invalid node_name type: {type(node_name)}. Returning empty string."
        )
        return ""

    segments = node_name.split(".")
    # Basic sanitization: replace non-alphanumeric (except _) with underscore
    safe_segments = ["".join(c if c.isalnum() else "_" for c in s) for s in segments]

    # Join sanitized segments
    sanitized_name = "__".join(safe_segments)

    # Truncate if necessary (adjust max length as needed)
    max_length = 64  # Example max length
    if len(sanitized_name) > max_length:
        # Simple truncation, could be smarter (e.g., keep start/end)
        return sanitized_name[:max_length]
    else:
        return sanitized_name


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

    # Instance attribute
    workspace_dir: str

    def __init__(self, workspace_dir: str):
        """
        Initialize the Tool.

        Args:
            workspace_dir: The absolute path to the workspace directory.
        """
        if not workspace_dir or not os.path.isdir(workspace_dir):
            raise ValueError(f"Invalid workspace directory provided: {workspace_dir}")
        self.workspace_dir = os.path.abspath(workspace_dir)

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

    def resolve_workspace_path(self, path: str) -> str:
        """
        Helper method to resolve a path relative to the tool's workspace directory.

        Args:
            path: The path string to resolve.

        Returns:
            The absolute path in the actual filesystem, constrained to the workspace.

        Raises:
            ValueError: If the resolved path is outside the workspace directory.
        """
        return resolve_workspace_path(self.workspace_dir, path)
