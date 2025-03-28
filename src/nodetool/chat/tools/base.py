"""
Base module providing the Tool class and common utility functions.

This module includes the fundamental Tool class that all tools inherit from,
and utility functions used by multiple tools.
"""

from typing import Any
import os
from datetime import datetime, timedelta
from nodetool.workflows.base_node import (
    BaseNode,
    get_node_class,
    get_registered_node_classes,
)
from nodetool.workflows.processing_context import ProcessingContext


def resolve_workspace_path(workspace_dir: str, path: str) -> str:
    """
    Resolve a path relative to the workspace directory.
    Handles paths with /workspace prefix by stripping it and resolving against the actual workspace directory.

    Args:
        path (str): The path, can be:
            - absolute with /workspace prefix
            - absolute within the actual workspace directory
            - relative to the workspace directory

    Returns:
        str: The absolute path in the actual filesystem
    """
    # Handle paths with /workspace prefix
    if path.startswith("/workspace/"):
        # Strip the /workspace prefix and treat as relative path
        relative_path = path[len("/workspace/") :]
        return os.path.normpath(os.path.join(workspace_dir, relative_path))

    elif path.startswith("workspace"):
        # Strip the /workspace prefix and treat as relative path
        relative_path = path[len("workspace") :]
        return os.path.normpath(os.path.join(workspace_dir, relative_path))

    # Handle absolute paths
    elif os.path.isabs(path):
        # Security check to ensure the path is inside the workspace
        abs_path = os.path.normpath(path)
        if not abs_path.startswith(workspace_dir):
            return os.path.normpath(os.path.join(workspace_dir, abs_path))
        return abs_path

    # Handle relative paths
    else:
        # Relative path within the workspace
        return os.path.normpath(os.path.join(workspace_dir, path))


def sanitize_node_name(node_name: str) -> str:
    """
    Sanitize a node name.

    Args:
        node_name (str): The node name.

    Returns:
        str: The sanitized node name.
    """
    segments = node_name.split(".")
    if len(node_name) > 50:
        return segments[0] + "__" + segments[-1]
    else:
        return "__".join(node_name.split("."))


# Tool registry to keep track of all tool subclasses
_tool_registry = {}


def get_registered_tools():
    """
    Get all registered tool classes.

    Returns:
        dict: A dictionary mapping tool class names to tool classes.
    """
    return _tool_registry.copy()


def get_tool_by_name(name):
    """
    Get a tool class by its name.

    Args:
        name (str): The name of the tool class.

    Returns:
        The tool class if found, None otherwise.
    """
    return _tool_registry.get(name)


class Tool:
    """Base class that all tools inherit from."""

    name: str
    description: str
    input_schema: Any
    workspace_dir: str

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def tool_param(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        return params

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register all Tool subclasses.
        This method is called when a subclass of Tool is created.
        """
        super().__init_subclass__(**kwargs)
        _tool_registry[cls.name] = cls

    def resolve_workspace_path(self, path: str) -> str:
        """
        Resolve a path relative to the workspace directory.
        Handles paths with /workspace prefix by stripping it and resolving against the actual workspace directory.

        Args:
            path (str): The path, can be:
                - absolute with /workspace prefix
                - absolute within the actual workspace directory
                - relative to the workspace directory

        Returns:
            str: The absolute path in the actual filesystem
        """
        return resolve_workspace_path(self.workspace_dir, path)
