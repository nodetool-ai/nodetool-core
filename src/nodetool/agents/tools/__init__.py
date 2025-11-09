"""Utilities for working with agent tools."""

from __future__ import annotations

from typing import Optional, Type

from .base import Tool
from .tool_registry import load_all_nodes, _tool_node_registry
from .node_tool import NodeTool
from nodetool.workflows.base_node import get_node_class, sanitize_node_name

from .browser_tools import BrowserTool
from .google_tools import GoogleGroundedSearchTool, GoogleImageGenerationTool
from .serp_tools import GoogleSearchTool
from .workflow_tool import WorkflowTool

__all__ = [
    "Tool",
    "get_tool_by_name",
    "BrowserTool",
    "GoogleSearchTool",
    "GoogleGroundedSearchTool",
    "GoogleImageGenerationTool",
    "WorkflowTool",
]


def get_tool_by_name(name: str) -> Optional[Type[Tool]]:
    """Retrieve a zero-argument tool class for the given registered name."""

    if not _tool_node_registry:
        load_all_nodes()

    metadata = _tool_node_registry.get(name)
    if metadata is None:
        metadata = _tool_node_registry.get(sanitize_node_name(name))
    if metadata is None:
        return None

    node_class = get_node_class(metadata.node_type)
    if node_class is None:
        return None

    class _NodeToolWrapper(NodeTool):
        def __init__(self) -> None:
            super().__init__(node_class)

    _NodeToolWrapper.__name__ = f"{node_class.__name__}Tool"
    _NodeToolWrapper.__qualname__ = _NodeToolWrapper.__name__
    return _NodeToolWrapper
