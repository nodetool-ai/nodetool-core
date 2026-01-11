"""Utilities for working with agent tools."""

from __future__ import annotations

from typing import Optional

from nodetool.workflows.base_node import get_node_class, sanitize_node_name

from .base import Tool
from .browser_tools import BrowserTool
from .google_tools import GoogleGroundedSearchTool, GoogleImageGenerationTool
from .node_tool import NodeTool
from .serp_tools import GoogleSearchTool
from .tool_registry import _tool_node_registry, load_all_nodes
from .workflow_tool import WorkflowTool

__all__ = [
    "BrowserTool",
    "GoogleGroundedSearchTool",
    "GoogleImageGenerationTool",
    "GoogleSearchTool",
    "Tool",
    "WorkflowTool",
    "get_tool_by_name",
]


def get_tool_by_name(name: str) -> type[Tool] | None:
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
