"""Utilities for working with agent tools."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nodetool.agents.tools.base import Tool
    from nodetool.agents.tools.browser_tools import BrowserTool
    from nodetool.agents.tools.google_tools import GoogleGroundedSearchTool, GoogleImageGenerationTool
    from nodetool.agents.tools.node_tool import NodeTool
    from nodetool.agents.tools.serp_tools import GoogleSearchTool
    from nodetool.agents.tools.workflow_tool import WorkflowTool

__all__ = [
    "BrowserTool",
    "GoogleGroundedSearchTool",
    "GoogleImageGenerationTool",
    "GoogleSearchTool",
    "Tool",
    "WorkflowTool",
    "get_tool_by_name",
]

# Lazy attribute mapping: name -> (module, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Tool": ("nodetool.agents.tools.base", "Tool"),
    "BrowserTool": ("nodetool.agents.tools.browser_tools", "BrowserTool"),
    "GoogleGroundedSearchTool": ("nodetool.agents.tools.google_tools", "GoogleGroundedSearchTool"),
    "GoogleImageGenerationTool": ("nodetool.agents.tools.google_tools", "GoogleImageGenerationTool"),
    "NodeTool": ("nodetool.agents.tools.node_tool", "NodeTool"),
    "GoogleSearchTool": ("nodetool.agents.tools.serp_tools", "GoogleSearchTool"),
    "WorkflowTool": ("nodetool.agents.tools.workflow_tool", "WorkflowTool"),
    "_tool_node_registry": ("nodetool.agents.tools.tool_registry", "_tool_node_registry"),
    "load_all_nodes": ("nodetool.agents.tools.tool_registry", "load_all_nodes"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_tool_by_name(name: str) -> type[Tool] | None:
    """Retrieve a zero-argument tool class for the given registered name."""
    from nodetool.agents.tools.node_tool import NodeTool
    from nodetool.agents.tools.tool_registry import _tool_node_registry, load_all_nodes
    from nodetool.workflows.base_node import get_node_class, sanitize_node_name

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
