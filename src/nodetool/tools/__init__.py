"""Reusable tools module for NodeTool.

This module provides tool functions that can be called from both:
- MCP (Model Context Protocol) server via @mcp.tool() decorators
- CLI commands via direct function calls

The functions are organized by category and are designed to be:
- Async-first (for MCP compatibility)
- Type-hinted (for CLI auto-completion)
- Well-documented (for help text)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .workflow_tools import WorkflowTools
    from .asset_tools import AssetTools
    from .node_tools import NodeTools
    from .model_tools import ModelTools
    from .collection_tools import CollectionTools
    from .job_tools import JobTools
    from .agent_tools import AgentTools
    from .storage_tools import StorageTools
    from .hf_tools import HfTools

# Re-export all tool categories for easy importing
__all__ = [
    "WorkflowTools",
    "AssetTools",
    "NodeTools",
    "ModelTools",
    "CollectionTools",
    "JobTools",
    "AgentTools",
    "StorageTools",
    "HfTools",
    "get_all_tool_functions",
]

_TOOL_MODULES = {
    "WorkflowTools": "workflow_tools",
    "AssetTools": "asset_tools",
    "NodeTools": "node_tools",
    "ModelTools": "model_tools",
    "CollectionTools": "collection_tools",
    "JobTools": "job_tools",
    "AgentTools": "agent_tools",
    "StorageTools": "storage_tools",
    "HfTools": "hf_tools",
}


def __getattr__(name: str) -> Any:
    if name in _TOOL_MODULES:
        module_name = _TOOL_MODULES[name]
        module = importlib.import_module(f".{module_name}", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_all_tool_functions() -> dict[str, Any]:
    """
    Get all tool functions organized by category.

    Returns:
        Dictionary mapping tool names to async functions
    """
    all_tools = {}

    for module_name in _TOOL_MODULES.values():
        try:
            module = importlib.import_module(f".{module_name}", __package__)
            if hasattr(module, "get_tool_functions"):
                all_tools.update(module.get_tool_functions())
        except ImportError as e:
            # Log error but don't fail if a tool module is broken
            # We can't use logger here easily without potentially circular imports or eager logging init
            pass

    return all_tools
