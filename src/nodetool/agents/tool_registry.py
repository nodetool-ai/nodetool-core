"""Tool registry for mapping tool names to Tool instances.

This module provides a registry of available tools that can be specified
by name in agent YAML configuration files.
"""

from typing import Dict, Type

from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.browser_tools import (
    BrowserClickTool,
    BrowserNavigateTool,
    BrowserQueryElementsTool,
    BrowserScreenshotTool,
    BrowserTypeTool,
)
from nodetool.agents.tools.chroma_tools import (
    ChromaHybridSearchTool,
    ChromaIndexTool,
    ChromaMarkdownSplitAndIndexTool,
)
from nodetool.agents.tools.code_tools import ExecutePythonTool
from nodetool.agents.tools.filesystem_tools import (
    DeleteFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from nodetool.agents.tools.math_tools import CalculatorTool
from nodetool.agents.tools.serp_tools import (
    GoogleImagesTool,
    GoogleNewsTool,
    GoogleSearchTool,
)

# Registry mapping tool names to their classes
TOOL_REGISTRY: Dict[str, Type[Tool]] = {
    # Browser tools
    "browser_navigate": BrowserNavigateTool,
    "browser_screenshot": BrowserScreenshotTool,
    "browser_query": BrowserQueryElementsTool,
    "browser_click": BrowserClickTool,
    "browser_type": BrowserTypeTool,
    "browser": BrowserNavigateTool,  # Alias for navigation

    # Filesystem tools
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "list_directory": ListDirectoryTool,
    "delete_file": DeleteFileTool,
    "filesystem": ReadFileTool,  # Alias, can read files

    # Code execution
    "python": ExecutePythonTool,
    "execute_python": ExecutePythonTool,

    # Math tools
    "calculator": CalculatorTool,
    "math": CalculatorTool,

    # Search tools
    "google_search": GoogleSearchTool,
    "google_news": GoogleNewsTool,
    "google_images": GoogleImagesTool,
    "search": GoogleSearchTool,  # Alias

    # ChromaDB tools (require collection parameter, handled separately)
    "chroma_index": ChromaIndexTool,
    "chroma_search": ChromaHybridSearchTool,
    "chroma_markdown": ChromaMarkdownSplitAndIndexTool,
}


def get_tool_instance(tool_name: str, **kwargs) -> Tool:
    """Get a tool instance by name.

    Args:
        tool_name: Name of the tool to instantiate
        **kwargs: Additional arguments to pass to the tool constructor

    Returns:
        Tool instance

    Raises:
        ValueError: If the tool name is not recognized
    """
    tool_class = TOOL_REGISTRY.get(tool_name)

    if tool_class is None:
        available_tools = ", ".join(sorted(TOOL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown tool: {tool_name}. Available tools: {available_tools}"
        )

    return tool_class(**kwargs)


def get_available_tools() -> list[str]:
    """Get a list of all available tool names.

    Returns:
        Sorted list of tool names
    """
    return sorted(TOOL_REGISTRY.keys())
