"""
Tools package for nodetool chat functionality.

This package provides various utility tools for interacting with files,
services, APIs, and other resources. Tools are grouped by category in
separate modules but exposed here as a flat API for easy importing.

It uses lazy loading via PEP 562 (__getattr__, __dir__) to only import
submodules when a tool class from them is accessed.

Each tool inherits from the base Tool class and implements:
- input_schema: JSON schema defining the tool's parameters
- process(): Async method to execute the tool's functionality

Tools are used by AI agents to perform operations and integrate with various services.
"""

import importlib
from typing import Any, List

# Base helpers (imported directly)
from nodetool.agents.tools.base import sanitize_node_name, get_tool_by_name

# Email helpers (imported directly)
from nodetool.agents.tools.email_tools import (
    create_gmail_connection,
    parse_email_message,
)

# Map class names to the relative module path where they are defined
_CLASS_SUBMODULE_MAP = {
    # Base
    "Tool": ".base",
    # Web and browser
    "BrowserTool": ".browser_tools",
    "ScreenshotTool": ".browser_tools",
    "GoogleSearchTool": ".browser_tools",
    "WebFetchTool": ".browser_tools",
    "DownloadFileTool": ".browser_tools",
    "GoogleNewsTool": ".browser_tools",
    "GoogleImagesTool": ".browser_tools",
    "GoogleGroundedSearchTool": ".google_tools",
    "GoogleImageGenerationTool": ".google_tools",
    # Openai
    "OpenAIWebSearchTool": ".openai_tools",
    "OpenAIImageGenerationTool": ".openai_tools",
    "OpenAITextToSpeechTool": ".openai_tools",
    # PDF
    "ExtractPDFTextTool": ".pdf",
    "ExtractPDFTablesTool": ".pdf",
    "ConvertPDFToMarkdownTool": ".pdf",
    # Search and database
    "ChromaIndexTool": ".chroma_tools",
    "ChromaTextSearchTool": ".chroma_tools",
    "ChromaHybridSearchTool": ".chroma_tools",
    # Email
    "SearchEmailTool": ".email_tools",
    "ArchiveEmailTool": ".email_tools",
    "AddLabelTool": ".email_tools",
    # HTTP
    "DownloadFileTool": ".http_tools",
    # Workspace tools
    "WriteFileTool": ".workspace_tools",
    "ReadFileTool": ".workspace_tools",
    # Asset tools
    "ListAssetsDirectoryTool": ".asset_tools",
    "ReadAssetTool": ".asset_tools",
    "SaveAssetTool": ".asset_tools",
}

# Names to be exported (dunder all)
# Defined statically to help linters/static analysis tools
__all__ = [
    "AddLabelTool",
    "ArchiveEmailTool",
    "BrowserTool",
    "ChromaHybridSearchTool",
    "ChromaIndexTool",
    "ChromaTextSearchTool",
    "ConvertPDFToMarkdownTool",
    "DownloadFileTool",
    "ExtractPDFTablesTool",
    "ExtractPDFTextTool",
    "GoogleImagesTool",
    "GoogleNewsTool",
    "GoogleSearchTool",
    "GoogleGroundedSearchTool",
    "GoogleImageGenerationTool",
    "ListAssetsDirectoryTool",
    "OpenAIWebSearchTool",
    "OpenAIImageGenerationTool",
    "OpenAITextToSpeechTool",
    "ReadAssetTool",
    "ReadFileTool",
    "SaveAssetTool",
    "ScreenshotTool",
    "SearchEmailTool",
    "Tool",  # From .base, handled by __getattr__
    "WebFetchTool",
    "WriteFileTool",
    # Directly imported helpers
    "create_gmail_connection",
    "get_tool_by_name",
    "parse_email_message",
    "sanitize_node_name",
]


def __getattr__(name: str) -> Any:
    """
    Lazily loads tool classes upon first access.

    Args:
        name: The name of the attribute (class) being accessed.

    Returns:
        The requested class after importing its module.

    Raises:
        AttributeError: If the name is not a known tool class.
    """
    if name in _CLASS_SUBMODULE_MAP:
        module_path = _CLASS_SUBMODULE_MAP[name]
        module = importlib.import_module(module_path, package=__name__)
        attr = getattr(module, name)
        # Cache it in the module's globals for future access
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """
    Provides list of available names, including lazily loaded ones,
    for introspection tools like dir() and tab completion.
    """
    # Combine globals (like directly imported functions) and lazy-loadable class names
    return sorted(list(globals().keys()) + list(_CLASS_SUBMODULE_MAP.keys()))
