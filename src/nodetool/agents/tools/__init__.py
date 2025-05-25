"""
Tools package for nodetool chat functionality.

This package provides various utility tools for interacting with files,
services, APIs, and other resources. Tools are grouped by category in
separate modules but exposed here as a flat API for easy importing.

Each tool inherits from the base Tool class and implements:
- input_schema: JSON schema defining the tool's parameters
- process(): Async method to execute the tool's functionality

Tools are used by AI agents to perform operations and integrate with various services.
"""

# Base helpers
from nodetool.agents.tools.base import Tool, get_tool_by_name, sanitize_node_name

# Browser tools
from .browser_tools import BrowserTool, ScreenshotTool
from .browser_use_tools import BrowserUseTool

# HTTP tools
from .http_tools import DownloadFileTool

# SERP tools
from .serp_tools import (
    GoogleImagesTool,
    GoogleNewsTool,
    GoogleSearchTool,
    GoogleLensTool,
    GoogleMapsTool,
    GoogleShoppingTool,
    GoogleFinanceTool,
    GoogleJobsTool,
)

# Google tools
from .google_tools import GoogleGroundedSearchTool, GoogleImageGenerationTool

# OpenAI tools
from .openai_tools import (
    OpenAIImageGenerationTool,
    OpenAITextToSpeechTool,
    OpenAIWebSearchTool,
)

# MCP tools
from .mcp_tools import MCPTool

# PDF tools
from .pdf_tools import (
    ConvertPDFToMarkdownTool,
    ExtractPDFTablesTool,
    ExtractPDFTextTool,
)

# Search and database tools
from .chroma_tools import ChromaHybridSearchTool, ChromaIndexTool, ChromaTextSearchTool

# Email tools
from nodetool.agents.tools.email_tools import (
    AddLabelTool,
    ArchiveEmailTool,
    SearchEmailTool,
    create_gmail_connection,
    parse_email_message,
)

# Workspace tools
from .workspace_tools import ReadFileTool, WriteFileTool

# Asset tools
from .asset_tools import ListAssetsDirectoryTool, ReadAssetTool, SaveAssetTool

# Names to be exported (dunder all)
__all__ = [
    "AddLabelTool",
    "ArchiveEmailTool",
    "BrowserTool",
    "BrowserUseTool",
    "ChromaHybridSearchTool",
    "ChromaIndexTool",
    "ChromaTextSearchTool",
    "ConvertPDFToMarkdownTool",
    "DownloadFileTool",
    "ExtractPDFTablesTool",
    "ExtractPDFTextTool",
    "GoogleGroundedSearchTool",
    "GoogleImageGenerationTool",
    "GoogleImagesTool",
    "GoogleJobsTool",
    "GoogleLensTool",
    "GoogleMapsTool",
    "GoogleNewsTool",
    "GoogleSearchTool",
    "GoogleShoppingTool",
    "GoogleFinanceTool",
    "ListAssetsDirectoryTool",
    "OpenAIImageGenerationTool",
    "OpenAITextToSpeechTool",
    "OpenAIWebSearchTool",
    "MCPTool",
    "ReadAssetTool",
    "ReadFileTool",
    "SaveAssetTool",
    "ScreenshotTool",
    "SearchEmailTool",
    "Tool",
    "WriteFileTool",
    # Directly imported helpers
    "create_gmail_connection",
    "get_tool_by_name",
    "parse_email_message",
    "sanitize_node_name",
]
