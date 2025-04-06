"""
Tools package for nodetool chat functionality.

This package provides various utility tools for interacting with files,
services, APIs, and other resources. Tools are grouped by category in
separate modules but exposed here as a flat API for easy importing.

File Operations:
- ReadFileTool: Read contents of files
- WriteFileTool: Write/append content to files
- ListDirectoryTool: List directory contents
- SearchFileTool: Search files for text patterns
- ExtractPDFTextTool: Extract text from PDFs
- ExtractPDFTablesTool: Extract tables from PDFs
- ConvertPDFToMarkdownTool: Convert PDFs to markdown

Web & Browser:
- BrowserTool: Control web browser automation
- ScreenshotTool: Take browser screenshots

Search & Database:
- ChromaTextSearchTool: Semantic search in ChromaDB
- ChromaHybridSearchTool: Combined semantic/keyword search
- SemanticDocSearchTool: Search documentation semantically
- KeywordDocSearchTool: Search documentation by keywords

Email (Gmail):
- SearchEmailTool: Search Gmail messages
- ArchiveEmailTool: Archive Gmail messages
- AddLabelTool: Add labels to Gmail messages

Apple Notes:
- CreateAppleNoteTool: Create notes in Apple Notes
- ReadAppleNotesTool: Read from Apple Notes

Google Services:
- GoogleSearchTool: Search Google via SerpAPI
- GoogleFinanceTool: Retrieve financial data via Google Finance
- GoogleFlightsTool: Search flight information via Google Flights
- GoogleNewsTool: Search Google News for articles and topics

System:
- ExecuteShellTool: Run shell commands
- ProcessNodeTool: Process workflow nodes
- TestTool: Tool for integration testing
- FindNodeTool: Find nodes in node library

Workspace Management:
- ReadWorkspaceFileTool: Read contents of files in the workspace
- WriteWorkspaceFileTool: Write content to a file in the agent workspace, creating it if it doesn't exist
- UpdateWorkspaceFileTool: Update existing files in the workspace
- DeleteWorkspaceFileTool: Delete files from the workspace
- ListWorkspaceContentsTool: List contents of the workspace
- ExecuteWorkspaceCommandTool: Execute commands in the workspace

Task Management:
- ListTasksTool: List tasks from a markdown file
- AddTaskTool: Add a new task to the list
- CompleteTaskTool: Mark a task as complete
- UncompleteTaskTool: Mark a task as incomplete
- DeleteTaskTool: Delete a task from the list
- UpdateTaskTool: Update a task's text
- MoveTaskTool: Move a task to a different position or parent

Each tool inherits from the base Tool class and implements:
- input_schema: JSON schema defining the tool's parameters
- process(): Async method to execute the tool's functionality

Tools are used by AI agents to perform operations and integrate with various services.
"""

# Base tools
from nodetool.chat.tools.base import Tool, sanitize_node_name, get_tool_by_name

# System tools
from nodetool.chat.tools.system import (
    ExecuteShellTool,
    TestTool,
    FindNodeTool,
)

# Web and browser tools
from nodetool.chat.tools.browser import (
    BrowserTool,
    ScreenshotTool,
    GoogleSearchTool,
    WebFetchTool,
    DownloadFileTool,
)

# PDF tools
from nodetool.chat.tools.pdf import (
    ExtractPDFTextTool,
    ExtractPDFTablesTool,
    ConvertPDFToMarkdownTool,
)

# Search and database tools
from nodetool.chat.tools.chroma import (
    ChromaTextSearchTool,
    ChromaHybridSearchTool,
    ChromaIndexTool,
)

# Email tools
from nodetool.chat.tools.email import (
    SearchEmailTool,
    ArchiveEmailTool,
    AddLabelTool,
    create_gmail_connection,
    parse_email_message,
)

# Apple Notes tools
from nodetool.chat.tools.apple_notes import (
    CreateAppleNoteTool,
    ReadAppleNotesTool,
)

# Workspace tools
from nodetool.chat.tools.workspace import (
    WriteWorkspaceFileTool,
    ReadWorkspaceFileTool,
    UpdateWorkspaceFileTool,
    ListWorkspaceContentsTool,
    ExecuteWorkspaceCommandTool,
)

# Asset tools
from nodetool.chat.tools.assets import (
    ListAssetsDirectoryTool,
    ReadAssetTool,
    SaveAssetTool,
)

__all__ = [
    # Base
    "Tool",
    "sanitize_node_name",
    # System
    "ExecuteShellTool",
    "TestTool",
    "FindNodeTool",
    # Web and browser
    "BrowserTool",
    "ScreenshotTool",
    "GoogleSearchTool",
    "WebFetchTool",
    "DownloadFileTool",
    # PDF
    "ExtractPDFTextTool",
    "ExtractPDFTablesTool",
    "ConvertPDFToMarkdownTool",
    # Search and database
    "ChromaIndexTool",
    "ChromaTextSearchTool",
    "ChromaHybridSearchTool",
    # Email
    "SearchEmailTool",
    "ArchiveEmailTool",
    "AddLabelTool",
    "create_gmail_connection",
    "parse_email_message",
    # Apple Notes
    "CreateAppleNoteTool",
    "ReadAppleNotesTool",
    # Workspace tools
    "WriteWorkspaceFileTool",
    "ReadWorkspaceFileTool",
    "UpdateWorkspaceFileTool",
    "ListWorkspaceContentsTool",
    "ExecuteWorkspaceCommandTool",
    # Asset tools
    "ListAssetsDirectoryTool",
    "ReadAssetTool",
    "SaveAssetTool",
    # Other
    "get_tool_by_name",
]
