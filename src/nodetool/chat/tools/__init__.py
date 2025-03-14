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
- CreateWorkspaceFileTool: Create a new file in the agent workspace
- ReadWorkspaceFileTool: Read contents of files in the workspace
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
from nodetool.chat.tools.base import Tool, sanitize_node_name

# File operations tools
from nodetool.chat.tools.file_operations import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    SearchFileTool,
)

# System tools
from nodetool.chat.tools.system import (
    ExecuteShellTool,
    ProcessNodeTool,
    TestTool,
    FindNodeTool,
)

# Web and browser tools
from nodetool.chat.tools.browser import (
    BrowserTool,
    ScreenshotTool,
    GoogleSearchTool,
)

# PDF tools
from nodetool.chat.tools.pdf import (
    ExtractPDFTextTool,
    ExtractPDFTablesTool,
    ConvertPDFToMarkdownTool,
)

# Search and database tools
from nodetool.chat.tools.search import (
    ChromaTextSearchTool,
    ChromaHybridSearchTool,
    SemanticDocSearchTool,
    KeywordDocSearchTool,
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
    WorkspaceBaseTool,
    CreateWorkspaceFileTool,
    ReadWorkspaceFileTool,
    UpdateWorkspaceFileTool,
    DeleteWorkspaceFileTool,
    ListWorkspaceContentsTool,
    ExecuteWorkspaceCommandTool,
)

# Task management tools
from nodetool.chat.tools.task_management import (
    TaskBaseTool,
    AddTaskTool,
    FinishTaskTool,
    TaskList,
)


__all__ = [
    # Base
    "Tool",
    "sanitize_node_name",
    # File operations
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    "SearchFileTool",
    # System
    "ExecuteShellTool",
    "ProcessNodeTool",
    "TestTool",
    "FindNodeTool",
    # Web and browser
    "BrowserTool",
    "ScreenshotTool",
    "GoogleSearchTool",
    # PDF
    "ExtractPDFTextTool",
    "ExtractPDFTablesTool",
    "ConvertPDFToMarkdownTool",
    # Search and database
    "ChromaTextSearchTool",
    "ChromaHybridSearchTool",
    "SemanticDocSearchTool",
    "KeywordDocSearchTool",
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
    "WorkspaceBaseTool",
    "CreateWorkspaceFileTool",
    "ReadWorkspaceFileTool",
    "UpdateWorkspaceFileTool",
    "DeleteWorkspaceFileTool",
    "ListWorkspaceContentsTool",
    "ExecuteWorkspaceCommandTool",
    # Task management tools
    "TaskBaseTool",
    "AddTaskTool",
    "FinishTaskTool",
    "TaskList",
]
