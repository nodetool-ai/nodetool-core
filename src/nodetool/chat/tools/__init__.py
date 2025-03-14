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

System:
- ExecuteShellTool: Run shell commands
- ProcessNodeTool: Process workflow nodes
- TestTool: Tool for integration testing
- FindNodeTool: Find nodes in node library

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
]
