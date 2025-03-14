"""
File operation tools module.

This module provides tools for file system operations:
- ReadFileTool: Read contents of files
- WriteFileTool: Write/append content to files
- ListDirectoryTool: List directory contents
- SearchFileTool: Search files for text patterns
"""

import os
import re
from pathlib import Path
from typing import Any, List

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class ReadFileTool(Tool):
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read the contents of a file at the specified path",
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum number of characters to read (optional)",
                    "default": 100000,
                },
            },
            "required": ["path"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            path = os.path.expanduser(params["path"])  # Resolve ~ to home directory
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(params.get("max_length"))
                return {
                    "content": content,
                    "truncated": len(content) >= params.get("max_length", 100000),
                }
        except Exception as e:
            return {"error": str(e)}


class WriteFileTool(Tool):
    def __init__(self):
        super().__init__(
            name="write_file",
            description="Write content to a file at the specified path",
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path where the file should be written",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
                "mode": {
                    "type": "string",
                    "description": "Write mode: 'w' for overwrite, 'a' for append",
                    "enum": ["w", "a"],
                    "default": "w",
                },
            },
            "required": ["path", "content"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            path = os.path.expanduser(params["path"])  # Resolve ~ to home directory
            with open(path, params.get("mode", "w"), encoding="utf-8") as f:
                f.write(params["content"])
            return {"success": True, "path": path}
        except Exception as e:
            return {"error": str(e)}


class ListDirectoryTool(Tool):
    def __init__(self):
        super().__init__(
            name="list_directory",
            description="List files and directories at the specified path",
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list",
                },
            },
            "required": ["path"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            from pathlib import Path

            path = Path(
                os.path.expanduser(params["path"])
            )  # Resolve ~ to home directory

            if not path.is_dir():
                return {"error": f"'{params['path']}' is not a directory"}

            files = list(path.iterdir())

            return {
                "files": [str(Path(f).relative_to(path)) for f in files],
                "count": len(files),
            }
        except Exception as e:
            return {"error": str(e)}


class SearchFileTool(Tool):
    def __init__(self):
        super().__init__(
            name="search_file",
            description="Search for text patterns in files using grep-like functionality",
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to search in (file or directory)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories",
                    "default": False,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether to perform case-sensitive search",
                    "default": False,
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file extensions to search (e.g., ['.txt', '.py']). Empty means all files.",
                    "default": [
                        ".txt",
                        ".py",
                        ".md",
                        ".json",
                        ".yaml",
                        ".yml",
                        ".ini",
                        ".cfg",
                    ],
                },
            },
            "required": ["path", "pattern"],
        }

    def is_binary(self, file_path):
        """Check if a file is binary by reading its first few bytes"""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                return b"\0" in chunk  # Binary files typically contain null bytes
        except Exception:
            return True

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            import re
            from pathlib import Path

            path = Path(os.path.expanduser(params["path"]))
            pattern = params["pattern"]
            if not params.get("case_sensitive", False):
                pattern = re.compile(pattern, re.IGNORECASE)
            else:
                pattern = re.compile(pattern)

            allowed_extensions = params.get(
                "file_extensions",
                [".txt", ".py", ".md", ".json", ".yaml", ".yml", ".ini", ".cfg"],
            )
            results = []

            def search_file(file_path):
                matches = []
                # Skip binary files and check file extension if specified
                if (
                    allowed_extensions and file_path.suffix not in allowed_extensions
                ) or self.is_binary(file_path):
                    return matches

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f, 1):
                            if pattern.search(line):
                                matches.append(
                                    {
                                        "line_number": i,
                                        "line": line.strip(),
                                        "file": str(file_path),
                                    }
                                )
                except UnicodeDecodeError:
                    # Skip files that can't be decoded as UTF-8
                    pass
                except Exception as e:
                    matches.append(
                        {
                            "file": str(file_path),
                            "error": f"Error reading file: {str(e)}",
                        }
                    )
                return matches

            if path.is_file():
                results.extend(search_file(path))
            elif path.is_dir():
                if params.get("recursive", False):
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            results.extend(search_file(file_path))
                else:
                    for file_path in path.glob("*"):
                        if file_path.is_file():
                            results.extend(search_file(file_path))

            return {
                "matches": results,
                "count": len(results),
                "searched_path": str(path),
            }
        except Exception as e:
            return {"error": str(e)}
