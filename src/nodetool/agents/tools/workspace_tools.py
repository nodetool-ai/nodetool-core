"""
Workspace management tools module.

This module provides tools for managing an agent's workspace:
- ReadFileTool: Read contents of files in the workspace
- WriteFileTool: Write content to a file in the agent workspace, creating it if it doesn't exist
"""

import os
import subprocess
import shutil

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file in the agent workspace, creating it if it doesn't exist"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write, relative to the workspace directory",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
            "append": {
                "type": "boolean",
                "description": "Whether to append to the file instead of overwriting",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            path = params["path"]
            content = params["content"]
            append = params.get("append", False)

            full_path = context.resolve_workspace_path(path)

            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            mode = "a" if append else "w"
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(content)

            file_existed = os.path.exists(full_path)
            return {
                "success": True,
                "path": path,
                "full_path": full_path,
                "append": append,
                "created": not file_existed,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a text file in the agent workspace (cannot read binary files)"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read, relative to the workspace directory",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum number of characters to read (optional)",
                "default": 100000,
            },
        },
        "required": ["path"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            path = params["path"]
            max_length = params.get("max_length", 100000)

            full_path = context.resolve_workspace_path(path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"File {path} does not exist",
                }

            if not os.path.isfile(full_path):
                return {
                    "success": False,
                    "error": f"{path} is not a file",
                }

            # Check if the file is binary
            try:
                # Try to open and read a small part of the file as text
                with open(full_path, "r", encoding="utf-8") as f:
                    f.read(1024)  # Just read a small chunk to test

                # If we're here, the file is text, so read it normally
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read(max_length)

                return {
                    "success": True,
                    "content": content,
                    "path": path,
                    "full_path": full_path,
                    "truncated": len(content) >= max_length,
                    "is_binary": False,
                }
            except UnicodeDecodeError:
                # File is binary
                return {
                    "success": False,
                    "error": f"The file {path} contains binary data that cannot be processed",
                    "is_binary": True,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}
