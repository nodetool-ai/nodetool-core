from typing import Any, ClassVar

"""
Filesystem management tools module.

This module provides tools for managing the filesystem:
- ReadFileTool: Read contents of files, with token counting and line range control
- WriteFileTool: Write content to a file, creating it if it doesn't exist
- ListDirectoryTool: List the contents of a directory
"""

import os
import re

import aiofiles
import tiktoken

from nodetool.workflows.processing_context import ProcessingContext

from .base import Tool


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file, creating it if it doesn't exist"
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write. Can be absolute or relative to current working directory.",
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

            full_path = os.path.abspath(path)

            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            mode = "a" if append else "w"
            async with aiofiles.open(full_path, mode, encoding="utf-8") as f:
                await f.write(content)

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

    def user_message(self, params: dict) -> str:
        path = params.get("path")
        append = params.get("append", False)
        action = "Appending to" if append else "Writing to"
        msg = f"{action} file {path}..."
        if len(msg) > 160:
            msg = f"{action} a file..."
        return msg


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a text file. Automatically counts tokens and supports reading specific line ranges. Cannot read binary files."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read. Can be absolute or relative to current working directory.",
            },
            "start_line": {
                "type": "integer",
                "description": "The line number to start reading from (1-based index, optional).",
            },
            "end_line": {
                "type": "integer",
                "description": "The line number to end reading at (1-based index, inclusive, optional).",
            },
        },
        "required": ["path"],
    }

    def count_tokens(self, text):
        """Count the number of tokens in a text string using cl100k_base encoding."""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    async def process(self, context: ProcessingContext, params: dict):
        try:
            path = params["path"]
            start_line = params.get("start_line")
            end_line = params.get("end_line")
            max_tokens = 25000
            max_length_per_file = 100000

            full_path = os.path.abspath(path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Path {path} does not exist",
                }

            if os.path.isfile(full_path):
                # Handle reading a single file
                try:
                    # Check if the file is binary
                    async with aiofiles.open(full_path, encoding="utf-8") as f:
                        await f.read(1024)  # Test read

                    # Read the file content
                    async with aiofiles.open(full_path, encoding="utf-8") as f:
                        if start_line is not None or end_line is not None:
                            # Read specified line range
                            lines = await f.readlines()
                            total_lines = len(lines)

                            # Adjust for 1-based indexing and validate range
                            start_idx = max(0, (start_line or 1) - 1)
                            end_idx = min(total_lines, (end_line or total_lines))

                            if start_idx >= total_lines or start_idx > end_idx:
                                return {
                                    "success": False,
                                    "is_directory": False,
                                    "path": path,
                                    "full_path": full_path,
                                    "error": f"Invalid line range: start_line={start_line}, end_line={end_line}, total lines={total_lines}",
                                    "suggested_ranges": [
                                        f"1-{min(500, total_lines)}",
                                        f"{max(1, total_lines - 500)}-{total_lines}",
                                    ],
                                }

                            lines = lines[start_idx:end_idx]
                            content = "".join(lines)
                            line_info = {
                                "start_line": start_line or 1,
                                "end_line": end_idx,
                                "total_lines": total_lines,
                            }
                        else:
                            # Read the whole file or up to max_length_per_file
                            content = await f.read(max_length_per_file)
                            line_info = {"total_lines": len(re.findall(r"\n", content)) + 1}

                    # Always count tokens
                    token_count = self.count_tokens(content)
                    token_info = {
                        "count": token_count,
                        "model": "cl100k_base",
                    }

                    # If token count exceeds max_tokens, return error with suggestions
                    if max_tokens and token_count > max_tokens:
                        total_lines = line_info.get("total_lines", 0)
                        approx_lines_per_token = total_lines / max(1, token_count)
                        suggested_line_count = int(approx_lines_per_token * max_tokens * 0.9)  # 90% to be safe

                        if start_line:
                            # Suggest reducing the end line
                            suggested_end = min(start_line + suggested_line_count - 1, total_lines)
                            suggested_ranges = [f"{start_line}-{suggested_end}"]
                        else:
                            # Suggest chunks of the file
                            chunk_size = min(suggested_line_count, 500)
                            suggested_ranges = []
                            for i in range(1, total_lines, chunk_size):
                                suggested_ranges.append(f"{i}-{min(i + chunk_size - 1, total_lines)}")
                                if len(suggested_ranges) >= 3:  # Limit to 3 suggestions
                                    break

                        return {
                            "success": False,
                            "is_directory": False,
                            "path": path,
                            "full_path": full_path,
                            "error": f"Token count ({token_count}) exceeds maximum ({max_tokens}). Please read only a portion of the file.",
                            "token_info": token_info,
                            "line_info": line_info,
                            "suggested_ranges": suggested_ranges,
                        }

                    return {
                        "success": True,
                        "is_directory": False,
                        "path": path,
                        "full_path": full_path,
                        "content": content,
                        "truncated": len(content) >= max_length_per_file,
                        "is_binary": False,
                        "line_info": line_info,
                        "token_info": token_info,
                    }
                except UnicodeDecodeError:
                    return {
                        "success": False,
                        "is_directory": False,
                        "path": path,
                        "full_path": full_path,
                        "error": f"The file {path} contains binary data that cannot be processed",
                        "is_binary": True,
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "is_directory": False,
                        "path": path,
                        "error": str(e),
                    }

            elif os.path.isdir(full_path):
                return {
                    "success": False,
                    "error": f"Path {path} is a directory, not a file",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict) -> str:
        path = params.get("path")
        msg = f"Reading content from {path}..."
        if len(msg) > 80:
            msg = "Reading content from a path..."
        return msg


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List the contents (files and subdirectories) of a directory."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory to list. Can be absolute or relative to current working directory.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to list contents recursively (optional).",
                "default": False,
            },
        },
        "required": ["path"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            path = params["path"]
            recursive = params.get("recursive", False)
            full_path = os.path.abspath(path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Path {path} does not exist",
                }

            if not os.path.isdir(full_path):
                return {
                    "success": False,
                    "error": f"Path {path} is not a directory",
                }

            results = {"files": [], "directories": []}
            errors = []

            if recursive:
                for root, dirs, files in os.walk(full_path):
                    for dir_name in dirs:
                        results["directories"].append(os.path.join(root, dir_name))
                    for file_name in files:
                        results["files"].append(os.path.join(root, file_name))
            else:
                try:
                    for item in os.listdir(full_path):
                        item_path = os.path.join(full_path, item)
                        if os.path.isdir(item_path):
                            results["directories"].append(item_path)
                        elif os.path.isfile(item_path):
                            results["files"].append(item_path)
                except Exception as e:
                    errors.append(f"Error listing contents of {path}: {str(e)}")

            return {
                "success": True,
                "path": path,
                "full_path": full_path,
                "recursive": recursive,
                "contents": results,
                "list_errors": errors if errors else None,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict) -> str:
        path = params.get("path")
        msg = f"Listing contents of directory {path}..."
        if len(msg) > 80:
            msg = "Listing directory contents..."
        return msg
