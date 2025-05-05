"""
Workspace management tools module.

This module provides tools for managing an agent's workspace:
- ReadFileTool: Read contents of files in the workspace
- WriteFileTool: Write content to a file in the agent workspace, creating it if it doesn't exist
"""

import os

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

    def user_message(self, params: dict) -> str:
        path = params.get("path")
        append = params.get("append", False)
        action = "Appending to" if append else "Writing to"
        msg = f"{action} file {path}..."
        if len(msg) > 80:
            msg = f"{action} a file..."
        return msg


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a text file or all text files in a folder within the agent workspace. Cannot read binary files."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or folder to read, relative to the workspace directory",
            },
            "max_length_per_file": {
                "type": "integer",
                "description": "Maximum number of characters to read per file (optional). For folders, this applies to each file.",
                "default": 100000,
            },
            "recursive": {
                "type": "boolean",
                "description": "If path is a folder, whether to read files in subdirectories recursively (optional).",
                "default": False,
            },
        },
        "required": ["path"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            path = params["path"]
            max_length_per_file = params.get("max_length_per_file", 100000)
            recursive = params.get("recursive", False)

            full_path = context.resolve_workspace_path(path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Path {path} does not exist",
                }

            if os.path.isfile(full_path):
                # Handle reading a single file
                try:
                    # Check if the file is binary
                    with open(full_path, "r", encoding="utf-8") as f:
                        f.read(1024)  # Test read

                    # Read the actual content
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read(max_length_per_file)

                    return {
                        "success": True,
                        "is_directory": False,
                        "path": path,
                        "full_path": full_path,
                        "content": content,
                        "truncated": len(content) >= max_length_per_file,
                        "is_binary": False,
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
                # Handle reading a directory
                results = {}
                errors = []
                for root, dirs, files in os.walk(full_path):
                    if not recursive and root != full_path:
                        dirs[:] = []  # Prune subdirectories
                        continue

                    for filename in files:
                        file_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(
                            file_path, context.workspace_dir
                        )
                        try:
                            # Check if text
                            with open(file_path, "r", encoding="utf-8") as f:
                                f.read(1024)  # Test read
                            # Read content
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read(max_length_per_file)
                            results[relative_path] = {
                                "content": content,
                                "truncated": len(content) >= max_length_per_file,
                                "is_binary": False,
                            }
                        except UnicodeDecodeError:
                            results[relative_path] = {
                                "error": f"File {relative_path} contains binary data",
                                "is_binary": True,
                            }
                            errors.append(f"Skipped binary file: {relative_path}")
                        except Exception as e:
                            error_msg = f"Error reading file {relative_path}: {str(e)}"
                            results[relative_path] = {
                                "error": error_msg,
                                "is_binary": False,
                            }
                            errors.append(error_msg)

                return {
                    "success": True,
                    "is_directory": True,
                    "path": path,
                    "full_path": full_path,
                    "recursive": recursive,
                    "files_data": results,
                    "read_errors": errors if errors else None,
                }
            else:
                # Path exists but is neither a file nor a directory (e.g., socket, fifo)
                return {
                    "success": False,
                    "error": f"Path {path} is not a file or directory",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict) -> str:
        path = params.get("path")
        msg = f"Reading content from {path}..."
        if len(msg) > 80:
            msg = f"Reading content from a path..."
        return msg


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List the contents (files and subdirectories) of a directory within the agent workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory to list, relative to the workspace directory",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to list contents recursively (optional).",
                "default": False,
            },
            # Consider adding include_details: size, modified_time, type later if needed
        },
        "required": ["path"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            path = params["path"]
            recursive = params.get("recursive", False)
            full_path = context.resolve_workspace_path(path)

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
                    rel_root = os.path.relpath(root, context.workspace_dir)
                    # Add directories found in this level relative to the input path
                    for dir_name in dirs:
                        dir_path = os.path.join(rel_root, dir_name)
                        # Ensure we don't list the root path itself if path was "."
                        if dir_path != ".":
                            results["directories"].append(dir_path)
                    # Add files found in this level relative to the input path
                    for file_name in files:
                        file_path = os.path.join(rel_root, file_name)
                        results["files"].append(file_path)
            else:
                try:
                    for item in os.listdir(full_path):
                        item_path = os.path.join(full_path, item)
                        rel_item_path = os.path.join(path, item)
                        if os.path.isdir(item_path):
                            results["directories"].append(rel_item_path)
                        elif os.path.isfile(item_path):
                            results["files"].append(rel_item_path)
                        # Ignore other types (links, etc.) for now
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
            msg = f"Listing directory contents..."
        return msg
