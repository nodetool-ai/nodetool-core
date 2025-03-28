"""
Workspace management tools module.

This module provides tools for managing an agent's workspace:
- ReadWorkspaceFileTool: Read contents of files in the workspace
- UpdateWorkspaceFileTool: Update existing files in the workspace
- DeleteWorkspaceFileTool: Delete files from the workspace
- ListWorkspaceContentsTool: List contents of the workspace
- ExecuteWorkspaceCommandTool: Execute commands in the workspace
"""

import os
import subprocess
import shutil

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class WriteWorkspaceFileTool(Tool):
    name = "write_workspace_file"
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

            full_path = self.resolve_workspace_path(path)

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


class ReadWorkspaceFileTool(Tool):
    name = "read_workspace_file"
    description = "Read the contents of a file in the agent workspace"

    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir)
        files = self.list_workspace_contents(self.workspace_dir)
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read, relative to the workspace directory",
                    "enum": files,
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum number of characters to read (optional)",
                    "default": 100000,
                },
            },
            "required": ["path"],
        }

    def list_workspace_contents(self, workspace_dir: str):
        files = []
        for root, dirs, filenames in os.walk(workspace_dir):
            # Exclude the traces directory
            if "traces" in dirs:
                dirs.remove("traces")

            # Add all files with their relative paths
            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, self.workspace_dir)
                files.append(rel_path)

        return files

    async def process(self, context: ProcessingContext, params: dict):
        try:
            path = params["path"]
            max_length = params.get("max_length", 100000)

            full_path = self.resolve_workspace_path(path)

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

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read(max_length)

            return {
                "success": True,
                "content": content,
                "path": path,
                "full_path": full_path,
                "truncated": len(content) >= max_length,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


class UpdateWorkspaceFileTool(Tool):
    name = "update_workspace_file"
    description = "Update an existing file in the agent workspace"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to update, relative to the workspace directory",
            },
            "content": {
                "type": "string",
                "description": "New content to write to the file",
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

            full_path = self.resolve_workspace_path(path)

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

            mode = "a" if append else "w"
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(content)

            return {
                "success": True,
                "path": path,
                "full_path": full_path,
                "append": append,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


class DeleteWorkspaceFileTool(Tool):
    name = "delete_workspace_file"
    description = "Delete a file or directory from the agent workspace"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or directory to delete, relative to the workspace directory",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to recursively delete directories",
                "default": False,
            },
        },
        "required": ["path"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            path = params["path"]
            recursive = params.get("recursive", False)

            full_path = self.resolve_workspace_path(path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Path {path} does not exist",
                }

            if os.path.isfile(full_path):
                os.remove(full_path)
                return {
                    "success": True,
                    "path": path,
                    "full_path": full_path,
                    "type": "file",
                }
            elif os.path.isdir(full_path):
                if recursive:
                    shutil.rmtree(full_path)
                    return {
                        "success": True,
                        "path": path,
                        "full_path": full_path,
                        "type": "directory",
                        "recursive": True,
                    }
                else:
                    try:
                        os.rmdir(full_path)
                        return {
                            "success": True,
                            "path": path,
                            "full_path": full_path,
                            "type": "directory",
                            "recursive": False,
                        }
                    except OSError as e:
                        return {
                            "success": False,
                            "error": f"Directory {path} is not empty. Set recursive=True to remove non-empty directories.",
                        }

        except Exception as e:
            return {"success": False, "error": str(e)}


class ListWorkspaceContentsTool(Tool):
    name = "list_workspace_contents"
    description = "List contents of the agent workspace directory"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory to list, relative to the workspace directory. Default is the root workspace.",
                "default": ".",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to recursively list subdirectories",
                "default": False,
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Whether to include hidden files (starting with .)",
                "default": False,
            },
        },
        "required": [],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            path = params.get("path", ".")
            recursive = params.get("recursive", False)
            include_hidden = params.get("include_hidden", False)

            full_path = self.resolve_workspace_path(path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Path {path} does not exist",
                }

            if not os.path.isdir(full_path):
                return {
                    "success": False,
                    "error": f"{path} is not a directory",
                }

            result = {
                "success": True,
                "path": path,
                "full_path": full_path,
                "contents": [],
            }

            if recursive:
                for root, dirs, files in os.walk(full_path):
                    rel_root = os.path.relpath(root, self.workspace_dir)

                    if not include_hidden and any(
                        part.startswith(".") for part in rel_root.split(os.sep) if part
                    ):
                        continue

                    for file in files:
                        if not include_hidden and file.startswith("."):
                            continue
                        rel_path = os.path.join(rel_root, file)
                        file_stat = os.stat(os.path.join(root, file))
                        result["contents"].append(
                            {
                                "path": rel_path,
                                "type": "file",
                                "size": file_stat.st_size,
                                "modified": file_stat.st_mtime,
                            }
                        )

                    for dir_name in dirs:
                        if not include_hidden and dir_name.startswith("."):
                            continue
                        rel_path = os.path.join(rel_root, dir_name)
                        dir_stat = os.stat(os.path.join(root, dir_name))
                        result["contents"].append(
                            {
                                "path": rel_path,
                                "type": "directory",
                                "modified": dir_stat.st_mtime,
                            }
                        )
            else:
                # Non-recursive listing
                for entry in os.scandir(full_path):
                    if not include_hidden and entry.name.startswith("."):
                        continue

                    rel_path = os.path.relpath(entry.path, self.workspace_dir)
                    entry_info = {
                        "path": rel_path,
                        "type": "directory" if entry.is_dir() else "file",
                        "modified": entry.stat().st_mtime,
                    }

                    if entry.is_file():
                        entry_info["size"] = entry.stat().st_size

                    result["contents"].append(entry_info)

            # Sort by path for consistency
            result["contents"].sort(key=lambda x: x["path"])

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}


class ExecuteWorkspaceCommandTool(Tool):
    name = "execute_workspace_command"
    description = "Execute a shell command in the agent workspace directory"
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds",
                "default": 60,
            },
        },
        "required": ["command"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            command = params["command"]
            timeout = params.get("timeout", 60)

            try:
                # Execute the command in the workspace directory
                process = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.workspace_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                return {
                    "success": True,
                    "exit_code": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}
