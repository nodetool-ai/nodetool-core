"""
Development tools module.

This module provides tools for software development activities:
- RunNodeJSTool: Execute Node.js scripts
- RunNpmCommandTool: Run npm commands (install, test, etc.)
- RunEslintTool: Validate code with ESLint and get structured results
- DebugJavaScriptTool: Debug JavaScript code
- RunJestTestTool: Run Jest unit tests
"""

import os
import json
import subprocess
from typing import Any, Dict, List

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool
from .workspace import WorkspaceBaseTool


class RunNodeJSTool(WorkspaceBaseTool):
    def __init__(self, workspace_dir: str):
        super().__init__(
            name="run_nodejs",
            description="Execute a Node.js script in the workspace",
            workspace_dir=workspace_dir,
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "script_path": {
                    "type": "string",
                    "description": "Path to the JavaScript file to run, relative to the workspace directory",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command line arguments to pass to the script",
                    "default": [],
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds",
                    "default": 60,
                },
            },
            "required": ["script_path"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            script_path = params["script_path"]
            args = params.get("args", [])
            timeout = params.get("timeout", 60)

            full_path = self.resolve_workspace_path(script_path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Script file {script_path} does not exist",
                }

            cmd = ["node", full_path] + args

            try:
                process = subprocess.run(
                    cmd,
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
                    "error": f"Script execution timed out after {timeout} seconds",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}


class RunNpmCommandTool(WorkspaceBaseTool):
    def __init__(self, workspace_dir: str):
        super().__init__(
            name="run_npm_command",
            description="Run an npm command in the workspace",
            workspace_dir=workspace_dir,
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The npm command to run (e.g., 'install', 'test', 'run build')",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional arguments for the npm command",
                    "default": [],
                },
                "path": {
                    "type": "string",
                    "description": "Path to run the command in, relative to the workspace directory",
                    "default": ".",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds",
                    "default": 300,
                },
            },
            "required": ["command"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            command = params["command"]
            args = params.get("args", [])
            path = params.get("path", ".")
            timeout = params.get("timeout", 300)

            full_path = self.resolve_workspace_path(path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Path {path} does not exist",
                }

            cmd = ["npm", command] + args

            try:
                process = subprocess.run(
                    cmd,
                    cwd=full_path,
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
                    "error": f"npm command timed out after {timeout} seconds",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}


class RunEslintTool(WorkspaceBaseTool):
    def __init__(self, workspace_dir: str):
        super().__init__(
            name="run_eslint",
            description="Validate JavaScript/TypeScript code with ESLint and get structured results",
            workspace_dir=workspace_dir,
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory to lint, relative to the workspace directory",
                },
                "fix": {
                    "type": "boolean",
                    "description": "Whether to automatically fix problems when possible",
                    "default": False,
                },
                "format": {
                    "type": "string",
                    "description": "Output format (json, stylish, etc.)",
                    "default": "json",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds",
                    "default": 60,
                },
            },
            "required": ["path"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            path = params["path"]
            fix = params.get("fix", False)
            format_type = params.get("format", "json")
            timeout = params.get("timeout", 60)

            full_path = self.resolve_workspace_path(path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Path {path} does not exist",
                }

            # Build the eslint command
            cmd = ["npx", "eslint", full_path, f"--format={format_type}"]
            if fix:
                cmd.append("--fix")

            try:
                process = subprocess.run(
                    cmd,
                    cwd=self.workspace_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                result = {
                    "success": True,
                    "exit_code": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }

                # If format is json and we have output, parse it
                if format_type == "json" and process.stdout.strip():
                    try:
                        parsed_results = json.loads(process.stdout)
                        result["lint_results"] = parsed_results
                    except json.JSONDecodeError:
                        result["lint_results"] = None
                        result["parse_error"] = "Failed to parse ESLint JSON output"

                return result

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"ESLint execution timed out after {timeout} seconds",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}


class DebugJavaScriptTool(WorkspaceBaseTool):
    def __init__(self, workspace_dir: str):
        super().__init__(
            name="debug_javascript",
            description="Debug a JavaScript file using Node.js inspector protocol",
            workspace_dir=workspace_dir,
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "script_path": {
                    "type": "string",
                    "description": "Path to the JavaScript file to debug, relative to the workspace directory",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command line arguments to pass to the script",
                    "default": [],
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds",
                    "default": 60,
                },
                "inspect_options": {
                    "type": "string",
                    "description": "Node.js inspect options",
                    "default": "--inspect-brk",
                },
            },
            "required": ["script_path"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            script_path = params["script_path"]
            args = params.get("args", [])
            timeout = params.get("timeout", 60)
            inspect_options = params.get("inspect_options", "--inspect-brk")

            full_path = self.resolve_workspace_path(script_path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Script file {script_path} does not exist",
                }

            cmd = ["node", inspect_options, full_path] + args

            try:
                process = subprocess.run(
                    cmd,
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
                    "debug_info": "To connect to the debugger, open Chrome DevTools and look for the Node.js icon or visit chrome://inspect",
                }

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Debug session timed out after {timeout} seconds",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}


class RunJestTestTool(WorkspaceBaseTool):
    def __init__(self, workspace_dir: str):
        super().__init__(
            name="run_jest_tests",
            description="Run Jest unit tests in the workspace",
            workspace_dir=workspace_dir,
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "test_path": {
                    "type": "string",
                    "description": "Path to the test file or directory, relative to the workspace directory. If empty, runs all tests.",
                    "default": "",
                },
                "test_name_pattern": {
                    "type": "string",
                    "description": "Run only tests with a name that matches the regex pattern",
                    "default": "",
                },
                "json_output": {
                    "type": "boolean",
                    "description": "Output test results in JSON format",
                    "default": True,
                },
                "coverage": {
                    "type": "boolean",
                    "description": "Collect test coverage",
                    "default": False,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds",
                    "default": 120,
                },
            },
            "required": [],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            test_path = params.get("test_path", "")
            test_name_pattern = params.get("test_name_pattern", "")
            json_output = params.get("json_output", True)
            coverage = params.get("coverage", False)
            timeout = params.get("timeout", 120)

            # Build the Jest command
            cmd = ["npx", "jest"]

            if test_path:
                full_path = self.resolve_workspace_path(test_path)
                if not os.path.exists(full_path):
                    return {
                        "success": False,
                        "error": f"Test path {test_path} does not exist",
                    }
                cmd.append(full_path)

            if test_name_pattern:
                cmd.extend(["-t", test_name_pattern])

            if json_output:
                cmd.append("--json")

            if coverage:
                cmd.append("--coverage")

            try:
                process = subprocess.run(
                    cmd,
                    cwd=self.workspace_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                result = {
                    "success": True,
                    "exit_code": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }

                # If JSON output was requested and we have output, parse it
                if json_output and process.stdout.strip():
                    try:
                        parsed_results = json.loads(process.stdout)
                        result["test_results"] = parsed_results
                    except json.JSONDecodeError:
                        result["test_results"] = None
                        result["parse_error"] = "Failed to parse Jest JSON output"

                return result

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Jest execution timed out after {timeout} seconds",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}


class ValidateJavaScriptTool(WorkspaceBaseTool):
    def __init__(self, workspace_dir: str):
        super().__init__(
            name="validate_javascript",
            description="Validate JavaScript syntax without executing the code",
            workspace_dir=workspace_dir,
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the JavaScript file to validate, relative to the workspace directory",
                },
                "check_type": {
                    "type": "string",
                    "description": "Type of check to perform: 'syntax' or 'parse'",
                    "default": "syntax",
                    "enum": ["syntax", "parse"],
                },
            },
            "required": ["file_path"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            file_path = params["file_path"]
            check_type = params.get("check_type", "syntax")

            full_path = self.resolve_workspace_path(file_path)

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"File {file_path} does not exist",
                }

            # Validate syntax without executing code
            if check_type == "syntax":
                cmd = ["node", "--check", full_path]
            else:  # parse
                cmd = ["node", "-e", f"require('{full_path}')"]

            process = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
            )

            is_valid = process.returncode == 0

            return {
                "success": True,
                "is_valid": is_valid,
                "exit_code": process.returncode,
                "errors": process.stderr if not is_valid else "",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
