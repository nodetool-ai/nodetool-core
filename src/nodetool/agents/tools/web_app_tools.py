"""
Web application development and validation tools.

This module provides tools for developing and validating web applications:
- ValidateJavaScriptTool: Validate JavaScript code using ESLint
"""

import os
import json
import subprocess
import tempfile

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class ValidateJavaScriptTool(Tool):
    name = "validate_javascript"
    description = "Validate JavaScript code using ESLint"
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "JavaScript code to validate"},
            "config": {
                "type": "object",
                "description": "ESLint configuration override (optional)",
            },
            "save_to_file": {
                "type": "string",
                "description": "Path to save the code before validation (optional)",
            },
        },
        "required": ["code"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            code = params["code"]
            config = params.get("config", {})
            save_path = params.get("save_to_file")

            # Create a temporary file for the code if not saving to a specific path
            if save_path:
                file_path = context.resolve_workspace_path(save_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            else:
                temp_file = tempfile.NamedTemporaryFile(suffix=".js", delete=False)
                file_path = temp_file.name
                temp_file.close()

            # Save the code to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Create a temporary ESLint config if provided
            if config:
                config_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                with open(config_file.name, "w", encoding="utf-8") as f:
                    json.dump(config, f)
                config_path = config_file.name
            else:
                config_path = None

            # Run ESLint
            cmd = ["npx", "eslint", "--format", "json", file_path]
            if config_path:
                cmd.extend(["--config", config_path])

            process = subprocess.run(
                cmd, capture_output=True, text=True, cwd=context.workspace_dir
            )

            if not save_path:
                os.unlink(file_path)
            if config_path:
                os.unlink(config_path)

            # Parse ESLint output
            if process.returncode == 0:
                return {"success": True, "valid": True, "issues": []}
            else:
                try:
                    lint_results = json.loads(process.stdout)
                    issues = []
                    for result in lint_results:
                        for message in result.get("messages", []):
                            issues.append(
                                {
                                    "line": message.get("line"),
                                    "column": message.get("column"),
                                    "severity": message.get("severity"),
                                    "message": message.get("message"),
                                    "rule": message.get("ruleId"),
                                }
                            )

                    return {
                        "success": True,
                        "valid": len(issues) == 0,
                        "issues": issues,
                    }
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": "Failed to parse ESLint output",
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                    }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict) -> str:
        target = params.get("save_to_file", "JavaScript code")
        msg = f"Validating {target} using ESLint..."
        if len(msg) > 80:
            msg = "Validating JavaScript using ESLint..."
        return msg
