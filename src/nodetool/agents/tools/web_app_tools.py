"""
Web application development and validation tools.

This module provides tools for developing and validating web applications:
- ValidateJavaScriptTool: Validate JavaScript code using ESLint
- ValidateCSSStyleTool: Validate CSS code using stylelint
- ValidateHTMLTool: Validate HTML code using html-validate
- FormatCodeTool: Format code using Prettier
- BundleWebAppTool: Bundle web application using webpack/parcel
- StartDevServerTool: Start a development server
- RunWebTestsTool: Run tests using Jest or other test runners
- InstallNpmPackageTool: Install npm packages
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


class ValidateCSSStyleTool(Tool):
    name = "validate_css"
    description = "Validate CSS code using stylelint"
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "CSS code to validate"},
            "config": {
                "type": "object",
                "description": "Stylelint configuration override (optional)",
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
                temp_file = tempfile.NamedTemporaryFile(suffix=".css", delete=False)
                file_path = temp_file.name
                temp_file.close()

            # Save the code to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Create a temporary stylelint config if provided
            if config:
                config_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                with open(config_file.name, "w", encoding="utf-8") as f:
                    json.dump(config, f)
                config_path = config_file.name
            else:
                config_path = None

            # Run stylelint
            cmd = ["npx", "stylelint", "--formatter", "json", file_path]
            if config_path:
                cmd.extend(["--config", config_path])

            process = subprocess.run(
                cmd, capture_output=True, text=True, cwd=context.workspace_dir
            )

            if not save_path:
                os.unlink(file_path)
            if config_path:
                os.unlink(config_path)

            # Parse stylelint output
            if process.returncode == 0:
                return {"success": True, "valid": True, "issues": []}
            else:
                try:
                    lint_results = json.loads(process.stdout)
                    issues = []
                    for result in lint_results:
                        for warning in result.get("warnings", []):
                            issues.append(
                                {
                                    "line": warning.get("line"),
                                    "column": warning.get("column"),
                                    "text": warning.get("text"),
                                    "rule": warning.get("rule"),
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
                        "error": "Failed to parse stylelint output",
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                    }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict) -> str:
        target = params.get("save_to_file", "CSS code")
        msg = f"Validating {target} using stylelint..."
        if len(msg) > 80:
            msg = "Validating CSS using stylelint..."
        return msg


class ValidateHTMLTool(Tool):
    name = "validate_html"
    description = "Validate HTML code using html-validate"
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "HTML code to validate"},
            "config": {
                "type": "object",
                "description": "HTML validation configuration override (optional)",
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
                temp_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
                file_path = temp_file.name
                temp_file.close()

            # Save the code to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Create a temporary html-validate config if provided
            if config:
                config_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                with open(config_file.name, "w", encoding="utf-8") as f:
                    json.dump(config, f)
                config_path = config_file.name
            else:
                config_path = None

            # Run html-validate
            cmd = ["npx", "html-validate", "--format", "json", file_path]
            if config_path:
                cmd.extend(["--config", config_path])

            process = subprocess.run(
                cmd, capture_output=True, text=True, cwd=context.workspace_dir
            )

            if not save_path:
                os.unlink(file_path)
            if config_path:
                os.unlink(config_path)

            # Parse html-validate output
            try:
                validate_results = json.loads(process.stdout)
                issues = []

                for result in validate_results.get("results", []):
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

                return {"success": True, "valid": len(issues) == 0, "issues": issues}
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Failed to parse HTML validation output",
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict) -> str:
        target = params.get("save_to_file", "HTML code")
        msg = f"Validating {target} using html-validate..."
        if len(msg) > 80:
            msg = "Validating HTML using html-validate..."
        return msg


class FormatCodeTool(Tool):
    name = "format_code"
    description = "Format code using Prettier"
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Code to format"},
            "language": {
                "type": "string",
                "description": "Language of the code (js, jsx, ts, tsx, css, html)",
                "enum": ["js", "jsx", "ts", "tsx", "css", "html", "json"],
                "default": "js",
            },
            "options": {
                "type": "object",
                "description": "Prettier formatting options (optional)",
            },
        },
        "required": ["code"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            code = params["code"]
            language = params.get("language", "js")
            options = params.get("options", {})

            # Map language to file extension
            ext_map = {
                "js": ".js",
                "jsx": ".jsx",
                "ts": ".ts",
                "tsx": ".tsx",
                "css": ".css",
                "html": ".html",
                "json": ".json",
            }
            file_ext = ext_map.get(language, ".js")

            # Create a temporary file for the code
            temp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
            file_path = temp_file.name
            temp_file.close()

            # Save the code to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Create a temporary Prettier config if options provided
            if options:
                config_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                with open(config_file.name, "w", encoding="utf-8") as f:
                    json.dump(options, f)
                config_path = config_file.name
            else:
                config_path = None

            # Run Prettier
            cmd = ["npx", "prettier", "--write", file_path]
            if config_path:
                cmd.extend(["--config", config_path])

            process = subprocess.run(
                cmd, capture_output=True, text=True, cwd=context.workspace_dir
            )

            formatted_code = ""
            if process.returncode == 0:
                with open(file_path, "r", encoding="utf-8") as f:
                    formatted_code = f.read()

            # Clean up temporary files
            os.unlink(file_path)
            if config_path:
                os.unlink(config_path)

            if process.returncode == 0:
                return {"success": True, "formatted_code": formatted_code}
            else:
                return {
                    "success": False,
                    "error": "Failed to format code",
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict) -> str:
        target = params.get("save_to_file", "code")
        msg = f"Formatting {target} using Prettier..."
        if len(msg) > 80:
            msg = "Formatting code using Prettier..."
        return msg


class InstallNpmPackageTool(Tool):
    name = "install_npm_package"
    description = "Install npm packages needed for development or testing"
    input_schema = {
        "type": "object",
        "properties": {
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of npm packages to install",
            },
            "dev": {
                "type": "boolean",
                "description": "Whether to install as dev dependencies",
                "default": False,
            },
        },
        "required": ["packages"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            packages = params["packages"]
            dev = params.get("dev", False)

            if not packages or len(packages) == 0:
                return {
                    "success": False,
                    "error": "No packages specified for installation",
                }

            cmd = ["npm", "install"]
            if dev:
                cmd.append("--save-dev")
            cmd.extend(packages)

            process = subprocess.run(
                cmd, capture_output=True, text=True, cwd=context.workspace_dir
            )

            if process.returncode == 0:
                return {
                    "success": True,
                    "installed_packages": packages,
                    "dev_dependencies": dev,
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to install packages",
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict) -> str:
        packages = params.get("packages", "npm package(s)")
        dev = params.get("dev")
        where = "development dependency" if dev else "dependency"
        packages_str = ", ".join(packages) if isinstance(packages, list) else packages
        msg = f"Installing {packages_str} as {where}..."
        if len(msg) > 80:
            msg = f"Installing {where}..."
        return msg


class InitializeReactAppTool(Tool):
    """Tool for initializing a new React application using create-react-app."""

    name = "initialize_react_app"
    description = "Initialize a new React application using npm create vite"
    input_schema = {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "Name of the React application",
                "default": "my-react-app",
            },
        },
        "required": ["app_name"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            app_name = params.get("app_name", "my-react-app")

            cmd = [
                "npm",
                "create",
                "vite@latest",
                app_name,
                "--",
                "--template",
                "react",
            ]

            process = subprocess.run(
                cmd, capture_output=True, text=True, cwd=context.workspace_dir
            )

            if process.returncode == 0:
                return {
                    "success": True,
                    "app_name": app_name,
                    "method": "vite",
                    "message": f"Successfully initialized React app '{app_name}'",
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to initialize React app",
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def user_message(self, params: dict) -> str:
        app_name = params.get("app_name", "React app")
        msg = f"Initializing {app_name} using npm create vite..."
        if len(msg) > 80:
            msg = "Initializing React app..."
        return msg
