"""
System tools module.

This module provides tools for system operations:
- ExecuteShellTool: Run shell commands
- ProcessNodeTool: Process workflow nodes
- TestTool: Tool for integration testing
- FindNodeTool: Find nodes in node library
"""

import os
from datetime import datetime
import asyncio
from typing import Any

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import (
    BaseNode,
    get_node_class,
    get_registered_node_classes,
)
from .base import Tool, sanitize_node_name


class ExecuteShellTool(Tool):
    def __init__(self):
        super().__init__(
            name="execute_shell",
            description="Execute a shell command and return its output (use with caution)",
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds",
                    "default": 30,
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for command execution",
                    "default": ".",
                },
            },
            "required": ["command"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            import asyncio
            import os
            from pathlib import Path

            working_dir = Path(params.get("working_dir", ".")).absolute()
            process = await asyncio.create_subprocess_shell(
                params["command"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=params.get("timeout", 30)
                )
                return {
                    "success": process.returncode == 0,
                    "return_code": process.returncode,
                    "stdout": stdout.decode().strip(),
                    "stderr": stderr.decode().strip(),
                }
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "error": f"Command timed out after {params.get('timeout', 30)} seconds"
                }
        except Exception as e:
            return {"error": str(e)}


class ProcessNodeTool(Tool):
    node_name: str
    node_type: type[BaseNode]

    def __init__(self, node_name: str):
        super().__init__(
            name=sanitize_node_name(node_name),
            description=f"Process node {node_name}",
        )
        self.node_name = node_name
        node_type = get_node_class(self.node_name)
        if node_type is None:
            raise ValueError(f"Node {self.node_name} does not exist")
        self.node_type = node_type
        self.input_schema = self.node_type.get_json_schema()

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        node = self.node_type(id="")

        for key, value in params.items():
            node.assign_property(key, value)

        res = await node.process(context)
        out = await node.convert_output(context, res)
        return out


class TestTool(Tool):
    def __init__(self):
        super().__init__(name="test", description="A test tool for integration testing")
        self.input_schema = {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Test message to echo back",
                },
                "delay": {
                    "type": "number",
                    "description": "Optional delay in seconds",
                    "default": 0,
                },
            },
            "required": ["message"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        if params.get("delay", 0) > 0:
            import asyncio

            await asyncio.sleep(params["delay"])
        return {
            "echo": params["message"],
            "timestamp": datetime.now().isoformat(),
        }


class FindNodeTool(Tool):
    def __init__(self):
        super().__init__(
            name="find_node",
            description="Find a node in the node library",
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "node_name": {
                    "type": "string",
                    "description": "Name of the node to find",
                },
            },
        }

    async def process(self, context: ProcessingContext, params: dict) -> dict:
        node_classes = get_registered_node_classes()
        query = params["node_name"]
        for node_class in node_classes:
            if query in node_class.__name__:
                return {
                    "node_class": node_class,
                    "description": node_class.get_description(),
                    "node_type": node_class.get_node_type(),
                    "properties": node_class.properties(),
                }

        return {"error": "Node not found"}
