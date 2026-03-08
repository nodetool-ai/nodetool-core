"""
UI Tool Proxy — forwards tool calls to the frontend via tool bridge.
"""

import asyncio
import uuid

from nodetool.agents.tools import Tool
from nodetool.workflows.processing_context import ProcessingContext


class UIToolProxy(Tool):
    """Proxy tool that forwards tool calls to the frontend."""

    def __init__(self, tool_manifest: dict):
        # Configure base Tool fields expected by providers
        self.name = tool_manifest["name"]
        self.description = tool_manifest.get("description", "UI tool")
        # Providers expect JSON schema under input_schema
        self.input_schema = tool_manifest.get("parameters", {})

    async def process(self, context: ProcessingContext, params: dict) -> dict:
        """Forward tool call to frontend and wait for result."""
        if not context.tool_bridge:
            raise ValueError("Tool bridge not available")

        # Generate a unique tool call ID
        tool_call_id = str(uuid.uuid4())

        # Forward to frontend
        tool_call_message = {
            "type": "tool_call",
            "tool_call_id": tool_call_id,
            "name": self.name,
            "args": params,
            "thread_id": getattr(context, "thread_id", ""),
        }

        await context.send_message(tool_call_message)  # type: ignore

        # Wait for result with timeout
        try:
            payload = await asyncio.wait_for(context.tool_bridge.create_waiter(tool_call_id), timeout=60.0)

            if payload.get("ok"):
                return payload.get("result", {})

            error_msg = payload.get("error", "Unknown error")
            # Return a tool result shaped like other tool errors so the model can retry.
            return {"error": f"Frontend tool execution failed: {error_msg}"}

        except TimeoutError:
            return {"error": f"Frontend tool {self.name} timed out after 60 seconds"}

    def user_message(self, params: dict) -> str:
        """Generate user-friendly message for tool execution."""
        return f"Executing frontend tool: {self.name}"
