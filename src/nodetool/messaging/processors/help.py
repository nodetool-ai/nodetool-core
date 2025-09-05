"""
Help message processor module.

This module provides the processor for help mode messages.
"""

import logging
import asyncio
import json
import uuid
from typing import List
import httpx
from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from pydantic import BaseModel

from nodetool.metadata.types import (
    Message,
    ToolCall,
)
from nodetool.workflows.types import (
    Chunk,
    ToolCallUpdate,
)
from nodetool.chat.help import SYSTEM_PROMPT
from nodetool.agents.tools.help_tools import (
    SearchNodesTool,
    SearchExamplesTool,
)

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.providers.base import ChatProvider
from nodetool.agents.tools.base import Tool
from .base import MessageProcessor

log = logging.getLogger(__name__)
# Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)


class UIToolProxy(Tool):
    """Proxy tool that forwards tool calls to the frontend."""

    def __init__(self, tool_manifest: dict):
        # Configure base Tool fields expected by providers
        self.name = tool_manifest["name"]
        self.description = tool_manifest.get("description", "UI tool")
        # Providers expect JSON schema under input_schema
        self.input_schema = tool_manifest.get("parameters", {})

    async def process(self, context: ProcessingContext, args: dict) -> dict:
        """Forward tool call to frontend and wait for result."""
        if not context.tool_bridge:
            raise ValueError("Tool bridge not available")

        # Generate a unique tool call ID
        import uuid

        tool_call_id = str(uuid.uuid4())

        # Forward to frontend
        tool_call_message = {
            "type": "tool_call",
            "tool_call_id": tool_call_id,
            "name": self.name,
            "args": args,
            "thread_id": getattr(context, "thread_id", ""),
        }

        await context.send_message(tool_call_message)  # type: ignore

        # Wait for result with timeout
        try:
            payload = await asyncio.wait_for(
                context.tool_bridge.create_waiter(tool_call_id), timeout=60.0
            )

            if payload.get("ok"):
                return payload.get("result", {})
            else:
                error_msg = payload.get("error", "Unknown error")
                raise ValueError(f"Frontend tool execution failed: {error_msg}")

        except asyncio.TimeoutError:
            raise ValueError(f"Frontend tool {self.name} timed out after 60 seconds")

    def user_message(self, args: dict) -> str:
        """Generate user-friendly message for tool execution."""
        return f"Executing frontend tool: {self.name}"


class HelpMessageProcessor(MessageProcessor):
    """
    Processor for help mode messages.

    This processor handles help requests using the integrated help system
    with access to help-specific tools and documentation.
    """

    def __init__(self, provider: ChatProvider):
        super().__init__()
        self.provider = provider

    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ):
        """Process help messages with integrated help system."""
        last_message = chat_history[-1]

        try:
            if not last_message.provider:
                raise ValueError("Model provider is not set")

            log.debug(f"Processing help messages with model: {last_message.model}")

            # Create help tools combined with all available tools
            help_tools = [
                SearchNodesTool(),
                SearchExamplesTool(),
            ]
            help_tools_by_name = {t.name: t for t in help_tools}
            if last_message.tools:
                tools = await asyncio.gather(
                    *[
                        resolve_tool_by_name(name, processing_context.user_id)
                        for name in last_message.tools
                    ]
                )
            else:
                tools = []

            # Create UI proxy tools from manifest
            ui_tools = []
            if (
                hasattr(processing_context, "tool_bridge")
                and processing_context.tool_bridge
                and hasattr(processing_context, "client_tools_manifest")
                and processing_context.client_tools_manifest
            ):
                # Create proxy tools for each UI tool in the manifest
                for (
                    tool_name,
                    tool_manifest,
                ) in processing_context.client_tools_manifest.items():
                    ui_tools.append(UIToolProxy(tool_manifest))

            # Create effective messages with help system prompt
            effective_messages = [
                Message(role="system", content=SYSTEM_PROMPT)
            ] + chat_history

            accumulated_content = ""
            unprocessed_messages = []

            # Process messages with tool execution
            while True:
                messages_to_send = effective_messages + unprocessed_messages
                unprocessed_messages = []
                assert last_message.model, "Model is required"

                async for chunk in self.provider.generate_messages(
                    messages=messages_to_send,
                    model=last_message.model,
                    tools=help_tools + tools + ui_tools,
                ):  # type: ignore
                    if isinstance(chunk, Chunk):
                        accumulated_content += chunk.content
                        await self.send_message(
                            {"type": "chunk", "content": chunk.content, "done": False}
                        )
                    elif isinstance(chunk, ToolCall):
                        log.debug(f"Processing help tool call: {chunk.name}")

                        # Check if this is a UI tool
                        if (
                            hasattr(processing_context, "ui_tool_names")
                            and chunk.name in processing_context.ui_tool_names
                        ):
                            # Handle UI tool call using provider tool_call id to satisfy OpenAI API
                            tool_call_id = chunk.id
                            assert tool_call_id is not None, "Tool call id is required"
                            tool_call_message = {
                                "type": "tool_call",
                                "tool_call_id": tool_call_id,
                                "name": chunk.name,
                                "args": chunk.args,
                                "thread_id": last_message.thread_id,
                            }

                            await self.send_message(tool_call_message)

                            # Wait for result from frontend
                            try:
                                payload = await asyncio.wait_for(
                                    processing_context.tool_bridge.create_waiter(
                                        tool_call_id
                                    ),
                                    timeout=60.0,
                                )

                                if payload.get("ok"):
                                    result = payload.get("result", {})
                                else:
                                    error_msg = payload.get("error", "Unknown error")
                                    raise ValueError(
                                        f"Frontend tool execution failed: {error_msg}"
                                    )

                            except asyncio.TimeoutError:
                                raise ValueError(
                                    f"Frontend tool {chunk.name} timed out after 60 seconds"
                                )

                            # Create tool result with the same id
                            tool_result = ToolCall(
                                id=tool_call_id,
                                name=chunk.name,
                                args=chunk.args,
                                result=result,
                            )
                        else:
                            # Handle built-in help tools locally
                            if chunk.name in help_tools_by_name:
                                tool_impl = help_tools_by_name[chunk.name]
                                # Notify client about tool execution
                                await self.send_message(
                                    ToolCallUpdate(
                                        name=chunk.name,
                                        args=chunk.args,
                                        message=tool_impl.user_message(chunk.args),
                                    ).model_dump()
                                )
                                result = await tool_impl.process(
                                    processing_context, chunk.args
                                )
                                tool_result = ToolCall(
                                    id=chunk.id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    result=result,
                                )
                            else:
                                # Process regular server tool call
                                tool_result = await self._run_tool(
                                    processing_context, chunk
                                )

                        log.debug(
                            f"Help tool {chunk.name} execution complete, id={tool_result.id}"
                        )

                        # Add tool messages to unprocessed messages
                        assistant_msg = Message(
                            role="assistant",
                            tool_calls=[chunk],
                            thread_id=last_message.thread_id,
                            workflow_id=last_message.workflow_id,
                            provider=last_message.provider,
                            model=last_message.model,
                            agent_mode=last_message.agent_mode or False,
                            help_mode=True,
                        )
                        unprocessed_messages.append(assistant_msg)
                        await self.send_message(assistant_msg.model_dump())

                        # Convert result to JSON
                        converted_result = self._recursively_model_dump(
                            tool_result.result
                        )
                        tool_result_json = json.dumps(converted_result)
                        tool_msg = Message(
                            role="tool",
                            tool_call_id=tool_result.id,
                            content=tool_result_json,
                        )
                        unprocessed_messages.append(tool_msg)
                        await self.send_message(tool_msg.model_dump())

                # If no more unprocessed messages, we're done
                if not unprocessed_messages:
                    break

            # Signal the end of the help stream
            await self.send_message({"type": "chunk", "content": "", "done": True})
            await self.send_message(
                Message(
                    role="assistant",
                    content=accumulated_content if accumulated_content else None,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=last_message.agent_mode or False,
                    help_mode=True,
                ).model_dump()
            )

        except httpx.ConnectError as e:
            # Handle connection errors
            error_msg = self._format_connection_error(e)
            log.error(
                f"httpx.ConnectError in _process_help_messages: {e}", exc_info=True
            )

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "connection_error",
                }
            )

            # Signal the end of the help stream with error
            await self.send_message({"type": "chunk", "content": "", "done": True})

            # Return an error message
            await self.send_message(
                Message(
                    role="assistant",
                    content=f"I encountered a connection error while processing the help request: {error_msg}. Please check your network connection and try again.",
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=last_message.agent_mode or False,
                    help_mode=True,
                ).model_dump()
            )

        finally:
            # Always mark processing as complete
            self.is_processing = False

    async def _run_tool(
        self,
        context: ProcessingContext,
        tool_call: ToolCall,
    ) -> ToolCall:
        """Execute a tool call and return the result."""
        from nodetool.agents.tools.tool_registry import resolve_tool_by_name

        tool = await resolve_tool_by_name(tool_call.name, context.user_id)
        log.debug(
            f"Executing tool {tool_call.name} (id={tool_call.id}) with args: {tool_call.args}"
        )

        # Send tool call to client
        await self.send_message(
            ToolCallUpdate(
                name=tool_call.name,
                args=tool_call.args,
                message=tool.user_message(tool_call.args),
            ).model_dump()
        )

        result = await tool.process(context, tool_call.args)
        log.debug(f"Tool {tool_call.name} returned: {result}")

        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.args,
            result=result,
        )

    def _recursively_model_dump(self, obj):
        """Recursively convert BaseModel instances to dictionaries."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: self._recursively_model_dump(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._recursively_model_dump(item) for item in obj]
        else:
            return obj

    def _format_connection_error(self, e: httpx.ConnectError) -> str:
        """Format connection error message."""
        error_msg = str(e)
        if "nodename nor servname provided" in error_msg:
            return "Connection error: Unable to resolve hostname. Please check your network connection and API endpoint configuration."
        else:
            return f"Connection error: {error_msg}"
