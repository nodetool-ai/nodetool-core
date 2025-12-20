"""
Claude Agent SDK message processor module.

This module provides the processor for agent mode messages using
Anthropic's native tool_runner (Claude Agent SDK) instead of the
custom nodetool agent implementation.
"""

import asyncio
import logging
from typing import Any, List
from uuid import uuid4

import anthropic
from anthropic import AsyncAnthropic
from anthropic.lib.tools import BetaAsyncFunctionTool
from anthropic.types.beta import BetaMessageParam

from nodetool.agents.tools.base import Tool
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    Provider,
)
from nodetool.providers.anthropic_provider import AnthropicProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

from .message_processor import MessageProcessor

log = get_logger(__name__)
log.setLevel(logging.DEBUG)


def _create_anthropic_tool_wrapper(tool: Tool, context: ProcessingContext) -> BetaAsyncFunctionTool:
    """
    Create an Anthropic BetaAsyncFunctionTool wrapper for a nodetool Tool.

    This wraps a nodetool Tool's process method as an async function
    that Anthropic's tool_runner can call.

    Args:
        tool: The nodetool Tool to wrap
        context: The ProcessingContext to pass to the tool

    Returns:
        BetaAsyncFunctionTool: An Anthropic-compatible async function tool
    """

    async def tool_wrapper(**kwargs: Any) -> Any:
        """Wrapper function that calls the nodetool Tool's process method."""
        log.debug(f"Tool wrapper called for {tool.name} with args: {kwargs}")
        result = await tool.process(context, kwargs)
        log.debug(f"Tool {tool.name} returned: {result}")
        # Convert result to string if necessary for Anthropic
        if isinstance(result, (dict, list)):
            import json

            return json.dumps(result)
        return str(result)

    # Create the BetaAsyncFunctionTool with the tool's schema
    return BetaAsyncFunctionTool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        func=tool_wrapper,
    )


def _convert_message_to_anthropic(message: Message) -> BetaMessageParam | None:
    """
    Convert a nodetool Message to Anthropic's BetaMessageParam format.

    Args:
        message: The nodetool Message to convert

    Returns:
        BetaMessageParam or None if the message should be skipped
    """
    if message.role == "system":
        # System messages are handled separately in Anthropic
        return None
    elif message.role == "user":
        if isinstance(message.content, str):
            return {"role": "user", "content": message.content}
        elif isinstance(message.content, list):
            content_blocks = []
            for part in message.content:
                if isinstance(part, MessageTextContent):
                    content_blocks.append({"type": "text", "text": part.text})
            return {"role": "user", "content": content_blocks}
    elif message.role == "assistant":
        if message.tool_calls:
            return {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.args,
                    }
                    for tc in message.tool_calls
                ],
            }
        elif isinstance(message.content, str) and message.content:
            return {"role": "assistant", "content": message.content}
    elif message.role == "tool":
        if message.tool_call_id:
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": str(message.content),
                    }
                ],
            }
    return None


def _extract_system_message(messages: List[Message]) -> str:
    """Extract system message from chat history."""
    for message in messages:
        if message.role == "system":
            if isinstance(message.content, str):
                return message.content
            elif isinstance(message.content, list):
                text_parts = []
                for part in message.content:
                    if isinstance(part, MessageTextContent):
                        text_parts.append(part.text)
                return " ".join(text_parts)
    return "You are a helpful assistant."


class ClaudeAgentMessageProcessor(MessageProcessor):
    """
    Processor for agent mode messages using Claude Agent SDK.

    This processor uses Anthropic's native tool_runner (beta) to handle
    complex tasks with automatic tool calling, instead of the custom
    nodetool Agent implementation.

    Benefits of using Claude Agent SDK:
    - Simpler implementation with less custom code
    - Native handling of tool calls by Anthropic
    - Automatic conversation management
    - Built-in retry logic and error handling
    """

    def __init__(self, provider: AnthropicProvider):
        super().__init__()
        if not isinstance(provider, AnthropicProvider):
            raise ValueError("ClaudeAgentMessageProcessor requires an AnthropicProvider")
        self.provider = provider

    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ):
        """Process messages using the Claude Agent SDK's tool_runner."""
        last_message = chat_history[-1]
        assert last_message.model, "Model is required for agent mode"
        assert last_message.provider == Provider.Anthropic, "Claude Agent SDK requires Anthropic provider"

        # Generate a unique execution ID for this agent session
        agent_execution_id = str(uuid4())

        # Resolve tools from message.tools
        selected_tools: list[BetaAsyncFunctionTool] = []
        if last_message.tools:
            from nodetool.agents.tools.tool_registry import resolve_tool_by_name

            nodetool_tools = await asyncio.gather(
                *[resolve_tool_by_name(name, processing_context.user_id) for name in last_message.tools]
            )
            # Wrap each tool for Anthropic
            for tool in nodetool_tools:
                if tool:
                    wrapped_tool = _create_anthropic_tool_wrapper(tool, processing_context)
                    selected_tools.append(wrapped_tool)
            log.debug(f"Selected tools for Claude agent: {[t.name for t in selected_tools]}")

        # Extract system message
        system_message = _extract_system_message(chat_history)

        # Convert messages to Anthropic format
        anthropic_messages: list[BetaMessageParam] = []
        for msg in chat_history:
            if msg.role != "system" and msg.role != "agent_execution":
                converted = _convert_message_to_anthropic(msg)
                if converted:
                    anthropic_messages.append(converted)

        try:
            # Get the Anthropic client from the provider
            client: AsyncAnthropic = self.provider.client

            log.info(f"Starting Claude Agent SDK execution with model {last_message.model}")

            # Send planning update
            await self.send_message(
                {
                    "type": "message",
                    "role": "agent_execution",
                    "execution_event_type": "planning_update",
                    "agent_execution_id": agent_execution_id,
                    "content": {
                        "type": "planning_update",
                        "phase": "initialization",
                        "status": "Running",
                        "content": "Starting Claude Agent SDK execution",
                    },
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                    "provider": last_message.provider,
                    "model": last_message.model,
                    "agent_mode": True,
                }
            )

            # Create the tool runner
            runner = client.beta.messages.tool_runner(
                model=last_message.model,
                messages=anthropic_messages,
                tools=selected_tools,
                system=system_message,
                max_tokens=8192,
                stream=True,  # Enable streaming
            )

            # Collect the full response content
            full_content = ""

            # Iterate through the tool runner's events
            async for item in runner:
                log.debug(f"Claude Agent SDK yielded item: {type(item)}")

                # Handle different event types from the tool runner
                if hasattr(item, "type"):
                    item_type = item.type

                    if item_type == "content_block_delta":
                        # Text delta during streaming
                        delta = getattr(item, "delta", None)
                        if delta:
                            text = getattr(delta, "text", None)
                            if text:
                                full_content += text
                                await self.send_message(
                                    {
                                        "type": "chunk",
                                        "content": text,
                                        "done": False,
                                    }
                                )

                    elif item_type == "content_block_stop":
                        # A content block finished
                        content_block = getattr(item, "content_block", None)
                        if content_block:
                            block_type = getattr(content_block, "type", "")
                            if block_type == "tool_use":
                                # Tool call completed
                                tool_id = getattr(content_block, "id", "")
                                tool_name = getattr(content_block, "name", "")
                                tool_input = getattr(content_block, "input", {})

                                await self.send_message(
                                    {
                                        "type": "tool_call_update",
                                        "tool_call_id": tool_id,
                                        "name": tool_name,
                                        "message": f"Calling {tool_name}...",
                                        "args": tool_input,
                                        "agent_execution_id": agent_execution_id,
                                    }
                                )

                    elif item_type == "message_stop":
                        # Message completed
                        log.debug("Claude Agent SDK message completed")

                    elif item_type == "tool_result":
                        # Tool result received
                        log.debug(f"Tool result received: {item}")

            # Get the final message from the runner
            final_message = await runner.until_done()
            log.debug(f"Claude Agent SDK final message: {final_message}")

            # Extract final content
            content = ""
            if hasattr(final_message, "content"):
                for block in final_message.content:
                    if hasattr(block, "text"):
                        content += block.text
            if not content:
                content = full_content

            # Send completion planning update
            await self.send_message(
                {
                    "type": "message",
                    "role": "agent_execution",
                    "execution_event_type": "planning_update",
                    "agent_execution_id": agent_execution_id,
                    "content": {
                        "type": "planning_update",
                        "phase": "completion",
                        "status": "Success",
                        "content": "Claude Agent SDK execution completed",
                    },
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                    "provider": last_message.provider,
                    "model": last_message.model,
                    "agent_mode": True,
                }
            )

            # Send the final assistant message
            await self.send_message(
                Message(
                    role="assistant",
                    content=content,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=True,
                ).model_dump()
            )

            # Signal completion
            await self.send_message({"type": "chunk", "content": "", "done": True})

        except anthropic.APIError as e:
            log.error(f"Anthropic API error in Claude Agent SDK: {e}", exc_info=True)
            error_msg = f"Claude Agent SDK error: {str(e)}"
            await self.send_message({"type": "error", "message": error_msg, "error_type": "anthropic_error"})
            await self.send_message({"type": "chunk", "content": "", "done": True})
            await self.send_message(
                Message(
                    role="assistant",
                    content=error_msg,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=True,
                ).model_dump()
            )

        except Exception as e:
            log.error(f"Error in Claude Agent SDK execution: {e}", exc_info=True)
            error_msg = f"Agent execution error: {str(e)}"
            await self.send_message({"type": "error", "message": error_msg, "error_type": "agent_error"})
            await self.send_message({"type": "chunk", "content": "", "done": True})
            await self.send_message(
                Message(
                    role="assistant",
                    content=error_msg,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=True,
                ).model_dump()
            )

        finally:
            self.is_processing = False

    def _extract_objective(self, message: Message) -> str:
        """Extract objective from message content."""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list) and message.content:
            for content_item in message.content:
                if isinstance(content_item, MessageTextContent):
                    return content_item.text
        return "Complete the requested task"
