"""
Anthropic provider implementation for chat completions.

This module implements the ChatProvider interface for Anthropic Claude models,
handling message conversion, streaming, and tool integration.
"""

import json
import os
from typing import Any, AsyncGenerator, Sequence

import anthropic
from anthropic.types.message_param import MessageParam
from anthropic.types.image_block_param import ImageBlockParam
from anthropic.types.tool_param import ToolParam
from pydantic import BaseModel

from nodetool.chat.providers.base import ChatProvider, Chunk
from nodetool.metadata.types import (
    Message,
    ToolCall,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
    FunctionModel,
)
from nodetool.common.environment import Environment


class AnthropicProvider(ChatProvider):
    """
    Anthropic implementation of the ChatProvider interface.

    Handles conversion between internal message format and Anthropic's API format,
    as well as streaming completions and tool calling.

    Anthropic's message structure follows a specific format:

    1. Message Format:
       - Messages are exchanged as alternating 'user' and 'assistant' roles
       - Each message has a 'role' and 'content'
       - Content can be a string or an array of content blocks (e.g., text, images, tool use)

    2. Content Block Types:
       - TextBlock: Simple text content ({"type": "text", "text": "content"})
       - ToolUseBlock: Used when Claude wants to call a tool
         ({"type": "tool_use", "id": "tool_id", "name": "tool_name", "input": {...}})
       - Images and other media types are also supported

    3. Response Structure:
       - id: Unique identifier for the response
       - model: The Claude model used
       - type: Always "message"
       - role: Always "assistant"
       - content: Array of content blocks
       - stop_reason: Why generation stopped (e.g., "end_turn", "max_tokens", "tool_use")
       - stop_sequence: The sequence that triggered stopping (if applicable)
       - usage: Token usage statistics

    4. Tool Use Flow:
       - Claude requests to use a tool via a ToolUseBlock
       - The application executes the tool and returns results
       - Results are provided back as a tool_result message

    For more details, see: https://docs.anthropic.com/claude/reference/messages_post
    """

    def __init__(self):
        """Initialize the Anthropic provider with client credentials."""
        env = Environment.get_environment()
        api_key = env.get("ANTHROPIC_API_KEY")
        assert api_key, "ANTHROPIC_API_KEY is not set"
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

    def convert_message(self, message: Message) -> MessageParam:
        """Convert an internal message to Anthropic's format."""
        if message.role == "tool":
            assert message.tool_call_id is not None, "Tool call ID must not be None"
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
        elif message.role == "system":
            return {
                "role": "assistant",
                "content": str(message.content),
            }
        elif message.role == "user":
            assert message.content is not None, "User message content must not be None"
            if isinstance(message.content, str):
                return {"role": "user", "content": message.content}
            else:
                content = []
                for part in message.content:
                    if isinstance(part, MessageTextContent):
                        content.append({"type": "text", "text": part.text})
                    elif isinstance(part, MessageImageContent):
                        content.append(
                            ImageBlockParam(
                                type="image",
                                source={
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": part.image.uri,
                                },
                            )
                        )
                return {"role": "user", "content": content}
        elif message.role == "assistant":
            if isinstance(message.content, str):
                return {"role": "assistant", "content": message.content}
            elif isinstance(message.content, list):
                content = []
                assert (
                    message.content is not None
                ), "Assistant message content must not be None"
                for part in message.content:
                    if isinstance(part, MessageTextContent):
                        content.append({"type": "text", "text": part.text})
                return {"role": "assistant", "content": content}
            elif message.tool_calls:
                return {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": tool_call.name,
                            "id": tool_call.id,
                            "input": tool_call.args,
                        }
                        for tool_call in message.tool_calls
                    ],
                }
            else:
                raise ValueError(
                    f"Unknown message content type {type(message.content)}"
                )
        else:
            raise ValueError(f"Unknown message role {message.role}")

    def format_tools(self, tools: Sequence[Any]) -> list[ToolParam]:
        """Convert tools to Anthropic's format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: FunctionModel,
        tools: Sequence[Any] = [],
        **kwargs,
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """Generate streaming completions from Anthropic."""
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096

        system_messages = [message for message in messages if message.role == "system"]
        system_message = (
            str(system_messages[0].content)
            if len(system_messages) > 0
            else "You are a helpful assistant."
        )

        # Convert messages and tools to Anthropic format
        anthropic_messages = [
            self.convert_message(msg) for msg in messages if msg.role != "system"
        ]

        anthropic_tools = self.format_tools(tools)

        async with self.client.messages.stream(
            model=model.name,
            messages=anthropic_messages,
            system=system_message,
            tools=anthropic_tools,
            **kwargs,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield Chunk(content=event.delta.text, done=False)
                elif event.type == "content_block_stop":
                    if event.content_block.type == "tool_use":
                        yield ToolCall(
                            id=str(event.content_block.id),
                            name=event.content_block.name,
                            args=event.content_block.input,  # type: ignore
                        )
                elif event.type == "message_stop":
                    yield Chunk(content="", done=True)
