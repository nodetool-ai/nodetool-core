"""
Anthropic provider implementation for chat completions.

This module implements the ChatProvider interface for Anthropic Claude models,
handling message conversion, streaming, and tool integration.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Sequence
import random
import anthropic
from anthropic.types.message_param import MessageParam
from anthropic.types.image_block_param import ImageBlockParam
from anthropic.types.tool_param import ToolParam
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import (
    Message,
    ToolCall,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
)
from nodetool.common.environment import Environment
from nodetool.workflows.types import Chunk


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
        super().__init__()
        env = Environment.get_environment()
        api_key = env.get("ANTHROPIC_API_KEY")
        assert api_key, "ANTHROPIC_API_KEY is not set"
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
        )
        # Initialize usage tracking
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    def convert_message(self, message: Message) -> MessageParam | None:
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
            # Skip assistant messages with empty content
            if not message.content and not message.tool_calls:
                return None  # Will be filtered out later

            if message.tool_calls:
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
            elif isinstance(message.content, str):
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
        model: str,
        tools: Sequence[Any] = [],
        **kwargs,
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """Generate streaming completions from Anthropic."""
        if "max_tokens" not in kwargs:
            if "haiku" in model:
                kwargs["max_tokens"] = 4096
            else:
                kwargs["max_tokens"] = 8192

        # Handle response_format parameter
        response_format = kwargs.pop("response_format", None)

        system_messages = [message for message in messages if message.role == "system"]
        system_message = (
            str(system_messages[0].content)
            if len(system_messages) > 0
            else "You are a helpful assistant."
        )

        # If JSON format is requested via schema (OpenAI compatibility)
        if isinstance(response_format, dict) and "schema" in response_format:
            # Create a JSON output tool based on the schema
            json_tool = {
                "name": "json_output",
                "description": "Use this tool to output JSON according to the specified schema.",
                "input_schema": response_format["schema"],
            }
            # Add the JSON tool to the list of tools
            tools = list(tools) + [json_tool]
            # Add instruction to use the JSON output tool in system message
            system_message = f"{system_message}\nWhen you need to provide a JSON response, use the json_output tool."

        # if "thinking" in kwargs:
        #     kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4096}
        #     if "haiku" in model:
        #         kwargs.pop("thinking")

        # Convert messages and tools to Anthropic format
        anthropic_messages = [
            msg
            for msg in [
                self.convert_message(msg) for msg in messages if msg.role != "system"
            ]
            if msg is not None
        ]

        anthropic_tools = self.format_tools(tools)

        async with self.client.messages.stream(
            model=model,
            messages=anthropic_messages,
            system=system_message,
            tools=anthropic_tools,
            **kwargs,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield Chunk(content=event.delta.text, done=False)
                    elif event.delta.type == "thinking_delta":
                        yield Chunk(content=event.delta.thinking, done=False)
                elif event.type == "content_block_start":
                    if (
                        hasattr(event, "content_block")
                        and event.content_block.type == "thinking"
                    ):
                        # Handle start of a thinking block if needed
                        pass
                elif event.type == "content_block_stop":
                    if event.content_block.type == "tool_use":
                        tool_call = ToolCall(
                            id=str(event.content_block.id),
                            name=event.content_block.name,
                            args=event.content_block.input,  # type: ignore
                        )
                        # If this is the json_output tool, convert it to a normal text chunk
                        if tool_call.name == "json_output":
                            json_str = json.dumps(tool_call.args)
                            yield Chunk(content=json_str, done=False)
                        else:
                            yield tool_call
                    elif event.content_block.type == "thinking":
                        # Handle complete thinking blocks if needed
                        pass
                elif event.type == "message_stop":
                    # Update usage statistics when the message is complete
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        usage = event.message.usage
                        self.usage["input_tokens"] += usage.input_tokens
                        self.usage["output_tokens"] += usage.output_tokens
                        if usage.cache_creation_input_tokens:
                            self.usage[
                                "cache_creation_input_tokens"
                            ] += usage.cache_creation_input_tokens
                        if usage.cache_read_input_tokens:
                            self.usage[
                                "cache_read_input_tokens"
                            ] += usage.cache_read_input_tokens

                    yield Chunk(content="", done=True)

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        use_code_interpreter: bool = False,
        **kwargs,
    ) -> Message:
        """Generate a complete non-streaming message from Anthropic.

        Similar to generate_messages but returns a complete response rather than streaming.

        Args:
            messages: The messages to send to the model
            model: The model to use
            tools: Tools the model can use
            **kwargs: Additional parameters to pass to the Anthropic API

        Returns:
            A complete Message object
        """
        if "max_tokens" not in kwargs:
            if "haiku" in model:
                kwargs["max_tokens"] = 4096
            else:
                kwargs["max_tokens"] = 8192

        # Handle response_format parameter
        response_format = kwargs.pop("response_format", None)

        system_messages = [message for message in messages if message.role == "system"]
        system_message = (
            str(system_messages[0].content)
            if len(system_messages) > 0
            else "You are a helpful assistant."
        )

        if isinstance(response_format, dict) and "schema" in response_format:
            # Create a JSON output tool based on the schema
            json_tool = {
                "name": "json_output",
                "description": "Use this tool to output JSON according to the specified schema.",
                "input_schema": response_format["schema"],
            }
            tools = list(tools) + [json_tool]
            system_message = f"{system_message}\nYou must call the json_output tool to output JSON according to the specified schema."

        # Convert messages and tools to Anthropic format
        anthropic_messages = [
            msg
            for msg in [
                self.convert_message(msg) for msg in messages if msg.role != "system"
            ]
            if msg is not None
        ]

        anthropic_tools = self.format_tools(tools)

        response: anthropic.types.message.Message = await self.client.messages.create(
            model=model,
            messages=anthropic_messages,
            system=system_message,
            tools=anthropic_tools,
            **kwargs,
        )

        # Update usage statistics
        if hasattr(response, "usage"):
            usage = response.usage
            self.usage["input_tokens"] += usage.input_tokens
            self.usage["output_tokens"] += usage.output_tokens
            if usage.cache_creation_input_tokens:
                self.usage[
                    "cache_creation_input_tokens"
                ] += usage.cache_creation_input_tokens
            if usage.cache_read_input_tokens:
                self.usage["cache_read_input_tokens"] += usage.cache_read_input_tokens

        content = []
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(block.id),
                        name=block.name,
                        args=block.input,  # type: ignore
                    )
                )
            elif block.type == "text":
                content.append(block.text)

        # Check if the json_output tool was used and return its content directly
        for tool_call in tool_calls:
            if tool_call.name == "json_output":
                return Message(
                    role="assistant",
                    content=json.dumps(tool_call.args),
                    tool_calls=[],
                )

        return Message(
            role="assistant",
            content="\n".join(content),
            tool_calls=tool_calls,
        )
