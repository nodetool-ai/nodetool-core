"""
Anthropic provider implementation for chat completions.

This module implements the ChatProvider interface for Anthropic Claude models,
handling message conversion, streaming, and tool integration.
"""

from typing import Any, AsyncGenerator, Sequence
import random
import anthropic
from anthropic.types.message_param import MessageParam
from anthropic.types.image_block_param import ImageBlockParam
from anthropic.types.tool_param import ToolParam
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
        model: FunctionModel,
        tools: Sequence[Any] = [],
        **kwargs,
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """Generate streaming completions from Anthropic."""
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 8192

        # Handle response_format parameter
        response_format = kwargs.pop("response_format", None)

        system_messages = [message for message in messages if message.role == "system"]
        system_message = (
            str(system_messages[0].content)
            if len(system_messages) > 0
            else "You are a helpful assistant."
        )

        # If JSON format is requested, modify the system prompt
        if response_format == "json_schema":
            system_message = f"{system_message}\nYou must respond with JSON only, without any explanations or conversation."
            # Add anthropic-specific response_format parameter
            kwargs["response_format"] = {"type": "json_object"}

        if "thinking" in kwargs:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4096}
            if model.name < "claude-3-7":
                kwargs.pop("thinking")

        # Convert messages and tools to Anthropic format
        anthropic_messages = [
            msg
            for msg in [
                self.convert_message(msg) for msg in messages if msg.role != "system"
            ]
            if msg is not None
        ]

        anthropic_tools = self.format_tools(tools)

        # Add retry logic with exponential backoff
        import asyncio
        from anthropic import APIStatusError

        max_retries = 5
        base_delay = 1  # Start with 1 second delay

        for retry in range(max_retries):
            try:
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
                                yield ToolCall(
                                    id=str(event.content_block.id),
                                    name=event.content_block.name,
                                    args=event.content_block.input,  # type: ignore
                                )
                            elif event.content_block.type == "thinking":
                                # Handle complete thinking blocks if needed
                                pass
                        elif event.type == "message_stop":
                            # Update usage statistics when the message is complete
                            if hasattr(event, "message") and hasattr(
                                event.message, "usage"
                            ):
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
                # If we get here, the API call was successful, so break out of the retry loop
                break

            except APIStatusError as e:
                # Only retry on 429 (rate limit), 500, 502, 503, 504 (server errors)
                if (
                    e.status_code == 429
                    or e.status_code >= 500
                    and retry < max_retries - 1
                ):
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2**retry) + (random.random() * 0.5)
                    # Log the error and retry attempt
                    print(
                        f"Anthropic API error: {e}. Retrying in {delay:.2f} seconds (attempt {retry+1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # If we've exhausted retries or it's not a retryable error, re-raise
                    raise
