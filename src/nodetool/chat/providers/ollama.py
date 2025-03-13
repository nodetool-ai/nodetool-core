"""
Ollama provider implementation for chat completions.

This module implements the ChatProvider interface for Ollama models,
handling message conversion, streaming, and tool integration.
"""

import json
from typing import Any, AsyncGenerator, Sequence, Dict

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
from nodetool.chat.ollama_service import get_ollama_client


class OllamaProvider(ChatProvider):
    """
    Ollama implementation of the ChatProvider interface.

    Handles conversion between internal message format and Ollama's API format,
    as well as streaming completions and tool calling.
    """

    def __init__(self):
        """Initialize the Ollama provider."""
        self.client = get_ollama_client()

    def convert_message(self, message: Message) -> Dict[str, Any]:
        """Convert an internal message to Ollama's format."""
        if message.role == "tool":
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            else:
                content = json.dumps(message.content)
            return {"role": "tool", "content": content, "name": message.name}
        elif message.role == "system":
            return {"role": "system", "content": message.content}
        elif message.role == "user":
            assert message.content is not None, "User message content must not be None"
            message_dict: Dict[str, Any] = {"role": "user"}

            if isinstance(message.content, str):
                message_dict["content"] = message.content
            else:
                # Handle text content
                text_parts = [
                    part.text
                    for part in message.content
                    if isinstance(part, MessageTextContent)
                ]
                message_dict["content"] = "\n".join(text_parts)

                # Handle image content
                image_parts = [
                    part.image.uri
                    for part in message.content
                    if isinstance(part, MessageImageContent)
                ]
                if image_parts:
                    message_dict["images"] = image_parts

            return message_dict
        elif message.role == "assistant":
            return {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.args,
                        },
                    }
                    for tool_call in message.tool_calls or []
                ],
            }
        else:
            raise ValueError(f"Unknown message role {message.role}")

    def format_tools(self, tools: Sequence[Any]) -> list:
        """Convert tools to Ollama's format."""
        return [tool.tool_param() for tool in tools]

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: FunctionModel,
        tools: Sequence[Any] = [],
        **kwargs,
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """Generate streaming completions from Ollama."""
        ollama_messages = [self.convert_message(m) for m in messages]

        if len(tools) > 0:
            kwargs["tools"] = self.format_tools(tools)

        completion = await self.client.chat(
            model=model.name, messages=ollama_messages, stream=True, **kwargs
        )

        async for response in completion:
            if response.message.tool_calls is not None:
                for tool_call in response.message.tool_calls:
                    yield ToolCall(
                        name=tool_call.function.name,
                        args=dict(tool_call.function.arguments),
                    )
            yield Chunk(
                content=response.message.content or "",
                done=response.done or False,
            )
