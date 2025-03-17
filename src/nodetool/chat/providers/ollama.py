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

    Ollama's message structure follows a specific format:

    1. Message Format:
       - Each message is a dict with "role" and "content" fields
       - Role can be: "user", "assistant", or "tool"
       - Content contains the message text (string)
       - The message history is passed as a list of these message objects

    2. Tool Calls:
       - When a model wants to call a tool, the response includes a "tool_calls" field
       - Each tool call contains:
         - "function": An object with "name" and "arguments" (dict)
         - "arguments" contains the parameters to be passed to the function
       - When responding to a tool call, you provide a message with:
         - "role": "tool"
         - "name": The name of the function that was called
         - "content": The result of the function call

    3. Response Structure:
       - response["message"] contains the model's response
       - It includes fields like "role", "content", and optionally "tool_calls"
       - The response message format is consistent with the input message format
       - If a tool is called, response["message"]["tool_calls"] will be present

    4. Tool Call Flow:
       - Model generates a response with tool_calls
       - Application executes the tool(s) based on arguments
       - Result is sent back as a "tool" role message
       - Model generates a new response incorporating tool results

    For more details, see: https://ollama.com/blog/tool-support

    """

    def __init__(self):
        """Initialize the Ollama provider."""
        super().__init__()
        self.client = get_ollama_client()
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

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

        if "thinking" in kwargs:
            kwargs.pop("thinking")

        print(ollama_messages)

        completion = await self.client.chat(
            model=model.name, messages=ollama_messages, stream=True, **kwargs
        )

        async for response in completion:
            # Track usage metrics when we receive the final response
            if response.done:
                # Accumulate token counts in self.usage
                prompt_tokens = getattr(response, "prompt_eval_count", 0)
                completion_tokens = getattr(response, "eval_count", 0)

                self.usage["prompt_tokens"] += prompt_tokens
                self.usage["completion_tokens"] += completion_tokens
                self.usage["total_tokens"] += prompt_tokens + completion_tokens

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
