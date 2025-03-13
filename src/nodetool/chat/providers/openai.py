"""
OpenAI provider implementation for chat completions.

This module implements the ChatProvider interface for OpenAI models,
handling message conversion, streaming, and tool integration.
"""

import json
import os
from typing import Any, AsyncGenerator, Sequence

import openai
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

from nodetool.chat.providers.base import ChatProvider, Chunk
from nodetool.chat.tools import Tool
from nodetool.metadata.types import (
    Message,
    ToolCall,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
    FunctionModel,
)
from nodetool.common.environment import Environment


class OpenAIProvider(ChatProvider):
    """
    OpenAI implementation of the ChatProvider interface.

    Handles conversion between internal message format and OpenAI's API format,
    as well as streaming completions and tool calling.
    """

    def __init__(self):
        """Initialize the OpenAI provider with client credentials."""
        env = Environment.get_environment()
        api_key = env.get("OPENAI_API_KEY")
        assert api_key, "OPENAI_API_KEY is not set"
        self.client = openai.AsyncClient(api_key=api_key)

    def message_content_to_openai_content_part(
        self, content: MessageContent
    ) -> ChatCompletionContentPartParam:
        """Convert a message content to an OpenAI content part."""
        if isinstance(content, MessageTextContent):
            return {"type": "text", "text": content.text}
        elif isinstance(content, MessageImageContent):
            return {"type": "image_url", "image_url": {"url": content.image.uri}}
        else:
            raise ValueError(f"Unknown content type {content}")

    def convert_message(self, message: Message) -> ChatCompletionMessageParam:
        """Convert an internal message to OpenAI's format."""
        if message.role == "tool":
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            else:
                content = json.dumps(message.content)
            assert message.tool_call_id is not None, "Tool call ID must not be None"
            return ChatCompletionToolMessageParam(
                role=message.role,
                content=content,
                tool_call_id=message.tool_call_id,
            )
        elif message.role == "system":
            return ChatCompletionSystemMessageParam(
                role=message.role, content=str(message.content)
            )
        elif message.role == "user":
            assert message.content is not None, "User message content must not be None"
            if isinstance(message.content, str):
                content = message.content
            elif message.content is not None:
                content = [
                    self.message_content_to_openai_content_part(c)
                    for c in message.content
                ]
            else:
                raise ValueError(
                    f"Unknown message content type {type(message.content)}"
                )
            return ChatCompletionUserMessageParam(role=message.role, content=content)
        elif message.role == "assistant":
            tool_calls = [
                ChatCompletionMessageToolCallParam(
                    type="function",
                    id=tool_call.id,
                    function=Function(
                        name=tool_call.name,
                        arguments=json.dumps(
                            tool_call.args, default=self._default_serializer
                        ),
                    ),
                )
                for tool_call in message.tool_calls or []
            ]
            if isinstance(message.content, str):
                content = message.content
            elif message.content is not None:
                content = [
                    self.message_content_to_openai_content_part(c)
                    for c in message.content
                ]
            else:
                content = None
            if len(tool_calls) == 0:
                return ChatCompletionAssistantMessageParam(
                    role=message.role,
                    content=content,  # type: ignore
                )
            else:
                return ChatCompletionAssistantMessageParam(
                    role=message.role, content=content, tool_calls=tool_calls  # type: ignore
                )
        else:
            raise ValueError(f"Unknown message role {message.role}")

    def _default_serializer(self, obj: Any) -> dict:
        """Serialize Pydantic models to dict."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        raise TypeError("Type not serializable")

    def format_tools(self, tools: Sequence[Tool]) -> list[ChatCompletionToolParam]:
        """Convert tools to OpenAI's format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
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
        """Generate streaming completions from OpenAI."""
        # Convert system messages to user messages for O1/O3 models
        if model.name.startswith("o1") or model.name.startswith("o3"):
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", 1000)
            kwargs.pop("temperature", None)
            converted_messages = []
            for msg in messages:
                if msg.role == "system":
                    converted_messages.append(
                        Message(
                            role="user",
                            content=f"Instructions: {msg.content}",
                            thread_id=msg.thread_id,
                        )
                    )
                else:
                    converted_messages.append(msg)
            messages = converted_messages

        if len(tools) > 0:
            kwargs["tools"] = self.format_tools(tools)

        openai_messages = [self.convert_message(m) for m in messages]

        completion = await self.client.chat.completions.create(
            model=model.name,
            messages=openai_messages,
            stream=True,
            **kwargs,
        )

        current_tool_call = None
        async for chunk in completion:
            delta = chunk.choices[0].delta

            if delta.content or chunk.choices[0].finish_reason == "stop":
                yield Chunk(
                    content=delta.content or "",
                    done=chunk.choices[0].finish_reason == "stop",
                )

            if chunk.choices[0].finish_reason == "tool_calls":
                if current_tool_call:
                    yield ToolCall(
                        id=current_tool_call["id"],
                        name=current_tool_call["name"],
                        args=json.loads(current_tool_call["args"]),
                    )
                else:
                    raise ValueError("No tool call found")

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.id:
                        current_tool_call = {
                            "id": tool_call.id,
                            "name": (
                                tool_call.function.name if tool_call.function else ""
                            ),
                            "args": "",
                        }
                    if tool_call.function and current_tool_call:
                        if tool_call.function.arguments:
                            current_tool_call["args"] += tool_call.function.arguments

                    assert (
                        current_tool_call is not None
                    ), "Current tool call must not be None"
