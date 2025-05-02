"""
OpenAI provider implementation for chat completions.

This module implements the ChatProvider interface for OpenAI models,
handling message conversion, streaming, and tool integration.

"""

import base64
import json
import io
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
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel
import requests
from pydub import AudioSegment

from nodetool.chat.providers.base import ChatProvider
from nodetool.agents.tools.base import Tool
from nodetool.chat.providers.openai_prediction import calculate_chat_cost
from nodetool.metadata.types import (
    Message,
    ToolCall,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
    MessageAudioContent,
    AudioRef,
)
from nodetool.common.environment import Environment
from nodetool.workflows.types import Chunk


class OpenAIProvider(ChatProvider):
    """
    OpenAI implementation of the ChatProvider interface.

    Handles conversion between internal message format and OpenAI's API format,
    as well as streaming completions and tool calling.

    OpenAI's message structure follows a specific format:

    1. Message Format:
       - Each message is a dict with "role" and "content" fields
       - Role can be: "system", "user", "assistant", or "tool"
       - Content contains the message text (string) or content blocks (for multimodal)
       - Messages can have optional "name" field to identify specific users/assistants

    2. Tool Calls:
       - When a model wants to call a tool, the response includes a "tool_calls" field
       - Each tool call contains:
         - "id": A unique identifier for the tool call
         - "function": An object with "name" and "arguments" (JSON string)
       - When responding to a tool call, you provide a message with:
         - "role": "tool"
         - "tool_call_id": The ID of the tool call being responded to
         - "name": The name of the function that was called
         - "content": The result of the function call

    3. Response Structure:
       - response.choices[0].message contains the model's response
       - It includes fields like "role", "content", and optionally "tool_calls"
       - response.usage contains token usage statistics
         - "prompt_tokens": Number of tokens in the input
         - "completion_tokens": Number of tokens in the output
         - "total_tokens": Total tokens used

    4. Tool Call Flow:
       - Model generates a response with tool_calls
       - Application executes the tool(s) based on arguments
       - Result is sent back as a "tool" message
       - Model generates a new response incorporating tool results

    For more details, see: https://platform.openai.com/docs/guides/function-calling
    """

    has_code_interpreter: bool = False

    def __init__(self):
        """Initialize the OpenAI provider with client credentials."""
        super().__init__()
        env = Environment.get_environment()
        api_key = env.get("OPENAI_API_KEY")
        assert api_key, "OPENAI_API_KEY is not set"
        self.cost = 0.0
        self.client = openai.AsyncClient(api_key=api_key)
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }

    def uri_to_base64(self, uri: str) -> str:
        """Convert a URI to a base64 encoded data: URI string.
        If the URI points to an audio file, it converts it to MP3 first.
        """
        response = requests.get(uri)
        response.raise_for_status()  # Raise an exception for bad status codes
        mime_type = response.headers.get("content-type", "application/octet-stream")

        if mime_type.startswith("audio/") and mime_type != "audio/mpeg":
            try:
                audio = AudioSegment.from_file(io.BytesIO(response.content))
                with io.BytesIO() as buffer:
                    audio.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                mime_type = "audio/mpeg"  # Update mime type to mp3
                content_b64 = base64.b64encode(mp3_data).decode("utf-8")
            except Exception as e:
                print(
                    f"Warning: Failed to convert audio URI {uri} to MP3: {e}. Using original content."
                )
                content_b64 = base64.b64encode(response.content).decode("utf-8")
        else:
            content_b64 = base64.b64encode(response.content).decode("utf-8")

        return f"data:{mime_type};base64,{content_b64}"

    def message_content_to_openai_content_part(
        self, content: MessageContent
    ) -> ChatCompletionContentPartParam:
        """Convert a message content to an OpenAI content part."""
        if isinstance(content, MessageTextContent):
            return {"type": "text", "text": content.text}
        elif isinstance(content, MessageAudioContent):
            print(f"Audio content: {content.audio}")
            if content.audio.uri:
                # uri_to_base64 now handles conversion and returns MP3 data URI
                data_uri = self.uri_to_base64(content.audio.uri)
                # Extract base64 data part for OpenAI API
                base64_data = data_uri.split(",", 1)[1]
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "format": "mp3",
                        "data": base64_data,
                    },
                }
            else:
                # Convert raw bytes data to MP3 using pydub
                try:
                    audio = AudioSegment.from_file(io.BytesIO(content.audio.data))
                    with io.BytesIO() as buffer:
                        audio.export(buffer, format="mp3")
                        mp3_data = buffer.getvalue()
                    data = base64.b64encode(mp3_data).decode("utf-8")
                except Exception as e:
                    print(
                        f"Warning: Failed to convert raw audio data to MP3: {e}. Sending original data."
                    )
                    # Fallback to sending original data if conversion fails
                    data = base64.b64encode(content.audio.data).decode("utf-8")

                return {
                    "type": "input_audio",
                    "input_audio": {
                        "format": "mp3",
                        "data": data,
                    },
                }
        elif isinstance(content, MessageImageContent):
            if content.image.uri:
                # For images, use the original uri_to_base64 logic (implicitly called)
                return {
                    "type": "image_url",
                    "image_url": {"url": self.uri_to_base64(content.image.uri)},
                }
            else:
                # Base64 encode raw image data
                data = base64.b64encode(content.image.data).decode("utf-8")
                # Assuming PNG for raw data, adjust if needed
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{data}"},
                }
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
        formatted_tools = []

        for tool in tools:
            if tool.name == "code_interpreter":
                # Handle code_interpreter tool specially
                formatted_tools.append({"type": "code_interpreter"})
            else:
                # Handle regular function tools
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    }
                )

        return formatted_tools

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        context_window: int = 128000,
        response_format: dict | None = None,
        audio: dict | None = None,
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """Generate streaming completions from OpenAI."""

        modalities = ["text"]
        if audio:
            modalities.append("audio")
        # Convert system messages to user messages for O1/O3 models
        kwargs = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "response_format": response_format,
            "audio": audio,
            "stream": True,
            "modalities": modalities,
            "stream_options": {"include_usage": True},
        }
        if len(tools) > 0:
            kwargs["tools"] = self.format_tools(tools)

        if model.startswith("o"):
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

        self._log_api_request(
            "chat_stream",
            messages,
            model,
            tools,
            stream=True,
            modalities=modalities,
            max_completion_tokens=max_tokens,
            response_format=response_format,
            stream_options={"include_usage": True},
        )

        openai_messages = [self.convert_message(m) for m in messages]

        # if "thinking" in kwargs:
        #     kwargs.pop("thinking")
        #     if model.startswith("o1") or model.startswith("o3"):
        #         kwargs["reasoning_effort"] = "high"

        completion = await self.client.chat.completions.create(
            messages=openai_messages,
            **kwargs,
        )
        delta_tool_calls = {}
        current_chunk = ""

        async for chunk in completion:
            chunk: ChatCompletionChunk = chunk
            # Track usage information (only available in the final chunk)
            if chunk.usage:
                self.usage["prompt_tokens"] += chunk.usage.prompt_tokens
                self.usage["completion_tokens"] += chunk.usage.completion_tokens
                self.usage["total_tokens"] += chunk.usage.total_tokens
                self.cost += await calculate_chat_cost(
                    model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens
                )
                if (
                    chunk.usage.prompt_tokens_details
                    and chunk.usage.prompt_tokens_details.cached_tokens
                ):
                    self.usage[
                        "cached_prompt_tokens"
                    ] += chunk.usage.prompt_tokens_details.cached_tokens
                if (
                    chunk.usage.completion_tokens_details
                    and chunk.usage.completion_tokens_details.reasoning_tokens
                ):
                    self.usage[
                        "reasoning_tokens"
                    ] += chunk.usage.completion_tokens_details.reasoning_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, "audio") and "data" in delta.audio:  # type: ignore
                yield Chunk(
                    content=delta.audio["data"],  # type: ignore
                    content_type="audio",
                )

            if delta.content or chunk.choices[0].finish_reason == "stop":
                current_chunk += delta.content or ""
                if chunk.choices[0].finish_reason == "stop":
                    self._log_api_response(
                        "chat_stream",
                        Message(
                            role="assistant",
                            content=current_chunk,
                        ),
                    )
                yield Chunk(
                    content=delta.content or "",
                    done=chunk.choices[0].finish_reason == "stop",
                )

            if chunk.choices[0].finish_reason == "tool_calls":
                if delta_tool_calls:
                    for tc in delta_tool_calls.values():
                        assert tc is not None, "Tool call must not be None"
                        tool_call = ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            args=json.loads(tc["function"]["arguments"]),
                        )
                        self._log_tool_call(tool_call)
                        yield tool_call
                else:
                    raise ValueError("No tool call found")

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tc: dict[str, Any] | None = None
                    if tool_call.index in delta_tool_calls:
                        tc = delta_tool_calls[tool_call.index]
                    else:
                        tc = {
                            "id": tool_call.id,
                        }
                        delta_tool_calls[tool_call.index] = tc
                    assert tc is not None, "Tool call must not be None"

                    if tool_call.id:
                        tc["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        tc["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        if "function" not in tc:
                            tc["function"] = {}
                        if "arguments" not in tc["function"]:
                            tc["function"]["arguments"] = ""
                        tc["function"]["arguments"] += tool_call.function.arguments

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        context_window: int = 128000,
        response_format: dict | None = None,
    ) -> Message:
        """Generate a non-streaming completion from OpenAI.

        Args:
            messages: The message history
            model: The model to use
            tools: Optional tools to provide to the model
            max_tokens: The maximum number of tokens to generate
            context_window: The maximum number of tokens to consider for the context
            response_format: The format of the response
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            A Message object containing the model's response
        """
        kwargs = {
            "max_completion_tokens": max_tokens,
            "response_format": response_format,
        }

        # Convert system messages to user messages for O1/O3 models
        if model.startswith("o1") or model.startswith("o3"):
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

        self._log_api_request("chat", messages, model, tools, **kwargs)

        if len(tools) > 0:
            kwargs["tools"] = self.format_tools(tools)

        openai_messages = [self.convert_message(m) for m in messages]

        # Make non-streaming call to OpenAI
        completion = await self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            stream=False,
            **kwargs,
        )

        # Update usage stats
        if completion.usage:
            self.usage["prompt_tokens"] += completion.usage.prompt_tokens
            self.usage["completion_tokens"] += completion.usage.completion_tokens
            self.usage["total_tokens"] += completion.usage.total_tokens
            self.cost += await calculate_chat_cost(
                model,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )

        choice = completion.choices[0]
        response_message = choice.message

        def try_parse_args(args: Any) -> Any:
            try:
                return json.loads(args)
            except Exception:
                print(f"Warning: Error parsing tool call arguments: {args}")
                return {}

        # Create tool calls if present
        tool_calls = None
        if response_message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=try_parse_args(tool_call.function.arguments),
                )
                for tool_call in response_message.tool_calls
            ]

        message = Message(
            role="assistant",
            content=response_message.content,
            tool_calls=tool_calls,
        )

        self._log_api_response("chat", message)

        return message

    def get_usage(self) -> dict:
        """Return the current accumulated token usage statistics."""
        return self.usage.copy()

    def reset_usage(self) -> None:
        """Reset the usage counters to zero."""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
