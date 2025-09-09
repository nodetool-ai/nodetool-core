"""
OpenAI provider implementation for chat completions.

This module implements the ChatProvider interface for OpenAI models,
handling message conversion, streaming, and tool integration.

"""

import base64
import json
import io
from typing import Any, AsyncGenerator, AsyncIterator, Sequence

import httpx
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
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    Function,
    ChatCompletionMessageFunctionToolCallParam,
)
from pydantic import BaseModel
from pydub import AudioSegment
from urllib.parse import unquote_to_bytes

from nodetool.chat.providers.base import ChatProvider
from nodetool.agents.tools.base import Tool
from nodetool.chat.providers.openai_prediction import calculate_chat_cost
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    Provider,
    ToolCall,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
    MessageAudioContent,
)
from nodetool.config.environment import Environment
from nodetool.workflows.types import Chunk

log = get_logger(__name__)


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
    provider: Provider = Provider.OpenAI

    def __init__(self):
        """Initialize the OpenAI provider with client credentials."""
        super().__init__()
        env = Environment.get_environment()
        self.api_key = env.get("OPENAI_API_KEY")
        assert self.api_key, "OPENAI_API_KEY is not set"
        self.client = None
        self.cost = 0.0
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }
        log.debug("OpenAIProvider initialized. API key present: True")

    def get_container_env(self) -> dict[str, str]:
        env_vars = {"OPENAI_API_KEY": self.api_key} if self.api_key else {}
        log.debug(f"Container environment variables: {list(env_vars.keys())}")
        return env_vars

    def get_client(
        self,
    ) -> openai.AsyncClient:
        log.debug("Creating OpenAI async client")
        client = openai.AsyncClient(
            api_key=self.api_key,
            http_client=httpx.AsyncClient(
                follow_redirects=True, timeout=600, verify=False
            ),
        )
        log.debug("OpenAI async client created successfully")
        return client

    def get_context_length(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        log.debug(f"Getting context length for model: {model}")

        if (
            model.startswith("gpt-4o")
            or model.startswith("chatgpt-4o")
            or model.startswith("o3")
        ):
            log.debug("Using context length: 128000")
            return 128000
        elif model.startswith("gpt-4.1"):
            log.debug("Using context length: 1000000")
            return 1000000
        elif model.startswith("gpt-5"):
            log.debug("Using context length: 400000")
            return 400000
        elif model.startswith("o4-mini"):
            log.debug("Using context length: 200000")
            return 200000
        else:
            log.error(f"Unsupported model: {model}")
            raise ValueError(f"Unsupported model: {model}")

    async def uri_to_base64(self, uri: str) -> str:
        """Convert a URI to a base64 encoded data: URI string.
        If the URI points to an audio file, it converts it to MP3 first.
        """
        log.debug(f"Converting URI to base64: {uri[:50]}...")

        # Handle data URIs directly without fetching
        if uri.startswith("data:"):
            log.debug("Processing data URI directly")
            return self._normalize_data_uri(uri)

        log.debug(f"Fetching data from URI: {uri}")
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=600, verify=False
        ) as client:
            response = await client.get(uri)
            response.raise_for_status()  # Raise an exception for bad status codes

        mime_type = response.headers.get("content-type", "application/octet-stream")
        log.debug(
            f"Detected mime type: {mime_type}, content length: {len(response.content)}"
        )

        if mime_type.startswith("audio/") and mime_type != "audio/mpeg":
            log.debug("Converting audio to MP3 format")
            try:
                audio = AudioSegment.from_file(io.BytesIO(response.content))
                with io.BytesIO() as buffer:
                    audio.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                mime_type = "audio/mpeg"  # Update mime type to mp3
                content_b64 = base64.b64encode(mp3_data).decode("utf-8")
                log.debug(f"Audio converted to MP3, new length: {len(mp3_data)}")
            except Exception as e:
                log.warning(
                    f"Failed to convert audio URI {uri} to MP3: {e}. Using original content."
                )
                print(
                    f"Warning: Failed to convert audio URI {uri} to MP3: {e}. Using original content."
                )
                content_b64 = base64.b64encode(response.content).decode("utf-8")
        else:
            log.debug("Encoding content to base64")
            content_b64 = base64.b64encode(response.content).decode("utf-8")

        result = f"data:{mime_type};base64,{content_b64}"
        log.debug(f"Created data URI with mime type: {mime_type}")
        return result

    def _normalize_data_uri(self, uri: str) -> str:
        """Normalize a data URI and convert audio/* to MP3 base64 data URI.

        Returns a string in the form: data:<mime>;base64,<base64data>
        """
        log.debug(f"Normalizing data URI: {uri[:50]}...")

        # Format: data:[<mediatype>][;base64],<data>
        try:
            header, data_part = uri.split(",", 1)
        except ValueError:
            log.error(f"Invalid data URI format: {uri[:64]}...")
            raise ValueError(f"Invalid data URI: {uri[:64]}...")

        is_base64 = ";base64" in header
        mime_type = "application/octet-stream"
        if header[5:]:  # after 'data:'
            mime_type = header[5:].split(";", 1)[0] or mime_type

        log.debug(f"Data URI mime type: {mime_type}, is_base64: {is_base64}")

        # Decode payload to bytes
        if is_base64:
            try:
                raw_bytes = base64.b64decode(data_part)
                log.debug(f"Decoded base64 data, length: {len(raw_bytes)}")
            except Exception as e:
                log.error(f"Failed to decode base64 data URI: {e}")
                raise ValueError(f"Failed to decode base64 data URI: {e}")
        else:
            # Percent-decoded textual payload → bytes
            raw_bytes = unquote_to_bytes(data_part)
            log.debug(f"Decoded percent-encoded data, length: {len(raw_bytes)}")

        # If audio and not mp3, convert to mp3; otherwise keep as-is
        if mime_type.startswith("audio/") and mime_type != "audio/mpeg":
            log.debug("Converting audio data to MP3 format")
            try:
                audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
                with io.BytesIO() as buffer:
                    audio.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                mime_type = "audio/mpeg"
                content_b64 = base64.b64encode(mp3_data).decode("utf-8")
                log.debug(f"Audio converted to MP3, new length: {len(mp3_data)}")
            except Exception as e:
                log.warning(
                    f"Failed to convert data URI audio to MP3: {e}. Using original content."
                )
                print(
                    f"Warning: Failed to convert data URI audio to MP3: {e}. Using original content."
                )
                content_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        else:
            log.debug("Encoding data to base64")
            content_b64 = base64.b64encode(raw_bytes).decode("utf-8")

        result = f"data:{mime_type};base64,{content_b64}"
        log.debug(f"Normalized data URI with mime type: {mime_type}")
        return result

    async def message_content_to_openai_content_part(
        self, content: MessageContent
    ) -> ChatCompletionContentPartParam:
        """Convert a message content to an OpenAI content part."""
        log.debug(f"Converting message content type: {type(content)}")

        if isinstance(content, MessageTextContent):
            log.debug(f"Converting text content: {content.text[:50]}...")
            return {"type": "text", "text": content.text}
        elif isinstance(content, MessageAudioContent):
            log.debug("Converting audio content")
            if content.audio.uri:
                # uri_to_base64 now handles conversion and returns MP3 data URI
                data_uri = await self.uri_to_base64(content.audio.uri)
                # Extract base64 data part for OpenAI API
                base64_data = data_uri.split(",", 1)[1]
                log.debug(f"Audio URI processed, data length: {len(base64_data)}")
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "format": "mp3",
                        "data": base64_data,
                    },
                }
            else:
                log.debug("Converting raw audio data to MP3")
                # Convert raw bytes data to MP3 using pydub
                try:
                    audio = AudioSegment.from_file(io.BytesIO(content.audio.data))
                    with io.BytesIO() as buffer:
                        audio.export(buffer, format="mp3")
                        mp3_data = buffer.getvalue()
                    data = base64.b64encode(mp3_data).decode("utf-8")
                    log.debug(f"Audio converted to MP3, data length: {len(data)}")
                except Exception as e:
                    log.warning(
                        f"Failed to convert raw audio data to MP3: {e}. Sending original data."
                    )
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
            log.debug("Converting image content")
            if content.image.uri:
                # For images, use the original uri_to_base64 logic (implicitly called)
                image_url = await self.uri_to_base64(content.image.uri)
                log.debug(f"Image URI processed: {image_url[:50]}...")
                return {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            else:
                log.debug("Converting raw image data")
                # Base64 encode raw image data
                data = base64.b64encode(content.image.data).decode("utf-8")
                # Assuming PNG for raw data, adjust if needed
                image_url = f"data:image/png;base64,{data}"
                log.debug(f"Raw image data processed, length: {len(data)}")
                return {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
        else:
            log.error(f"Unknown content type {content}")
            raise ValueError(f"Unknown content type {content}")

    async def convert_message(self, message: Message) -> ChatCompletionMessageParam:
        """Convert an internal message to OpenAI's format."""
        log.debug(f"Converting message with role: {message.role}")

        if message.role == "tool":
            log.debug(f"Converting tool message, tool_call_id: {message.tool_call_id}")
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            elif isinstance(message.content, dict):
                content = json.dumps(message.content)
            elif isinstance(message.content, list):
                content = json.dumps([part.model_dump() for part in message.content])
            elif isinstance(message.content, str):
                content = message.content
            else:
                content = json.dumps(message.content)
            log.debug(f"Tool message content type: {type(message.content)}")
            assert message.tool_call_id is not None, "Tool call ID must not be None"
            return ChatCompletionToolMessageParam(
                role=message.role,
                content=content,
                tool_call_id=message.tool_call_id,
            )
        elif message.role == "system":
            log.debug("Converting system message")
            return ChatCompletionSystemMessageParam(
                role=message.role, content=str(message.content)
            )
        elif message.role == "user":
            log.debug("Converting user message")
            assert message.content is not None, "User message content must not be None"
            if isinstance(message.content, str):
                content = message.content
                log.debug("User message has string content")
            elif message.content is not None:
                log.debug(f"Converting {len(message.content)} content parts")
                content = [
                    await self.message_content_to_openai_content_part(c)
                    for c in message.content
                ]
            else:
                log.error(f"Unknown message content type {type(message.content)}")
                raise ValueError(
                    f"Unknown message content type {type(message.content)}"
                )
            return ChatCompletionUserMessageParam(role=message.role, content=content)
        elif message.role == "assistant":
            log.debug("Converting assistant message")
            tool_calls = [
                ChatCompletionMessageFunctionToolCallParam(
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
            log.debug(f"Assistant message has {len(tool_calls)} tool calls")

            if isinstance(message.content, str):
                content = message.content
                log.debug("Assistant message has string content")
            elif message.content is not None:
                log.debug(f"Converting {len(message.content)} assistant content parts")
                content = [
                    await self.message_content_to_openai_content_part(c)
                    for c in message.content
                ]
            else:
                content = None
                log.debug("Assistant message has no content")

            if len(tool_calls) == 0:
                log.debug("Returning assistant message without tool calls")
                return ChatCompletionAssistantMessageParam(
                    role=message.role,
                    content=content,  # type: ignore
                )
            else:
                log.debug("Returning assistant message with tool calls")
                return ChatCompletionAssistantMessageParam(
                    role=message.role, content=content, tool_calls=tool_calls  # type: ignore
                )
        else:
            log.error(f"Unknown message role: {message.role}")
            raise ValueError(f"Unknown message role {message.role}")

    def _default_serializer(self, obj: Any) -> dict:
        """Serialize Pydantic models to dict."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        raise TypeError("Type not serializable")

    def format_tools(
        self, tools: Sequence[Tool]
    ) -> list[ChatCompletionMessageFunctionToolCallParam]:
        """Convert tools to OpenAI's format."""
        log.debug(f"Formatting {len(tools)} tools for OpenAI API")
        formatted_tools = []

        for tool in tools:
            log.debug(f"Formatting tool: {tool.name}")
            if tool.name == "code_interpreter":
                # Handle code_interpreter tool specially
                formatted_tools.append({"type": "code_interpreter"})
                log.debug("Added code_interpreter tool")
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
                log.debug(f"Added function tool: {tool.name}")

        log.debug(f"Formatted {len(formatted_tools)} tools total")
        return formatted_tools

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        context_window: int = 128000,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Generate streaming completions from OpenAI."""
        log.debug(f"Starting streaming generation for model: {model}")
        log.debug(f"Streaming with {len(messages)} messages, {len(tools)} tools")

        # Convert system messages to user messages for O1/O3 models
        _kwargs = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "response_format": response_format,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        log.debug(f"Initial kwargs: {_kwargs}")

        if kwargs.get("audio", None):
            _kwargs["audio"] = kwargs.get("audio", None)
            _kwargs["modalities"] = ["text", "audio"]
            if not kwargs.get("audio", None):
                _kwargs["audio"] = {
                    "voice": "alloy",
                    "format": "pcm16",
                }
            log.debug("Added audio modalities to request")

        if len(tools) > 0:
            _kwargs["tools"] = self.format_tools(tools)
            log.debug(f"Added {len(tools)} tools to request")

        if model.startswith("o"):
            log.debug("Converting system messages for O-series model")
            _kwargs.pop("temperature", None)
            converted_messages = []
            for msg in messages:
                if msg.role == "system":
                    log.debug(
                        "Converting system message to user message for O-series model"
                    )
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
            log.debug(
                f"Converted {len(converted_messages)} messages for O-series model"
            )

        kwargs_for_log = _kwargs.copy()
        kwargs_for_log.pop("tools", None)
        kwargs_for_log.pop("model", None)

        self._log_api_request(
            "chat_stream",
            messages,
            model,
            tools,
            **kwargs_for_log,
        )

        log.debug(f"Converting {len(messages)} messages to OpenAI format")
        openai_messages = [await self.convert_message(m) for m in messages]
        log.debug("Making streaming API call to OpenAI")

        completion = await self.get_client().chat.completions.create(
            messages=openai_messages,
            **_kwargs,
        )
        log.debug("Streaming response initialized")
        delta_tool_calls = {}
        current_chunk = ""
        chunk_count = 0

        async for chunk in completion:
            chunk: ChatCompletionChunk = chunk
            chunk_count += 1

            # Track usage information (only available in the final chunk)
            if chunk.usage:
                log.debug("Processing usage statistics from chunk")
                self.usage["prompt_tokens"] += chunk.usage.prompt_tokens
                self.usage["completion_tokens"] += chunk.usage.completion_tokens
                self.usage["total_tokens"] += chunk.usage.total_tokens
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
                log.debug(f"Updated usage stats: {self.usage}")

            if not chunk.choices:
                log.debug("Chunk has no choices, skipping")
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, "audio") and "data" in delta.audio:  # type: ignore
                log.debug("Yielding audio chunk")
                yield Chunk(
                    content=delta.audio["data"],  # type: ignore
                    content_type="audio",
                )

            if delta.content or chunk.choices[0].finish_reason == "stop":
                current_chunk += delta.content or ""
                finish_reason = chunk.choices[0].finish_reason
                log.debug(
                    f"Content chunk - finish_reason: {finish_reason}, content length: {len(delta.content or '')}"
                )

                if finish_reason == "stop":
                    log.debug("Final chunk received, logging response")
                    self._log_api_response(
                        "chat_stream",
                        Message(
                            role="assistant",
                            content=current_chunk,
                        ),
                    )

                content_to_yield = delta.content or ""
                yield Chunk(
                    content=content_to_yield,
                    done=finish_reason == "stop",
                )

            if chunk.choices[0].finish_reason == "tool_calls":
                log.debug("Processing tool calls completion")
                if delta_tool_calls:
                    log.debug(f"Yielding {len(delta_tool_calls)} tool calls")
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
                    log.error("No tool call found in delta_tool_calls")
                    raise ValueError("No tool call found")

            if delta.tool_calls:
                log.debug(f"Processing {len(delta.tool_calls)} tool call deltas")
                for tool_call in delta.tool_calls:
                    log.debug(f"Processing tool call delta at index {tool_call.index}")
                    tc: dict[str, Any] | None = None
                    if tool_call.index in delta_tool_calls:
                        tc = delta_tool_calls[tool_call.index]
                        log.debug(
                            f"Found existing tool call at index {tool_call.index}"
                        )
                    else:
                        tc = {
                            "id": tool_call.id,
                        }
                        delta_tool_calls[tool_call.index] = tc
                        log.debug(f"Created new tool call at index {tool_call.index}")
                    assert tc is not None, "Tool call must not be None"

                    if tool_call.id:
                        tc["id"] = tool_call.id
                        log.debug(f"Set tool call ID: {tool_call.id}")
                    if tool_call.function and tool_call.function.name:
                        tc["name"] = tool_call.function.name
                        log.debug(f"Set tool call name: {tool_call.function.name}")
                    if tool_call.function and tool_call.function.arguments:
                        if "function" not in tc:
                            tc["function"] = {}
                        if "arguments" not in tc["function"]:
                            tc["function"]["arguments"] = ""
                        tc["function"]["arguments"] += tool_call.function.arguments
                        log.debug(
                            f"Added arguments to tool call: {len(tool_call.function.arguments)} chars"
                        )

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
        log.debug(f"Generating non-streaming message for model: {model}")
        log.debug(f"Non-streaming with {len(messages)} messages, {len(tools)} tools")

        kwargs = {
            "max_completion_tokens": max_tokens,
            "response_format": response_format,
        }
        log.debug(f"Request kwargs: {kwargs}")

        # Convert system messages to user messages for O1/O3 models
        if model.startswith("o1") or model.startswith("o3"):
            log.debug("Converting system messages for O-series model")
            converted_messages = []
            for msg in messages:
                if msg.role == "system":
                    log.debug("Converting system message to user message")
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
            log.debug(
                f"Converted {len(converted_messages)} messages for O-series model"
            )

        self._log_api_request("chat", messages, model, tools, **kwargs)

        if len(tools) > 0:
            kwargs["tools"] = self.format_tools(tools)
            log.debug(f"Added {len(tools)} tools to request")

        log.debug(f"Converting {len(messages)} messages to OpenAI format")
        openai_messages = [await self.convert_message(m) for m in messages]
        log.debug("Making non-streaming API call to OpenAI")

        # Make non-streaming call to OpenAI
        completion = await self.get_client().chat.completions.create(
            model=model,
            messages=openai_messages,
            stream=False,
            **kwargs,
        )
        log.debug("Received response from OpenAI API")

        # Update usage stats
        if completion.usage:
            log.debug("Processing usage statistics")
            self.usage["prompt_tokens"] += completion.usage.prompt_tokens
            self.usage["completion_tokens"] += completion.usage.completion_tokens
            self.usage["total_tokens"] += completion.usage.total_tokens
            cost = await calculate_chat_cost(
                model,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )
            self.cost += cost
            log.debug(f"Updated usage: {self.usage}, cost: {cost}")

        choice = completion.choices[0]
        response_message = choice.message
        log.debug(f"Response content length: {len(response_message.content or '')}")

        def try_parse_args(args: Any) -> Any:
            try:
                return json.loads(args)
            except Exception:
                log.warning(f"Error parsing tool call arguments: {args}")
                print(f"Warning: Error parsing tool call arguments: {args}")
                return {}

        # Create tool calls if present
        tool_calls = None
        if response_message.tool_calls:
            log.debug(f"Processing {len(response_message.tool_calls)} tool calls")
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,  # type: ignore
                    args=try_parse_args(tool_call.function.arguments),  # type: ignore
                )
                for tool_call in response_message.tool_calls
            ]
        else:
            log.debug("Response contains no tool calls")

        message = Message(
            role="assistant",
            content=response_message.content,
            tool_calls=tool_calls,
        )

        self._log_api_response("chat", message)
        log.debug("Returning generated message")

        return message

    def get_usage(self) -> dict:
        """Return the current accumulated token usage statistics."""
        log.debug(f"Getting usage stats: {self.usage}")
        return self.usage.copy()

    def reset_usage(self) -> None:
        """Reset the usage counters to zero."""
        log.debug("Resetting usage counters")
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }
        self.cost = 0.0

    def is_context_length_error(self, error: Exception) -> bool:
        """Detect OpenAI context window errors."""
        msg = str(error).lower()
        is_context_error = "context length" in msg or "maximum context" in msg
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error
