"""
OpenAI provider implementation for chat completions.

This module implements the ChatProvider interface for OpenAI models,
handling message conversion, streaming, and tool integration.

"""

import base64
import inspect
import json
import io
import numpy as np
from typing import Any, AsyncGenerator, AsyncIterator, List, Sequence

import aiohttp
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

from nodetool.providers.base import (
    BaseProvider,
    ProviderCapability,
    register_provider,
)
from nodetool.agents.tools.base import Tool
from nodetool.providers.openai_prediction import calculate_chat_cost
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    Provider,
    ToolCall,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
    MessageAudioContent,
    LanguageModel,
    TTSModel,
)
from nodetool.config.environment import Environment
from nodetool.workflows.types import Chunk
from nodetool.io.uri_utils import fetch_uri_bytes_and_mime
from nodetool.media.image.image_utils import image_data_to_base64_jpeg

log = get_logger(__name__)


@register_provider(Provider.OpenAI)
class OpenAIProvider(BaseProvider):
    """OpenAI implementation of the ChatProvider interface.

    Handles conversion between internal message format and OpenAI's API format,
    streaming completions, and tool calling.

    Overview of OpenAI chat constructs used:

    1. Message Format:
       - Each message is a dict with "role" and "content" fields
       - Role can be: "system", "user", "assistant", or "tool"
       - Content contains the message text (string) or content blocks (multimodal)
       - Messages may include an optional "name" field

    2. Tool Calls:
       - When a model wants to call a tool, the response includes "tool_calls"
       - Each tool call has an "id" and a "function" with name and arguments
       - To respond, send a message with role "tool" including tool_call_id

    3. Response Structure:
       - ``response.choices[0].message`` holds the model response
       - ``response.usage`` contains token usage stats

    4. Flow with Tools:
       - Model returns tool_calls → App executes tools → App replies with role
         "tool" → Model continues using results.

    For details, see the OpenAI function calling guide.
    """

    has_code_interpreter: bool = False
    provider: Provider = Provider.OpenAI

    def __init__(self):
        """Initialize the OpenAI provider with client credentials.

        Reads ``OPENAI_API_KEY`` from environment and prepares usage tracking.
        """
        super().__init__()
        env = Environment.get_environment()
        self.api_key = env.get("OPENAI_API_KEY")
        # Do not assert API key at init; tests mock API calls.
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

    def get_capabilities(self) -> set[ProviderCapability]:
        """OpenAI provider supports message generation and text-to-speech capabilities."""
        return {
            ProviderCapability.GENERATE_MESSAGE,
            ProviderCapability.GENERATE_MESSAGES,
            ProviderCapability.TEXT_TO_SPEECH,
        }

    def get_container_env(self) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``OPENAI_API_KEY`` if available; otherwise empty.
        """
        env_vars = {"OPENAI_API_KEY": self.api_key} if self.api_key else {}
        log.debug(f"Container environment variables: {list(env_vars.keys())}")
        return env_vars

    def get_client(
        self,
    ) -> openai.AsyncClient:
        """Create and return an OpenAI async client.

        Returns:
            An initialized ``openai.AsyncClient`` with reasonable timeouts.
        """
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
        """Return an approximate maximum token limit for a given model.

        Provides reasonable defaults for common OpenAI model names and a
        conservative fallback for unknown models.

        Args:
            model: Model identifier string.

        Returns:
            Approximate maximum number of tokens the model can handle.
        """
        log.debug(f"Getting context length for model: {model}")

        # Explicit mappings for tests and common models
        mapping: dict[str, int] = {
            # Classic GPT-3.5/4 families
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
        }

        if model in mapping:
            return mapping[model]

        # Broader families/prefixes
        if (
            model.startswith("gpt-4o")
            or model.startswith("chatgpt-4o")
            or model.startswith("o3")
        ):
            return 128000
        if model.startswith("gpt-4.1"):
            return 1_000_000
        if model.startswith("gpt-5"):
            return 400_000
        if model.startswith("o4-mini"):
            return 200_000

        # Fallback for unknown models
        log.debug("Unknown model; returning conservative default context length: 4096")
        return 4096

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        Most OpenAI models support function calling, with the notable exception
        of the O1 and O3 reasoning models which do not support tools.

        Args:
            model: Model identifier string.

        Returns:
            True if the model supports function calling, False otherwise.
        """
        log.debug(f"Checking tool support for model: {model}")

        # O1 and O3 series reasoning models do not support tools
        if model.startswith("o1") or model.startswith("o3"):
            log.debug(f"Model {model} is a reasoning model without tool support")
            return False

        # All other modern OpenAI models support tools (GPT-3.5-turbo, GPT-4, GPT-4o, GPT-5, etc.)
        log.debug(f"Model {model} supports tool calling")
        return True

    async def get_available_language_models(self) -> List[LanguageModel]:
        """
        Get available OpenAI models.

        Fetches models dynamically from the OpenAI API if an API key is available.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of LanguageModel instances for OpenAI
        """
        if not self.api_key:
            log.debug("No OpenAI API key configured, returning empty model list")
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=3)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers
            ) as session:
                async with session.get("https://api.openai.com/v1/models") as response:
                    if response.status != 200:
                        log.warning(
                            f"Failed to fetch OpenAI models: HTTP {response.status}"
                        )
                        return []
                    payload = await response.json()
                    data = payload.get("data", [])

                    models: List[LanguageModel] = []
                    for item in data:
                        model_id = item.get("id")
                        if not model_id:
                            continue
                        models.append(
                            LanguageModel(
                                id=model_id,
                                name=model_id,
                                provider=Provider.OpenAI,
                            )
                        )
                    log.debug(f"Fetched {len(models)} OpenAI models")
                    return models
        except Exception as e:
            log.error(f"Error fetching OpenAI models: {e}")
            return []

    async def get_available_tts_models(self) -> List[TTSModel]:
        """
        Get available OpenAI TTS models.

        Returns TTS models with their supported voices.
        Returns an empty list if no API key is configured.

        Returns:
            List of TTSModel instances for OpenAI TTS
        """
        if not self.api_key:
            log.debug("No OpenAI API key configured, returning empty TTS model list")
            return []

        # OpenAI TTS models and their voices
        # Source: https://platform.openai.com/docs/guides/text-to-speech
        tts_models_config = [
            {
                "id": "tts-1",
                "name": "TTS 1",
                "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            },
            {
                "id": "tts-1-hd",
                "name": "TTS 1 HD",
                "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            },
        ]

        models: List[TTSModel] = []
        for config in tts_models_config:
            models.append(
                TTSModel(
                    id=config["id"],
                    name=config["name"],
                    provider=Provider.OpenAI,
                    voices=config["voices"],
                )
            )

        log.debug(f"Returning {len(models)} OpenAI TTS models")
        return models

    async def uri_to_base64(self, uri: str) -> str:
        """Convert a URI to a base64-encoded ``data:`` URI string.

        If the URI points to audio, convert it to MP3 first for compatibility.

        Args:
            uri: Source URI. Supports standard URLs and ``data:`` URIs.

        Returns:
            A ``data:<mime>;base64,<data>`` string suitable for OpenAI APIs.
        """
        log.debug(f"Converting URI to base64: {uri[:50]}...")

        # Handle data URIs via normalizer to ensure audio normalization
        if uri.startswith("data:"):
            log.debug("Processing data URI directly via normalizer")
            return self._normalize_data_uri(uri)

        # Use shared utility for consistent fetching across providers
        mime_type, data_bytes = await fetch_uri_bytes_and_mime(uri)
        log.debug(
            f"Fetched bytes via utility. Mime: {mime_type}, length: {len(data_bytes)}"
        )

        # Convert audio to mp3 if needed
        if mime_type.startswith("audio/") and mime_type != "audio/mpeg":
            log.debug("Converting audio to MP3 format")
            try:
                audio = AudioSegment.from_file(io.BytesIO(data_bytes))
                with io.BytesIO() as buffer:
                    audio.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                mime_type = "audio/mpeg"
                content_b64 = base64.b64encode(mp3_data).decode("utf-8")
                log.debug(f"Audio converted to MP3, new length: {len(mp3_data)}")
            except Exception as e:
                log.warning(
                    f"Failed to convert audio URI {uri} to MP3: {e}. Using original content."
                )
                print(
                    f"Warning: Failed to convert audio URI {uri} to MP3: {e}. Using original content."
                )
                content_b64 = base64.b64encode(data_bytes).decode("utf-8")
        else:
            log.debug("Encoding content to base64")
            content_b64 = base64.b64encode(data_bytes).decode("utf-8")

        result = f"data:{mime_type};base64,{content_b64}"
        log.debug(f"Created data URI with mime type: {mime_type}")
        return result

    def _normalize_data_uri(self, uri: str) -> str:
        """Normalize a data URI and convert audio to MP3 when necessary.

        Args:
            uri: A ``data:`` URI string.

        Returns:
            A normalized ``data:<mime>;base64,<base64data>`` string.
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
        """Convert a message content to an OpenAI content part.

        Args:
            content: Internal message content variant (text, image, audio).

        Returns:
            A content part dictionary per OpenAI's chat API specification.
        """
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
                # Normalize to JPEG base64 using shared helper
                data = image_data_to_base64_jpeg(content.image.data)
                image_url = f"data:image/jpeg;base64,{data}"
                log.debug(f"Raw image data processed, length: {len(data)}")
                return {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
        else:
            log.error(f"Unknown content type {content}")
            raise ValueError(f"Unknown content type {content}")

    async def convert_message(self, message: Message) -> ChatCompletionMessageParam:
        """Convert an internal message to OpenAI's message param format.

        Args:
            message: Internal ``Message`` instance.

        Returns:
            OpenAI chat message structure matching the input role/content.
        """
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
        """Convert internal tools to OpenAI function/tool definitions.

        Args:
            tools: Iterable of tools to expose to the model.

        Returns:
            List of OpenAI-compatible tool specifications.
        """
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
        """Stream assistant deltas and tool calls from OpenAI.

        Args:
            messages: Conversation history to send.
            model: Target OpenAI model.
            tools: Optional tool definitions to provide.
            max_tokens: Maximum tokens to generate.
            context_window: Maximum tokens considered for context.
            response_format: Optional response schema.
            **kwargs: Additional OpenAI parameters such as temperature.

        Yields:
            Text ``Chunk`` items and ``ToolCall`` objects when the model
            requests tool execution.
        """
        log.debug(f"Starting streaming generation for model: {model}")
        log.debug(f"Streaming with {len(messages)} messages, {len(tools)} tools")

        if not messages:
            raise ValueError("messages must not be empty")

        # Convert system messages to user messages for O1/O3 models
        _kwargs: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if response_format is not None:
            _kwargs["response_format"] = response_format
        # Common sampling params if provided
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if key in kwargs and kwargs[key] is not None:
                _kwargs[key] = kwargs[key]
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

        self._log_api_request(
            "chat_stream",
            messages,
            **_kwargs,
        )

        log.debug(f"Converting {len(messages)} messages to OpenAI format")
        openai_messages = [await self.convert_message(m) for m in messages]
        log.debug("Making streaming API call to OpenAI")

        create_result = self.get_client().chat.completions.create(
            messages=openai_messages,
            **_kwargs,
        )
        if inspect.isawaitable(create_result):
            completion = await create_result
        else:
            completion = create_result
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

            # Process tool call deltas before checking finish_reason
            if delta.tool_calls:
                log.debug(f"Processing {len(delta.tool_calls)} tool call deltas")
                for tool_call in delta.tool_calls:
                    log.debug(f"Processing tool call delta at index {tool_call.index}")
                    tc: dict[str, Any] | None = None
                    if tool_call.index in delta_tool_calls:
                        tc = delta_tool_calls[tool_call.index]
                    else:
                        tc = {"id": tool_call.id}
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

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        context_window: int = 128000,
        response_format: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        **kwargs,
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

        if not messages:
            raise ValueError("messages must not be empty")

        request_kwargs: dict[str, Any] = {
            "max_completion_tokens": max_tokens,
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        # Common sampling params (pass-through if provided via caller)
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if presence_penalty is not None:
            request_kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            request_kwargs["frequency_penalty"] = frequency_penalty
        log.debug(f"Request kwargs: {request_kwargs}")

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

        self._log_api_request("chat", messages, **request_kwargs)

        if len(tools) > 0:
            request_kwargs["tools"] = self.format_tools(tools)
            log.debug(f"Added {len(tools)} tools to request")

        log.debug(f"Converting {len(messages)} messages to OpenAI format")
        openai_messages = [await self.convert_message(m) for m in messages]
        log.debug("Making non-streaming API call to OpenAI")

        # Make non-streaming call to OpenAI
        create_result = self.get_client().chat.completions.create(
            model=model,
            messages=openai_messages,
            stream=False,
            **request_kwargs,
        )
        if inspect.isawaitable(create_result):
            completion = await create_result
        else:
            completion = create_result
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

    async def text_to_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        speed: float = 1.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[np.ndarray[Any, np.dtype[np.int16]], None]:
        """Generate speech audio from text using OpenAI TTS with streaming.

        Uses OpenAI's streaming TTS API to yield audio chunks as they're generated,
        enabling real-time playback.

        Args:
            text: Input text to convert to speech
            model: Model identifier (e.g., "tts-1", "tts-1-hd", "gpt-4o-mini-tts")
            voice: Voice identifier (e.g., "alloy", "echo", "fable", "onyx", "nova", "shimmer")
            speed: Speech speed multiplier (0.25 to 4.0)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            **kwargs: Additional OpenAI parameters

        Yields:
            numpy.ndarray: Int16 audio chunks at 24kHz mono

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(
            f"Generating streaming speech for model: {model}, voice: {voice}, speed: {speed}"
        )

        if not text:
            raise ValueError("text must not be empty")

        # Default voice to "alloy" if not specified
        voice = voice or "alloy"

        # Clamp speed to OpenAI's supported range
        speed = max(0.25, min(4.0, speed))
        log.debug(
            f"Making streaming TTS API call with model={model}, voice={voice}, speed={speed}"
        )

        try:

            # Use streaming response
            async with self.get_client().audio.speech.with_streaming_response.create(
                model=model,
                input=text,
                voice=voice,  # type: ignore
                speed=speed,
                response_format="pcm",
            ) as response:
                log.debug("TTS streaming API call started")

                # Collect all chunks first (OpenAI sends complete MP3)
                async for chunk in response.iter_bytes(chunk_size=4096):
                    yield np.frombuffer(chunk, dtype=np.int16)

                log.debug(f"TTS streaming completed")

            self._log_api_response("text_to_speech")

        except Exception as e:
            log.error(f"OpenAI TTS streaming failed: {e}")
            raise RuntimeError(f"OpenAI TTS generation failed: {str(e)}")

    def get_usage(self) -> dict:
        """Return the current accumulated token usage statistics.

        Returns:
            A shallow copy of the usage counters collected so far.
        """
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
        """Detect whether an exception represents a context window error.

        Args:
            error: Exception to inspect.

        Returns:
            True if the error message suggests a context length violation.
        """
        msg = str(error).lower()
        is_context_error = "context length" in msg or "maximum context" in msg
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error