"""
Anthropic provider implementation for chat completions.

This module implements the ChatProvider interface for Anthropic Claude models,
handling message conversion, streaming, and tool integration.
"""

import base64
import json
from typing import TYPE_CHECKING, Any, Dict, List, cast
from collections.abc import AsyncIterator, Sequence
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    import asyncio

    import httpx

import aiohttp
import anthropic
from anthropic.types.base64_image_source_param import Base64ImageSourceParam
from anthropic.types.image_block_param import ImageBlockParam
from anthropic.types.message_param import MessageParam
from anthropic.types.tool_param import ToolParam
from pydantic import BaseModel

from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.io.media_fetch import fetch_uri_bytes_and_mime_sync
from nodetool.metadata.types import (
    LanguageModel,
    Message,
    MessageImageContent,
    MessageTextContent,
    Provider,
    ToolCall,
)
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.openai_prediction import calculate_chat_cost
from nodetool.workflows.base_node import ApiKeyMissingError
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

log = get_logger(__name__)


@register_provider(Provider.Anthropic)
class AnthropicProvider(BaseProvider):
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

    provider_name: str = "anthropic"

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["ANTHROPIC_API_KEY"]

    def _prepare_json_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Prepare a schema for Anthropic structured output."""
        if not isinstance(schema, dict):
            return schema

        # Copy to avoid mutating original
        new_schema = schema.copy()

        # Add additionalProperties: false to object types
        if new_schema.get("type") == "object":
            new_schema["additionalProperties"] = False

        # Recursively process properties
        if "properties" in new_schema and isinstance(new_schema["properties"], dict):
            new_schema["properties"] = {k: self._prepare_json_schema(v) for k, v in new_schema["properties"].items()}

        # Recursively process array items
        if "items" in new_schema and isinstance(new_schema["items"], dict):
            new_schema["items"] = self._prepare_json_schema(new_schema["items"])

        # Process definitions/$defs if present
        for key in ["definitions", "$defs"]:
            if key in new_schema and isinstance(new_schema[key], dict):
                new_schema[key] = {k: self._prepare_json_schema(v) for k, v in new_schema[key].items()}

        # Remove unsupported keys
        # "Not supported: ... Numerical constraints (minimum, maximum...), String constraints (minLength...)"
        unsupported_keys = [
            "default",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "minProperties",
            "maxProperties",
            "minItems",
            "maxItems",
            "uniqueItems",
        ]

        for k in unsupported_keys:
            if k in new_schema:
                del new_schema[k]

        return new_schema

    def __init__(self, secrets: dict[str, str]):
        """Initialize the Anthropic provider with client credentials."""
        api_key = secrets.get("ANTHROPIC_API_KEY")
        if not api_key or not api_key.strip():
            raise ApiKeyMissingError("ANTHROPIC_API_KEY is not configured in nodetool settings")
        self.api_key = api_key
        log.debug("AnthropicProvider initialized")
        # Cache clients per event loop to avoid sharing httpx sessions across threads/loops
        self._clients: WeakKeyDictionary[asyncio.AbstractEventLoop, anthropic.AsyncAnthropic] = WeakKeyDictionary()

    def get_client(self) -> anthropic.AsyncAnthropic:
        """Return an Anthropic async client for the current event loop."""
        import asyncio

        loop = asyncio.get_running_loop()
        if loop not in self._clients:
            log.debug(f"Creating Anthropic AsyncClient for loop {id(loop)}")
            self._clients[loop] = anthropic.AsyncAnthropic(
                api_key=self.api_key,
            )
        return self._clients[loop]

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        return {"ANTHROPIC_API_KEY": self.api_key} if self.api_key else {}

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        All Anthropic Claude models support function calling.

        Args:
            model: Model identifier string.

        Returns:
            True for all Claude models as they all support function calling.
        """
        log.debug(f"Checking tool support for model: {model}")
        log.debug(f"Model {model} supports tool calling (all Claude models do)")
        return True

    async def get_available_language_models(self) -> list[LanguageModel]:
        """
        Get available Anthropic models.

        Fetches models dynamically from the Anthropic API if an API key is available.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of LanguageModel instances for Anthropic
        """
        if not self.api_key:
            log.debug("No Anthropic API key configured, returning empty model list")
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=3)
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                session.get("https://api.anthropic.com/v1/models") as response,
            ):
                if response.status != 200:
                    log.warning(f"Failed to fetch Anthropic models: HTTP {response.status}")
                    return []
                payload: dict[str, Any] = await response.json()
                data = payload.get("data", [])

                models: list[LanguageModel] = []
                for item in data:
                    model_id = item.get("id") or item.get("name")
                    if not model_id:
                        continue
                    models.append(
                        LanguageModel(
                            id=model_id,
                            name=model_id,
                            provider=Provider.Anthropic,
                        )
                    )
                log.debug(f"Fetched {len(models)} Anthropic models")
                return models
        except Exception as e:
            log.error(f"Error fetching Anthropic models: {e}")
            return []

    def convert_message(self, message: Message) -> MessageParam | None:
        """Convert an internal message to Anthropic's format."""
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
            log.debug("Converting system message")
            return {
                "role": "assistant",
                "content": str(message.content),
            }
        elif message.role == "user":
            log.debug("Converting user message")
            assert message.content is not None, "User message content must not be None"
            if isinstance(message.content, str):
                log.debug("User message has string content")
                return {"role": "user", "content": message.content}
            else:
                log.debug(f"Converting {len(message.content)} content parts")
                content = []
                for part in message.content:
                    if isinstance(part, MessageTextContent):
                        content.append({"type": "text", "text": part.text})
                    elif isinstance(part, MessageImageContent):
                        log.debug("Converting image content")
                        # Always convert images to base64 for Anthropic
                        # (Anthropic only supports HTTPS URLs, so base64 is more reliable)
                        uri = part.image.uri or ""
                        if part.image.data:
                            # Use raw image data if available
                            log.debug("Handling raw image data")
                            data = base64.b64encode(part.image.data).decode("utf-8")
                            media_type = "image/png"
                        elif uri:
                            # Fetch image from URI and convert to base64
                            log.debug(f"Fetching image and converting to base64: {uri[:50]}...")
                            try:
                                mime_type, data_bytes = fetch_uri_bytes_and_mime_sync(uri)
                                data = base64.b64encode(data_bytes).decode("utf-8")
                                media_type = mime_type or "image/png"
                            except Exception as e:
                                log.error(f"Failed to fetch image from {uri}: {e}")
                                raise
                        else:
                            log.error("Invalid image reference with no uri or data")
                            raise ValueError("Invalid image reference with no uri or data")
                        image_source = Base64ImageSourceParam(
                            type="base64",
                            media_type=media_type,  # type: ignore
                            data=data,
                        )

                        content.append(
                            ImageBlockParam(
                                type="image",
                                source=image_source,
                            )
                        )
                log.debug(f"Converted to {len(content)} content parts")
                return {"role": "user", "content": content}
        elif message.role == "assistant":
            log.debug("Converting assistant message")
            # Skip assistant messages with empty content
            if not message.content and not message.tool_calls:
                log.debug("Skipping assistant message with no content and no tool calls")
                return None  # Will be filtered out later

            if message.tool_calls:
                log.debug(f"Assistant message has {len(message.tool_calls)} tool calls")
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
                log.debug("Assistant message has string content")
                return {"role": "assistant", "content": message.content}
            elif isinstance(message.content, list):
                log.debug(f"Assistant message has {len(message.content)} content parts")
                content = []
                assert message.content is not None, "Assistant message content must not be None"
                for part in message.content:
                    if isinstance(part, MessageTextContent):
                        content.append({"type": "text", "text": part.text})
                return {"role": "assistant", "content": content}
            else:
                log.error(f"Unknown message content type {type(message.content)}")
                raise ValueError(f"Unknown message content type {type(message.content)}")
        else:
            log.error(f"Unknown message role: {message.role}")
            raise ValueError(f"Unknown message role {message.role}")

    def format_tools(self, tools: Sequence[Any]) -> list[ToolParam]:
        """Convert tools to Anthropic's format."""
        log.debug(f"Formatting {len(tools or [])} tools for Anthropic API")
        formatted_tools = []
        for tool in tools:
            input_schema = self._prepare_json_schema(tool.input_schema)
            formatted_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": input_schema,
                }
            )
        log.debug(f"Formatted tools: {[tool['name'] for tool in formatted_tools]}")
        return formatted_tools

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] | None = None,
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Generate streaming completions from Anthropic."""
        log.debug(f"Starting streaming generation for model: {model}")
        log.debug(f"Streaming with {len(messages)} messages, {len(tools or [])} tools")

        # Handle response_format parameter
        local_tools = list(tools) if tools else []  # Make a mutable copy
        output_format = None

        system_messages = [message for message in messages if message.role == "system"]
        if len(system_messages) > 0:
            raw = system_messages[0].content
            if isinstance(raw, str):
                system_message = raw
            elif isinstance(raw, list):
                # Extract text from text content blocks; join with spaces
                text_parts: list[str] = []
                for part in raw:
                    if isinstance(part, MessageTextContent):
                        text_parts.append(part.text)
                system_message = " ".join(text_parts) if len(text_parts) > 0 else str(raw)
            else:
                system_message = str(raw)
        else:
            system_message = "You are a helpful assistant."
        log.debug(f"System message: {system_message[:50]}...")

        # Convert messages and tools to Anthropic format
        log.debug("Converting messages to Anthropic format")
        anthropic_messages = [
            msg for msg in [self.convert_message(msg) for msg in messages if msg.role != "system"] if msg is not None
        ]
        log.debug(f"Converted to {len(anthropic_messages)} Anthropic messages")

        # Use the potentially modified local_tools list
        anthropic_tools = self.format_tools(local_tools)

        log.debug(f"Starting streaming API call to Anthropic with model: {model}")

        # Prepare common kwargs and include optional sampling params if provided
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "system": system_message,
            "max_tokens": max_tokens,
        }
        if anthropic_tools:
            request_kwargs["tools"] = anthropic_tools

        for key in ("temperature", "top_p", "top_k"):
            if kwargs.get(key) is not None:
                request_kwargs[key] = kwargs[key]

        if output_format:
            raise ValueError("Output format is not supported for Anthropic")

        log.debug("Streaming response initialized")
        log.debug("Streaming response initialized")
        client = self.get_client()
        try:
            async with client.messages.stream(**request_kwargs) as ctx_stream:  # type: ignore
                async for event in ctx_stream:  # type: ignore
                    etype = getattr(event, "type", "")
                    if etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        # Prefer text; fall back to partial_json/thinking if present
                        text = getattr(delta, "text", None)
                        getattr(delta, "partial_json", None)
                        thinking = getattr(delta, "thinking", None)
                        if isinstance(text, str):
                            yield Chunk(content=text, done=False)
                        # Note: partial_json contains tool call input fragments and should NOT
                        # be yielded as message content. The complete tool call is emitted
                        # at content_block_stop event.
                        elif isinstance(thinking, str):
                            yield Chunk(content=thinking, done=False)
                    elif etype == "content_block_stop":
                        # Tool use may appear here in real SDK; tests often omit attributes
                        content_block = getattr(event, "content_block", None)
                        if content_block is not None and getattr(content_block, "type", "") == "tool_use":
                            tool_call = ToolCall(
                                id=str(getattr(content_block, "id", "")),
                                name=getattr(content_block, "name", ""),
                                args=getattr(content_block, "input", {}) or {},  # type: ignore
                            )
                            yield tool_call
                    elif etype == "message_stop":
                        yield Chunk(content="", done=True)
        except anthropic.AnthropicError as exc:
            raise self._as_httpx_status_error(exc) from exc

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] | None = None,
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> Message:
        """Generate a complete non-streaming message from Anthropic.

        Similar to generate_messages but returns a complete response rather than streaming.

        Args:
            messages: The messages to send to the model
            model: The model to use
            tools: Tools the model can use
            max_tokens: Maximum tokens to generate
            response_format: Format of the response
            **kwargs: Additional parameters to pass to the Anthropic API

        Returns:
            A complete Message object
        """
        log.debug(f"Generating non-streaming message for model: {model}")
        log.debug(f"Non-streaming with {len(messages)} messages, {len(tools or [])} tools")

        if response_format:
            raise ValueError("Output format is not supported for Anthropic")

        system_messages = [message for message in messages if message.role == "system"]
        if len(system_messages) > 0:
            raw = system_messages[0].content
            if isinstance(raw, str):
                system_message = raw
            elif isinstance(raw, list):
                text_parts: list[str] = []
                for part in raw:
                    if isinstance(part, MessageTextContent):
                        text_parts.append(part.text)
                system_message = " ".join(text_parts) if len(text_parts) > 0 else str(raw)
            else:
                system_message = str(raw)
        else:
            system_message = "You are a helpful assistant."
        log.debug(f"System message: {system_message[:50]}...")

        # Convert messages and tools to Anthropic format
        log.debug("Converting messages to Anthropic format")
        anthropic_messages = [
            msg for msg in [self.convert_message(msg) for msg in messages if msg.role != "system"] if msg is not None
        ]
        log.debug(f"Converted to {len(anthropic_messages)} Anthropic messages")

        # Use the tools from parameter, or format the local tools
        local_tools = list(tools) if tools else []
        anthropic_tools = self.format_tools(local_tools)

        log.debug(f"Making non-streaming API call to Anthropic with model: {model}")
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "system": system_message,
            "max_tokens": max_tokens,
        }
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools

        # Handle temperature, top_p, top_k from kwargs if provided
        if "temperature" in kwargs:
            create_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            create_kwargs["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            create_kwargs["top_k"] = kwargs["top_k"]

        try:
            client = self.get_client()
            response: anthropic.types.message.Message = await client.messages.create(**create_kwargs)
        except anthropic.AnthropicError as exc:
            raise self._as_httpx_status_error(exc) from exc
        log.debug("Received response from Anthropic API")

        # Update cost
        if hasattr(response, "usage"):
            log.debug("Processing usage statistics")
            usage = response.usage
            cost = await calculate_chat_cost(
                model,
                usage.input_tokens,
                usage.output_tokens,
            )
            self.cost += cost
            log.debug(f"Updated cost: {cost}")

        log.debug(f"Processing {len(response.content)} content blocks")
        content = []
        tool_calls = []
        for block in response.content:
            log.debug(f"Processing content block type: {block.type}")
            if block.type == "tool_use":
                log.debug(f"Found tool call: {block.name}")
                tool_calls.append(
                    ToolCall(
                        id=str(block.id),
                        name=block.name,
                        args=block.input,  # type: ignore
                    )
                )
            elif block.type == "text":
                content.append(block.text)

        log.debug(f"Response has {len(content)} text parts and {len(tool_calls)} tool calls")
        message = Message(
            role="assistant",
            content="\n".join(content),
            tool_calls=tool_calls,
        )

        self._log_api_response("chat", message)
        log.debug("Returning generated message")

        return message

    @staticmethod
    def _as_httpx_status_error(
        exc: anthropic.AnthropicError,
    ) -> "httpx.HTTPStatusError":
        """Normalize Anthropic SDK exceptions to `httpx.HTTPStatusError`."""
        import httpx

        maybe_response = getattr(exc, "response", None)
        status_code = getattr(maybe_response, "status_code", None) or getattr(exc, "status_code", 500)

        request = getattr(maybe_response, "request", None)
        if not isinstance(request, httpx.Request):
            request = httpx.Request(
                "POST",
                "https://api.anthropic.com/v1/messages",
            )

        response = maybe_response if isinstance(maybe_response, httpx.Response) else None
        if response is None:
            response = httpx.Response(status_code=int(status_code), request=request)

        return httpx.HTTPStatusError(str(exc), request=request, response=response)

    def is_context_length_error(self, error: Exception) -> bool:
        """Detect Anthropic context window errors robustly."""
        msg = str(error).lower()
        try:
            body = getattr(error, "body", {}) or {}
            if isinstance(body, dict):
                err = body.get("error") or {}
                message = str(err.get("message", "")).lower()
                code = str(err.get("code", "")).lower()
                if (
                    "context" in message
                    or "too long" in message
                    or "maximum context" in message
                    or code == "context_length_exceeded"
                ):
                    log.debug("Detected context length error from error body")
                    return True
        except Exception:
            pass

        is_context_error = (
            "context length" in msg or "context window" in msg or "token limit" in msg or "too long" in msg
        )
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error
