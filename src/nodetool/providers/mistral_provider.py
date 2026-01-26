"""
Mistral AI provider implementation for chat completions.

This module implements the ChatProvider interface for Mistral AI,
which provides access to Mistral models through an OpenAI-compatible API.

Mistral AI API Documentation: https://docs.mistral.ai/api/
Mistral AI Models: https://docs.mistral.ai/getting-started/models/
Authentication: Uses MISTRAL_API_KEY
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Sequence

import aiohttp
import openai

if TYPE_CHECKING:
    from nodetool.workflows.processing_context import ProcessingContext

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    LanguageModel,
    Message,
    Provider,
    ToolCall,
)
from nodetool.providers.base import register_provider
from nodetool.providers.openai_provider import OpenAIProvider
from nodetool.workflows.types import Chunk

log = get_logger(__name__)


@register_provider(Provider.Mistral)
class MistralProvider(OpenAIProvider):
    """Mistral AI implementation of the ChatProvider interface.

    Mistral AI provides access to Mistral language models through an OpenAI-compatible API.
    This provider extends OpenAIProvider with Mistral-specific configuration.

    Key differences from OpenAI:
    1. Base URL: https://api.mistral.ai/v1
    2. Uses MISTRAL_API_KEY for authentication
    3. Different set of available models (Mistral, Mixtral, etc.)

    For details, see: https://docs.mistral.ai/api/
    """

    provider: Provider = Provider.Mistral

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["MISTRAL_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the Mistral provider with client credentials.

        Reads ``MISTRAL_API_KEY`` from secrets.
        """
        assert "MISTRAL_API_KEY" in secrets, "MISTRAL_API_KEY is required"
        self.api_key = secrets["MISTRAL_API_KEY"]
        self.client = None
        self.cost = 0.0
        log.debug("MistralProvider initialized. API key present: True")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``MISTRAL_API_KEY`` if available; otherwise empty.
        """
        return {"MISTRAL_API_KEY": self.api_key} if self.api_key else {}

    def get_client(self) -> openai.AsyncClient:
        """Create and return a Mistral async client.

        Uses OpenAI SDK with Mistral's base URL and API key.

        Returns:
            An initialized ``openai.AsyncClient`` configured for Mistral.
        """
        log.debug("Creating Mistral async client")

        # Use ResourceScope's HTTP client if available
        from nodetool.runtime.resources import require_scope

        http_client = require_scope().get_http_client()

        # Configure client for Mistral
        client = openai.AsyncClient(
            api_key=self.api_key,
            base_url="https://api.mistral.ai/v1",
            http_client=http_client,
        )
        log.debug("Mistral async client created successfully")
        return client

    async def generate_messages(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        json_schema: dict | None = None,
        response_format: dict | None = None,
        **kwargs,
    ):
        from openai._types import NotGiven

        if TYPE_CHECKING:
            from openai.types.chat import ChatCompletionChunk

        log.debug(f"Starting streaming generation for model: {model}")
        log.debug(f"Streaming with {len(messages)} messages, {len(tools)} tools")

        if not messages:
            raise ValueError("messages must not be empty")

        _kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if response_format is None:
            response_format = kwargs.get("response_format")
        if response_format is not None and json_schema is not None:
            raise ValueError("response_format and json_schema are mutually exclusive")
        if response_format is not None:
            _kwargs["response_format"] = response_format
        elif json_schema is not None:
            _kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if key in kwargs and kwargs[key] is not None:
                _kwargs[key] = kwargs[key]
        log.debug(f"Initial kwargs: {_kwargs}")

        if len(tools) > 0:
            _kwargs["tools"] = self.format_tools(tools)
            log.debug(f"Added {len(tools)} tools to request")

        self._log_api_request(
            "chat_stream",
            messages,
            **_kwargs,
        )

        log.debug(f"Converting {len(messages)} messages to OpenAI format")
        openai_messages = [await self.convert_message(m) for m in messages]
        log.debug("Making streaming API call to Mistral")

        create_result = self.get_client().chat.completions.create(
            messages=openai_messages,
            **_kwargs,
        )
        import inspect

        if inspect.isawaitable(create_result):
            completion = await create_result
        else:
            completion = create_result
        log.debug("Streaming response initialized")
        delta_tool_calls = {}
        current_chunk = ""
        chunk_count = 0

        try:
            import asyncio

            async for chunk in completion:
                chunk: ChatCompletionChunk = chunk
                chunk_count += 1

                if not chunk.choices:
                    log.debug("Chunk has no choices, skipping")
                    continue

                delta = chunk.choices[0].delta

                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
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

        except asyncio.CancelledError:
            log.info("Mistral streaming cancelled by user")
            raise
        finally:
            if hasattr(completion, "close"):
                try:
                    completion.close()
                except Exception:
                    pass
            elif hasattr(completion, "response") and hasattr(completion.response, "close"):
                try:
                    await completion.response.close()
                except Exception:
                    pass
            log.debug("Mistral streaming cleanup completed")

    async def generate_message(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        json_schema: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        import json

        log.debug(f"Generating non-streaming message for model: {model}")
        log.debug(f"Non-streaming with {len(messages)} messages, {len(tools)} tools")

        if not messages:
            raise ValueError("messages must not be empty")

        request_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
        }
        if response_format is None:
            response_format = kwargs.get("response_format")
        if response_format is not None and json_schema is not None:
            raise ValueError("response_format and json_schema are mutually exclusive")
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        elif json_schema is not None:
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if presence_penalty is not None:
            request_kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            request_kwargs["frequency_penalty"] = frequency_penalty
        log.debug(f"Request kwargs: {request_kwargs}")

        self._log_api_request("chat", messages, **request_kwargs)

        if len(tools) > 0:
            request_kwargs["tools"] = self.format_tools(tools)
            log.debug(f"Added {len(tools)} tools to request")

        log.debug(f"Converting {len(messages)} messages to OpenAI format")
        openai_messages = [await self.convert_message(m) for m in messages]
        log.debug("Making non-streaming API call to Mistral")

        response = await self.get_client().chat.completions.create(
            messages=openai_messages,
            model=model,
            **request_kwargs,
        )

        log.debug("Received response from Mistral")

        response_message = response.choices[0].message

        log.debug("Logging API response")
        self._log_api_response("chat", Message(role="assistant", content=response_message.content))

        tool_calls = []
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call.function and tool_call.function.name:
                    log.debug(f"Tool call: {tool_call.function.name}")
                    tool_calls.append(
                        ToolCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            args=json.loads(tool_call.function.arguments),
                        )
                    )
                    self._log_tool_call(tool_calls[-1])

        message = Message(
            role="assistant",
            content=response_message.content,
            tool_calls=tool_calls if tool_calls else None,
        )

        log.debug(f"Generated message with {len(tool_calls)} tool calls")
        return message

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        Mistral models support function calling for compatible models.

        Args:
            model: Model identifier string.

        Returns:
            True if the model supports function calling, False otherwise.
        """
        log.debug(f"Checking tool support for model: {model}")
        # Mistral models generally support tool calling
        log.debug(f"Model {model} supports tool calling")
        return True

    async def get_available_language_models(self) -> list[LanguageModel]:
        """
        Get available Mistral models.

        Fetches models dynamically from the Mistral API if an API key is available.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of LanguageModel instances for Mistral
        """
        if not self.api_key:
            log.debug("No Mistral API key configured, returning empty model list")
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                session.get("https://api.mistral.ai/v1/models") as response,
            ):
                if response.status != 200:
                    log.warning(f"Failed to fetch Mistral models: HTTP {response.status}")
                    return []
                payload = await response.json()
                data = payload.get("data", [])

                models: list[LanguageModel] = []
                for item in data:
                    model_id = item.get("id")
                    if not model_id:
                        continue

                    # Use the name field if available, otherwise use id
                    model_name = item.get("name", model_id)

                    models.append(
                        LanguageModel(
                            id=model_id,
                            name=model_name,
                            provider=Provider.Mistral,
                        )
                    )
                log.debug(f"Fetched {len(models)} Mistral models")
                return models
        except Exception as e:
            log.error(f"Error fetching Mistral models: {e}")
            return []
