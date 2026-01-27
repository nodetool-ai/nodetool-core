"""
AIME provider implementation for chat completions.

This module implements the ChatProvider interface for AIME,
which provides access to AI models through the AIME OpenAI-compatible API.

AIME API Documentation: https://www.aime.info/api
OpenAI-compatible endpoint: https://api.aime.info/v1/chat/completions
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Sequence

import httpx
import openai

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    LanguageModel,
    Message,
    Provider,
    ToolCall,
)
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.openai_compat import OpenAICompat
from nodetool.workflows.types import Chunk

if TYPE_CHECKING:
    from nodetool.agents.tools.base import Tool
    from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


@register_provider(Provider.AIME)
class AIMEProvider(BaseProvider, OpenAICompat):
    """AIME implementation of the ChatProvider interface using OpenAI-compatible API.

    AIME provides access to AI models through an OpenAI-compatible API endpoint.
    This provider extends BaseProvider with OpenAICompat for message/tool formatting.

    Key features:
    1. Base URL: https://api.aime.info/v1
    2. Uses AIME_API_KEY for authentication (as Bearer token)
    3. OpenAI-compatible chat completions API

    For details, see: https://www.aime.info/api
    """

    provider: Provider = Provider.AIME
    provider_name: str = "aime"

    DEFAULT_BASE_URL = "https://api.aime.info/v1"
    DEFAULT_TIMEOUT = 300.0  # 5 minutes for long requests

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["AIME_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the AIME provider with API key.

        Reads ``AIME_API_KEY`` from secrets.
        """
        super().__init__(secrets=secrets)
        assert "AIME_API_KEY" in secrets, "AIME_API_KEY is required"
        self.api_key = secrets["AIME_API_KEY"]
        self._base_url = self.DEFAULT_BASE_URL
        self._timeout = self.DEFAULT_TIMEOUT
        self.cost = 0.0
        self._fallback_http_client: httpx.AsyncClient | None = None
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }
        log.debug("AIMEProvider initialized. API key present: True")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``AIME_API_KEY`` if available.
        """
        env = {}
        if self.api_key:
            env["AIME_API_KEY"] = self.api_key
        return env

    def _ensure_client(self) -> openai.AsyncClient:
        """Get the OpenAI async client for AIME.

        Uses a dedicated HTTP client to avoid ResourceScope lifecycle issues.

        Returns:
            Configured OpenAI AsyncClient instance.
        """
        # Always use a dedicated HTTP client for AIME to avoid premature closure
        # during streaming responses
        if self._fallback_http_client is None or self._fallback_http_client.is_closed:
            log.debug("Creating dedicated HTTP client for AIME")
            self._fallback_http_client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=self._timeout,
            )

        return openai.AsyncClient(
            base_url=self._base_url,
            api_key=self.api_key,
            http_client=self._fallback_http_client,
        )

    def format_tools(self, tools: Sequence[Tool]) -> list[dict[str, Any]]:
        """Convert tools to OpenAI-compatible format.

        Args:
            tools: Sequence of Tool instances

        Returns:
            List of tool definitions in OpenAI format.
        """
        return super().format_tools(tools)

    async def generate_message(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] | None = None,
        max_tokens: int = 16384,
        json_schema: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        """Generate a non-streaming completion from AIME.

        Args:
            messages: The message history
            model: The model to use
            tools: Optional tools to provide to the model
            max_tokens: The maximum number of tokens to generate
            json_schema: Optional JSON schema for structured output
            temperature: Optional sampling temperature
            top_p: Optional nucleus sampling parameter
            response_format: Optional response format specification
            **kwargs: Additional arguments to pass to the API

        Returns:
            A Message object containing the model's response
        """
        log.debug(f"AIME generating non-streaming message for model: {model}")
        log.debug(f"AIME non-streaming with {len(messages)} messages")

        if not messages:
            raise ValueError("messages must not be empty")

        client = self._ensure_client()
        request_payload: dict[str, Any] = {
            "max_completion_tokens": max_tokens,
            "stream": False,
        }

        if response_format is not None:
            request_payload["response_format"] = response_format
        elif json_schema is not None:
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        if tools:
            request_payload["tools"] = self.format_tools(tools)

        if temperature is not None:
            request_payload["temperature"] = temperature
        if top_p is not None:
            request_payload["top_p"] = top_p

        self._log_api_request("chat", messages, model=model, **request_payload)

        openai_messages = [await self.convert_message(m) for m in messages]

        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                **request_payload,
            )
        except openai.OpenAIError as exc:
            log.error(f"AIME API error: {exc}")
            raise

        if completion.usage:
            self._usage["prompt_tokens"] += completion.usage.prompt_tokens
            self._usage["completion_tokens"] += completion.usage.completion_tokens
            self._usage["total_tokens"] += completion.usage.total_tokens
            if completion.usage.prompt_tokens_details and completion.usage.prompt_tokens_details.cached_tokens:
                self._usage["cached_prompt_tokens"] += completion.usage.prompt_tokens_details.cached_tokens
            if (
                completion.usage.completion_tokens_details
                and completion.usage.completion_tokens_details.reasoning_tokens
            ):
                self._usage["reasoning_tokens"] += completion.usage.completion_tokens_details.reasoning_tokens

        choice = completion.choices[0]
        response_message = choice.message

        tool_calls = None
        if response_message.tool_calls:
            tool_calls = []
            for tool_call in response_message.tool_calls:
                args_raw = tool_call.function.arguments if tool_call.function else ""
                try:
                    parsed_args = json.loads(args_raw)
                except Exception:
                    parsed_args = {}
                name = tool_call.function.name if tool_call.function else ""
                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=name,
                        args=parsed_args,
                    )
                )

        message = Message(role="assistant", content=response_message.content, tool_calls=tool_calls)
        self._log_api_response("chat", message)
        return message

    async def generate_messages(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] | None = None,
        max_tokens: int = 16384,
        json_schema: dict | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Stream assistant deltas and tool calls from AIME.

        Args:
            messages: Conversation history to send.
            model: Model identifier to use for generation.
            tools: Optional tool definitions.
            max_tokens: Maximum new tokens to generate.
            json_schema: Optional response schema.
            response_format: Optional response format specification.
            **kwargs: Additional OpenAI-compatible parameters.

        Yields:
            Chunk objects for text deltas and ToolCall entries when
            the model requests tool execution.

        Raises:
            ValueError: If messages is empty.
        """
        log.debug(f"AIME starting streaming generation for model: {model}")
        log.debug(f"AIME streaming with {len(messages)} messages")

        if not messages:
            raise ValueError("messages must not be empty")

        client = self._ensure_client()
        request_payload: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if response_format is not None:
            request_payload["response_format"] = response_format
        elif json_schema is not None:
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        if tools:
            request_payload["tools"] = self.format_tools(tools)

        for param in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if param in kwargs and kwargs[param] is not None:
                request_payload[param] = kwargs[param]

        self._log_api_request("chat_stream", messages, **request_payload)

        openai_messages = [await self.convert_message(m) for m in messages]
        completion = await client.chat.completions.create(
            messages=openai_messages,
            **request_payload,
        )

        delta_tool_calls: dict[int, dict[str, Any]] = {}
        current_chunk = ""

        async for chunk in completion:
            if chunk.usage:
                self._usage["prompt_tokens"] += chunk.usage.prompt_tokens
                self._usage["completion_tokens"] += chunk.usage.completion_tokens
                self._usage["total_tokens"] += chunk.usage.total_tokens
                if chunk.usage.prompt_tokens_details and chunk.usage.prompt_tokens_details.cached_tokens:
                    self._usage["cached_prompt_tokens"] += chunk.usage.prompt_tokens_details.cached_tokens
                if chunk.usage.completion_tokens_details and chunk.usage.completion_tokens_details.reasoning_tokens:
                    self._usage["reasoning_tokens"] += chunk.usage.completion_tokens_details.reasoning_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tc = delta_tool_calls.get(tool_call.index) or {"id": tool_call.id}
                    delta_tool_calls[tool_call.index] = tc
                    if tool_call.id:
                        tc["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        tc["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        tc.setdefault("function", {})
                        tc["function"].setdefault("arguments", "")
                        tc["function"]["arguments"] += tool_call.function.arguments

            if delta.content or finish_reason == "stop":
                current_chunk += delta.content or ""
                if finish_reason == "stop":
                    self._log_api_response(
                        "chat_stream",
                        Message(role="assistant", content=current_chunk),
                    )
                yield Chunk(content=delta.content or "", done=finish_reason == "stop")

            if finish_reason == "tool_calls" and delta_tool_calls:
                for tc in delta_tool_calls.values():
                    function = tc.get("function", {})
                    arguments = function.get("arguments", "") if isinstance(function, dict) else ""
                    try:
                        args = json.loads(arguments)
                    except Exception:
                        args = {}
                    tool_call = ToolCall(
                        id=tc.get("id") or "",
                        name=tc.get("name") or "",
                        args=args or {},
                    )
                    self._log_tool_call(tool_call)
                    yield tool_call

    async def get_available_language_models(self) -> list[LanguageModel]:
        """Get available AIME language models.

        Queries the AIME server's /models endpoint to discover available models.
        Returns an empty list if the server is not accessible.

        Returns:
            List of LanguageModel instances for AIME
        """
        try:
            client = self._ensure_client()
            models_response = await client.models.list()
            models: list[LanguageModel] = []

            for model in models_response.data:
                models.append(
                    LanguageModel(
                        id=model.id,
                        name=model.id,
                        provider=Provider.AIME,
                    )
                )

            log.debug(f"Fetched {len(models)} AIME models")
            return models
        except Exception as e:
            log.error(f"Error fetching AIME models: {e}")
            # Return a fallback list of known models
            return [
                LanguageModel(
                    id="gpt-oss:20b",
                    name="GPT OSS 20B",
                    provider=Provider.AIME,
                ),
            ]

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        AIME OpenAI-compatible API supports function calling for compatible models.

        Args:
            model: Model identifier string.

        Returns:
            True if the model supports function calling, False otherwise.
        """
        # AIME models with OpenAI-compatible API generally support tool calling
        return True

    def get_usage(self) -> dict[str, int]:
        """Return a shallow copy of accumulated usage counters.

        Returns:
            Dictionary with token usage statistics.
        """
        return self._usage.copy()

    def reset_usage(self) -> None:
        """Reset all usage counters to zero."""
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }
