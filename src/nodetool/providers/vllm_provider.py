"""vLLM OpenAI-compatible provider implementation.

This module implements the BaseProvider interface for vLLM models, handling message
conversion, streaming, and tool integration. vLLM provides an OpenAI-compatible API
endpoint for serving large language models with optimized inference.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, List, Sequence

import httpx
import openai

from nodetool.agents.tools.base import Tool
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.openai_compat import OpenAICompat
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Message, Provider, ToolCall, LanguageModel
from nodetool.workflows.types import Chunk

log = get_logger(__name__)

# Only register the provider if VLLM_BASE_URL is explicitly set
_vllm_base_url = Environment.get_environment().get("VLLM_BASE_URL")


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse a string value to boolean.

    Args:
        value: String value to parse (1, true, yes, on for True; 0, false, no, off for False)
        default: Default value if parsing fails

    Returns:
        Parsed boolean value or default
    """
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


class VllmProvider(BaseProvider, OpenAICompat):
    """OpenAI-compatible provider backed by a vLLM server.

    The provider assumes an external vLLM instance is already running and exposes
    the OpenAI-compatible REST API as described in the vLLM documentation
    (https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

    Connection details are sourced from environment variables:

    - VLLM_BASE_URL: Base URL for the vLLM server (default http://127.0.0.1:8000)
    - VLLM_API_KEY: Optional API key to include in requests
    - VLLM_HTTP_TIMEOUT: Request timeout in seconds (default 600)
    - VLLM_VERIFY_TLS: 1 to enable TLS verification (default disabled)
    - VLLM_CONTEXT_WINDOW: Default context window hint (default 128000)

    Attributes:
        provider_name: Provider identifier used by the application.
    """

    provider_name: str = "vllm"

    def __init__(self) -> None:
        """Initialize the vLLM provider with environment configuration."""
        super().__init__()
        env = Environment.get_environment()
        # No default URL - VLLM_BASE_URL must be explicitly set
        base_url = env.get("VLLM_BASE_URL")
        if not base_url:
            raise ValueError("VLLM_BASE_URL environment variable must be set to use vLLM provider")
        self._base_url: str = base_url.rstrip("/")
        self._api_key: str | None = env.get("VLLM_API_KEY")
        timeout_raw = env.get("VLLM_HTTP_TIMEOUT", 600)
        try:
            self._timeout = float(timeout_raw)
        except (TypeError, ValueError):
            self._timeout = 600.0
        verify_env = env.get("VLLM_VERIFY_TLS")
        self._verify_tls = _parse_bool(
            verify_env if verify_env is None else str(verify_env), False
        )
        ctx_window_raw = env.get("VLLM_CONTEXT_WINDOW", 128000)
        try:
            self._default_context_window = int(ctx_window_raw)
        except (TypeError, ValueError):
            self._default_context_window = 128000

        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._client: openai.AsyncClient | None = None

    def get_container_env(self) -> dict[str, str]:
        """Return environment variables for containerized execution.

        Returns:
            Dictionary of environment variables needed for vLLM connection.
        """
        env_vars: dict[str, str] = {"VLLM_BASE_URL": self._base_url}
        if self._api_key:
            env_vars["VLLM_API_KEY"] = self._api_key
        env_vars["VLLM_HTTP_TIMEOUT"] = str(int(self._timeout))
        env_vars["VLLM_VERIFY_TLS"] = "1" if self._verify_tls else "0"
        env_vars["VLLM_CONTEXT_WINDOW"] = str(self._default_context_window)
        return env_vars

    def _ensure_client(self) -> openai.AsyncClient:
        """Get or create the OpenAI async client for vLLM.

        Returns:
            Configured OpenAI AsyncClient instance.
        """
        if self._client is None:
            api_key = self._api_key or "sk-no-key-required"
            self._client = openai.AsyncClient(
                base_url=f"{self._base_url}/v1",
                api_key=api_key,
                http_client=httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=self._timeout,
                    verify=self._verify_tls,
                ),
            )
        return self._client

    def get_context_length(self, model: str) -> int:
        """Return the context window size for the given model.

        Args:
            model: Model identifier (unused, returns default)

        Returns:
            Default context window size from configuration.
        """
        return self._default_context_window

    async def get_available_language_models(self) -> List[LanguageModel]:
        """Get available vLLM models.

        Queries the vLLM server's /models endpoint to discover available models.
        Returns an empty list if the server is not accessible.

        Returns:
            List of LanguageModel instances for vLLM
        """
        try:
            client = self._ensure_client()
            models_response = await client.models.list()
            models: List[LanguageModel] = []

            for model in models_response.data:
                models.append(
                    LanguageModel(
                        id=model.id,
                        name=model.id,
                        provider=Provider.VLLM,
                    )
                )

            log.debug(f"Fetched {len(models)} vLLM models")
            return models
        except Exception as e:
            log.error(f"Error fetching vLLM models: {e}")
            return []

    def format_tools(self, tools: Sequence[Tool]) -> list[dict[str, Any]]:
        """Convert tools to OpenAI-compatible format.

        Args:
            tools: Sequence of Tool instances

        Returns:
            List of tool definitions in OpenAI format.
        """
        return super().format_tools(tools)

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
        """Stream assistant deltas and tool calls from vLLM.

        Args:
            messages: Conversation history to send.
            model: Model identifier to use for generation.
            tools: Optional tool definitions.
            max_tokens: Maximum new tokens to generate.
            context_window: Maximum context window size.
            response_format: Optional response schema.
            **kwargs: Additional OpenAI-compatible parameters.

        Yields:
            Chunk objects for text deltas and ToolCall entries when
            the model requests tool execution.

        Raises:
            ValueError: If messages is empty.
        """
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

        if tools:
            request_payload["tools"] = self.format_tools(tools)

        for param in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if param in kwargs and kwargs[param] is not None:
                request_payload[param] = kwargs[param]

        extra_body: dict[str, Any] = {}
        if context_window is not None:
            extra_body["max_prompt_tokens"] = context_window

        log_payload = request_payload.copy()
        if extra_body:
            log_payload["extra_body"] = extra_body

        self._log_api_request("chat_stream", messages, **log_payload)

        openai_messages = [await self.convert_message(m) for m in messages]
        completion_kwargs = request_payload.copy()
        if extra_body:
            completion_kwargs["extra_body"] = extra_body
        completion = await client.chat.completions.create(
            messages=openai_messages,
            **completion_kwargs,
        )

        delta_tool_calls: dict[int, dict[str, Any]] = {}
        current_chunk = ""

        async for chunk in completion:
            chunk = chunk  # type: ignore[assignment]
            if chunk.usage:
                self._usage["prompt_tokens"] += chunk.usage.prompt_tokens
                self._usage["completion_tokens"] += chunk.usage.completion_tokens
                self._usage["total_tokens"] += chunk.usage.total_tokens
                if (
                    chunk.usage.prompt_tokens_details
                    and chunk.usage.prompt_tokens_details.cached_tokens
                ):
                    self._usage[
                        "cached_prompt_tokens"
                    ] += chunk.usage.prompt_tokens_details.cached_tokens
                if (
                    chunk.usage.completion_tokens_details
                    and chunk.usage.completion_tokens_details.reasoning_tokens
                ):
                    self._usage[
                        "reasoning_tokens"
                    ] += chunk.usage.completion_tokens_details.reasoning_tokens

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
                    arguments = (
                        function.get("arguments", "")
                        if isinstance(function, dict)
                        else ""
                    )
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

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        context_window: int = 128000,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        """Return a single, non-streaming assistant message.

        Args:
            messages: Conversation history to send.
            model: Model identifier to use for generation.
            tools: Optional tool definitions.
            max_tokens: Maximum new tokens to generate.
            context_window: Maximum context window size.
            response_format: Optional response schema.
            **kwargs: Additional OpenAI-compatible parameters.

        Returns:
            Final assistant Message with optional tool_calls.

        Raises:
            ValueError: If messages is empty.
        """
        if not messages:
            raise ValueError("messages must not be empty")

        client = self._ensure_client()
        request_payload: dict[str, Any] = {
            "max_completion_tokens": max_tokens,
            "stream": False,
        }

        if response_format is not None:
            request_payload["response_format"] = response_format

        if tools:
            request_payload["tools"] = self.format_tools(tools)

        for param in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if param in kwargs and kwargs[param] is not None:
                request_payload[param] = kwargs[param]

        extra_body: dict[str, Any] = {}
        if context_window is not None:
            extra_body["max_prompt_tokens"] = context_window

        log_payload = request_payload.copy()
        if extra_body:
            log_payload["extra_body"] = extra_body

        self._log_api_request("chat", messages, model=model, **log_payload)

        openai_messages = [await self.convert_message(m) for m in messages]
        completion_kwargs = request_payload.copy()
        if extra_body:
            completion_kwargs["extra_body"] = extra_body
        completion = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            **completion_kwargs,
        )

        if completion.usage:
            self._usage["prompt_tokens"] += completion.usage.prompt_tokens
            self._usage["completion_tokens"] += completion.usage.completion_tokens
            self._usage["total_tokens"] += completion.usage.total_tokens
            if (
                completion.usage.prompt_tokens_details
                and completion.usage.prompt_tokens_details.cached_tokens
            ):
                self._usage[
                    "cached_prompt_tokens"
                ] += completion.usage.prompt_tokens_details.cached_tokens
            if (
                completion.usage.completion_tokens_details
                and completion.usage.completion_tokens_details.reasoning_tokens
            ):
                self._usage[
                    "reasoning_tokens"
                ] += completion.usage.completion_tokens_details.reasoning_tokens

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

        message = Message(
            role="assistant", content=response_message.content, tool_calls=tool_calls
        )
        self._log_api_response("chat", message)
        return message

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


# Conditionally register the provider only if VLLM_BASE_URL is set
if _vllm_base_url:
    register_provider(Provider.VLLM)(VllmProvider)
    log.info(f"vLLM provider registered with base URL: {_vllm_base_url}")
else:
    log.debug("vLLM provider not registered: VLLM_BASE_URL not set")
