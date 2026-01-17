"""LM Studio OpenAI-compatible provider implementation.

This module implements the BaseProvider interface for LM Studio models, handling message
conversion, streaming, and tool integration. LM Studio provides an OpenAI-compatible API
endpoint for serving large language models locally.

LM Studio Documentation: https://lmstudio.ai/docs/developer/openai-compat
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator, List, Sequence

import httpx
import openai

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import LanguageModel, Message, Provider, ToolCall
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.openai_compat import OpenAICompat
from nodetool.runtime.resources import require_scope
from nodetool.workflows.types import Chunk

if TYPE_CHECKING:
    from nodetool.agents.tools.base import Tool
    from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


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


class LMStudioProvider(BaseProvider, OpenAICompat):
    """OpenAI-compatible provider backed by LM Studio.

    The provider assumes an external LM Studio instance is already running and exposes
    the OpenAI-compatible REST API as described in the LM Studio documentation
    (https://lmstudio.ai/docs/developer/openai-compat).

    Connection details are sourced from environment variables:

    - LMSTUDIO_API_URL: Base URL for the LM Studio server (default http://127.0.0.1:1234)
    - LMSTUDIO_API_KEY: Optional API key to include in requests
    - LMSTUDIO_HTTP_TIMEOUT: Request timeout in seconds (default 600)
    - LMSTUDIO_VERIFY_TLS: 1 to enable TLS verification (default disabled)

    Attributes:
        provider_name: Provider identifier used by the application.
    """

    provider_name: str = "lmstudio"

    @classmethod
    def required_secrets(cls) -> list[str]:
        # LM Studio doesn't require any secrets - API key is optional
        return []

    def __init__(self, secrets: dict[str, str]) -> None:
        """Initialize the LM Studio provider with environment configuration.

        Args:
            secrets: Optional secrets dictionary. LMSTUDIO_API_KEY is optional
                    and only needed if the LM Studio server requires authentication.
        """
        super().__init__(secrets)
        env = Environment.get_environment()
        # Default URL for LM Studio
        base_url = env.get("LMSTUDIO_API_URL", "http://127.0.0.1:1234")
        self._base_url: str = base_url.rstrip("/")
        # API key is optional - LM Studio doesn't require it by default
        self._api_key: str | None = secrets.get("LMSTUDIO_API_KEY")
        timeout_raw = env.get("LMSTUDIO_HTTP_TIMEOUT", 600)
        try:
            self._timeout = float(timeout_raw)
        except (TypeError, ValueError):
            self._timeout = 600.0
        verify_env = env.get("LMSTUDIO_VERIFY_TLS")
        self._verify_tls = _parse_bool(verify_env if verify_env is not None else None, False)

        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        log.info(
            f"LMStudioProvider initialized. base_url={self._base_url}, "
            f"timeout={self._timeout}s, verify_tls={self._verify_tls}"
        )

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables needed for containerized execution."""
        env_vars = {}
        if self._base_url:
            env_vars["LMSTUDIO_API_URL"] = self._base_url
        if self._api_key:
            env_vars["LMSTUDIO_API_KEY"] = self._api_key
        log.debug(f"Container environment variables: {list(env_vars.keys())}")
        return env_vars

    def _create_client(self) -> openai.AsyncOpenAI:
        """Create an OpenAI client instance configured for LM Studio."""
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            verify=self._verify_tls,
        )
        return openai.AsyncOpenAI(
            base_url=f"{self._base_url}/v1",
            api_key=self._api_key or "lm-studio",  # LM Studio doesn't require an API key
            http_client=http_client,
        )

    async def get_available_language_models(self) -> List[LanguageModel]:
        """Get available LM Studio models.

        Returns models available in the local LM Studio installation.
        Returns an empty list if LM Studio is not available.

        Returns:
            List of LanguageModel instances for LM Studio
        """
        try:
            client = self._create_client()
            models_response = await client.models.list()
            models: List[LanguageModel] = []
            for model in models_response.data:
                model_id = model.id
                if model_id:
                    models.append(
                        LanguageModel(
                            id=model_id,
                            name=model_id,
                            provider=Provider.LMStudio,
                        )
                    )
            log.debug(f"Fetched {len(models)} LM Studio models")
            return models
        except Exception as e:
            log.error(f"Error fetching LM Studio models: {e}")
            return []

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] | None = None,
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        """Generate a complete message from LM Studio without streaming.

        Args:
            messages: The conversation history
            model: The model to use
            tools: Optional tools to make available to the model
            max_tokens: Maximum tokens to generate
            response_format: Optional response format specification
            **kwargs: Additional parameters to pass to the LM Studio API
                - context_window: Optional context window override

        Returns:
            Message: The complete response message
        """
        # Extract optional parameters from kwargs
        context_window = kwargs.pop("context_window", None)

        log.debug(f"Generating complete message for model: {model}")
        tools_list = tools if tools is not None else []
        log.debug(f"Non-streaming with {len(messages)} messages, {len(tools_list)} tools")

        client = self._create_client()

        # Convert messages to OpenAI format
        openai_messages = [await self.convert_message(m) for m in messages]

        # Prepare request parameters
        params: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # Add tools if provided
        if tools_list:
            params["tools"] = self.format_tools(tools_list)

        # Add response format if provided
        if response_format:
            params["response_format"] = response_format

        # Add any additional kwargs
        params.update(kwargs)

        log.debug(f"Making non-streaming chat request with params: {list(params.keys())}")

        try:
            response = await client.chat.completions.create(**params)
            log.debug("Received complete response from LM Studio")

            # Update usage stats
            if response.usage:
                self._usage["prompt_tokens"] += response.usage.prompt_tokens or 0
                self._usage["completion_tokens"] += response.usage.completion_tokens or 0
                self._usage["total_tokens"] += response.usage.total_tokens or 0

            choice = response.choices[0]
            content = choice.message.content or ""
            log.debug(f"Response content length: {len(content)}")

            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        args=json.loads(tc.function.arguments),
                    )
                    for tc in choice.message.tool_calls
                ]
                log.debug(f"Response contains {len(tool_calls)} tool calls")

            return Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )
        except Exception as e:
            log.error(f"Error generating message from LM Studio: {e}")
            raise

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] | None = None,
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Generate streaming completions from LM Studio.

        Args:
            messages: The conversation history
            model: The model to use
            tools: Optional tools to make available to the model
            max_tokens: Maximum tokens to generate
            response_format: Optional response format specification
            **kwargs: Additional parameters to pass to the LM Studio API
                - context_window: Optional context window override

        Yields:
            Chunk | ToolCall: Content chunks or tool calls
        """
        # Extract optional parameters from kwargs
        context_window = kwargs.pop("context_window", None)

        log.debug(f"Starting streaming generation for model: {model}")
        tools_list = tools if tools is not None else []
        log.debug(f"Streaming with {len(messages)} messages, {len(tools_list)} tools")

        client = self._create_client()

        # Convert messages to OpenAI format
        openai_messages = [await self.convert_message(m) for m in messages]

        # Prepare request parameters
        params: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        # Add tools if provided
        if tools_list:
            params["tools"] = self.format_tools(tools_list)

        # Add response format if provided
        if response_format:
            params["response_format"] = response_format

        # Add any additional kwargs
        params.update(kwargs)

        log.debug("Starting streaming chat request")

        try:
            stream = await client.chat.completions.create(**params)
            log.debug("Streaming response initialized")

            chunk_count = 0
            tool_call_count = 0
            tool_calls_buffer: dict[int, dict[str, Any]] = {}

            async for chunk in stream:
                chunk_count += 1

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Handle tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc_delta.id:
                            tool_calls_buffer[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_buffer[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments

                # Handle content
                if delta.content:
                    yield Chunk(
                        content=delta.content,
                        done=False,
                    )

                # Check if streaming is done
                if choice.finish_reason:
                    # Yield any accumulated tool calls
                    for tc_data in tool_calls_buffer.values():
                        if tc_data["name"]:
                            tool_call_count += 1
                            log.debug(f"Yielding tool call: {tc_data['name']}")
                            try:
                                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            yield ToolCall(
                                id=tc_data["id"],
                                name=tc_data["name"],
                                args=args,
                            )

                    # Yield final chunk
                    yield Chunk(content="", done=True)
                    log.debug(f"Streaming completed. Total chunks: {chunk_count}, tool calls: {tool_call_count}")
                    break

        except Exception as e:
            log.error(f"Error during streaming generation from LM Studio: {e}")
            raise

    def is_context_length_error(self, error: Exception) -> bool:
        """Check if an error is due to context length exceeded."""
        msg = str(error).lower()
        is_context_error = (
            "context length" in msg
            or "context window" in msg
            or "token limit" in msg
            or "request too large" in msg
            or "413" in msg
            or "maximum context length" in msg
        )
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error

    def get_usage(self) -> dict:
        """Return the current accumulated token usage statistics."""
        log.debug(f"Getting usage stats: {self._usage}")
        return self._usage.copy()

    def reset_usage(self) -> None:
        """Reset the usage counters to zero."""
        log.debug("Resetting usage counters")
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


# Always register the provider (like other providers do)
# Users can configure it via LMSTUDIO_API_URL environment variable
register_provider(Provider.LMStudio)(LMStudioProvider)
