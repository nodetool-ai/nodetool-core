"""
Qwen provider implementation for chat completions and vision.

This module implements the ChatProvider interface for Qwen (Alibaba Cloud),
which provides access to Qwen models through an OpenAI-compatible API.

Supported capabilities:
- Chat completions: All Qwen language models (qwen-plus, qwen-max, qwen-turbo, etc.)
- Vision (image-to-text): Qwen-VL models
- Tool calling: Supported by Qwen models

Qwen API Documentation: https://www.alibabacloud.com/help/en/model-studio/
Authentication: Uses DASHSCOPE_API_KEY (also known as QWEN_API_KEY)
Base URL: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Sequence

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

# Qwen API base URL for international (Singapore) region
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


@register_provider(Provider.Qwen)
class QwenProvider(OpenAIProvider):
    """Qwen (Alibaba Cloud) implementation of the ChatProvider interface.

    Qwen provides access to Qwen language models through an OpenAI-compatible API.
    This provider extends OpenAIProvider with Qwen-specific configuration.

    Key differences from OpenAI:
    1. Base URL: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    2. Uses DASHSCOPE_API_KEY for authentication
    3. Different set of available models (Qwen, Qwen-VL, Qwen-Coder, etc.)

    For details, see: https://www.alibabacloud.com/help/en/model-studio/
    """

    provider: Provider = Provider.Qwen

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["DASHSCOPE_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the Qwen provider with client credentials.

        Reads ``DASHSCOPE_API_KEY`` from secrets.
        """
        assert "DASHSCOPE_API_KEY" in secrets, "DASHSCOPE_API_KEY is required"
        self.api_key = secrets["DASHSCOPE_API_KEY"]
        self.client = None
        self.cost = 0.0
        log.debug("QwenProvider initialized. API key present: True")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``DASHSCOPE_API_KEY`` if available; otherwise empty.
        """
        return {"DASHSCOPE_API_KEY": self.api_key} if self.api_key else {}

    def get_client(self) -> openai.AsyncClient:
        """Create and return a Qwen async client.

        Uses OpenAI SDK with Qwen's base URL and API key.

        Returns:
            An initialized ``openai.AsyncClient`` configured for Qwen.
        """
        log.debug("Creating Qwen async client")

        # Use ResourceScope's HTTP client if available
        from nodetool.runtime.resources import require_scope

        http_client = require_scope().get_http_client()

        # Configure client for Qwen
        client = openai.AsyncClient(
            api_key=self.api_key,
            base_url=QWEN_BASE_URL,
            http_client=http_client,
        )
        log.debug("Qwen async client created successfully")
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
        log.debug("Making streaming API call to Qwen")

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
            log.info("Qwen streaming cancelled by user")
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
            log.debug("Qwen streaming cleanup completed")

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
        log.debug("Making non-streaming API call to Qwen")

        response = await self.get_client().chat.completions.create(
            messages=openai_messages,
            model=model,
            **request_kwargs,
        )

        log.debug("Received response from Qwen")

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

        Qwen models support function calling.

        Args:
            model: Model identifier string.

        Returns:
            True if the model supports function calling, False otherwise.
        """
        log.debug(f"Checking tool support for model: {model}")
        # Qwen models generally support tool calling
        log.debug(f"Model {model} supports tool calling")
        return True

    async def get_available_language_models(self) -> list[LanguageModel]:
        """
        Get available Qwen models.

        Returns a curated list of Qwen models since the API may not provide
        a models listing endpoint.

        Returns:
            List of LanguageModel instances for Qwen
        """
        if not self.api_key:
            log.debug("No Qwen API key configured, returning empty model list")
            return []

        # Curated list of Qwen models based on documentation
        # Source: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        qwen_models = [
            # Commercial models
            {"id": "qwen-plus", "name": "Qwen Plus"},
            {"id": "qwen-plus-latest", "name": "Qwen Plus Latest"},
            {"id": "qwen-max", "name": "Qwen Max"},
            {"id": "qwen-max-latest", "name": "Qwen Max Latest"},
            {"id": "qwen-turbo", "name": "Qwen Turbo"},
            {"id": "qwen-turbo-latest", "name": "Qwen Turbo Latest"},
            # Qwen3 models
            {"id": "qwen3-235b-a22b", "name": "Qwen3 235B A22B"},
            {"id": "qwen3-32b", "name": "Qwen3 32B"},
            {"id": "qwen3-30b-a3b", "name": "Qwen3 30B A3B"},
            {"id": "qwen3-14b", "name": "Qwen3 14B"},
            {"id": "qwen3-8b", "name": "Qwen3 8B"},
            {"id": "qwen3-4b", "name": "Qwen3 4B"},
            {"id": "qwen3-1.7b", "name": "Qwen3 1.7B"},
            {"id": "qwen3-0.6b", "name": "Qwen3 0.6B"},
            # Coder models
            {"id": "qwen-coder-plus", "name": "Qwen Coder Plus"},
            {"id": "qwen-coder-plus-latest", "name": "Qwen Coder Plus Latest"},
            {"id": "qwen-coder-turbo", "name": "Qwen Coder Turbo"},
            {"id": "qwen-coder-turbo-latest", "name": "Qwen Coder Turbo Latest"},
            # Vision models
            {"id": "qwen-vl-plus", "name": "Qwen VL Plus"},
            {"id": "qwen-vl-plus-latest", "name": "Qwen VL Plus Latest"},
            {"id": "qwen-vl-max", "name": "Qwen VL Max"},
            {"id": "qwen-vl-max-latest", "name": "Qwen VL Max Latest"},
            # Math models
            {"id": "qwen-math-plus", "name": "Qwen Math Plus"},
            {"id": "qwen-math-plus-latest", "name": "Qwen Math Plus Latest"},
            {"id": "qwen-math-turbo", "name": "Qwen Math Turbo"},
            {"id": "qwen-math-turbo-latest", "name": "Qwen Math Turbo Latest"},
        ]

        models: list[LanguageModel] = []
        for item in qwen_models:
            models.append(
                LanguageModel(
                    id=item["id"],
                    name=item["name"],
                    provider=Provider.Qwen,
                )
            )

        log.debug(f"Returning {len(models)} Qwen models")
        return models
