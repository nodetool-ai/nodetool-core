"""
Together AI provider implementation for chat completions.

This module implements the ChatProvider interface for Together AI,
which provides access to open-source LLMs through an OpenAI-compatible API.

Together AI API Documentation: https://docs.together.ai/reference/chat-completions
Together AI Models: https://docs.together.ai/docs/inference-models
Authentication: https://docs.together.ai/reference/authentication
"""

from __future__ import annotations

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


@register_provider(Provider.Together)
class TogetherProvider(OpenAIProvider):
    """Together AI implementation of the ChatProvider interface.

    Together AI provides access to open-source LLMs through an OpenAI-compatible API.
    This provider extends OpenAIProvider with Together-specific configuration.

    Key differences from OpenAI:
    1. Base URL: https://api.together.xyz/v1
    2. Uses TOGETHER_API_KEY for authentication
    3. Different set of available models (primarily open-source models)

    For details, see: https://docs.together.ai/reference/chat-completions
    """

    provider: Provider = Provider.Together

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["TOGETHER_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the Together AI provider with client credentials.

        Reads ``TOGETHER_API_KEY`` from secrets.
        """
        assert "TOGETHER_API_KEY" in secrets, "TOGETHER_API_KEY is required"
        self.api_key = secrets["TOGETHER_API_KEY"]
        self.client = None
        self.cost = 0.0
        log.debug("TogetherProvider initialized. API key present: True")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``TOGETHER_API_KEY`` if available; otherwise empty.
        """
        return {"TOGETHER_API_KEY": self.api_key} if self.api_key else {}

    def get_client(self) -> openai.AsyncClient:
        """Create and return a Together AI async client.

        Uses OpenAI SDK with Together's base URL and API key.

        Returns:
            An initialized ``openai.AsyncClient`` configured for Together AI.
        """
        log.debug("Creating Together AI async client")

        # Use ResourceScope's HTTP client if available
        from nodetool.runtime.resources import require_scope

        http_client = require_scope().get_http_client()

        # Configure client for Together AI
        client = openai.AsyncClient(
            api_key=self.api_key,
            base_url="https://api.together.xyz/v1",
            http_client=http_client,
        )
        log.debug("Together AI async client created successfully")
        return client

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        Together AI supports function calling for compatible models.

        Args:
            model: Model identifier string.

        Returns:
            True if the model supports function calling, False otherwise.
        """
        log.debug(f"Checking tool support for model: {model}")
        # Together AI models generally support tool calling for newer models
        log.debug(f"Model {model} supports tool calling")
        return True

    async def get_available_language_models(self) -> list[LanguageModel]:
        """
        Get available Together AI models.

        Fetches models dynamically from the Together AI API if an API key is available.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of LanguageModel instances for Together AI
        """
        if not self.api_key:
            log.debug("No Together AI API key configured, returning empty model list")
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                session.get("https://api.together.xyz/v1/models") as response,
            ):
                if response.status != 200:
                    log.warning(f"Failed to fetch Together AI models: HTTP {response.status}")
                    return []
                payload = await response.json()
                data = payload.get("data", [])

                models: list[LanguageModel] = []
                for item in data:
                    model_id = item.get("id")
                    if not model_id:
                        continue

                    # Filter for chat/language models only
                    model_type = item.get("type", "")
                    if model_type not in ["chat", "language"]:
                        continue

                    # Use the display_name field if available, otherwise use id
                    model_name = item.get("display_name", model_id)

                    models.append(
                        LanguageModel(
                            id=model_id,
                            name=model_name,
                            provider=Provider.Together,
                        )
                    )
                log.debug(f"Fetched {len(models)} Together AI models")
                return models
        except Exception as e:
            log.error(f"Error fetching Together AI models: {e}")
            return []

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
        """Generate a non-streaming completion from Together AI.

        This method extends OpenAI's generate_message for Together AI.

        Args:
            messages: The message history
            model: The model to use
            tools: Optional tools to provide to the model
            max_tokens: The maximum number of tokens to generate
            json_schema: Optional JSON schema for structured output
            temperature: Optional sampling temperature (0.0-1.0)
            top_p: Optional nucleus sampling parameter (0.0-1.0)
            presence_penalty: Optional presence penalty (-2.0 to 2.0)
            frequency_penalty: Optional frequency penalty (-2.0 to 2.0)
            response_format: Optional format specification for the response (e.g., {"type": "json_object"})
            **kwargs: Additional arguments to pass to the API

        Returns:
            A Message object containing the model's response
        """
        import json

        log.debug(f"Generating non-streaming message for model: {model}")
        log.debug(f"Non-streaming with {len(messages)} messages, {len(tools)} tools")

        if not messages:
            raise ValueError("messages must not be empty")

        request_kwargs: dict[str, Any] = {
            "max_completion_tokens": max_tokens,
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

        # Common sampling params
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
        log.debug("Making non-streaming API call to Together AI")

        # Make non-streaming call
        try:
            create_result = self.get_client().chat.completions.create(
                model=model,
                messages=openai_messages,
                stream=False,
                **request_kwargs,
            )
            import inspect

            if inspect.isawaitable(create_result):
                completion = await create_result
            else:
                completion = create_result
        except openai.OpenAIError as exc:
            raise self._as_httpx_status_error(exc) from exc
        log.debug("Received response from Together AI API")

        # Process usage statistics if available
        if completion.usage:
            log.debug("Processing usage statistics")
            log.debug(
                f"Usage - prompt_tokens: {completion.usage.prompt_tokens}, "
                f"completion_tokens: {completion.usage.completion_tokens}"
            )

        choice = completion.choices[0]
        response_message = choice.message
        log.debug(f"Response content length: {len(response_message.content or '')}")

        def try_parse_args(args: Any) -> Any:
            try:
                return json.loads(args)
            except Exception:
                log.warning(f"Error parsing tool call arguments: {args}")
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

    async def generate_messages(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        json_schema: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Stream assistant deltas and tool calls from Together AI.

        This method extends OpenAI's generate_messages for Together AI.

        Args:
            messages: Conversation history to send.
            model: Target model.
            tools: Optional tool definitions to provide.
            max_tokens: Maximum tokens to generate.
            json_schema: Optional response schema.
            **kwargs: Additional parameters such as temperature.

        Yields:
            Text ``Chunk`` items and ``ToolCall`` objects when the model
            requests tool execution.
        """
        import json

        log.debug(f"Starting streaming generation for model: {model}")
        log.debug(f"Streaming with {len(messages)} messages, {len(tools)} tools")

        if not messages:
            raise ValueError("messages must not be empty")

        _kwargs: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if "response_format" in kwargs and kwargs["response_format"] is not None:
            _kwargs["response_format"] = kwargs["response_format"]
        elif json_schema is not None:
            _kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        # Common sampling params if provided
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
        log.debug("Making streaming API call to Together AI")

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

        async for chunk in completion:
            chunk_count += 1

            # Track usage from streaming
            if chunk.usage:
                log.debug("Processing usage statistics from chunk")

            if not chunk.choices:
                log.debug("Chunk has no choices, skipping")
                continue

            delta = chunk.choices[0].delta

            # Process tool call deltas before checking finish_reason
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
                log.debug(f"Content chunk - finish_reason: {finish_reason}, content length: {len(delta.content or '')}")

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
