"""
OpenRouter provider implementation for chat completions.

This module implements the ChatProvider interface for OpenRouter,
which provides access to multiple AI models through a unified OpenAI-compatible API.

OpenRouter API Documentation: https://openrouter.ai/docs/api/reference/overview
OpenRouter Usage Accounting: https://openrouter.ai/docs/use-cases/usage-accounting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, List, Sequence

import aiohttp
import openai

if TYPE_CHECKING:
    from nodetool.workflows.processing_context import ProcessingContext

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    ImageModel,
    LanguageModel,
    Message,
    ModelPricing,
    Provider,
    ToolCall,
)
from nodetool.providers.base import register_provider
from nodetool.providers.openai_provider import OpenAIProvider
from nodetool.workflows.types import Chunk

log = get_logger(__name__)


@register_provider(Provider.OpenRouter)
class OpenRouterProvider(OpenAIProvider):
    """OpenRouter implementation of the ChatProvider interface.

    OpenRouter provides access to multiple AI models through an OpenAI-compatible API.
    This provider extends OpenAIProvider with OpenRouter-specific configuration.

    Key differences from OpenAI:
    1. Base URL: https://openrouter.ai/api/v1
    2. Additional headers: HTTP-Referer and X-Title for tracking
    3. Model listing via OpenRouter's models endpoint
    4. Different cost tracking mechanism

    For details, see: https://openrouter.ai/docs/api/reference/overview
    """

    has_code_interpreter: bool = False
    provider: Provider = Provider.OpenRouter

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["OPENROUTER_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the OpenRouter provider with client credentials.

        Reads ``OPENROUTER_API_KEY`` from environment.
        """
        assert "OPENROUTER_API_KEY" in secrets, "OPENROUTER_API_KEY is required"
        self.api_key = secrets["OPENROUTER_API_KEY"]
        self.client = None
        self.cost = 0.0
        log.debug("OpenRouterProvider initialized. API key present: True")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``OPENROUTER_API_KEY`` if available; otherwise empty.
        """
        return {"OPENROUTER_API_KEY": self.api_key} if self.api_key else {}

    def get_client(self) -> openai.AsyncClient:
        """Create and return an OpenRouter async client.

        Uses OpenAI SDK with OpenRouter's base URL and API key.

        Returns:
            An initialized ``openai.AsyncClient`` configured for OpenRouter.
        """
        log.debug("Creating OpenRouter async client")

        # Use ResourceScope's HTTP client if available
        from nodetool.runtime.resources import require_scope

        http_client = require_scope().get_http_client()

        # Configure client for OpenRouter with additional headers
        client = openai.AsyncClient(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            http_client=http_client,
            default_headers={
                "HTTP-Referer": "https://github.com/nodetool-ai/nodetool-core",
                "X-Title": "NodeTool",
            },
        )
        log.debug("OpenRouter async client created successfully")
        return client

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        OpenRouter supports function calling for most modern models.

        Args:
            model: Model identifier string.

        Returns:
            True if the model supports function calling, False otherwise.
        """
        log.debug(f"Checking tool support for model: {model}")

        model_lower = model.lower()

        # Models known to NOT support function calling
        no_tool_models = [
            "o1",  # OpenAI o1 models don't support tools
            "o3",  # OpenAI o3 models don't support tools
        ]

        for pattern in no_tool_models:
            if pattern in model_lower:
                log.debug(f"Model {model} does not support tool calling")
                return False

        # Most modern models on OpenRouter support function calling
        log.debug(f"Model {model} supports tool calling")
        return True

    async def get_available_language_models(self) -> list[LanguageModel]:
        """
        Get available OpenRouter models.

        Fetches models dynamically from the OpenRouter API if an API key is available.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of LanguageModel instances for OpenRouter
        """
        if not self.api_key:
            log.debug("No OpenRouter API key configured, returning empty model list")
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/nodetool-ai/nodetool-core",
                "X-Title": "NodeTool",
            }
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                session.get("https://openrouter.ai/api/v1/models") as response,
            ):
                if response.status != 200:
                    log.warning(f"Failed to fetch OpenRouter models: HTTP {response.status}")
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
                            provider=Provider.OpenRouter,
                        )
                    )
                log.debug(f"Fetched {len(models)} OpenRouter models")
                return models
        except Exception as e:
            log.error(f"Error fetching OpenRouter models: {e}")
            return []

    async def get_available_image_models(self) -> list[ImageModel]:
        """
        Get available OpenRouter image generation models.

        Fetches image models from the OpenRouter API that support image generation.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of ImageModel instances for OpenRouter image generation
        """
        from nodetool.metadata.types import ImageModel

        if not self.api_key:
            log.debug("No OpenRouter API key configured, returning empty image model list")
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/nodetool-ai/nodetool-core",
                "X-Title": "NodeTool",
            }
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                session.get("https://openrouter.ai/api/v1/models") as response,
            ):
                if response.status != 200:
                    log.warning(f"Failed to fetch OpenRouter models: HTTP {response.status}")
                    return []
                payload = await response.json()
                data = payload.get("data", [])

                models: list[ImageModel] = []
                for item in data:
                    model_id = item.get("id")
                    if not model_id:
                        continue

                    # Check if model supports image generation
                    # OpenRouter uses "image" in the architecture field or specific model families
                    architecture = item.get("architecture", {})
                    modality = architecture.get("modality") if isinstance(architecture, dict) else None

                    # Common image generation model patterns
                    is_image_model = (
                        modality == "image"
                        or "dall-e" in model_id.lower()
                        or "stable-diffusion" in model_id.lower()
                        or "flux" in model_id.lower()
                        or "midjourney" in model_id.lower()
                        or "imagen" in model_id.lower()
                    )

                    if is_image_model:
                        model_name = item.get("name", model_id)
                        models.append(
                            ImageModel(
                                id=model_id,
                                name=model_name,
                                provider=Provider.OpenRouter,
                            )
                        )
                log.debug(f"Fetched {len(models)} OpenRouter image models")
                return models
        except Exception as e:
            log.error(f"Error fetching OpenRouter image models: {e}")
            return []

    async def get_pricing(self, endpoint_ids: list[str] | None = None) -> list[ModelPricing]:
        """Get pricing information for OpenRouter models.

        Fetches models from the OpenRouter API and extracts pricing information.
        OpenRouter embeds pricing in the model metadata.

        Args:
            endpoint_ids: Optional list of specific model IDs to get pricing for.
                         If None, returns pricing for all available models.

        Returns:
            List of ModelPricing instances with pricing information.
        """
        if not self.api_key:
            log.debug("No OpenRouter API key configured, returning empty pricing list")
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/nodetool-ai/nodetool-core",
                "X-Title": "NodeTool",
            }
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                session.get("https://openrouter.ai/api/v1/models") as response,
            ):
                if response.status != 200:
                    log.warning(f"Failed to fetch OpenRouter models for pricing: HTTP {response.status}")
                    return []
                payload = await response.json()
                data = payload.get("data", [])

                pricing_list: list[ModelPricing] = []
                for item in data:
                    model_id = item.get("id")
                    if not model_id:
                        continue

                    # Skip if we're filtering and this model isn't in the list
                    if endpoint_ids and model_id not in endpoint_ids:
                        continue

                    # Extract pricing from the model data
                    pricing_data = item.get("pricing", {})
                    if not pricing_data:
                        continue

                    # OpenRouter uses per-token pricing (prompt/completion)
                    prompt_price = pricing_data.get("prompt", "0")
                    completion_price = pricing_data.get("completion", "0")
                    request_price = pricing_data.get("request", "0")
                    image_price = pricing_data.get("image", "0")

                    # Convert string prices to float
                    try:
                        prompt_price = float(prompt_price) if prompt_price else 0.0
                        completion_price = float(completion_price) if completion_price else 0.0
                        request_price = float(request_price) if request_price else 0.0
                        image_price = float(image_price) if image_price else 0.0
                    except (ValueError, TypeError):
                        prompt_price = 0.0
                        completion_price = 0.0
                        request_price = 0.0
                        image_price = 0.0

                    # Use completion price as the primary unit_price for LLMs
                    unit_price = completion_price if completion_price else prompt_price

                    pricing_list.append(
                        ModelPricing(
                            endpoint_id=model_id,
                            provider=Provider.OpenRouter,
                            unit_price=unit_price,
                            unit="token",
                            currency="USD",
                            prompt_price=prompt_price if prompt_price else None,
                            completion_price=completion_price if completion_price else None,
                            request_price=request_price if request_price else None,
                            image_price=image_price if image_price else None,
                        )
                    )

                log.debug(f"Fetched {len(pricing_list)} pricing entries from OpenRouter")
                return pricing_list
        except Exception as e:
            log.error(f"Error fetching OpenRouter pricing: {e}")
            return []

    async def text_to_image(
        self,
        params: Any,  # TextToImageParams
        timeout_s: int | None = None,
        context: Any = None,  # ProcessingContext
        node_id: str | None = None,
    ) -> bytes:
        """Generate an image from a text prompt using OpenRouter's image generation models.

        OpenRouter supports multiple image generation models including DALL-E, Stable Diffusion,
        and Flux through an OpenAI-compatible API.

        Args:
            params: Text-to-image generation parameters including:
                - prompt: Text description of the desired image
                - model: ImageModel with OpenRouter model ID (e.g., "openai/dall-e-3")
                - width: Desired width (optional, model-dependent)
                - height: Desired height (optional, model-dependent)
                - negative_prompt: What to avoid in the image (optional)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            node_id: Optional node ID for progress tracking

        Returns:
            Raw image bytes (PNG, JPEG, etc.)

        Raises:
            ValueError: If required parameters are missing or invalid
            RuntimeError: If generation fails
        """
        import base64

        from nodetool.io.uri_utils import fetch_uri_bytes_and_mime

        if not params.prompt:
            raise ValueError("The input prompt cannot be empty.")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for image generation.")

        model_id = params.model.id
        if not model_id:
            raise ValueError("A text-to-image model with a valid id must be specified for image generation.")

        prompt = params.prompt.strip()
        if params.negative_prompt:
            prompt = f"{prompt}\n\nDo not include: {params.negative_prompt.strip()}"

        log.debug(f"Generating image with OpenRouter model: {model_id}")
        log.debug(f"Prompt: {prompt[:100]}...")

        try:
            request_timeout = timeout_s if timeout_s and timeout_s > 0 else 120
            client = self.get_client()

            # Build request parameters
            request_params: dict[str, Any] = {
                "model": model_id,
                "prompt": prompt,
                "n": 1,
                "timeout": request_timeout,
            }

            # Add size if both width and height are specified
            if params.width and params.height:
                if params.width <= 0 or params.height <= 0:
                    raise ValueError("width and height must be positive integers.")
                # OpenRouter may support size parameter for compatible models
                size = self._resolve_image_size(int(params.width), int(params.height))
                if size:
                    request_params["size"] = size

            log.debug(f"Making image generation request with params: {request_params}")

            # Call the API and handle potential awaitable
            create_result = client.images.generate(**request_params)
            import inspect

            if inspect.isawaitable(create_result):
                response = await create_result
            else:
                response = create_result

            data = response.data or []
            if len(data) == 0:
                raise RuntimeError("OpenRouter image generation returned no data.")

            image_entry = data[0]
            image_bytes: bytes | None = None

            # Try to get image from base64 first
            b64_data = image_entry.b64_json
            if b64_data:
                image_bytes = base64.b64decode(b64_data)
                log.debug("Retrieved image from base64 response")
            else:
                # Fall back to URL
                image_url = image_entry.url
                if image_url:
                    log.debug(f"Fetching image from URL: {image_url}")
                    _, image_bytes = await fetch_uri_bytes_and_mime(image_url)

            if not image_bytes:
                raise RuntimeError("OpenRouter image generation returned no image bytes.")

            log.debug(f"Successfully generated image ({len(image_bytes)} bytes)")
            return image_bytes

        except openai.APIStatusError as api_error:
            log.error(
                "OpenRouter text-to-image generation failed (status=%s): %s",
                api_error.status_code,
                api_error.message,
            )
            raise RuntimeError(
                f"OpenRouter text-to-image generation failed with status {api_error.status_code}: {api_error.message}"
            ) from api_error
        except Exception as exc:
            log.error(f"OpenRouter text-to-image generation failed: {exc}")
            raise RuntimeError(f"OpenRouter text-to-image generation failed: {exc}") from exc

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
        """Generate a non-streaming completion from OpenRouter with cost tracking.

        This method extends OpenAI's generate_message to include OpenRouter's
        usage accounting, which returns cost information in USD per request.

        Args:
            messages: The message history
            model: The model to use
            tools: Optional tools to provide to the model
            max_tokens: The maximum number of tokens to generate
            json_schema: Optional JSON schema for structured output
            temperature: Optional sampling temperature
            top_p: Optional nucleus sampling parameter
            presence_penalty: Optional presence penalty
            frequency_penalty: Optional frequency penalty
            response_format: The format of the response
            **kwargs: Additional arguments to pass to the API

        Returns:
            A Message object containing the model's response with cost tracking
        """
        import json

        log.debug(f"Generating non-streaming message for model: {model}")
        log.debug(f"Non-streaming with {len(messages)} messages, {len(tools)} tools")

        if not messages:
            raise ValueError("messages must not be empty")

        request_kwargs: dict[str, Any] = {
            "max_completion_tokens": max_tokens,
            # Enable OpenRouter usage accounting to get cost information
            "extra_body": {"usage": {"include": True}},
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
            log.debug(f"Converted {len(converted_messages)} messages for O-series model")

        self._log_api_request("chat", messages, **request_kwargs)

        if len(tools) > 0:
            request_kwargs["tools"] = self.format_tools(tools)
            log.debug(f"Added {len(tools)} tools to request")

        log.debug(f"Converting {len(messages)} messages to OpenAI format")
        openai_messages = [await self.convert_message(m) for m in messages]
        log.debug("Making non-streaming API call to OpenRouter")

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
        log.debug("Received response from OpenRouter API")

        # Extract OpenRouter-specific cost
        message_cost = 0.0
        if completion.usage:
            log.debug("Processing usage statistics")
            # Extract OpenRouter cost from usage (in USD)
            # OpenRouter returns cost in the usage object
            if hasattr(completion.usage, "cost") and completion.usage.cost is not None:
                message_cost = float(completion.usage.cost)
                self.cost += message_cost
                log.debug(f"OpenRouter cost for this request: ${message_cost:.6f} USD")
            else:
                log.debug("No cost information returned from OpenRouter")

            log.debug(f"Total cost: ${self.cost:.6f}")

        choice = completion.choices[0]
        response_message = choice.message
        log.debug(f"Response content length: {len(response_message.content or '')}")

        def try_parse_args(args: Any) -> Any:
            try:
                return json.loads(args)
            except Exception:
                log.warning("Error parsing tool call arguments: %s", args)
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
        """Stream assistant deltas and tool calls from OpenRouter with cost tracking.

        This method extends OpenAI's generate_messages to include OpenRouter's
        usage accounting in streaming mode.

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

        # Convert system messages to user messages for O1/O3 models
        _kwargs: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Enable OpenRouter usage accounting for streaming
            "extra_body": {"usage": {"include": True}},
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

        if kwargs.get("audio"):
            _kwargs["audio"] = kwargs.get("audio")
            _kwargs["modalities"] = ["text", "audio"]
            if not kwargs.get("audio"):
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
                    log.debug("Converting system message to user message for O-series model")
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
            log.debug(f"Converted {len(converted_messages)} messages for O-series model")

        self._log_api_request(
            "chat_stream",
            messages,
            **_kwargs,
        )

        log.debug(f"Converting {len(messages)} messages to OpenAI format")
        openai_messages = [await self.convert_message(m) for m in messages]
        log.debug("Making streaming API call to OpenRouter")

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

            # Track OpenRouter cost from streaming usage
            if chunk.usage:
                log.debug("Processing usage statistics from chunk")
                # Extract OpenRouter cost from streaming usage
                if hasattr(chunk.usage, "cost") and chunk.usage.cost is not None:
                    message_cost = float(chunk.usage.cost)
                    self.cost += message_cost
                    log.debug(f"OpenRouter streaming cost: ${message_cost:.6f} USD")

                log.debug(f"Total cost: ${self.cost:.6f}")

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
