"""
HuggingFace provider implementation for chat completions.

This module implements the ChatProvider interface for HuggingFace models using their
Inference Providers API with the AsyncInferenceClient from huggingface_hub.
"""

import asyncio
import base64
import json
import logging
import os
import traceback
from typing import Any, AsyncGenerator, List, Literal, Sequence
from weakref import WeakKeyDictionary

import aiohttp
import httpx
import numpy as np
import requests
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel

from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.io.media_fetch import fetch_uri_bytes_and_mime_sync
from nodetool.media.image.image_utils import image_data_to_base64_jpeg
from nodetool.metadata.types import (
    ImageModel,
    LanguageModel,
    Message,
    MessageImageContent,
    MessageTextContent,
    Provider,
    ToolCall,
    TTSModel,
    VideoModel,
)
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.types.model import CachedFileInfo
from nodetool.workflows.base_node import ApiKeyMissingError
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

log = get_logger(__name__)

# Simple cache for context window lookups (public models only)
_context_window_cache: dict[str, int | None] = {}

PROVIDER_T = Literal[
    "black-forest-labs",
    "cerebras",
    "cohere",
    "fal-ai",
    "featherless-ai",
    "fireworks-ai",
    "groq",
    "hf-inference",
    "hyperbolic",
    "nebius",
    "novita",
    "nscale",
    "openai",
    "replicate",
    "sambanova",
    "together",
]

if hasattr(CachedFileInfo, "model_rebuild"):
    CachedFileInfo.model_rebuild()

# Provider mapping for HuggingFace Hub API
HF_PROVIDER_MAPPING = {
    "cerebras": Provider.HuggingFaceCerebras,
    "cohere": Provider.HuggingFaceCohere,
    "fal-ai": Provider.HuggingFaceFalAI,
    "featherless-ai": Provider.HuggingFaceFeatherlessAI,
    "fireworks-ai": Provider.HuggingFaceFireworksAI,
    "groq": Provider.HuggingFaceGroq,
    "hf-inference": Provider.HuggingFaceHFInference,
    "hyperbolic": Provider.HuggingFaceHyperbolic,
    "nebius": Provider.HuggingFaceNebius,
    "novita": Provider.HuggingFaceNovita,
    "nscale": Provider.HuggingFaceNscale,
    "openai": Provider.HuggingFaceOpenAI,
    "replicate": Provider.HuggingFaceReplicate,
    "scaleway": Provider.HuggingFaceScaleway,
    "sambanova": Provider.HuggingFaceSambanova,
    "together": Provider.HuggingFaceTogether,
    "zai-org": Provider.HuggingFaceZAI,
}


def get_remote_context_window(model_id: str, token: str | None = None) -> int | None:
    """Fetch context window info from the model's Hugging Face config, if available."""
    # Note: We don't cache when token is provided since cache keys can't include tokens
    # For public models (no token), we can use a simple cache
    cache_key = f"{model_id}:{bool(token)}"

    # Try to get from cache if no token (public model)
    if not token:
        cached = _context_window_cache.get(cache_key)
        if cached is not None:
            return cached

    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(url, headers=headers, timeout=5)
    except Exception as exc:  # pragma: no cover - network failure fallback
        log.debug("Failed to fetch remote config for %s: %s", model_id, exc)
        return None

    if response.status_code != 200:
        log.debug(
            "Remote config request returned %s for model %s",
            response.status_code,
            model_id,
        )
        return None

    try:
        cfg = response.json()
    except ValueError as exc:  # pragma: no cover - invalid JSON
        log.debug("Invalid JSON in remote config for %s: %s", model_id, exc)
        return None

    for key in (
        "max_position_embeddings",
        "n_positions",
        "sequence_length",
        "context_length",
    ):
        if key in cfg:
            value = cfg[key]
            try:
                result = int(value)
                # Cache result for public models (no token)
                if not token:
                    _context_window_cache[cache_key] = result
                return result
            except (TypeError, ValueError):
                log.debug(
                    "Context length key %s in %s was non-integer: %s",
                    key,
                    model_id,
                    value,
                )
                continue

    # Cache None result for public models to avoid repeated failed requests
    if not token:
        _context_window_cache[cache_key] = None
    return None


def _message_contains_media(message: Message) -> tuple[bool, str]:
    """
    Check if a message contains media content and return the type of media found.

    Args:
        message: The message to check for media content

    Returns:
        Tuple of (has_media, media_type) where media_type is "image", "audio", or "none"
    """
    if not message.content:
        return False, "none"

    if isinstance(message.content, str):
        return False, "none"

    if isinstance(message.content, list):
        for part in message.content:
            if isinstance(part, MessageImageContent):
                return True, "image"
            # Check for audio content if MessageAudioContent exists
            if hasattr(part, "__class__") and "Audio" in part.__class__.__name__:
                return True, "audio"

    return False, "none"


async def fetch_image_models_from_hf_provider(
    provider: str, pipeline_tag: str, token: str | None = None
) -> List[ImageModel]:
    """
    Fetch image models from HuggingFace Hub API for a specific provider.

    Args:
        provider: The provider value (e.g., "fal-ai", "replicate", etc.)
        pipeline_tag: The pipeline tag value (e.g., "text-to-image")
        token: HuggingFace API token for authentication

    Returns:
        List of ImageModel instances
    """
    try:
        url = f"https://huggingface.co/api/models?inference_provider={provider}&pipeline_tag={pipeline_tag}&limit=100"

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url) as response:
            if response.status == 200:
                models_data = await response.json()

                models = []
                for model_data in models_data:
                    model_id = model_data.get("id", "")
                    if model_id:
                        # Use the model name from the API if available, otherwise use the ID
                        model_name = model_data.get("name") or model_id.split("/")[-1] if "/" in model_id else model_id

                        # Get the appropriate provider enum value
                        provider_enum = HF_PROVIDER_MAPPING.get(provider)
                        if provider_enum is None:
                            log.warning(f"Unknown image provider: {provider}, skipping model: {model_id}")
                            continue

                        task = (
                            "text_to_image"
                            if pipeline_tag == "text-to-image"
                            else "image_to_image"
                            if pipeline_tag == "image-to-image"
                            else None
                        )
                        supported = [task] if task else []
                        models.append(
                            ImageModel(
                                id=model_id,
                                name=model_name,
                                provider=provider_enum,
                                supported_tasks=supported,
                            )
                        )

                log.debug(f"Fetched {len(models)} image models from HF provider: {provider}")
                return models
            else:
                log.warning(f"Failed to fetch image models for provider {provider}: HTTP {response.status}")
                return []

    except Exception as e:
        log.error(f"Error fetching image models for provider {provider}: {e}")
        return []


async def fetch_tts_models_from_hf_provider(provider: str, pipeline_tag: str) -> List[TTSModel]:
    """
    Fetch TTS models from HuggingFace Hub API for a specific provider.

    Args:
        provider: The provider value (e.g., "hf-inference")
        pipeline_tag: The pipeline tag value (e.g., "text-to-speech")

    Returns:
        List of TTSModel instances
    """
    try:
        url = f"https://huggingface.co/api/models?inference_provider={provider}&pipeline_tag={pipeline_tag}&limit=1000"

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url) as response:
            if response.status == 200:
                models_data = await response.json()

                models = []
                for model_data in models_data:
                    model_id = model_data.get("id", "")
                    if model_id:
                        # Use the model name from the API if available, otherwise use the ID
                        model_name = model_data.get("name") or model_id.split("/")[-1] if "/" in model_id else model_id

                        # Get the appropriate provider enum value
                        provider_enum = HF_PROVIDER_MAPPING.get(provider)
                        if provider_enum is None:
                            log.warning(f"Unknown TTS provider: {provider}, skipping model: {model_id}")
                            continue

                        models.append(
                            TTSModel(
                                id=model_id,
                                name=model_name,
                                provider=provider_enum,
                                voices=[],  # HF TTS models typically don't have named voices
                            )
                        )

                log.debug(f"Fetched {len(models)} TTS models from HF provider: {provider}")
                return models
            else:
                log.warning(f"Failed to fetch TTS models for provider {provider}: HTTP {response.status}")
                return []

    except Exception as e:
        log.error(f"Error fetching TTS models for provider {provider}: {e}")
        return []


async def fetch_video_models_from_hf_provider(provider: str, pipeline_tag: str) -> List[VideoModel]:
    """
    Fetch video models from HuggingFace Hub API for a specific provider.

    Args:
        provider: The provider value (e.g., "fal-ai", "replicate", "novita")
        pipeline_tag: The pipeline tag value (e.g., "text-to-video")

    Returns:
        List of VideoModel instances
    """
    try:
        url = f"https://huggingface.co/api/models?inference_provider={provider}&pipeline_tag={pipeline_tag}&limit=100"

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url) as response:
            if response.status == 200:
                models_data = await response.json()

                models = []
                for model_data in models_data:
                    model_id = model_data.get("id", "")
                    if model_id:
                        # Use the model name from the API if available, otherwise use the ID
                        model_name = model_data.get("name") or model_id.split("/")[-1] if "/" in model_id else model_id

                        # Get the appropriate provider enum value
                        provider_enum = HF_PROVIDER_MAPPING.get(provider)
                        if provider_enum is None:
                            log.warning(f"Unknown video provider: {provider}, skipping model: {model_id}")
                            continue

                        task = (
                            "text_to_video"
                            if pipeline_tag == "text-to-video"
                            else "image_to_video"
                            if pipeline_tag == "image-to-video"
                            else None
                        )
                        supported = [task] if task else []
                        models.append(
                            VideoModel(
                                id=model_id,
                                name=model_name,
                                provider=provider_enum,
                                supported_tasks=supported,
                            )
                        )

                log.debug(f"Fetched {len(models)} video models from HF provider: {provider}")
                return models
            else:
                log.warning(f"Failed to fetch video models for provider {provider}: HTTP {response.status}")
                return []

    except Exception as e:
        log.error(f"Error fetching video models for provider {provider}: {e}")
        return []


async def fetch_models_from_hf_provider(provider: str, pipeline_tag: str) -> List[LanguageModel]:
    """
    Fetch language models from HuggingFace Hub API for a specific provider.

    Args:
        provider: The provider value (e.g., "groq", "cerebras", etc.)
        pipeline_tag: The pipeline tag value (e.g., "text-generation")

    Returns:
        List of LanguageModel instances
    """
    try:
        url = f"https://huggingface.co/api/models?inference_provider={provider}&pipeline_tag={pipeline_tag}&limit=1000"

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url) as response:
            if response.status == 200:
                models_data = await response.json()

                models = []
                for model_data in models_data:
                    model_id = model_data.get("id", "")
                    if model_id:
                        # Use the model name from the API if available, otherwise use the ID
                        model_name = model_data.get("name") or model_id.split("/")[-1] if "/" in model_id else model_id

                        # Get the appropriate provider enum value
                        provider_enum = HF_PROVIDER_MAPPING.get(provider)
                        if provider_enum is None:
                            log.warning(f"Unknown provider: {provider}, skipping model: {model_id}")
                            continue

                        models.append(
                            LanguageModel(
                                id=model_id,
                                name=model_name,
                                provider=provider_enum,
                            )
                        )

                # Preserve API order to match test expectations
                log.debug(f"Fetched {len(models)} language models from HF provider: {provider}")
                return models
            else:
                log.warning(f"Failed to fetch models for provider {provider}: HTTP {response.status}")
                return []

    except Exception as e:
        log.error(f"Error fetching models for provider {provider}: {e}")
        return []


@register_provider(Provider.HuggingFaceGroq, inference_provider="groq")
@register_provider(Provider.HuggingFaceCerebras, inference_provider="cerebras")
@register_provider(Provider.HuggingFaceCohere, inference_provider="cohere")
@register_provider(Provider.HuggingFaceFalAI, inference_provider="fal-ai")
@register_provider(Provider.HuggingFaceFeatherlessAI, inference_provider="featherless-ai")
@register_provider(Provider.HuggingFaceFireworksAI, inference_provider="fireworks-ai")
@register_provider(Provider.HuggingFaceHFInference, inference_provider="hf-inference")
@register_provider(Provider.HuggingFaceHyperbolic, inference_provider="hyperbolic")
@register_provider(Provider.HuggingFaceNebius, inference_provider="nebius")
@register_provider(Provider.HuggingFaceNovita, inference_provider="novita")
@register_provider(Provider.HuggingFaceNscale, inference_provider="nscale")
@register_provider(Provider.HuggingFaceOpenAI, inference_provider="openai")
@register_provider(Provider.HuggingFaceReplicate, inference_provider="replicate")
@register_provider(Provider.HuggingFaceSambanova, inference_provider="sambanova")
@register_provider(Provider.HuggingFaceTogether, inference_provider="together")
@register_provider(Provider.HuggingFaceScaleway, inference_provider="scaleway")
@register_provider(Provider.HuggingFaceZAI, inference_provider="zai-org")
class HuggingFaceProvider(BaseProvider):
    """
    HuggingFace implementation of the Provider interface.

    Uses the HuggingFace Inference Providers API via AsyncInferenceClient from huggingface_hub.
    This provider works with various inference providers (Cerebras, Cohere, Fireworks, etc.)
    that support the OpenAI-compatible chat completion format.

    HuggingFace's message structure follows the OpenAI format:

    1. Message Format:
       - Each message is a dict with "role" and "content" fields

       - Role can be: "system", "user", "assistant", or "tool"
       - Content contains the message text (string) or content blocks (for multimodal)

    2. Tool Calls:
       - When a model wants to call a tool, the response includes a "tool_calls" field
       - Each tool call contains:
         - "id": A unique identifier for the tool call
         - "function": An object with "name" and "arguments" (JSON string)

    3. Response Structure:
       - response.choices[0].message contains the model's response
       - It includes fields like "role", "content", and optionally "tool_calls"
       - response.usage contains token usage statistics

    For more details, see: https://huggingface.co/docs/hugs/en/guides/function-calling#using-tools-function-definitions
    """

    provider_name: str = "huggingface"

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["HF_TOKEN"]

    def __init__(self, secrets: dict[str, str], inference_provider: PROVIDER_T):
        """Initialize the HuggingFace provider with AsyncInferenceClient."""
        super().__init__()
        if "HF_TOKEN" not in secrets or not secrets["HF_TOKEN"]:
            # Fallback to environment variable if not in secrets
            if os.environ.get("HF_TOKEN"):
                secrets["HF_TOKEN"] = os.environ.get("HF_TOKEN")
            else:
                log.warning("HF_TOKEN not found in secrets, HuggingFace provider will not be initialized")
                raise ValueError("HF_TOKEN is required but not provided")
        self.api_key = secrets["HF_TOKEN"]
        self.inference_provider = inference_provider
        self.provider_name = f"huggingface_{inference_provider}"

        self.provider_name = f"huggingface_{inference_provider}"

        # Cache clients per event loop to avoid sharing httpx sessions across threads/loops
        self._clients: WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncInferenceClient] = WeakKeyDictionary()

    def get_client(self) -> AsyncInferenceClient:
        """Return a HuggingFace AsyncInferenceClient for the current event loop."""
        loop = asyncio.get_running_loop()
        if loop not in self._clients:
            log.debug(f"Creating AsyncInferenceClient for loop {id(loop)} with provider: {self.inference_provider}")
            self._clients[loop] = AsyncInferenceClient(
                api_key=self.api_key,
                provider=self.inference_provider,
            )
        return self._clients[loop]

        self.cost = 0.0
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        log.debug(f"HuggingFaceProvider initialized with provider: {self.inference_provider or 'default'}")

    async def __aenter__(self):
        """Async context manager entry."""
        log.debug("Entering async context manager")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - properly close client."""
        log.debug("Exiting async context manager")
        await self.close()

    async def close(self):
        """Close the async client for the current loop properly."""
        log.debug("Closing async client")
        try:
            loop = asyncio.get_running_loop()
            if loop in self._clients:
                client = self._clients[loop]
                if hasattr(client, "close"):
                    await client.close()  # type: ignore
                    log.debug("Async client closed successfully")
                    # Remove from cache
                    del self._clients[loop]
                else:
                    log.debug("Client does not have close method")
        except RuntimeError:
            log.debug("No running loop, cannot close client")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        env_vars = {}
        if self.api_key:
            env_vars["HF_TOKEN"] = self.api_key
        if hasattr(self, "inference_provider") and self.inference_provider:
            env_vars["HUGGINGFACE_PROVIDER"] = self.inference_provider
        log.debug(f"Container environment variables: {list(env_vars.keys())}")
        return env_vars

    async def get_available_language_models(self) -> List[LanguageModel]:
        """
        Get available HuggingFace models for this inference provider.

        Fetches models from the HuggingFace API based on the inference provider.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of LanguageModel instances for HuggingFace
        """
        if not self.api_key:
            log.debug("No HuggingFace API key configured, returning empty model list")
            return []

        try:
            assert self.inference_provider is not None, "Inference provider is not set"
            models = await fetch_models_from_hf_provider(self.inference_provider, "text-generation")
            log.debug(f"Fetched {len(models)} models for HF inference provider: {self.inference_provider}")
            return models
        except Exception as e:
            log.error(f"Error fetching HuggingFace models for provider {self.inference_provider}: {e}")
            return []

    def convert_message(self, message: Message) -> dict:
        """Convert an internal message to HuggingFace's OpenAI-compatible format."""
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
                "role": "tool",
                "content": content,
                "tool_call_id": message.tool_call_id,
            }
        elif message.role == "system":
            log.debug("Converting system message")
            return {
                "role": "system",
                "content": str(message.content),
            }
        elif message.role == "user":
            log.debug("Converting user message")
            if isinstance(message.content, str):
                log.debug("User message has string content")
                return {"role": "user", "content": message.content}
            elif message.content is not None:
                # Handle multimodal content
                log.debug(f"Converting {len(message.content)} content parts")
                content = []
                for part in message.content:
                    if isinstance(part, MessageTextContent):
                        content.append({"type": "text", "text": part.text})
                    elif isinstance(part, MessageImageContent):
                        # Always convert images to base64 data URI for consistency
                        # (handles local storage URLs and avoids timeout issues)
                        if part.image.data:
                            # Use raw bytes directly if available
                            log.debug("Converting raw image data to JPEG base64")
                            b64 = image_data_to_base64_jpeg(part.image.data)
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                }
                            )
                        elif part.image.uri:
                            # Fetch image and convert to data URI
                            uri = part.image.uri
                            log.debug(f"Fetching image and converting to base64: {uri[:50]}...")
                            mime, data = fetch_uri_bytes_and_mime_sync(uri)
                            b64 = base64.b64encode(data).decode("utf-8")
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                                }
                            )
                log.debug(f"Converted to {len(content)} content parts")
                return {"role": "user", "content": content}
            else:
                log.debug("User message has no content")
                return {"role": "user", "content": ""}
        elif message.role == "assistant":
            log.debug("Converting assistant message")
            result: dict[str, Any] = {"role": "assistant"}

            if message.content:
                result["content"] = str(message.content)
                log.debug("Assistant message has content")
            else:
                log.debug("Assistant message has no content")

            if message.tool_calls:
                log.debug(f"Assistant message has {len(message.tool_calls)} tool calls")
                result["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": (
                                json.dumps(tool_call.args) if isinstance(tool_call.args, dict) else str(tool_call.args)
                            ),
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            else:
                log.debug("Assistant message has no tool calls")

            return result
        else:
            log.error(f"Unsupported message role: {message.role}")
            raise ValueError(f"Unsupported message role: {message.role}")

    def format_tools(self, tools: Sequence[Tool]) -> list[dict]:
        """Format tools for HuggingFace API (OpenAI-compatible format)."""
        log.debug(f"Formatting {len(tools)} tools for HuggingFace API")
        formatted_tools = []
        for tool in tools:
            log.debug(f"Formatting tool: {tool.name}")
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
        log.debug(f"Formatted {len(formatted_tools)} tools")
        return formatted_tools

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        """
        Generate a single message completion from HuggingFace using AsyncInferenceClient.

        Args:
            messages: Sequence of Message objects representing the conversation
            model: Model identifier (can be repo_id like "microsoft/Phi-4-mini-flash-reasoning")
            tools: Available tools for the model to use
            max_tokens: Maximum number of tokens to generate
            response_format: Format of the response
            **kwargs: Additional provider-specific parameters

        Returns:
            A message returned by the provider.
        """
        log.debug(f"Generating message for model: {model}")
        log.debug(f"Input: {len(messages)} messages, {len(tools)} tools, max_tokens: {max_tokens}")

        if len(messages) == 0:
            raise ValueError("Empty messsages")

        # Convert messages to HuggingFace format
        log.debug("Converting messages to HuggingFace format")
        hf_messages = []
        for message in messages:
            converted = self.convert_message(message)
            if converted:  # Skip None messages
                hf_messages.append(converted)
        log.debug(f"Converted to {len(hf_messages)} HuggingFace messages")

        # Prepare request parameters - using HuggingFace's chat_completion method
        request_params: dict[str, Any] = {
            "messages": hf_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        log.debug(f"Request params: max_tokens={max_tokens}, stream=False")

        # Add tools if provided (following HuggingFace docs format)
        if tools:
            request_params["tools"] = self.format_tools(tools)
            request_params["tool_choice"] = "auto"  # As per HF docs
            log.debug("Added tools and tool_choice to request")

        # Add response format if specified
        if response_format:
            request_params["response_format"] = response_format
            log.debug("Added response format to request")

        max_retries = 3
        base_delay = 1.0
        log.debug(f"Starting API call with max_retries={max_retries}")

        completion: Any | None = None

        for attempt in range(max_retries + 1):
            try:
                log.debug(f"API call attempt {attempt + 1}/{max_retries + 1}")
                client = self.get_client()
                completion = await client.chat_completion(model=model, **request_params)
                log.debug("API call successful")
                break
            except httpx.HTTPStatusError as e:
                log.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                status = getattr(getattr(e, "response", None), "status_code", None)
                body_text = getattr(getattr(e, "response", None), "text", None)

                # Do not retry on client-side errors (4xx), including 429, 413, 404.
                if isinstance(status, int) and 400 <= status < 500:
                    log.error(f"Non-retryable client error {status}; aborting retries")

                    if status == 422:
                        has_media = any(_message_contains_media(msg)[0] for msg in messages)
                        if has_media:
                            media_types = [
                                _message_contains_media(msg)[1] for msg in messages if _message_contains_media(msg)[0]
                            ]
                            media_str = ", ".join(set(media_types))
                            raise httpx.HTTPStatusError(
                                (
                                    f"422 Model '{model}' cannot process {media_str} content. "
                                    f"The model may not support multimodal input or the {media_str} format is not supported. "
                                    f"Original error: {body_text or str(e)}"
                                ),
                                request=e.request,
                                response=e.response,
                            ) from e

                        raise httpx.HTTPStatusError(
                            (
                                f"422 Model '{model}' received unprocessable input. "
                                f"The model may not support the provided parameters or content format. "
                                f"Original error: {body_text or str(e)}"
                            ),
                            request=e.request,
                            response=e.response,
                        ) from e

                    raise httpx.HTTPStatusError(
                        f"{status} {body_text or str(e)}",
                        request=e.request,
                        response=e.response,
                    ) from e

                if attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    log.debug(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue

                log.error(f"All {max_retries + 1} attempts failed")
                traceback.print_exc()
                if isinstance(status, int):
                    raise httpx.HTTPStatusError(
                        f"{status} {body_text or str(e)}",
                        request=e.request,
                        response=e.response,
                    ) from e
                raise
            except Exception as e:
                log.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    log.debug(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue

                log.error(f"All {max_retries + 1} attempts failed")
                traceback.print_exc()
                raise

        if completion is None:
            raise RuntimeError("HuggingFace chat completion did not return a response")

        # Update usage statistics if available
        if hasattr(completion, "usage") and completion.usage:
            log.debug("Processing usage statistics")
            self.usage["prompt_tokens"] = completion.usage.prompt_tokens or 0
            self.usage["completion_tokens"] = completion.usage.completion_tokens or 0
            self.usage["total_tokens"] = completion.usage.total_tokens or 0
            log.debug(f"Updated usage: {self.usage}")

        # Extract the response message
        choice = completion.choices[0]
        message_data = choice.message
        log.debug(f"Response content length: {len(message_data.content or '')}")

        # Create the response message
        response_message = Message(
            role="assistant",
            content=message_data.content or "",
        )

        # Handle tool calls if present
        if hasattr(message_data, "tool_calls") and message_data.tool_calls:
            log.debug(f"Processing {len(message_data.tool_calls)} tool calls")
            tool_calls = []
            for tool_call in message_data.tool_calls:
                function = tool_call.function
                try:
                    # Parse arguments - they might be JSON string or dict
                    args = function.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                        log.debug(f"Parsed JSON arguments for tool: {function.name}")
                    else:
                        log.debug(f"Using dict arguments for tool: {function.name}")
                except (json.JSONDecodeError, AttributeError) as e:
                    log.warning(f"Failed to parse arguments for tool {function.name}: {e}")
                    args = {}

                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=function.name,
                        args=args,
                    )
                )
            response_message.tool_calls = tool_calls
            log.debug(f"Added {len(tool_calls)} tool calls to response")
        else:
            log.debug("Response contains no tool calls")

        log.debug("Returning generated message")
        return response_message

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """
        Generate message completions from HuggingFace, yielding chunks or tool calls.

        Uses AsyncInferenceClient's streaming capability for real-time token generation.

        Args:
            messages: Sequence of Message objects representing the conversation
            model: Model identifier
            tools: Available tools for the model to use
            max_tokens: Maximum number of tokens to generate
            response_format: Format of the response
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunk objects with content and completion status or ToolCall objects
        """
        log.debug(f"Starting streaming generation for model: {model}")
        log.debug(f"Streaming with {len(messages)} messages, {len(tools)} tools")

        # Convert messages to HuggingFace format
        log.debug("Converting messages to HuggingFace format")
        hf_messages = []
        for message in messages:
            converted = self.convert_message(message)
            if converted:  # Skip None messages
                hf_messages.append(converted)
        log.debug(f"Converted to {len(hf_messages)} HuggingFace messages")

        # Prepare request parameters for streaming
        request_params: dict[str, Any] = {
            "messages": hf_messages,
            "max_tokens": max_tokens,
            "stream": True,  # Enable streaming
        }
        log.debug("Prepared streaming request parameters")

        # Add tools if provided
        if tools:
            request_params["tools"] = self.format_tools(tools)
            request_params["tool_choice"] = "auto"
            log.debug("Added tools to streaming request")

        # Add response format if specified
        if response_format:
            request_params["response_format"] = response_format
            log.debug("Added response format to streaming request")

        # Create streaming completion using chat_completion method
        log.debug("Starting streaming API call")
        stream = await self.client.chat_completion(model=model, **request_params)

        # Track tool calls during streaming
        accumulated_tool_calls = {}
        chunk_count = 0

        try:
            async for chunk in stream:
                chunk_count += 1

                if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                    log.debug("Updating usage stats from streaming chunk")
                    usage = chunk.usage  # type: ignore[attr-defined]
                    self.usage["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                    self.usage["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
                    self.usage["total_tokens"] = getattr(usage, "total_tokens", 0) or 0
                    log.debug(f"Updated usage: {self.usage}")

                choices = getattr(chunk, "choices", None)
                if not choices:
                    log.debug("Chunk has no choices, skipping")
                    continue

                choice = choices[0]
                delta = getattr(choice, "delta", None)
                log.debug(f"Processing delta with finish_reason: {choice.finish_reason}")

                # Only yield content if we are not currently accumulating tool calls
                # This prevents "noise" (garbage tokens) that some providers emit during tool calling
                if delta and getattr(delta, "content", None) and not accumulated_tool_calls:
                    yield Chunk(
                        content=delta.content,
                        done=choice.finish_reason == "stop",
                    )

                if delta and getattr(delta, "tool_calls", None):
                    for tool_call_delta in delta.tool_calls:
                        index = getattr(tool_call_delta, "index", 0)

                        if index not in accumulated_tool_calls:
                            accumulated_tool_calls[index] = {
                                "id": tool_call_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                            log.debug(f"Created new tool call at index {index}")

                        if tool_call_delta.id:
                            accumulated_tool_calls[index]["id"] = tool_call_delta.id
                            log.debug(f"Set tool call ID: {tool_call_delta.id}")

                        function_delta = getattr(tool_call_delta, "function", None)
                        if function_delta:
                            if getattr(function_delta, "name", None):
                                accumulated_tool_calls[index]["name"] = function_delta.name or ""
                                log.debug(f"Set tool call name: {function_delta.name}")
                            if getattr(function_delta, "arguments", None):
                                accumulated_tool_calls[index]["arguments"] += function_delta.arguments or ""
                                log.debug(
                                    "Added arguments to tool call: %s chars",
                                    len(function_delta.arguments or ""),
                                )

                if choices and choice.finish_reason == "tool_calls" and accumulated_tool_calls:
                    log.debug(
                        "Streaming complete with %d tool calls",
                        len(accumulated_tool_calls),
                    )
                    for tool_call_data in accumulated_tool_calls.values():
                        try:
                            args = json.loads(tool_call_data["arguments"])
                            log.debug(
                                "Parsed arguments for tool: %s",
                                tool_call_data["name"],
                            )
                        except json.JSONDecodeError as err:
                            log.warning(
                                "Failed to parse arguments for tool %s: %s",
                                tool_call_data["name"],
                                err,
                            )
                            args = {}

                        yield ToolCall(
                            id=tool_call_data["id"],
                            name=tool_call_data["name"],
                            args=args,
                        )
                    log.debug("Yielded all accumulated tool calls")

                if choices and choice.finish_reason == "stop":
                    log.debug("Finish reason is stop; emitting synthetic done chunk")
                    yield Chunk(content="", done=True)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            body_text = getattr(getattr(e, "response", None), "text", None)
            if status is not None:
                # Provide better error messages for 422 status codes in streaming
                if status == 422:
                    # Check if the messages contain media that the model cannot process
                    has_media = any(_message_contains_media(msg)[0] for msg in messages)
                    if has_media:
                        media_types = [
                            _message_contains_media(msg)[1] for msg in messages if _message_contains_media(msg)[0]
                        ]
                        media_str = ", ".join(set(media_types))
                        raise Exception(
                            f"422 Model '{model}' cannot process {media_str} content in streaming mode. "
                            f"The model may not support multimodal input or the {media_str} format is not supported. "
                            f"Original error: {body_text or str(e)}"
                        ) from e
                    else:
                        raise Exception(
                            f"422 Model '{model}' received unprocessable input in streaming mode. "
                            f"The model may not support the provided parameters or content format. "
                            f"Original error: {body_text or str(e)}"
                        ) from e
                else:
                    raise Exception(f"{status} {body_text or str(e)}") from e
            raise

    def get_usage(self) -> dict:
        """Get token usage statistics."""
        log.debug(f"Getting usage stats: {self.usage}")
        return self.usage

    def reset_usage(self) -> None:
        """Reset token usage statistics."""
        log.debug("Resetting usage counters")
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def is_context_length_error(self, error: Exception) -> bool:
        """Check if the error is due to context length exceeding limits."""
        error_str = str(error).lower()
        is_context_error = any(
            phrase in error_str
            for phrase in [
                "context length",
                "maximum context",
                "token limit",
                "too long",
                "context size",
                "request too large",
                "413",
            ]
        )
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error

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
        """Generate speech audio from text using HuggingFace text-to-speech models.

        HuggingFace Inference API does not support streaming TTS, so this yields
        a single chunk with all audio bytes.

        Args:
            text: Input text to convert to speech
            model: Model identifier (HuggingFace model ID)
            voice: Voice identifier (not used by most HF TTS models)
            speed: Speech speed multiplier (not used by most HF TTS models)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            **kwargs: Additional HuggingFace parameters

        Yields:
            numpy.ndarray: Int16 audio chunks at 24kHz mono

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating speech with HuggingFace model: {model}")

        if not text:
            raise ValueError("text must not be empty")

        log.debug(f"Making HuggingFace TTS API call with model={model}")

        try:
            # Use the text_to_speech method from AsyncInferenceClient
            audio_bytes = await self.client.text_to_speech(
                text=text,
                model=model,
            )

            log.debug("HuggingFace TTS API call successful")

            # audio_bytes is already bytes from the API
            log.debug(f"Generated {len(audio_bytes)} bytes of audio")

            # Convert to 24kHz mono 16-bit numpy array
            from nodetool.media.audio.audio_helpers import (
                convert_audio_to_standard_format,
            )

            audio_array = convert_audio_to_standard_format(audio_bytes, target_sample_rate=24000)

            yield audio_array

            log.debug(f"Yielded {len(audio_array)} samples of 16-bit audio")

        except Exception as e:
            log.error(f"HuggingFace TTS generation failed: {e}")
            raise RuntimeError(f"HuggingFace TTS generation failed: {str(e)}") from e

    async def get_available_tts_models(self) -> List[TTSModel]:
        """
        Get available HuggingFace TTS models.

        Returns common TTS models available on HuggingFace.
        Returns an empty list if no API key is configured.

        Returns:
            List of TTSModel instances for HuggingFace TTS
        """
        if not self.api_key:
            log.debug("No HuggingFace API key configured, returning empty TTS model list")
            return []

        try:
            assert self.inference_provider is not None, "Inference provider is not set"
            models = await fetch_tts_models_from_hf_provider(self.inference_provider, "text-to-speech")
            log.debug(f"Fetched {len(models)} TTS models for HF inference provider: {self.inference_provider}")
            return models
        except Exception as e:
            log.error(f"Error fetching HuggingFace TTS models for provider {self.inference_provider}: {e}")
            return []

    async def text_to_image(
        self,
        params: Any,  # TextToImageParams
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate an image from a text prompt using HuggingFace text-to-image models.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling

        Returns:
            Raw image bytes (PNG format)

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating image with HuggingFace model: {params.model.id}")

        if not params.prompt:
            raise ValueError("prompt must not be empty")

        try:
            # Use the text_to_image method from AsyncInferenceClient
            image = await self.client.text_to_image(
                prompt=params.prompt,
                model=params.model.id,
                negative_prompt=params.negative_prompt or None,
                height=params.height if params.height else None,
                width=params.width if params.width else None,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
                seed=params.seed if params.seed and params.seed >= 0 else None,
                scheduler=params.scheduler if hasattr(params, "scheduler") and params.scheduler else None,
            )

            log.debug("HuggingFace text-to-image API call successful")

            # Convert PIL Image to bytes
            import io

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            result = img_bytes.read()
            log.debug(f"Generated {len(result)} bytes of image data")

            return result

        except Exception as e:
            log.error(f"HuggingFace text-to-image generation failed: {e}")
            raise RuntimeError(f"HuggingFace text-to-image generation failed: {str(e)}") from e

    async def image_to_image(
        self,
        image: bytes,
        params: Any,  # ImageToImageParams
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Transform an image based on a text prompt using HuggingFace image-to-image models.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling

        Returns:
            Raw image bytes (PNG format)

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Transforming image with HuggingFace model: {params.model.id}")

        if not params.prompt:
            raise ValueError("prompt must not be empty")

        try:
            # Convert bytes to PIL Image for the API
            import io

            from PIL import Image

            input_image = Image.open(io.BytesIO(image))

            # Use the image_to_image method from AsyncInferenceClient
            result_image = await self.client.image_to_image(
                image=input_image,
                prompt=params.prompt,
                model=params.model.id,
                negative_prompt=params.negative_prompt or None,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
                target_size={  # pyright: ignore[reportArgumentType]
                    "width": params.target_width,
                    "height": params.target_height,
                }
                if params.target_width and params.target_height
                else None,
            )

            log.debug("HuggingFace image-to-image API call successful")

            # Convert PIL Image to bytes
            img_bytes = io.BytesIO()
            result_image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            result = img_bytes.read()
            log.debug(f"Generated {len(result)} bytes of image data")

            return result

        except Exception as e:
            log.error(f"HuggingFace image-to-image generation failed: {e}")
            raise RuntimeError(f"HuggingFace image-to-image generation failed: {str(e)}") from e

    async def get_available_image_models(self) -> List[ImageModel]:
        """
        Get available HuggingFace image generation models for this inference provider.

        Fetches models from the HuggingFace API based on the inference provider.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of ImageModel instances for HuggingFace
        """
        if not self.api_key:
            log.debug("No HuggingFace API key configured, returning empty image model list")
            return []

        try:
            assert self.inference_provider is not None, "Inference provider is not set"
            # Fetch both text-to-image and image-to-image and union tasks by model id
            t2i = await fetch_image_models_from_hf_provider(self.inference_provider, "text-to-image", self.api_key)
            i2i = await fetch_image_models_from_hf_provider(self.inference_provider, "image-to-image", self.api_key)
            by_id: dict[str, ImageModel] = {}
            for m in t2i + i2i:
                if m.id not in by_id:
                    by_id[m.id] = m
                else:
                    # Union supported tasks
                    existing = by_id[m.id]
                    tasks = set(existing.supported_tasks or [])
                    for t in m.supported_tasks or []:
                        tasks.add(t)
                    existing.supported_tasks = list(tasks)
            models = list(by_id.values())
            log.debug(f"Fetched {len(models)} image models for HF inference provider: {self.inference_provider}")
            return models
        except Exception as e:
            log.error(f"Error fetching HuggingFace image models for provider {self.inference_provider}: {e}")
            return []

    async def get_available_video_models(self) -> List[VideoModel]:
        """
        Get available HuggingFace video generation models for this inference provider.

        Fetches models from the HuggingFace API based on the inference provider.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of VideoModel instances for HuggingFace
        """
        if not self.api_key:
            log.debug("No HuggingFace API key configured, returning empty video model list")
            return []

        try:
            assert self.inference_provider is not None, "Inference provider is not set"
            # Fetch both text-to-video and image-to-video and union tasks by model id
            t2v = await fetch_video_models_from_hf_provider(self.inference_provider, "text-to-video")
            i2v = await fetch_video_models_from_hf_provider(self.inference_provider, "image-to-video")
            by_id: dict[str, VideoModel] = {}
            for m in t2v + i2v:
                if m.id not in by_id:
                    by_id[m.id] = m
                else:
                    existing = by_id[m.id]
                    tasks = set(existing.supported_tasks or [])
                    for t in m.supported_tasks or []:
                        tasks.add(t)
                    existing.supported_tasks = list(tasks)
            models = list(by_id.values())
            log.debug(f"Fetched {len(models)} video models for HF inference provider: {self.inference_provider}")
            return models
        except Exception as e:
            log.error(f"Error fetching HuggingFace video models for provider {self.inference_provider}: {e}")
            return []

    async def text_to_video(
        self,
        params: Any,  # TextToVideoParams
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate a video from a text prompt using HuggingFace text-to-video models.

        Args:
            params: Text-to-video generation parameters including:
                - prompt: Text description of the video
                - negative_prompt: Optional elements to exclude
                - model: VideoModel with HuggingFace model ID
                - num_frames: Number of video frames to generate
                - guidance_scale: Guidance scale for generation
                - num_inference_steps: Number of denoising steps
                - seed: Random seed for reproducibility
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling

        Returns:
            Raw video bytes

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating video with HuggingFace model: {params.model.id}")

        if not params.prompt:
            raise ValueError("prompt must not be empty")

        try:
            # Prepare parameters for the text_to_video API call
            api_params = {}

            if hasattr(params, "num_frames") and params.num_frames:
                api_params["num_frames"] = params.num_frames

            if hasattr(params, "guidance_scale") and params.guidance_scale:
                api_params["guidance_scale"] = params.guidance_scale

            if hasattr(params, "negative_prompt") and params.negative_prompt:
                api_params["negative_prompt"] = params.negative_prompt

            if hasattr(params, "num_inference_steps") and params.num_inference_steps:
                api_params["num_inference_steps"] = params.num_inference_steps

            if hasattr(params, "seed") and params.seed and params.seed >= 0:
                api_params["seed"] = params.seed

            # Use the text_to_video method from AsyncInferenceClient
            video_bytes = await self.client.text_to_video(
                prompt=params.prompt,
                model=params.model.id,
                **api_params,
            )

            log.debug("HuggingFace text-to-video API call successful")

            # video_bytes should be raw video bytes from the API
            log.debug(f"Generated {len(video_bytes)} bytes of video data")

            return video_bytes

        except Exception as e:
            log.error(f"HuggingFace text-to-video generation failed: {e}")
            raise RuntimeError(f"HuggingFace text-to-video generation failed: {str(e)}") from e
