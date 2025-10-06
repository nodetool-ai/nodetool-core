"""
HuggingFace provider implementation for chat completions.

This module implements the ChatProvider interface for HuggingFace models using their
Inference Providers API with the AsyncInferenceClient from huggingface_hub.
"""

import json
import asyncio
import time
import traceback
from typing import Any, AsyncGenerator, List, Literal, Sequence
import aiohttp
import numpy as np

from huggingface_hub import AsyncInferenceClient

from nodetool.providers.base import (
    BaseProvider,
    ProviderCapability,
    register_provider,
)
from nodetool.agents.tools.base import Tool
from nodetool.config.logging_config import get_logger
import base64
from nodetool.media.image.image_utils import image_data_to_base64_jpeg
from nodetool.io.media_fetch import fetch_uri_bytes_and_mime_sync
from nodetool.metadata.types import (
    InferenceProvider,
    Message,
    Provider,
    ToolCall,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
    LanguageModel,
    TTSModel,
    ImageModel,
)
from nodetool.config.environment import Environment
from nodetool.workflows.base_node import ApiKeyMissingError
from nodetool.workflows.types import Chunk
from pydantic import BaseModel

log = get_logger(__name__)

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
        url = f"https://huggingface.co/api/models?inference_provider={provider}&pipeline_tag={pipeline_tag}&limit=1000"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    models_data = await response.json()

                    models = []
                    for model_data in models_data:
                        model_id = model_data.get("id", "")
                        if model_id:
                            # Use the model name from the API if available, otherwise use the ID
                            model_name = (
                                model_data.get("name") or model_id.split("/")[-1]
                                if "/" in model_id
                                else model_id
                            )

                            # Get the appropriate provider enum value
                            provider_enum = HF_PROVIDER_MAPPING.get(provider)
                            if provider_enum is None:
                                log.warning(
                                    f"Unknown image provider: {provider}, skipping model: {model_id}"
                                )
                                continue

                            models.append(
                                ImageModel(
                                    id=model_id,
                                    name=model_name,
                                    provider=provider_enum,
                                )
                            )

                    log.debug(
                        f"Fetched {len(models)} image models from HF provider: {provider}"
                    )
                    return models
                else:
                    log.warning(
                        f"Failed to fetch image models for provider {provider}: HTTP {response.status}"
                    )
                    return []

    except Exception as e:
        log.error(f"Error fetching image models for provider {provider}: {e}")
        return []


async def fetch_tts_models_from_hf_provider(
    provider: str, pipeline_tag: str
) -> List[TTSModel]:
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

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    models_data = await response.json()

                    models = []
                    for model_data in models_data:
                        model_id = model_data.get("id", "")
                        if model_id:
                            # Use the model name from the API if available, otherwise use the ID
                            model_name = (
                                model_data.get("name") or model_id.split("/")[-1]
                                if "/" in model_id
                                else model_id
                            )

                            # Get the appropriate provider enum value
                            provider_enum = HF_PROVIDER_MAPPING.get(provider)
                            if provider_enum is None:
                                log.warning(
                                    f"Unknown TTS provider: {provider}, skipping model: {model_id}"
                                )
                                continue

                            models.append(
                                TTSModel(
                                    id=model_id,
                                    name=model_name,
                                    provider=provider_enum,
                                    voices=[],  # HF TTS models typically don't have named voices
                                )
                            )

                    log.debug(
                        f"Fetched {len(models)} TTS models from HF provider: {provider}"
                    )
                    return models
                else:
                    log.warning(
                        f"Failed to fetch TTS models for provider {provider}: HTTP {response.status}"
                    )
                    return []

    except Exception as e:
        log.error(f"Error fetching TTS models for provider {provider}: {e}")
        return []


async def fetch_models_from_hf_provider(
    provider: str, pipeline_tag: str
) -> List[LanguageModel]:
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

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    models_data = await response.json()

                    models = []
                    for model_data in models_data:
                        model_id = model_data.get("id", "")
                        if model_id:
                            # Use the model name from the API if available, otherwise use the ID
                            model_name = (
                                model_data.get("name") or model_id.split("/")[-1]
                                if "/" in model_id
                                else model_id
                            )

                            # Get the appropriate provider enum value
                            provider_enum = HF_PROVIDER_MAPPING.get(provider)
                            if provider_enum is None:
                                log.warning(
                                    f"Unknown provider: {provider}, skipping model: {model_id}"
                                )
                                continue

                            models.append(
                                LanguageModel(
                                    id=model_id,
                                    name=model_name,
                                    provider=provider_enum,
                                )
                            )

                    # Preserve API order to match test expectations
                    log.debug(
                        f"Fetched {len(models)} language models from HF provider: {provider}"
                    )
                    return models
                else:
                    log.warning(
                        f"Failed to fetch models for provider {provider}: HTTP {response.status}"
                    )
                    return []

    except Exception as e:
        log.error(f"Error fetching models for provider {provider}: {e}")
        return []




@register_provider(Provider.HuggingFaceGroq, inference_provider="groq")
@register_provider(Provider.HuggingFaceCerebras, inference_provider="cerebras")
@register_provider(Provider.HuggingFaceCohere, inference_provider="cohere")
@register_provider(Provider.HuggingFaceFalAI, inference_provider="fal-ai")
@register_provider(
    Provider.HuggingFaceFeatherlessAI, inference_provider="featherless-ai"
)
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

    def __init__(self, inference_provider: PROVIDER_T):
        """Initialize the HuggingFace provider with AsyncInferenceClient."""
        super().__init__()
        env = Environment.get_environment()
        self.api_key = env.get("HF_TOKEN")
        self.inference_provider = inference_provider
        self.provider_name = f"huggingface_{inference_provider}"

        if not self.api_key:
            log.error("HF_TOKEN or HUGGINGFACE_API_KEY is not set")
            raise ApiKeyMissingError("HF_TOKEN or HUGGINGFACE_API_KEY is not set")

        # Initialize the AsyncInferenceClient
        log.debug(
            f"Creating AsyncInferenceClient with provider: {self.inference_provider}"
        )
        self.client = AsyncInferenceClient(
            api_key=self.api_key,
            provider=self.inference_provider,
        )

        self.cost = 0.0
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        log.debug(
            f"HuggingFaceProvider initialized with provider: {inference_provider or 'default'}"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        log.debug("Entering async context manager")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - properly close client."""
        log.debug("Exiting async context manager")
        await self.close()

    async def close(self):
        """Close the async client properly."""
        log.debug("Closing async client")
        if hasattr(self.client, "close"):
            await self.client.close()
            log.debug("Async client closed successfully")
        else:
            log.debug("Client does not have close method")

    def get_capabilities(self) -> set[ProviderCapability]:
        """HuggingFace provider supports message generation and text-to-speech."""
        return {
            ProviderCapability.GENERATE_MESSAGE,
            ProviderCapability.GENERATE_MESSAGES,
            ProviderCapability.TEXT_TO_SPEECH,
            ProviderCapability.TEXT_TO_IMAGE,
            ProviderCapability.IMAGE_TO_IMAGE,
        }

    def get_container_env(self) -> dict[str, str]:
        env_vars = {}
        if self.api_key:
            env_vars["HF_TOKEN"] = self.api_key
        if hasattr(self, "inference_provider") and self.inference_provider:
            env_vars["HUGGINGFACE_PROVIDER"] = self.inference_provider
        log.debug(f"Container environment variables: {list(env_vars.keys())}")
        return env_vars

    def get_context_length(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        log.debug(f"Getting context length for model: {model}")

        # Common HuggingFace model limits - this can be expanded based on specific models
        if "llama" in model.lower():
            log.debug("Using context length: 32768 (Llama)")
            return 32768  # Many Llama models support 32k context
        elif "qwen" in model.lower():
            log.debug("Using context length: 32768 (Qwen)")
            return 32768  # Qwen models often support large context
        elif "phi" in model.lower():
            log.debug("Using context length: 128000 (Phi)")
            return 128000  # Phi-4 supports 128k context
        elif "smol" in model.lower():
            log.debug("Using context length: 8192 (SmolLM)")
            return 8192  # SmolLM models typically have smaller context
        elif "gemma" in model.lower():
            log.debug("Using context length: 8192 (Gemma)")
            return 8192  # Gemma models typically support 8k context
        elif "deepseek" in model.lower():
            log.debug("Using context length: 32768 (DeepSeek)")
            return 32768  # DeepSeek models often support large context
        elif "mistral" in model.lower():
            log.debug("Using context length: 32768 (Mistral)")
            return 32768  # Mistral models support 32k context
        else:
            log.debug("Using default context length: 8192")
            return 8192  # Conservative default

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
            models = await fetch_models_from_hf_provider(
                self.inference_provider, "text-generation"
            )
            log.debug(
                f"Fetched {len(models)} models for HF inference provider: {self.inference_provider}"
            )
            return models
        except Exception as e:
            log.error(
                f"Error fetching HuggingFace models for provider {self.inference_provider}: {e}"
            )
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
                        # Normalize image inputs to data URI using shared helpers
                        if part.image.uri:
                            uri = part.image.uri
                            if uri.startswith("http"):
                                # Keep remote URLs as-is for HF compatibility
                                log.debug(
                                    f"Adding image content with URL: {uri[:50]}..."
                                )
                                content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": uri},
                                    }
                                )
                            else:
                                # Convert non-http URIs to data URI
                                log.debug(
                                    f"Converting non-HTTP image via helper: {uri[:50]}..."
                                )
                                mime, data = fetch_uri_bytes_and_mime_sync(uri)
                                b64 = base64.b64encode(data).decode("utf-8")
                                content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime};base64,{b64}"
                                        },
                                    }
                                )
                        elif part.image.data:
                            # Convert raw bytes to JPEG base64 data URI
                            log.debug("Converting raw image data to JPEG base64")
                            b64 = image_data_to_base64_jpeg(part.image.data)
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}"
                                    },
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
                                json.dumps(tool_call.args)
                                if isinstance(tool_call.args, dict)
                                else str(tool_call.args)
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
        context_window: int = 4096,
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
            context_window: Maximum number of tokens to keep in context
            response_format: Format of the response
            **kwargs: Additional provider-specific parameters

        Returns:
            A message returned by the provider.
        """
        log.debug(f"Generating message for model: {model}")
        log.debug(
            f"Input: {len(messages)} messages, {len(tools)} tools, max_tokens: {max_tokens}"
        )

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
                completion = await self.client.chat_completion(
                    model=model, **request_params
                )
                log.debug("API call successful")
                break
            except Exception as e:
                error_str = str(e).lower()
                log.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                # Do not retry on client-side errors (4xx), including 429, 413, 404.
                status = getattr(getattr(e, "response", None), "status_code", None)
                body_text = getattr(getattr(e, "response", None), "text", None)
                if isinstance(status, int) and 400 <= status < 500:
                    log.error(f"Non-retryable client error {status}; aborting retries")
                    raise Exception(f"{status} {body_text or str(e)}")
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    log.debug(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    log.error(f"All {max_retries + 1} attempts failed")
                    traceback.print_exc()
                    # Include status code/body text when available for clearer errors/tests
                    if status is not None:
                        raise Exception(f"{status} {body_text or str(e)}")
                    raise Exception(str(e))

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
                    log.warning(
                        f"Failed to parse arguments for tool {function.name}: {e}"
                    )
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
        context_window: int = 4096,
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
            context_window: Maximum number of tokens to keep in context
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
        log.debug("Streaming response initialized")

        # Track tool calls during streaming
        accumulated_tool_calls = {}
        chunk_count = 0

        try:
            async for chunk in stream:
                chunk_count += 1

                if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                    log.debug("Updating usage stats from streaming chunk")
                    usage = chunk.usage  # type: ignore[attr-defined]
                    self.usage["prompt_tokens"] = (
                        getattr(usage, "prompt_tokens", 0) or 0
                    )
                    self.usage["completion_tokens"] = (
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                    self.usage["total_tokens"] = getattr(usage, "total_tokens", 0) or 0
                    log.debug(f"Updated usage: {self.usage}")

                choices = getattr(chunk, "choices", None)
                if not choices:
                    log.debug("Chunk has no choices, skipping")
                    continue

                choice = choices[0]
                delta = getattr(choice, "delta", None)
                log.debug(
                    f"Processing delta with finish_reason: {choice.finish_reason}"
                )

                if delta and getattr(delta, "content", None):
                    yield Chunk(
                        content=delta.content,
                        done=choice.finish_reason == "stop",
                    )

                if delta and getattr(delta, "tool_calls", None):
                    log.debug(f"Processing {len(delta.tool_calls)} tool call deltas")
                    for tool_call_delta in delta.tool_calls:
                        index = getattr(tool_call_delta, "index", 0)
                        log.debug(f"Processing tool call delta at index {index}")

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
                                accumulated_tool_calls[index]["name"] = (
                                    function_delta.name or ""
                                )
                                log.debug(f"Set tool call name: {function_delta.name}")
                            if getattr(function_delta, "arguments", None):
                                accumulated_tool_calls[index]["arguments"] += (
                                    function_delta.arguments or ""
                                )
                                log.debug(
                                    "Added arguments to tool call: %s chars",
                                    len(function_delta.arguments or ""),
                                )

                if (
                    choices
                    and choice.finish_reason == "tool_calls"
                    and accumulated_tool_calls
                ):
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
                raise Exception(f"{status} {body_text or str(e)}")
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

            audio_array = convert_audio_to_standard_format(
                audio_bytes, target_sample_rate=24000
            )

            yield audio_array

            log.debug(f"Yielded {len(audio_array)} samples of 16-bit audio")

        except Exception as e:
            log.error(f"HuggingFace TTS generation failed: {e}")
            raise RuntimeError(f"HuggingFace TTS generation failed: {str(e)}")

    async def get_available_tts_models(self) -> List[TTSModel]:
        """
        Get available HuggingFace TTS models.

        Returns common TTS models available on HuggingFace.
        Returns an empty list if no API key is configured.

        Returns:
            List of TTSModel instances for HuggingFace TTS
        """
        if not self.api_key:
            log.debug(
                "No HuggingFace API key configured, returning empty TTS model list"
            )
            return []

        try:
            assert self.inference_provider is not None, "Inference provider is not set"
            models = await fetch_tts_models_from_hf_provider(
                self.inference_provider, "text-to-speech"
            )
            log.debug(
                f"Fetched {len(models)} TTS models for HF inference provider: {self.inference_provider}"
            )
            return models
        except Exception as e:
            log.error(
                f"Error fetching HuggingFace TTS models for provider {self.inference_provider}: {e}"
            )
            return []

    async def text_to_image(
        self,
        params: Any,  # TextToImageParams
        timeout_s: int | None = None,
        context: Any = None,
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
                scheduler=params.scheduler if hasattr(params, 'scheduler') and params.scheduler else None,
            )

            log.debug("HuggingFace text-to-image API call successful")

            # Convert PIL Image to bytes
            import io
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            result = img_bytes.read()
            log.debug(f"Generated {len(result)} bytes of image data")

            return result

        except Exception as e:
            log.error(f"HuggingFace text-to-image generation failed: {e}")
            raise RuntimeError(f"HuggingFace text-to-image generation failed: {str(e)}")

    async def image_to_image(
        self,
        image: bytes,
        params: Any,  # ImageToImageParams
        timeout_s: int | None = None,
        context: Any = None,
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
            from PIL import Image
            import io

            input_image = Image.open(io.BytesIO(image))

            # Use the image_to_image method from AsyncInferenceClient
            result_image = await self.client.image_to_image(
                image=input_image,
                prompt=params.prompt,
                model=params.model.id,
                negative_prompt=params.negative_prompt or None,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
                target_size={
                    "width": params.target_width,
                    "height": params.target_height,
                } if hasattr(params, 'target_width') and params.target_width and hasattr(params, 'target_height') and params.target_height else None,
            )

            log.debug("HuggingFace image-to-image API call successful")

            # Convert PIL Image to bytes
            img_bytes = io.BytesIO()
            result_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            result = img_bytes.read()
            log.debug(f"Generated {len(result)} bytes of image data")

            return result

        except Exception as e:
            log.error(f"HuggingFace image-to-image generation failed: {e}")
            raise RuntimeError(f"HuggingFace image-to-image generation failed: {str(e)}")

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
            models = await fetch_image_models_from_hf_provider(
                self.inference_provider, "text-to-image", self.api_key
            )
            log.debug(
                f"Fetched {len(models)} image models for HF inference provider: {self.inference_provider}"
            )
            return models
        except Exception as e:
            log.error(
                f"Error fetching HuggingFace image models for provider {self.inference_provider}: {e}"
            )
            return []

async def main():
    for provider in InferenceProvider.__members__.values():
        provider = HuggingFaceProvider(provider) # type: ignore
        models = await provider.get_available_image_models()
        print(provider)


if __name__ == "__main__":
    asyncio.run(main())