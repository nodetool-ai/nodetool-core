"""
OpenRouter provider implementation for chat completions.

This module implements the ChatProvider interface for OpenRouter,
which provides access to multiple AI models through a unified OpenAI-compatible API.

OpenRouter API Documentation: https://openrouter.ai/docs/api/reference/overview
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, List, Sequence

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
from nodetool.providers.openai_provider import OpenAIProvider
from nodetool.providers.base import register_provider
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

        Reads ``OPENROUTER_API_KEY`` from environment and prepares usage tracking.
        """
        assert "OPENROUTER_API_KEY" in secrets, "OPENROUTER_API_KEY is required"
        self.api_key = secrets["OPENROUTER_API_KEY"]
        self.client = None
        self.cost = 0.0
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }
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

    def get_context_length(self, model: str) -> int:
        """Return an approximate maximum token limit for a given model.

        OpenRouter supports many models with varying context lengths.
        This provides reasonable defaults based on common model families.

        Args:
            model: Model identifier string (e.g., "openai/gpt-4", "anthropic/claude-3-opus")

        Returns:
            Approximate maximum number of tokens the model can handle.
        """
        log.debug(f"Getting context length for model: {model}")

        # OpenRouter model IDs are in format "provider/model-name"
        model_lower = model.lower()

        # OpenAI models
        if "gpt-4o" in model_lower or "chatgpt-4o" in model_lower:
            return 128000
        if "gpt-4-turbo" in model_lower:
            return 128000
        if "gpt-4" in model_lower:
            if "32k" in model_lower:
                return 32768
            return 8192
        if "gpt-3.5" in model_lower:
            if "16k" in model_lower:
                return 16384
            return 4096

        # Anthropic Claude models
        if "claude" in model_lower:
            return 200000

        # Google Gemini models
        if "gemini" in model_lower:
            if "1.5" in model_lower:
                return 1000000
            return 32768

        # Meta Llama models
        if "llama" in model_lower:
            if "3.1" in model_lower or "3.2" in model_lower:
                return 128000
            return 8192

        # Mistral models
        if "mistral" in model_lower or "mixtral" in model_lower:
            return 32768

        # Default fallback
        log.debug("Unknown model; returning conservative default context length: 8192")
        return 8192

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

    async def get_available_language_models(self) -> List[LanguageModel]:
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
                    log.warning(
                        f"Failed to fetch OpenRouter models: HTTP {response.status}"
                    )
                    return []
                payload = await response.json()
                data = payload.get("data", [])

                models: List[LanguageModel] = []
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
