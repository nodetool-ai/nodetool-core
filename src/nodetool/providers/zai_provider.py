"""
Z.AI provider implementation for chat completions.

This module implements the ChatProvider interface for Z.AI,
which provides access to GLM models through an OpenAI-compatible API.

Z.AI API Documentation: https://docs.z.ai/api-reference/llm/chat-completion
Z.AI Models: https://docs.z.ai/devpack/overview
Authentication: Uses ZHIPU_API_KEY
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


@register_provider(Provider.ZAI)
class ZAIProvider(OpenAIProvider):
    """Z.AI implementation of the ChatProvider interface.

    Z.AI provides access to GLM models through an OpenAI-compatible API.
    This provider extends OpenAIProvider with Z.AI-specific configuration.

    Key differences from OpenAI:
    1. Base URL: https://api.z.ai/api/coding/paas/v4
    2. Uses ZHIPU_API_KEY for authentication
    3. Different set of available models (GLM family)

    For details, see: https://docs.z.ai/api-reference/llm/chat-completion
    """

    provider: Provider = Provider.ZAI

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["ZHIPU_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the Z.AI provider with client credentials.

        Reads ``ZHIPU_API_KEY`` from secrets.
        """
        assert "ZHIPU_API_KEY" in secrets, "ZHIPU_API_KEY is required"
        self.api_key = secrets["ZHIPU_API_KEY"]
        self.client = None
        self.cost = 0.0
        log.debug("ZAIProvider initialized. API key present: True")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``ZHIPU_API_KEY`` if available; otherwise empty.
        """
        return {"ZHIPU_API_KEY": self.api_key} if self.api_key else {}

    def get_client(self) -> openai.AsyncClient:
        """Create and return a Z.AI async client.

        Uses OpenAI SDK with Z.AI's base URL and API key.

        Returns:
            An initialized ``openai.AsyncClient`` configured for Z.AI.
        """
        log.debug("Creating Z.AI async client")

        # Use ResourceScope's HTTP client if available
        from nodetool.runtime.resources import require_scope

        http_client = require_scope().get_http_client()

        # Configure client for Z.AI
        client = openai.AsyncClient(
            api_key=self.api_key,
            base_url="https://api.z.ai/api/coding/paas/v4",
            http_client=http_client,
        )
        log.debug("Z.AI async client created successfully")
        return client

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        Z.AI GLM models support function calling for compatible models.

        Args:
            model: Model identifier string.

        Returns:
            True if the model supports function calling, False otherwise.
        """
        log.debug(f"Checking tool support for model: {model}")
        # Z.AI GLM models generally support tool calling
        log.debug(f"Model {model} supports tool calling")
        return True

    async def get_available_language_models(self) -> list[LanguageModel]:
        """
        Get available Z.AI models.

        Fetches models dynamically from the Z.AI API if an API key is available.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of LanguageModel instances for Z.AI
        """
        if not self.api_key:
            log.debug("No Z.AI API key configured, returning empty model list")
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                session.get("https://api.z.ai/api/coding/paas/v4/models") as response,
            ):
                if response.status != 200:
                    log.warning(f"Failed to fetch Z.AI models: HTTP {response.status}")
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
                            provider=Provider.ZAI,
                        )
                    )
                log.debug(f"Fetched {len(models)} Z.AI models")
                return models
        except Exception as e:
            log.error(f"Error fetching Z.AI models: {e}")
            return []
