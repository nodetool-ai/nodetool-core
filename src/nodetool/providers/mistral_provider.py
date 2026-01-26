"""
Mistral AI provider implementation for chat completions.

This module implements the ChatProvider interface for Mistral AI,
which provides access to Mistral models through an OpenAI-compatible API.

Mistral AI API Documentation: https://docs.mistral.ai/api/
Mistral AI Models: https://docs.mistral.ai/getting-started/models/
Authentication: Uses MISTRAL_API_KEY
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiohttp
import openai

if TYPE_CHECKING:
    from nodetool.workflows.processing_context import ProcessingContext

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
)
from nodetool.providers.base import register_provider
from nodetool.providers.openai_provider import OpenAIProvider

log = get_logger(__name__)


@register_provider(Provider.Mistral)
class MistralProvider(OpenAIProvider):
    """Mistral AI implementation of the ChatProvider interface.

    Mistral AI provides access to Mistral language models through an OpenAI-compatible API.
    This provider extends OpenAIProvider with Mistral-specific configuration.

    Key differences from OpenAI:
    1. Base URL: https://api.mistral.ai/v1
    2. Uses MISTRAL_API_KEY for authentication
    3. Different set of available models (Mistral, Mixtral, etc.)

    For details, see: https://docs.mistral.ai/api/
    """

    provider: Provider = Provider.Mistral

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["MISTRAL_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the Mistral provider with client credentials.

        Reads ``MISTRAL_API_KEY`` from secrets.
        """
        assert "MISTRAL_API_KEY" in secrets, "MISTRAL_API_KEY is required"
        self.api_key = secrets["MISTRAL_API_KEY"]
        self.client = None
        self.cost = 0.0
        log.debug("MistralProvider initialized. API key present: True")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``MISTRAL_API_KEY`` if available; otherwise empty.
        """
        return {"MISTRAL_API_KEY": self.api_key} if self.api_key else {}

    def get_client(self) -> openai.AsyncClient:
        """Create and return a Mistral async client.

        Uses OpenAI SDK with Mistral's base URL and API key.

        Returns:
            An initialized ``openai.AsyncClient`` configured for Mistral.
        """
        log.debug("Creating Mistral async client")

        # Use ResourceScope's HTTP client if available
        from nodetool.runtime.resources import require_scope

        http_client = require_scope().get_http_client()

        # Configure client for Mistral
        client = openai.AsyncClient(
            api_key=self.api_key,
            base_url="https://api.mistral.ai/v1",
            http_client=http_client,
        )
        log.debug("Mistral async client created successfully")
        return client

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        Mistral models support function calling for compatible models.

        Args:
            model: Model identifier string.

        Returns:
            True if the model supports function calling, False otherwise.
        """
        log.debug(f"Checking tool support for model: {model}")
        # Mistral models generally support tool calling
        log.debug(f"Model {model} supports tool calling")
        return True

    async def get_available_language_models(self) -> list[LanguageModel]:
        """
        Get available Mistral models.

        Fetches models dynamically from the Mistral API if an API key is available.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of LanguageModel instances for Mistral
        """
        if not self.api_key:
            log.debug("No Mistral API key configured, returning empty model list")
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                session.get("https://api.mistral.ai/v1/models") as response,
            ):
                if response.status != 200:
                    log.warning(f"Failed to fetch Mistral models: HTTP {response.status}")
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
                            provider=Provider.Mistral,
                        )
                    )
                log.debug(f"Fetched {len(models)} Mistral models")
                return models
        except Exception as e:
            log.error(f"Error fetching Mistral models: {e}")
            return []
