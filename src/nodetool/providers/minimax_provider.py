"""
MiniMax provider implementation for chat completions.

This module implements the ChatProvider interface for MiniMax models,
using their Anthropic-compatible API endpoint.

MiniMax Anthropic API Documentation: https://platform.minimaxi.com/docs/api-reference/text-anthropic-api
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import aiohttp
import anthropic

if TYPE_CHECKING:
    import asyncio

    from nodetool.workflows.processing_context import ProcessingContext

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    LanguageModel,
    Provider,
)
from nodetool.providers.anthropic_provider import AnthropicProvider
from nodetool.providers.base import register_provider

log = get_logger(__name__)

# MiniMax Anthropic-compatible API base URL
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"


@register_provider(Provider.MiniMax)
class MiniMaxProvider(AnthropicProvider):
    """MiniMax implementation of the ChatProvider interface.

    MiniMax provides an Anthropic-compatible API for their models.
    This provider extends AnthropicProvider with MiniMax-specific configuration.

    Key differences from Anthropic:
    1. Base URL: https://api.minimaxi.chat/v1
    2. API key: MINIMAX_API_KEY instead of ANTHROPIC_API_KEY
    3. Model listing via MiniMax's models endpoint

    For details, see: https://platform.minimaxi.com/docs/api-reference/text-anthropic-api
    """

    provider_name: str = "minimax"

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["MINIMAX_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the MiniMax provider with client credentials.

        Reads ``MINIMAX_API_KEY`` from secrets and configures the Anthropic client
        with MiniMax's base URL.
        """
        assert "MINIMAX_API_KEY" in secrets, "MINIMAX_API_KEY is required"
        self.api_key = secrets["MINIMAX_API_KEY"]

        log.debug("MiniMaxProvider initialized")
        self._clients: dict[int, anthropic.AsyncAnthropic] = {}

    def get_client(self) -> anthropic.AsyncAnthropic:
        """Return a MiniMax async client for the current event loop."""
        import asyncio

        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        if loop_id not in self._clients:
            log.debug(f"Creating MiniMax AsyncClient for loop {loop_id}")
            self._clients[loop_id] = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=MINIMAX_BASE_URL,
            )
        return self._clients[loop_id]

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``MINIMAX_API_KEY`` if available; otherwise empty.
        """
        return {"MINIMAX_API_KEY": self.api_key} if self.api_key else {}

    def get_context_length(self, model: str) -> int:
        """Get the maximum token limit for a given MiniMax model.

        MiniMax models typically support large context windows.
        Returns a conservative default.
        """
        log.debug(f"Getting context length for MiniMax model: {model}")
        # MiniMax models generally support 200k context like Claude
        return 200000

    async def get_available_language_models(self) -> list[LanguageModel]:
        """
        Get available MiniMax models.

        MiniMax doesn't provide a models discovery endpoint, so we return
        a hardcoded list of known MiniMax models.

        Known models:
        - MiniMax-M2.1
        - MiniMax-M2.1-lightning
        - MiniMax-M2

        Returns:
            List of LanguageModel instances for MiniMax
        """
        if not self.api_key:
            log.debug("No MiniMax API key configured, returning empty model list")
            return []

        # MiniMax doesn't have a models discovery endpoint
        # Return known models based on API documentation
        known_models = [
            "MiniMax-M2.1",
            "MiniMax-M2.1-lightning",
            "MiniMax-M2",
        ]

        models = [
            LanguageModel(
                id=model_id,
                name=model_id,
                provider=Provider.MiniMax,
            )
            for model_id in known_models
        ]

        log.debug(f"Returning {len(models)} known MiniMax models")
        return models
