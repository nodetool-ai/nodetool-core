"""
MiniMax provider implementation for chat completions and image generation.

This module implements the ChatProvider interface for MiniMax models,
using their Anthropic-compatible API endpoint for chat and their
image generation API for text-to-image.

MiniMax Anthropic API Documentation: https://platform.minimaxi.com/docs/api-reference/text-anthropic-api
MiniMax Image Generation API: https://platform.minimax.io/docs/guides/image-generation
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import aiohttp
import anthropic

if TYPE_CHECKING:
    from nodetool.providers.types import TextToImageParams
    from nodetool.workflows.processing_context import ProcessingContext

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    ImageModel,
    LanguageModel,
    Provider,
)
from nodetool.providers.anthropic_provider import AnthropicProvider
from nodetool.providers.base import register_provider

log = get_logger(__name__)

# MiniMax Anthropic-compatible API base URL
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"

# MiniMax Image Generation API base URL
MINIMAX_IMAGE_API_URL = "https://api.minimax.io/v1/text_to_image"

# Known MiniMax image models
MINIMAX_IMAGE_MODELS = [
    ImageModel(
        id="image-01",
        name="MiniMax Image-01",
        provider=Provider.MiniMax,
    ),
]


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

    async def get_available_image_models(self) -> list[ImageModel]:
        """Get available MiniMax image generation models.

        Returns a list of known MiniMax image models. MiniMax currently
        supports the image-01 model for text-to-image generation.

        Returns:
            List of ImageModel instances for MiniMax
        """
        if not self.api_key:
            log.debug("No MiniMax API key configured, returning empty image model list")
            return []

        log.debug(f"Returning {len(MINIMAX_IMAGE_MODELS)} known MiniMax image models")
        return MINIMAX_IMAGE_MODELS

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate an image from a text prompt using MiniMax's Image Generation API.

        Uses the MiniMax text_to_image endpoint which supports the image-01 model
        for high-quality image generation.

        API Reference: https://platform.minimax.io/docs/guides/image-generation

        Args:
            params: Text-to-image generation parameters including:
                - model: ImageModel with model ID (e.g., "image-01")
                - prompt: Text description of the desired image
                - width/height: Desired dimensions
                - aspect_ratio: Optional aspect ratio
            timeout_s: Optional timeout in seconds
            context: Processing context (unused, reserved)
            node_id: Node ID for progress reporting (unused)

        Returns:
            Raw image bytes

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        if not params.prompt:
            raise ValueError("The input prompt cannot be empty.")

        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required for image generation.")

        model_id = params.model.id if params.model and params.model.id else "image-01"

        prompt = params.prompt.strip()

        log.debug(f"Generating image with MiniMax model: {model_id}")
        log.debug(f"Prompt: {prompt[:100]}...")

        self._log_api_request("text_to_image", params=params)

        try:
            request_timeout = timeout_s if timeout_s and timeout_s > 0 else 120

            # Build the request payload according to MiniMax API
            payload: dict[str, Any] = {
                "model": model_id,
                "prompt": prompt,
            }

            # Add optional aspect_ratio if width/height suggest a specific ratio
            if params.width and params.height:
                aspect_ratio = self._calculate_aspect_ratio(params.width, params.height)
                if aspect_ratio:
                    payload["aspect_ratio"] = aspect_ratio

            # Add optional parameters if provided
            if hasattr(params, "n") and params.n:
                payload["n"] = params.n

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            timeout = aiohttp.ClientTimeout(total=request_timeout)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    MINIMAX_IMAGE_API_URL,
                    json=payload,
                    headers=headers,
                ) as response,
            ):
                if response.status != 200:
                    error_text = await response.text()
                    log.error(f"MiniMax API error: {response.status} - {error_text}")
                    raise RuntimeError(
                        f"MiniMax text-to-image generation failed with status {response.status}: {error_text}"
                    )

                result = await response.json()

                # Extract image data from response
                # MiniMax API returns base64-encoded image data
                if "data" in result and len(result["data"]) > 0:
                    image_data = result["data"][0]
                    if "b64_image" in image_data:
                        image_bytes = base64.b64decode(image_data["b64_image"])
                    elif "url" in image_data:
                        # Fallback to URL if base64 not provided
                        async with session.get(image_data["url"]) as img_response:
                            if img_response.status != 200:
                                raise RuntimeError("Failed to download generated image from URL")
                            image_bytes = await img_response.read()
                    else:
                        raise RuntimeError("No image data returned in MiniMax response")
                else:
                    raise RuntimeError("No image data returned in MiniMax response")

            log.debug(f"Generated image, size: {len(image_bytes)} bytes")
            self._log_api_response("text_to_image", image_bytes=len(image_bytes))

            return image_bytes

        except aiohttp.ClientError as e:
            log.error(f"MiniMax text-to-image request failed: {e}")
            raise RuntimeError(f"MiniMax text-to-image generation failed: {e}") from e
        except Exception as exc:
            log.error(f"MiniMax text-to-image generation failed: {exc}")
            raise RuntimeError(f"MiniMax text-to-image generation failed: {exc}") from exc

    def _calculate_aspect_ratio(self, width: int, height: int) -> str | None:
        """Calculate aspect ratio string from width and height.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Aspect ratio string (e.g., "16:9", "1:1") or None if non-standard
        """
        if width == height:
            return "1:1"

        # Common aspect ratios
        ratio = width / height
        common_ratios = {
            16 / 9: "16:9",
            9 / 16: "9:16",
            4 / 3: "4:3",
            3 / 4: "3:4",
            3 / 2: "3:2",
            2 / 3: "2:3",
            21 / 9: "21:9",
            9 / 21: "9:21",
        }

        # Find closest match with 5% tolerance
        for known_ratio, ratio_str in common_ratios.items():
            if abs(ratio - known_ratio) / known_ratio < 0.05:
                return ratio_str

        return None
