"""
Z.AI provider implementation for chat completions and image generation.

This module implements the ChatProvider interface for Z.AI,
which provides access to GLM models through an OpenAI-compatible API.

Z.AI supports two endpoints:
- Normal: https://api.z.ai/api/paas/v4 (for general use including video generation)
- Coding Plan: https://api.z.ai/api/coding/paas/v4 (for coding-specific features)

Z.AI API Documentation: https://docs.z.ai/api-reference/llm/chat-completion
Z.AI Image Generation: https://docs.z.ai/api-reference/image/generate-image
Z.AI Models: https://docs.z.ai/devpack/overview
Authentication: Uses ZHIPU_API_KEY
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import aiohttp
import openai

if TYPE_CHECKING:
    from nodetool.providers.types import TextToImageParams
    from nodetool.workflows.processing_context import ProcessingContext

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    ImageModel,
    LanguageModel,
    Provider,
)
from nodetool.providers.base import register_provider
from nodetool.providers.openai_provider import OpenAIProvider

log = get_logger(__name__)


@register_provider(Provider.ZAI)
class ZAIProvider(OpenAIProvider):
    """Z.AI implementation of the ChatProvider interface.

    Z.AI provides access to GLM models through an OpenAI-compatible API.
    This provider extends OpenAIProvider with Z.AI-specific configuration.

    Z.AI supports two endpoints:
    1. Normal endpoint: https://api.z.ai/api/paas/v4 (default)
       - General use including video generation
    2. Coding Plan endpoint: https://api.z.ai/api/coding/paas/v4
       - Specialized for coding tasks
       - Enabled by setting ZAI_USE_CODING_PLAN=true

    Key differences from OpenAI:
    1. Base URL: Configurable via ZAI_USE_CODING_PLAN setting
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

        Reads ``ZHIPU_API_KEY`` from secrets and ``ZAI_USE_CODING_PLAN`` from settings
        to determine which endpoint to use.
        """
        assert "ZHIPU_API_KEY" in secrets, "ZHIPU_API_KEY is required"
        self.api_key = secrets["ZHIPU_API_KEY"]
        self.client = None
        self.cost = 0.0

        # Determine which endpoint to use based on settings
        env = Environment.get_environment()
        use_coding_plan = env.get("ZAI_USE_CODING_PLAN", "false").lower() in ["true", "1", "yes"]

        if use_coding_plan:
            self.base_url = "https://api.z.ai/api/coding/paas/v4"
            log.debug("ZAIProvider initialized with coding plan endpoint")
        else:
            self.base_url = "https://api.z.ai/api/paas/v4"
            log.debug("ZAIProvider initialized with normal endpoint")

        log.debug(f"ZAIProvider base URL: {self.base_url}")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``ZHIPU_API_KEY`` if available; otherwise empty.
        """
        return {"ZHIPU_API_KEY": self.api_key} if self.api_key else {}

    def get_client(self) -> openai.AsyncClient:
        """Create and return a Z.AI async client.

        Uses OpenAI SDK with Z.AI's base URL (configured based on ZAI_USE_CODING_PLAN)
        and API key.

        Returns:
            An initialized ``openai.AsyncClient`` configured for Z.AI.
        """
        log.debug(f"Creating Z.AI async client with base URL: {self.base_url}")

        # Use ResourceScope's HTTP client if available
        from nodetool.runtime.resources import require_scope

        http_client = require_scope().get_http_client()

        # Configure client for Z.AI
        client = openai.AsyncClient(
            api_key=self.api_key,
            base_url=self.base_url,
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
        Uses the configured base URL (normal or coding plan endpoint).
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
            models_url = f"{self.base_url}/models"
            log.debug(f"Fetching Z.AI models from: {models_url}")

            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                session.get(models_url) as response,
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

    async def get_available_image_models(self) -> list[ImageModel]:
        """
        Get available Z.AI image generation models.

        Returns GLM-Image model for text-to-image generation.
        Returns an empty list if no API key is configured.

        GLM-Image features:
        - Hybrid architecture: autoregressive + diffusion decoder
        - Excellent text rendering in images
        - Supports various aspect ratios
        - Resolution range: 512px–2048px (multiples of 32)

        Returns:
            List of ImageModel instances for Z.AI
        """
        if not self.api_key:
            log.debug("No Z.AI API key configured, returning empty image model list")
            return []

        models = [
            ImageModel(
                id="glm-image",
                name="GLM-Image",
                provider=Provider.ZAI,
                supported_tasks=["text_to_image"],
            ),
        ]

        log.debug(f"Returning {len(models)} Z.AI image models")
        return models

    def _resolve_zai_image_size(self, width: int | None, height: int | None) -> str:
        """Convert requested dimensions to Z.AI-supported image sizes.

        Z.AI GLM-Image API constraints:
        - Both width and height must be within 512px–2048px
        - Each dimension must be a multiple of 32
        - Recommended sizes: 1280×1280, 1568×1056, 1056×1568, 1472×1088,
          1088×1472, 1728×960, 960×1728

        Args:
            width: Requested width
            height: Requested height

        Returns:
            Z.AI-compatible size string (e.g., "1280x1280")
        """
        if not width or not height:
            return "1280x1280"  # Default size

        # Clamp to Z.AI limits (512-2048)
        width = max(512, min(2048, width))
        height = max(512, min(2048, height))

        # Round to nearest multiple of 32
        width = round(width / 32) * 32
        height = round(height / 32) * 32

        # Ensure still within bounds after rounding
        width = max(512, min(2048, width))
        height = max(512, min(2048, height))

        return f"{width}x{height}"

    async def text_to_image(
        self,
        params: Any,  # TextToImageParams, but imported later to avoid circular deps
        timeout_s: int | None = None,
        context: Any = None,  # ProcessingContext, but imported later
        node_id: str | None = None,
    ) -> bytes:
        """Generate an image from a text prompt using Z.AI's GLM-Image API.

        Uses the images/generations endpoint which supports the GLM-Image model.
        GLM-Image uses a hybrid architecture of "autoregressive + diffusion decoder"
        for high-quality image generation with excellent text rendering.

        Args:
            params: Text-to-image generation parameters including:
                - model: ImageModel with model ID (e.g., "glm-image")
                - prompt: Text description of the desired image
                - width/height: Desired dimensions (mapped to supported sizes)
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
            raise ValueError("ZHIPU_API_KEY is required for image generation.")

        model_id = params.model.id
        if not model_id:
            raise ValueError("A text-to-image model with a valid id must be specified for image generation.")

        prompt = params.prompt.strip()
        if params.negative_prompt:
            prompt = f"{prompt}\n\nDo not include: {params.negative_prompt.strip()}"

        # Resolve size from width/height parameters
        size = self._resolve_zai_image_size(params.width, params.height)

        log.debug(f"Generating image with Z.AI model={model_id}, size={size}")

        try:
            request_timeout = timeout_s if timeout_s and timeout_s > 0 else 120

            # Build API request
            api_url = f"{self.base_url}/images/generations"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload: dict[str, Any] = {
                "model": model_id,
                "prompt": prompt,
                "size": size,
            }

            log.debug(f"Z.AI image generation request to {api_url}")

            timeout = aiohttp.ClientTimeout(total=request_timeout)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(api_url, headers=headers, json=payload) as response,
            ):
                if response.status != 200:
                    error_text = await response.text()
                    log.error(f"Z.AI image generation failed: HTTP {response.status}: {error_text}")
                    raise RuntimeError(
                        f"Z.AI image generation failed with status {response.status}: {error_text}"
                    )

                result = await response.json()

                # Extract image URL from response
                # Z.AI returns: {"data": [{"url": "..."}]}
                data = result.get("data", [])
                if not data:
                    raise RuntimeError("Z.AI image generation returned no image data.")

                image_url = data[0].get("url")
                if not image_url:
                    raise RuntimeError("Z.AI image generation returned no image URL.")

                log.debug(f"Z.AI returned image URL: {image_url[:100]}...")

            # Download the image from the URL
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(image_url) as img_response,
            ):
                if img_response.status != 200:
                    raise RuntimeError(f"Failed to download image from Z.AI URL: HTTP {img_response.status}")
                image_bytes = await img_response.read()

            log.debug(f"Generated image, size: {len(image_bytes)} bytes")
            return image_bytes

        except aiohttp.ClientError as e:
            log.error(f"Z.AI text-to-image generation failed (network error): {e}")
            raise RuntimeError(f"Z.AI text-to-image generation failed: {e}") from e
        except Exception as exc:
            log.error(f"Z.AI text-to-image generation failed: {exc}")
            raise RuntimeError(f"Z.AI text-to-image generation failed: {exc}") from exc
