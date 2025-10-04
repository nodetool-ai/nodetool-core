"""
Gemini image provider implementation.

This module implements the ImageProvider interface for Google's Gemini models
(Imagen and Gemini image generation models).
"""

from typing import List
from nodetool.image.providers.base import ImageProvider
from nodetool.image.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.config.environment import Environment
from nodetool.workflows.base_node import ApiKeyMissingError
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageModel, Provider
from google.genai import Client
from google.genai.types import GenerateImagesConfig, GenerateContentConfig
from google.genai.types import FinishReason
from io import BytesIO
from PIL import Image
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class GeminiImageProvider(ImageProvider):
    """Image provider for Google's Gemini and Imagen models."""

    provider_name = "gemini"

    def __init__(self):
        super().__init__()
        env = Environment.get_environment()
        api_key = env.get("GEMINI_API_KEY")
        if not api_key:
            raise ApiKeyMissingError(
                "GEMINI_API_KEY is not configured in the nodetool settings"
            )
        self.api_key = api_key
        self.client = Client(api_key=api_key).aio

    def get_container_env(self) -> dict[str, str]:
        """Return environment variables needed when running inside Docker."""
        return {"GEMINI_API_KEY": self.api_key}

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
    ) -> ImageBytes:
        """Generate an image from a text prompt using Gemini models.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling

        Returns:
            Raw image bytes as PNG

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        if not params.prompt:
            raise ValueError("The input prompt cannot be empty.")

        self._log_api_request("text_to_image", params)

        try:
            model_id = params.model.id

            # If a Gemini image-capable model is selected, use the generate_content API
            if model_id.startswith("gemini-"):
                log.info(f"Using Gemini image-capable model: {model_id}")

                response = await self.client.models.generate_content(
                    model=model_id,
                    contents=params.prompt,
                    config=GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )

                log.debug(f"Gemini API response: {response}")

                # Extract first inline image from response parts
                if not response or not response.candidates:
                    log.error("No response received from Gemini API")
                    raise RuntimeError("No response received from Gemini API")

                candidate = response.candidates[0]

                if candidate.finish_reason == FinishReason.PROHIBITED_CONTENT:
                    log.error("Prohibited content in the input prompt")
                    raise ValueError("Prohibited content in the input prompt")

                if (
                    not candidate
                    or not candidate.content
                    or not candidate.content.parts
                ):
                    log.error("Invalid response format from Gemini API")
                    raise RuntimeError("Invalid response format from Gemini API")

                image_bytes = None
                for part in candidate.content.parts:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and getattr(inline_data, "data", None):
                        image_bytes = inline_data.data
                        break

                if not image_bytes:
                    raise RuntimeError("No image bytes returned in response")

                self.usage["total_requests"] += 1
                self.usage["total_images"] += 1
                self._log_api_response("text_to_image", 1)

                return image_bytes

            # Otherwise, use the images generation API (Imagen models)
            config = GenerateImagesConfig(
                number_of_images=1,
            )

            response = await self.client.models.generate_images(
                model=model_id,
                prompt=params.prompt,
                config=config,
            )

            if not response.generated_images:
                raise RuntimeError("No images generated")

            image = response.generated_images[0].image
            if not image or not image.image_bytes:
                raise RuntimeError("No image bytes in response")

            self.usage["total_requests"] += 1
            self.usage["total_images"] += 1
            self._log_api_response("text_to_image", 1)

            return image.image_bytes

        except Exception as e:
            log.error(f"Gemini text-to-image generation failed: {e}")
            raise RuntimeError(f"Gemini text-to-image generation failed: {e}")

    async def image_to_image(
        self,
        image: ImageBytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
    ) -> ImageBytes:
        """Transform an image based on a text prompt using Gemini models.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling

        Returns:
            Raw image bytes as PNG

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        if not params.prompt:
            raise ValueError("The input prompt cannot be empty.")

        self._log_api_request("image_to_image", params)

        try:
            model_id = params.model.id

            # Only Gemini image-capable models support image-to-image
            if not model_id.startswith("gemini-"):
                raise ValueError(
                    f"Model {model_id} does not support image-to-image generation. "
                    "Only Gemini models (gemini-*) support this feature."
                )

            log.info(f"Using Gemini image-capable model for image-to-image: {model_id}")

            # Convert image bytes to PIL Image
            pil_image = Image.open(BytesIO(image))

            # Build contents with both prompt and image
            contents = [params.prompt, pil_image]

            response = await self.client.models.generate_content(
                model=model_id,
                contents=contents,
                config=GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            log.debug(f"Gemini API response: {response}")

            # Extract first inline image from response parts
            if not response or not response.candidates:
                log.error("No response received from Gemini API")
                raise RuntimeError("No response received from Gemini API")

            candidate = response.candidates[0]

            if candidate.finish_reason == FinishReason.PROHIBITED_CONTENT:
                log.error("Prohibited content in the input prompt or image")
                raise ValueError("Prohibited content in the input prompt or image")

            if not candidate or not candidate.content or not candidate.content.parts:
                log.error("Invalid response format from Gemini API")
                raise RuntimeError("Invalid response format from Gemini API")

            image_bytes = None
            for part in candidate.content.parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    image_bytes = inline_data.data
                    break

            if not image_bytes:
                raise RuntimeError("No image bytes returned in response")

            self.usage["total_requests"] += 1
            self.usage["total_images"] += 1
            self._log_api_response("image_to_image", 1)

            return image_bytes

        except Exception as e:
            log.error(f"Gemini image-to-image generation failed: {e}")
            raise RuntimeError(f"Gemini image-to-image generation failed: {e}")

    async def get_available_models(self) -> List[ImageModel]:
        """
        Get available Gemini image models.

        Returns models only if GEMINI_API_KEY is configured.

        Returns:
            List of ImageModel instances for Gemini
        """
        env = Environment.get_environment()
        if "GEMINI_API_KEY" not in env:
            return []

        models = [
            # Gemini image-capable models (support both text-to-image and image-to-image)
            ImageModel(
                id="gemini-2.0-flash-preview-image-generation",
                name="Gemini 2.0 Flash Preview (Image Gen)",
                provider=Provider.Gemini,
            ),
            ImageModel(
                id="gemini-2.5-flash-image-preview",
                name="Gemini 2.5 Flash (Image Preview)",
                provider=Provider.Gemini,
            ),
            # Imagen models (text-to-image only)
            ImageModel(
                id="imagen-3.0-generate-001",
                name="Imagen 3.0 Generate 001",
                provider=Provider.Gemini,
            ),
            ImageModel(
                id="imagen-3.0-generate-002",
                name="Imagen 3.0 Generate 002",
                provider=Provider.Gemini,
            ),
            ImageModel(
                id="imagen-4.0-generate-preview-06-06",
                name="Imagen 4.0 Preview",
                provider=Provider.Gemini,
            ),
            ImageModel(
                id="imagen-4.0-ultra-generate-preview-06-06",
                name="Imagen 4.0 Ultra Preview",
                provider=Provider.Gemini,
            ),
        ]

        return models
