"""
HuggingFace image provider implementation.

This module implements the ImageProvider interface for HuggingFace Inference API.
"""

import asyncio
from typing import Any, List, Set
import aiohttp
from nodetool.image.providers.base import ImageProvider
from nodetool.chat.providers.base import ProviderCapability
from nodetool.image.providers.registry import register_image_provider
from nodetool.image.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.config.environment import Environment
from nodetool.metadata.types import InferenceProvider, Provider, ImageModel
from nodetool.ml.models.image_models import CACHE_TTL, image_model_cache
from nodetool.workflows.base_node import ApiKeyMissingError
from huggingface_hub import AsyncInferenceClient
from PIL import Image
from io import BytesIO
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


# Provider mapping for HuggingFace Hub API (image generation providers)
HF_IMAGE_PROVIDER_MAPPING = {
    "black-forest-labs": Provider.HuggingFaceBlackForestLabs,
    "fal-ai": Provider.HuggingFaceFalAI,
    "hf-inference": Provider.HuggingFaceHFInference,
    "nebius": Provider.HuggingFaceNebius,
    "novita": Provider.HuggingFaceNovita,
    "replicate": Provider.HuggingFaceReplicate,
}


async def fetch_image_models_from_hf_provider(provider: str) -> List[ImageModel]:
    """
    Fetch image generation models from HuggingFace Hub API for a specific provider.

    Args:
        provider: The provider value (e.g., "fal-ai", "hf-inference", etc.)

    Returns:
        List of ImageModel instances
    """
    try:
        url = f"https://huggingface.co/api/models?inference_provider={provider}&pipeline_tag=text-to-image&limit=1000"

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
                            provider_enum = HF_IMAGE_PROVIDER_MAPPING.get(provider)
                            if provider_enum is None:
                                log.warning(
                                    f"Unknown provider: {provider}, skipping model: {model_id}"
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


async def get_hf_inference_image_models() -> List[ImageModel]:
    """
    Get HuggingFace image models from in-memory cache or fetch them dynamically.

    Returns:
        List of ImageModel instances from HuggingFace providers
    """
    cache_key = "hf_image_models"

    # Try to get from cache first
    cached_models = image_model_cache.get(cache_key)
    if cached_models is not None:
        log.debug("Using cached HuggingFace image models")
        return cached_models

    log.debug("Fetching HuggingFace image models from API")

    # List of providers to fetch from (image generation focused)
    providers = [
        "black-forest-labs",
        "fal-ai",
        "hf-inference",
        "nebius",
        "novita",
        "replicate",
    ]

    # Fetch models from all providers concurrently
    tasks = [fetch_image_models_from_hf_provider(provider) for provider in providers]
    provider_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine all models
    all_models = []
    for i, result in enumerate(provider_results):
        if isinstance(result, Exception):
            log.error(
                f"Error fetching image models for provider {providers[i]}: {result}"
            )
        elif isinstance(result, list):
            all_models.extend(result)

    # Cache the results in memory
    image_model_cache.set(cache_key, all_models, ttl=CACHE_TTL)
    log.info(f"Cached {len(all_models)} HuggingFace image models in memory")

    return all_models


class HuggingFaceInferenceImageProvider(ImageProvider):
    """Image provider for HuggingFace Inference API."""

    provider_name = "hf_inference"

    def __init__(self):
        super().__init__()
        env = Environment.get_environment()
        api_key = env.get("HF_TOKEN")
        if not api_key:
            raise ApiKeyMissingError("HF_TOKEN is not configured")
        self.api_key = api_key

    def get_capabilities(self) -> Set[ProviderCapability]:
        """HuggingFace Inference provider supports text-to-image generation."""
        return {
            ProviderCapability.TEXT_TO_IMAGE,
        }

    def get_container_env(self) -> dict[str, str]:
        """Return environment variables needed when running inside Docker."""
        return {"HF_TOKEN": self.api_key}

    def get_inference_provider(self, provider: Provider) -> InferenceProvider:
        """Get the HuggingFace inference provider."""
        if provider == Provider.HuggingFaceHFInference:
            return InferenceProvider.hf_inference
        elif provider == Provider.HuggingFaceFalAI:
            return InferenceProvider.fal_ai
        elif provider == Provider.HuggingFaceBlackForestLabs:
            return InferenceProvider.black_forest_labs
        elif provider == Provider.HuggingFaceNebius:
            return InferenceProvider.nebius
        elif provider == Provider.HuggingFaceNovita:
            return InferenceProvider.novita
        elif provider == Provider.HuggingFaceNscale:
            return InferenceProvider.nscale
        elif provider == Provider.HuggingFaceReplicate:
            return InferenceProvider.replicate
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _get_client(self, provider: Provider) -> AsyncInferenceClient:
        """Get the HuggingFace inference client for a specific provider."""
        return AsyncInferenceClient(
            api_key=self.api_key,
            provider=self.get_inference_provider(provider).value,  # type: ignore
        )

    async def text_to_image(
        self, params: TextToImageParams, timeout_s: int | None = None
    ) -> ImageBytes:
        """Generate an image from a text prompt using HuggingFace Inference API.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds

        Returns:
            Raw image bytes as PNG

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        self._log_api_request("text_to_image", params)

        client = self._get_client(params.model.provider)

        # Build kwargs for HF API
        kwargs: dict[str, Any] = {
            "prompt": params.prompt,
            "model": params.model.id,
        }

        # Add optional parameters if provided
        if params.guidance_scale is not None:
            kwargs["guidance_scale"] = params.guidance_scale
        if params.num_inference_steps is not None:
            kwargs["num_inference_steps"] = params.num_inference_steps
        if params.width and params.height:
            kwargs["width"] = params.width
            kwargs["height"] = params.height
        if params.negative_prompt:
            kwargs["negative_prompt"] = params.negative_prompt
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.scheduler:
            kwargs["scheduler"] = params.scheduler

        try:
            # Generate image
            output = await client.text_to_image(**kwargs)

            # Convert PIL Image to bytes
            img_buffer = BytesIO()
            output.save(img_buffer, format="PNG")
            image_bytes = img_buffer.getvalue()

            self.usage["total_requests"] += 1
            self.usage["total_images"] += 1
            self._log_api_response("text_to_image", 1)

            return image_bytes

        except Exception as e:
            raise RuntimeError(f"HuggingFace text-to-image generation failed: {e}")

    async def image_to_image(
        self,
        image: ImageBytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
    ) -> ImageBytes:
        """Transform an image based on a text prompt using HuggingFace Inference API.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds

        Returns:
            Raw image bytes as PNG

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        self._log_api_request("image_to_image", params)

        client = self._get_client(params.model.provider)

        # Build kwargs for HF API
        kwargs = {
            "image": image,
            "prompt": params.prompt,
            "model": params.model.id,
        }

        # Add optional parameters if provided
        if params.guidance_scale is not None:
            kwargs["guidance_scale"] = params.guidance_scale
        if params.num_inference_steps is not None:
            kwargs["num_inference_steps"] = params.num_inference_steps
        if params.negative_prompt:
            kwargs["negative_prompt"] = params.negative_prompt
        if params.strength is not None:
            kwargs["strength"] = params.strength
        if params.target_width and params.target_height:
            kwargs["target_size"] = {
                "width": params.target_width,
                "height": params.target_height,
            }
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.scheduler:
            kwargs["scheduler"] = params.scheduler

        try:
            # Generate image
            output = await client.image_to_image(**kwargs)

            # Convert PIL Image to bytes
            img_buffer = BytesIO()
            output.save(img_buffer, format="PNG")
            image_bytes = img_buffer.getvalue()

            self.usage["total_requests"] += 1
            self.usage["total_images"] += 1
            self._log_api_response("image_to_image", 1)

            return image_bytes

        except Exception as e:
            raise RuntimeError(f"HuggingFace image-to-image generation failed: {e}")

    async def get_available_models(self) -> List[ImageModel]:
        """
        Get available HuggingFace image models.

        Returns both dynamically fetched HF Inference models and locally cached models.

        Returns:
            List of ImageModel instances for HuggingFace
        """
        env = Environment.get_environment()
        models: List[ImageModel] = []

        # Only fetch from HF Inference if API key is available
        if "HF_TOKEN" in env or "HUGGINGFACE_API_KEY" in env:
            try:
                models.extend(await get_hf_inference_image_models())
            except Exception as e:
                log.warning(f"Failed to fetch HF Inference image models: {e}")

        return models
