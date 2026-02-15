"""
Meshy AI provider implementation for 3D model generation.

This module implements the Provider interface for Meshy AI's 3D generation APIs:
- Text-to-3D: Generate 3D models from text descriptions
- Image-to-3D: Generate 3D models from images

Meshy AI API Documentation: https://docs.meshy.ai/
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import aiohttp

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Model3DModel,
    Provider,
)
from nodetool.providers.base import BaseProvider, register_provider

if TYPE_CHECKING:
    from nodetool.providers.types import (
        ImageTo3DParams,
        TextTo3DParams,
    )

log = get_logger(__name__)

# Meshy API endpoints
MESHY_API_BASE_URL = "https://api.meshy.ai"

# Model definitions for Meshy AI
MESHY_3D_MODELS = [
    # Text-to-3D models
    Model3DModel(
        id="meshy-4",
        name="Meshy-4 Text-to-3D",
        provider=Provider.Meshy,
        supported_tasks=["text_to_3d"],
        output_formats=["glb", "fbx", "obj", "usdz"],
    ),
    Model3DModel(
        id="meshy-3-turbo",
        name="Meshy-3 Turbo Text-to-3D",
        provider=Provider.Meshy,
        supported_tasks=["text_to_3d"],
        output_formats=["glb", "fbx", "obj", "usdz"],
    ),
    # Image-to-3D models
    Model3DModel(
        id="meshy-4-image",
        name="Meshy-4 Image-to-3D",
        provider=Provider.Meshy,
        supported_tasks=["image_to_3d"],
        output_formats=["glb", "fbx", "obj", "usdz"],
    ),
    Model3DModel(
        id="meshy-3-turbo-image",
        name="Meshy-3 Turbo Image-to-3D",
        provider=Provider.Meshy,
        supported_tasks=["image_to_3d"],
        output_formats=["glb", "fbx", "obj", "usdz"],
    ),
]


@register_provider(Provider.Meshy)
class MeshyProvider(BaseProvider):
    """Provider for Meshy AI 3D generation services.

    Meshy AI offers text-to-3D and image-to-3D generation capabilities
    through their API. This provider supports both generation modes.
    """

    provider_name = "meshy"

    # Polling configuration
    _poll_interval: float = 5.0  # seconds between status checks
    _max_poll_attempts: int = 120  # 10 minutes max at 5s intervals

    @classmethod
    def required_secrets(cls) -> list[str]:
        """Return the required secrets for this provider."""
        return ["MESHY_API_KEY"]

    def __init__(self, secrets: dict[str, str] | None = None, **kwargs: Any):
        """Initialize the Meshy provider.

        Args:
            secrets: Dictionary containing MESHY_API_KEY
            **kwargs: Additional configuration options
        """
        super().__init__(secrets=secrets)
        self.api_key = secrets.get("MESHY_API_KEY") if secrets else None
        if not self.api_key:
            log.warning("Meshy API key not configured")

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _submit_text_to_3d_task(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        art_style: str | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> str:
        """Submit a text-to-3D generation task.

        Args:
            session: aiohttp session
            prompt: Text description of the 3D model
            art_style: Art style (realistic, cartoon, etc.)
            negative_prompt: Elements to avoid
            seed: Random seed for reproducibility

        Returns:
            Task ID for polling status
        """
        payload: dict[str, Any] = {
            "mode": "preview",
            "prompt": prompt,
        }

        if art_style:
            payload["art_style"] = art_style
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = seed

        url = f"{MESHY_API_BASE_URL}/v2/text-to-3d"
        async with session.post(url, json=payload, headers=self._get_headers()) as response:
            if response.status != 200 and response.status != 202:
                error_text = await response.text()
                raise RuntimeError(f"Meshy API error ({response.status}): {error_text}")
            data = await response.json()
            return data["result"]

    async def _submit_image_to_3d_task(
        self,
        session: aiohttp.ClientSession,
        image_url: str,
    ) -> str:
        """Submit an image-to-3D generation task.

        Args:
            session: aiohttp session
            image_url: URL of the input image

        Returns:
            Task ID for polling status
        """
        payload: dict[str, Any] = {
            "image_url": image_url,
        }

        url = f"{MESHY_API_BASE_URL}/v1/image-to-3d"
        async with session.post(url, json=payload, headers=self._get_headers()) as response:
            if response.status != 200 and response.status != 202:
                error_text = await response.text()
                raise RuntimeError(f"Meshy API error ({response.status}): {error_text}")
            data = await response.json()
            return data["result"]

    async def _poll_task_status(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        endpoint: str,
        poll_interval: float,
        max_attempts: int,
    ) -> dict[str, Any]:
        """Poll for task completion.

        Args:
            session: aiohttp session
            task_id: The task ID to poll
            endpoint: API endpoint for status check
            poll_interval: Seconds between polls
            max_attempts: Maximum polling attempts

        Returns:
            Task result data

        Raises:
            RuntimeError: If task fails or times out
        """
        url = f"{MESHY_API_BASE_URL}{endpoint}/{task_id}"

        for attempt in range(max_attempts):
            async with session.get(url, headers=self._get_headers()) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Meshy API poll error ({response.status}): {error_text}")

                data = await response.json()
                status = data.get("status", "").upper()

                if status == "SUCCEEDED":
                    return data
                elif status in ("FAILED", "EXPIRED"):
                    error_msg = data.get("task_error", {}).get("message", "Unknown error")
                    raise RuntimeError(f"Meshy task failed: {error_msg}")

                log.debug(f"Meshy task {task_id} status: {status} (attempt {attempt + 1}/{max_attempts})")

            await asyncio.sleep(poll_interval)

        raise RuntimeError(f"Meshy task {task_id} timed out after {max_attempts * poll_interval}s")

    async def _download_result(self, session: aiohttp.ClientSession, url: str) -> bytes:
        """Download the result file.

        Args:
            session: aiohttp session
            url: URL of the result file

        Returns:
            Raw bytes of the 3D model
        """
        async with session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Failed to download result ({response.status}): {error_text}")
            return await response.read()

    async def _upload_image(self, image: bytes) -> str:
        """Upload an image to Meshy for image-to-3D generation.

        Meshy requires images to be accessible via URL. For images that aren't
        already hosted, we need to upload them first.

        Args:
            image: Image bytes to upload

        Returns:
            URL of the uploaded image
        """
        import base64

        # For now, use data URI as Meshy supports base64 images
        base64_image = base64.b64encode(image).decode("utf-8")
        # Detect image type (simplified - assume PNG or JPEG)
        mime_type = "image/png" if image[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return f"data:{mime_type};base64,{base64_image}"

    async def get_available_3d_models(self) -> list[Model3DModel]:
        """Get available Meshy 3D generation models."""
        if not self.api_key:
            log.debug("No Meshy API key configured, returning empty 3D model list")
            return []
        return MESHY_3D_MODELS

    async def text_to_3d(
        self,
        params: TextTo3DParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate a 3D model from a text prompt using Meshy AI.

        Args:
            params: Text-to-3D generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            node_id: Optional node ID for tracking

        Returns:
            Raw 3D model bytes (GLB format by default)

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating 3D model with Meshy model: {params.model.id}")

        if not params.prompt:
            raise ValueError("prompt must not be empty")

        if not self.api_key:
            raise ValueError("Meshy API key is not configured")

        try:
            # Calculate max attempts from timeout if provided
            poll_interval = self._poll_interval
            max_attempts = self._max_poll_attempts
            if timeout_s is not None and timeout_s > 0:
                max_attempts = max(1, int(timeout_s / poll_interval))

            async with aiohttp.ClientSession() as session:
                # Submit the task
                task_id = await self._submit_text_to_3d_task(
                    session,
                    prompt=params.prompt,
                    art_style=params.art_style,
                    negative_prompt=params.negative_prompt,
                    seed=params.seed,
                )

                log.debug(f"Meshy text-to-3D task submitted: {task_id}")

                # Poll for completion
                result = await self._poll_task_status(
                    session,
                    task_id,
                    endpoint="/v2/text-to-3d",
                    poll_interval=poll_interval,
                    max_attempts=max_attempts,
                )

                # Get the model URL based on requested format
                output_format = params.output_format.lower()
                model_urls = result.get("model_urls", {})
                model_url = model_urls.get(output_format) or model_urls.get("glb")

                if not model_url:
                    raise RuntimeError(f"No model URL found in response for format: {output_format}")

                # Download the result
                model_bytes = await self._download_result(session, model_url)
                log.debug(f"Generated {len(model_bytes)} bytes of 3D model data")
                return model_bytes

        except Exception as e:
            log.error(f"Meshy text-to-3D generation failed: {e}")
            raise RuntimeError(f"Meshy text-to-3D generation failed: {str(e)}") from e

    async def image_to_3d(
        self,
        image: bytes,
        params: ImageTo3DParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate a 3D model from an image using Meshy AI.

        Args:
            image: Input image as bytes
            params: Image-to-3D generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            node_id: Optional node ID for tracking

        Returns:
            Raw 3D model bytes (GLB format by default)

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating 3D model from image with Meshy model: {params.model.id}")

        if not image:
            raise ValueError("image must not be empty")

        if not self.api_key:
            raise ValueError("Meshy API key is not configured")

        try:
            # Calculate max attempts from timeout if provided
            poll_interval = self._poll_interval
            max_attempts = self._max_poll_attempts
            if timeout_s is not None and timeout_s > 0:
                max_attempts = max(1, int(timeout_s / poll_interval))

            async with aiohttp.ClientSession() as session:
                # Upload image and get URL
                image_url = await self._upload_image(image)

                # Submit the task
                task_id = await self._submit_image_to_3d_task(session, image_url)

                log.debug(f"Meshy image-to-3D task submitted: {task_id}")

                # Poll for completion
                result = await self._poll_task_status(
                    session,
                    task_id,
                    endpoint="/v1/image-to-3d",
                    poll_interval=poll_interval,
                    max_attempts=max_attempts,
                )

                # Get the model URL based on requested format
                output_format = params.output_format.lower()
                model_urls = result.get("model_urls", {})
                model_url = model_urls.get(output_format) or model_urls.get("glb")

                if not model_url:
                    raise RuntimeError(f"No model URL found in response for format: {output_format}")

                # Download the result
                model_bytes = await self._download_result(session, model_url)
                log.debug(f"Generated {len(model_bytes)} bytes of 3D model data")
                return model_bytes

        except Exception as e:
            log.error(f"Meshy image-to-3D generation failed: {e}")
            raise RuntimeError(f"Meshy image-to-3D generation failed: {str(e)}") from e
