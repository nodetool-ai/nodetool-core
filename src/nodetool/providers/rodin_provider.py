"""
Rodin AI provider implementation for 3D model generation.

This module implements the Provider interface for Rodin AI's 3D generation APIs:
- Text-to-3D: Generate 3D models from text descriptions
- Image-to-3D: Generate 3D models from images

Rodin AI (by Hyperhuman) API Documentation: https://hyperhuman.deemos.com/api
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

# Rodin API endpoints
RODIN_API_BASE_URL = "https://hyperhuman.deemos.com/api"

# Model definitions for Rodin AI
RODIN_3D_MODELS = [
    # Image-to-3D models (Rodin's primary capability)
    Model3DModel(
        id="rodin-gen-1",
        name="Rodin Gen-1 Image-to-3D",
        provider=Provider.Rodin,
        supported_tasks=["image_to_3d"],
        output_formats=["glb", "fbx", "obj", "usdz"],
    ),
    Model3DModel(
        id="rodin-gen-1-turbo",
        name="Rodin Gen-1 Turbo Image-to-3D",
        provider=Provider.Rodin,
        supported_tasks=["image_to_3d"],
        output_formats=["glb", "fbx", "obj", "usdz"],
    ),
    # Text-to-3D models
    Model3DModel(
        id="rodin-sketch",
        name="Rodin Sketch Text-to-3D",
        provider=Provider.Rodin,
        supported_tasks=["text_to_3d"],
        output_formats=["glb", "fbx", "obj", "usdz"],
    ),
]


@register_provider(Provider.Rodin)
class RodinProvider(BaseProvider):
    """Provider for Rodin AI (Hyperhuman) 3D generation services.

    Rodin AI offers high-quality image-to-3D and text-to-3D generation
    capabilities through their API.
    """

    provider_name = "rodin"

    # Polling configuration
    _poll_interval: float = 5.0  # seconds between status checks
    _max_poll_attempts: int = 120  # 10 minutes max at 5s intervals

    @classmethod
    def required_secrets(cls) -> list[str]:
        """Return the required secrets for this provider."""
        return ["RODIN_API_KEY"]

    def __init__(self, secrets: dict[str, str] | None = None, **kwargs: Any):
        """Initialize the Rodin provider.

        Args:
            secrets: Dictionary containing RODIN_API_KEY
            **kwargs: Additional configuration options
        """
        super().__init__(secrets=secrets)
        self.api_key = secrets.get("RODIN_API_KEY") if secrets else None
        if not self.api_key:
            log.warning("Rodin API key not configured")

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _submit_rodin_task(
        self,
        session: aiohttp.ClientSession,
        images: list[dict[str, str]],
        prompt: str | None = None,
        seed: int | None = None,
        geometry_file_format: str = "glb",
    ) -> tuple[str, str]:
        """Submit a Rodin generation task.

        Args:
            session: aiohttp session
            images: List of image data with type and data fields
            prompt: Optional text prompt
            seed: Random seed for reproducibility
            geometry_file_format: Output format (glb, fbx, obj, usdz)

        Returns:
            Tuple of (task_uuid, subscription_key) for polling status
        """
        payload: dict[str, Any] = {
            "images": images,
            "geometry_file_format": geometry_file_format.upper(),
        }

        if prompt:
            payload["prompt"] = prompt
        if seed is not None:
            payload["seed"] = seed

        url = f"{RODIN_API_BASE_URL}/v2/rodin"
        async with session.post(url, json=payload, headers=self._get_headers()) as response:
            if response.status != 200 and response.status != 202:
                error_text = await response.text()
                raise RuntimeError(f"Rodin API error ({response.status}): {error_text}")
            data = await response.json()
            # Extract UUID and subscription_key from response
            uuids = data.get("uuids", [])
            if not uuids:
                raise RuntimeError("No task UUID returned from Rodin API")
            subscription_key = data.get("subscription_key", "")
            if not subscription_key:
                raise RuntimeError("No subscription key returned from Rodin API")
            return uuids[0], subscription_key

    async def _poll_task_status(
        self,
        session: aiohttp.ClientSession,
        subscription_key: str,
        poll_interval: float,
        max_attempts: int,
    ) -> dict[str, Any]:
        """Poll for task completion using subscription key.

        Args:
            session: aiohttp session
            subscription_key: The subscription key for the task
            poll_interval: Seconds between polls
            max_attempts: Maximum polling attempts

        Returns:
            Task result data

        Raises:
            RuntimeError: If task fails or times out
        """
        url = f"{RODIN_API_BASE_URL}/v2/status"

        for attempt in range(max_attempts):
            payload = {"subscription_key": subscription_key}
            async with session.post(url, json=payload, headers=self._get_headers()) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Rodin API poll error ({response.status}): {error_text}")

                data = await response.json()
                jobs = data.get("jobs", [])

                if not jobs:
                    log.debug(f"Rodin task status: No jobs yet (attempt {attempt + 1}/{max_attempts})")
                    await asyncio.sleep(poll_interval)
                    continue

                job = jobs[0]
                status = job.get("status", "").upper()

                if status == "DONE":
                    return job
                elif status in ("FAILED", "ERROR", "CANCELLED"):
                    error_msg = job.get("error", "Unknown error")
                    raise RuntimeError(f"Rodin task failed: {error_msg}")

                log.debug(f"Rodin task status: {status} (attempt {attempt + 1}/{max_attempts})")

            await asyncio.sleep(poll_interval)

        raise RuntimeError(f"Rodin task timed out after {max_attempts * poll_interval}s")

    async def _download_result(
        self,
        session: aiohttp.ClientSession,
        task_uuid: str,
    ) -> bytes:
        """Download the result file.

        Args:
            session: aiohttp session
            task_uuid: The task UUID

        Returns:
            Raw bytes of the 3D model
        """
        url = f"{RODIN_API_BASE_URL}/v2/download"
        payload = {"task_uuid": task_uuid}

        async with session.post(url, json=payload, headers=self._get_headers()) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Failed to get download URL ({response.status}): {error_text}")
            data = await response.json()
            download_url = data.get("model_url") or data.get("url")

            if not download_url:
                raise RuntimeError("No download URL in response")

        # Download the actual file
        async with session.get(download_url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Failed to download result ({response.status}): {error_text}")
            return await response.read()

    async def _encode_image(self, image: bytes) -> dict[str, str]:
        """Encode an image for Rodin API.

        Args:
            image: Image bytes

        Returns:
            Dictionary with type and base64-encoded data
        """
        import base64

        base64_image = base64.b64encode(image).decode("utf-8")

        # Detect image type
        if image[:8] == b"\x89PNG\r\n\x1a\n":
            image_type = "png"
        elif image[:3] == b"\xff\xd8\xff":
            image_type = "jpeg"
        else:
            image_type = "png"  # Default to PNG

        return {
            "type": image_type,
            "data": base64_image,
        }

    async def get_available_3d_models(self) -> list[Model3DModel]:
        """Get available Rodin 3D generation models."""
        if not self.api_key:
            log.debug("No Rodin API key configured, returning empty 3D model list")
            return []
        return RODIN_3D_MODELS

    async def text_to_3d(
        self,
        params: TextTo3DParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate a 3D model from a text prompt using Rodin AI.

        Note: Rodin's text-to-3D works by generating an image from the prompt
        first, then converting to 3D. The API handles this internally.

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
        log.debug(f"Generating 3D model with Rodin model: {params.model.id}")

        if not params.prompt:
            raise ValueError("prompt must not be empty")

        if not self.api_key:
            raise ValueError("Rodin API key is not configured")

        try:
            # Calculate max attempts from timeout if provided
            poll_interval = self._poll_interval
            max_attempts = self._max_poll_attempts
            if timeout_s is not None and timeout_s > 0:
                max_attempts = max(1, int(timeout_s / poll_interval))

            async with aiohttp.ClientSession() as session:
                # Rodin text-to-3D uses a sketch/prompt-based approach
                # Submit with empty images but with prompt
                task_uuid, subscription_key = await self._submit_rodin_task(
                    session,
                    images=[],  # Empty images for text-to-3D
                    prompt=params.prompt,
                    seed=params.seed,
                    geometry_file_format=params.output_format,
                )

                log.debug(f"Rodin text-to-3D task submitted: {task_uuid}")

                # Poll for completion
                await self._poll_task_status(
                    session,
                    subscription_key=subscription_key,
                    poll_interval=poll_interval,
                    max_attempts=max_attempts,
                )

                # Download the result
                model_bytes = await self._download_result(session, task_uuid)
                log.debug(f"Generated {len(model_bytes)} bytes of 3D model data")
                return model_bytes

        except Exception as e:
            log.error(f"Rodin text-to-3D generation failed: {e}")
            raise RuntimeError(f"Rodin text-to-3D generation failed: {str(e)}") from e

    async def image_to_3d(
        self,
        image: bytes,
        params: ImageTo3DParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate a 3D model from an image using Rodin AI.

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
        log.debug(f"Generating 3D model from image with Rodin model: {params.model.id}")

        if not image:
            raise ValueError("image must not be empty")

        if not self.api_key:
            raise ValueError("Rodin API key is not configured")

        try:
            # Calculate max attempts from timeout if provided
            poll_interval = self._poll_interval
            max_attempts = self._max_poll_attempts
            if timeout_s is not None and timeout_s > 0:
                max_attempts = max(1, int(timeout_s / poll_interval))

            async with aiohttp.ClientSession() as session:
                # Encode image
                encoded_image = await self._encode_image(image)

                # Submit the task and get both task_uuid and subscription_key
                task_uuid, subscription_key = await self._submit_rodin_task(
                    session,
                    images=[encoded_image],
                    prompt=params.prompt,
                    seed=params.seed,
                    geometry_file_format=params.output_format,
                )

                log.debug(f"Rodin image-to-3D task submitted: {task_uuid}")

                # Poll for completion
                await self._poll_task_status(
                    session,
                    subscription_key=subscription_key,
                    poll_interval=poll_interval,
                    max_attempts=max_attempts,
                )

                # Download the result
                model_bytes = await self._download_result(session, task_uuid)
                log.debug(f"Generated {len(model_bytes)} bytes of 3D model data")
                return model_bytes

        except Exception as e:
            log.error(f"Rodin image-to-3D generation failed: {e}")
            raise RuntimeError(f"Rodin image-to-3D generation failed: {str(e)}") from e
