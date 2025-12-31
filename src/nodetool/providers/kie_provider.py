"""
Kie.ai provider implementation for image, video, and audio generation.

This module implements the Provider interface for Kie.ai's various APIs:
- Text-to-Image: Flux2Pro, Seedream45, ZImage, NanoBanana, FluxKontext, etc.
- Image-to-Image: Flux2ProImageToImage, Seedream45Edit, etc.
- Text-to-Video: Sora2, Seedance, Kling, Hailuo, GrokImagine, etc.
- Image-to-Video: Sora2Pro, SeedanceImageToVideo, HailuoImageToVideo, etc.

Kie.ai API Documentation: https://api.kie.ai/docs
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, List

import aiohttp

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    ImageModel,
    Provider,
    VideoModel,
)
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.types import (
    ImageToImageParams,
    ImageToVideoParams,
    TextToImageParams,
    TextToVideoParams,
)

log = get_logger(__name__)

# Kie.ai API endpoints
KIE_API_BASE_URL = "https://api.kie.ai"


# Model definitions for Kie.ai
KIE_IMAGE_MODELS = [
    # Text-to-Image models
    ImageModel(
        id="flux-2/pro-text-to-image",
        name="Flux 2 Pro Text-to-Image",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="flux-2/flex-text-to-image",
        name="Flux 2 Flex Text-to-Image",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="seedream/4.5-text-to-image",
        name="Seedream 4.5 Text-to-Image",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="z-image",
        name="Z-Image Turbo",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="google/nano-banana",
        name="Nano Banana (Gemini 2.5)",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="nano-banana-pro",
        name="Nano Banana Pro (Gemini 3.0)",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="flux-kontext",
        name="Flux Kontext",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="grok-imagine/text-to-image",
        name="Grok Imagine Text-to-Image",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="qwen/text-to-image",
        name="Qwen Text-to-Image",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="google/imagen4-fast",
        name="Imagen 4 Fast",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="google/imagen4-ultra",
        name="Imagen 4 Ultra",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    ImageModel(
        id="google/imagen4",
        name="Imagen 4",
        provider=Provider.KIE,
        supported_tasks=["text_to_image"],
    ),
    # Image-to-Image models
    ImageModel(
        id="flux-2/pro-image-to-image",
        name="Flux 2 Pro Image-to-Image",
        provider=Provider.KIE,
        supported_tasks=["image_to_image"],
    ),
    ImageModel(
        id="flux-2/flex-image-to-image",
        name="Flux 2 Flex Image-to-Image",
        provider=Provider.KIE,
        supported_tasks=["image_to_image"],
    ),
    ImageModel(
        id="seedream/4.5-edit",
        name="Seedream 4.5 Edit",
        provider=Provider.KIE,
        supported_tasks=["image_to_image"],
    ),
    ImageModel(
        id="qwen/image-to-image",
        name="Qwen Image-to-Image",
        provider=Provider.KIE,
        supported_tasks=["image_to_image"],
    ),
    ImageModel(
        id="google/nano-banana-edit",
        name="Nano Banana Edit",
        provider=Provider.KIE,
        supported_tasks=["image_to_image"],
    ),
    # Upscaling / Enhancement models
    ImageModel(
        id="grok-imagine/upscale",
        name="Grok Imagine Upscale",
        provider=Provider.KIE,
        supported_tasks=["upscale"],
    ),
    ImageModel(
        id="topaz/image-upscale",
        name="Topaz Image Upscale",
        provider=Provider.KIE,
        supported_tasks=["upscale"],
    ),
    ImageModel(
        id="recraft/crisp-upscale",
        name="Recraft Crisp Upscale",
        provider=Provider.KIE,
        supported_tasks=["upscale"],
    ),
    ImageModel(
        id="recraft/remove-background",
        name="Recraft Remove Background",
        provider=Provider.KIE,
        supported_tasks=["remove_background"],
    ),
    ImageModel(
        id="ideogram/character-remix",
        name="Ideogram Character Remix",
        provider=Provider.KIE,
        supported_tasks=["image_to_image"],
    ),
    ImageModel(
        id="ideogram/v3-reframe",
        name="Ideogram V3 Reframe",
        provider=Provider.KIE,
        supported_tasks=["image_to_image"],
    ),
]

KIE_VIDEO_MODELS = [
    # Text-to-Video models
    VideoModel(
        id="kling-2.6/text-to-video",
        name="Kling 2.6 Text-to-Video",
        provider=Provider.KIE,
        supported_tasks=["text_to_video"],
    ),
    VideoModel(
        id="grok-imagine/text-to-video",
        name="Grok Imagine Text-to-Video",
        provider=Provider.KIE,
        supported_tasks=["text_to_video"],
    ),
    VideoModel(
        id="seedance/v1-lite-text-to-video",
        name="Seedance V1 Lite Text-to-Video",
        provider=Provider.KIE,
        supported_tasks=["text_to_video"],
    ),
    VideoModel(
        id="seedance/v1-pro-text-to-video",
        name="Seedance V1 Pro Text-to-Video",
        provider=Provider.KIE,
        supported_tasks=["text_to_video"],
    ),
    VideoModel(
        id="hailuo/2-3-text-to-video-pro",
        name="Hailuo 2.3 Text-to-Video Pro",
        provider=Provider.KIE,
        supported_tasks=["text_to_video"],
    ),
    VideoModel(
        id="hailuo/2-3-text-to-video-standard",
        name="Hailuo 2.3 Text-to-Video Standard",
        provider=Provider.KIE,
        supported_tasks=["text_to_video"],
    ),
    VideoModel(
        id="sora-2-pro-text-to-video",
        name="Sora 2 Pro Text-to-Video",
        provider=Provider.KIE,
        supported_tasks=["text_to_video"],
    ),
    VideoModel(
        id="sora-2-text-to-video",
        name="Sora 2 Text-to-Video",
        provider=Provider.KIE,
        supported_tasks=["text_to_video"],
    ),
    VideoModel(
        id="wan/v2-1-multi-shot-text-to-video-pro",
        name="Wan 2.1 Multi-Shot Text-to-Video Pro",
        provider=Provider.KIE,
        supported_tasks=["text_to_video"],
    ),
    VideoModel(
        id="wan/2-6-text-to-video",
        name="Wan 2.6 Text-to-Video",
        provider=Provider.KIE,
        supported_tasks=["text_to_video"],
    ),
    # Image-to-Video models
    VideoModel(
        id="kling-2.6/image-to-video",
        name="Kling 2.6 Image-to-Video",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
    VideoModel(
        id="grok-imagine/image-to-video",
        name="Grok Imagine Image-to-Video",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
    VideoModel(
        id="seedance/v1-lite-image-to-video",
        name="Seedance V1 Lite Image-to-Video",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
    VideoModel(
        id="seedance/v1-pro-image-to-video",
        name="Seedance V1 Pro Image-to-Video",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
    VideoModel(
        id="seedance/v1-pro-fast-image-to-video",
        name="Seedance V1 Pro Fast Image-to-Video",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
    VideoModel(
        id="hailuo/2-3-image-to-video-pro",
        name="Hailuo 2.3 Image-to-Video Pro",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
    VideoModel(
        id="hailuo/2-3-image-to-video-standard",
        name="Hailuo 2.3 Image-to-Video Standard",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
    VideoModel(
        id="sora-2-pro-image-to-video",
        name="Sora 2 Pro Image-to-Video",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
    VideoModel(
        id="sora-2-pro-story-board",
        name="Sora 2 Pro Storyboard",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
    VideoModel(
        id="wan/2-6-image-to-video",
        name="Wan 2.6 Image-to-Video",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
    VideoModel(
        id="wan/2-6-video-to-video",
        name="Wan 2.6 Video-to-Video",
        provider=Provider.KIE,
        supported_tasks=["video_to_video"],
    ),
    # Avatar models
    VideoModel(
        id="kling/v1-avatar-standard",
        name="Kling V1 Avatar Standard",
        provider=Provider.KIE,
        supported_tasks=["avatar"],
    ),
    VideoModel(
        id="kling/v1-avatar-pro",
        name="Kling V1 Avatar Pro",
        provider=Provider.KIE,
        supported_tasks=["avatar"],
    ),
    # Upscaling models
    VideoModel(
        id="topaz-video-upscaler",
        name="Topaz Video Upscaler",
        provider=Provider.KIE,
        supported_tasks=["upscale"],
    ),
    VideoModel(
        id="infinitalk/from-audio",
        name="Infinitalk From Audio",
        provider=Provider.KIE,
        supported_tasks=["image_to_video"],
    ),
]


@register_provider(Provider.KIE)
class KieProvider(BaseProvider):
    """Kie.ai implementation of the Provider interface.

    Kie.ai provides access to various image and video generation models
    through a unified API. This provider implements the text-to-image,
    image-to-image, text-to-video, and image-to-video generation methods.

    The API uses an async task-based approach:
    1. Submit a task (POST to createTask endpoint)
    2. Poll for task completion (GET recordInfo endpoint)
    3. Download the result from the result URL

    For details, see: https://api.kie.ai/docs
    """

    provider_name: str = "kie"

    # Polling configuration
    _poll_interval: float = 2.0
    _max_poll_attempts: int = 180  # 6 minutes max
    _image_poll_interval: float = 1.5
    _image_max_poll_attempts: int = 60
    _video_poll_interval: float = 8.0
    _video_max_poll_attempts: int = 240

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["KIE_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the Kie.ai provider with API credentials."""
        super().__init__(secrets)
        if "KIE_API_KEY" not in secrets or not secrets["KIE_API_KEY"]:
            log.warning("KIE_API_KEY not found in secrets")
            raise ValueError("KIE_API_KEY is required but not provided")
        self.api_key = secrets["KIE_API_KEY"]
        log.debug("KieProvider initialized")

    def _get_headers(self) -> dict[str, str]:
        """Get common headers for Kie.ai API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _extract_kie_params(self, params: Any, input_params: dict[str, Any]) -> None:
        """Extract Kie-specific parameters from generic params object.

        Follows parameters defined in Kie nodes in nodetool-base.
        If the default params do not cover those, it falls back to default params.
        """
        # List of attributes to check on the params object based on Kie nodes
        attributes = [
            "aspect_ratio",
            "resolution",
            "quality",
            "duration",
            "steps",
            "guidance_scale",
            "seed",
            "strength",
            "remove_watermark",
            "sound",
            "mode",
            "audio_url",
            "video_url",
            "image_url",
            "image_urls",
            "upscale_factor",
            "lyrics",
            "style",
            "instrumental",
            "negative_prompt",
            "image_size",
            "image_input",
            "rendering_speed",
            "expand_prompt",
            "num_images",
            "reference_image_urls",
            "reference_mask_urls",
            "output_format",
            "model_version",
        ]

        # Use getattr to pick up any extra fields allowed by ConfigDict(extra="allow")
        for attr in attributes:
            val = getattr(params, attr, None)
            if val is not None:
                # Map some internal names to Kie API names or handle special types
                if attr == "steps":
                    input_params["steps"] = val
                elif attr == "duration":
                    # Convert to string if it's an int, as some Kie APIs expect strings for duration
                    input_params["duration"] = str(val) if isinstance(val, int) else val
                elif attr == "aspect_ratio" and hasattr(val, "value"):
                    # Handle Enums
                    input_params["aspect_ratio"] = val.value
                elif attr == "resolution" and hasattr(val, "value"):
                    input_params["resolution"] = val.value
                elif attr == "quality" and hasattr(val, "value"):
                    input_params["quality"] = val.value
                elif attr == "mode" and hasattr(val, "value"):
                    input_params["mode"] = val.value
                elif attr == "style" and hasattr(val, "value"):
                    input_params["style"] = val.value
                elif attr == "image_size" and hasattr(val, "value"):
                    input_params["image_size"] = val.value
                elif attr == "rendering_speed" and hasattr(val, "value"):
                    input_params["rendering_speed"] = val.value
                elif attr == "model_version":
                     # Map model_version to model in input params (e.g. for Suno)
                    input_params["model"] = val.value if hasattr(val, "value") else val
                elif attr == "image_url" and "image_url" not in input_params:
                    input_params["image_url"] = val
                elif attr == "image_urls" and "image_urls" not in input_params:
                    input_params["image_urls"] = val
                else:
                    # Extra params from Kie nodes take precedence over default calculated params
                    input_params[attr] = val

    def _check_response_status(self, response_data: dict) -> None:
        """Check response status code and raise appropriate error."""
        try:
            status = int(response_data.get("code", 0))
        except (ValueError, TypeError):
            return

        error_map = {
            401: "Unauthorized - Authentication credentials are missing or invalid",
            402: "Insufficient Credits - Account does not have enough credits",
            404: "Not Found - The requested resource does not exist",
            422: "Validation Error - Request parameters failed validation",
            429: "Rate Limited - Request limit exceeded",
            455: "Service Unavailable - System undergoing maintenance",
            500: "Server Error - Unexpected error occurred",
            501: "Generation Failed - Content generation task failed",
            505: "Feature Disabled - Requested feature is currently disabled",
        }
        if status in error_map:
            raise ValueError(f"{error_map[status]}: {response_data}")

    async def _submit_task(
        self,
        session: aiohttp.ClientSession,
        model: str,
        input_params: dict[str, Any],
    ) -> str:
        """Submit a task to the Kie.ai API and return the task ID."""
        url = f"{KIE_API_BASE_URL}/api/v1/jobs/createTask"
        payload = {
            "model": model,
            "input": input_params,
        }
        headers = self._get_headers()

        log.debug(f"Submitting task to {url} with model: {model}")
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            self._check_response_status(response_data)

            if response.status != 200:
                raise ValueError(f"Failed to submit task: {response.status} - {response_data}")

            task_id = response_data.get("data", {}).get("taskId")
            if not task_id:
                raise ValueError(f"Could not extract taskId from response: {response_data}")

            log.debug(f"Task submitted with ID: {task_id}")
            return task_id

    async def _poll_status(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        poll_interval: float,
        max_attempts: int,
    ) -> dict[str, Any]:
        """Poll for task completion status."""
        url = f"{KIE_API_BASE_URL}/api/v1/jobs/recordInfo?taskId={task_id}"
        headers = self._get_headers()

        for attempt in range(max_attempts):
            log.debug(f"Polling task status (attempt {attempt + 1}/{max_attempts})")
            async with session.get(url, headers=headers) as response:
                status_data = await response.json()
                self._check_response_status(status_data)

                state = status_data.get("data", {}).get("state", "")

                if state == "success":
                    log.debug("Task completed successfully")
                    return status_data

                if state == "failed":
                    error_msg = status_data.get("data", {}).get("failMsg", "Unknown error")
                    raise ValueError(f"Task failed: {error_msg}")

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Task did not complete within {max_attempts * poll_interval} seconds")

    async def _download_result(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
    ) -> bytes:
        """Download the result from the completed task."""
        url = f"{KIE_API_BASE_URL}/api/v1/jobs/recordInfo?taskId={task_id}"
        headers = self._get_headers()

        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                response_text = await response.text()
                raise ValueError(f"Failed to get result: {response.status} - {response_text}")

            status_data = await response.json()
            self._check_response_status(status_data)
            result_json_str = status_data.get("data", {}).get("resultJson", "")

            if not result_json_str:
                raise ValueError("No resultJson in response")

            result_data = json.loads(result_json_str)
            result_urls = result_data.get("resultUrls", [])

            if not result_urls:
                raise ValueError("No resultUrls in resultJson")

            result_url = result_urls[0]
            log.debug(f"Downloading result from {result_url}")

            async with session.get(result_url) as result_response:
                if result_response.status != 200:
                    raise ValueError(f"Failed to download result from URL: {result_url}")
                return await result_response.read()

    async def _execute_task(
        self,
        model: str,
        input_params: dict[str, Any],
        poll_interval: float,
        max_attempts: int,
    ) -> bytes:
        """Execute the full task workflow: submit, poll, download."""
        async with aiohttp.ClientSession() as session:
            task_id = await self._submit_task(session, model, input_params)
            await self._poll_status(session, task_id, poll_interval, max_attempts)
            return await self._download_result(session, task_id)

    # Model Discovery Methods

    async def get_available_image_models(self) -> List[ImageModel]:
        """Get available Kie.ai image generation models."""
        if not self.api_key:
            log.debug("No Kie.ai API key configured, returning empty image model list")
            return []
        return KIE_IMAGE_MODELS

    async def get_available_video_models(self) -> List[VideoModel]:
        """Get available Kie.ai video generation models."""
        if not self.api_key:
            log.debug("No Kie.ai API key configured, returning empty video model list")
            return []
        return KIE_VIDEO_MODELS

    # Image Generation Methods

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate an image from a text prompt using Kie.ai text-to-image models.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            node_id: Optional node ID for tracking

        Returns:
            Raw image bytes (PNG format)

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating image with Kie.ai model: {params.model.id}")

        if not params.prompt:
            raise ValueError("prompt must not be empty")

        try:
            # Build input parameters based on the model
            input_params: dict[str, Any] = {
                "prompt": params.prompt,
            }

            # Add optional parameters if provided
            if params.negative_prompt:
                input_params["negative_prompt"] = params.negative_prompt

            # Convert width/height to aspect_ratio if needed
            if params.width and params.height:
                # Calculate aspect ratio
                if params.width == params.height:
                    input_params["aspect_ratio"] = "1:1"
                elif params.width > params.height:
                    ratio = params.width / params.height
                    if abs(ratio - 16 / 9) < 0.1:
                        input_params["aspect_ratio"] = "16:9"
                    elif abs(ratio - 4 / 3) < 0.1:
                        input_params["aspect_ratio"] = "4:3"
                else:
                    ratio = params.height / params.width
                    if abs(ratio - 16 / 9) < 0.1:
                        input_params["aspect_ratio"] = "9:16"
                    elif abs(ratio - 4 / 3) < 0.1:
                        input_params["aspect_ratio"] = "3:4"

            if params.guidance_scale:
                input_params["guidance_scale"] = params.guidance_scale

            if params.num_inference_steps:
                input_params["steps"] = params.num_inference_steps

            if params.seed is not None and params.seed >= 0:
                input_params["seed"] = params.seed

            # Extract any extra Kie-specific params (aspect_ratio, resolution, quality, steps, etc.)
            self._extract_kie_params(params, input_params)

            result = await self._execute_task(
                model=params.model.id,
                input_params=input_params,
                poll_interval=self._image_poll_interval,
                max_attempts=self._image_max_poll_attempts,
            )

            log.debug(f"Generated {len(result)} bytes of image data")
            return result

        except Exception as e:
            log.error(f"Kie.ai text-to-image generation failed: {e}")
            raise RuntimeError(f"Kie.ai text-to-image generation failed: {str(e)}") from e

    async def image_to_image(
        self,
        image: bytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Transform an image based on a text prompt using Kie.ai image-to-image models.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            node_id: Optional node ID for tracking

        Returns:
            Raw image bytes (PNG format)

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Transforming image with Kie.ai model: {params.model.id}")

        if not params.prompt:
            raise ValueError("prompt must not be empty")

        try:
            # First, upload the image to get a URL
            image_url = await self._upload_image(image)

            # Build input parameters
            input_params: dict[str, Any] = {
                "prompt": params.prompt,
            }

            # Different models use different field names for input image
            if "topaz" in params.model.id or "grok" in params.model.id:
                input_params["image"] = image_url
            elif "qwen" in params.model.id:
                input_params["image_url"] = image_url
            else:
                input_params["input_urls"] = [image_url]

            # Add optional parameters
            if params.negative_prompt:
                input_params["negative_prompt"] = params.negative_prompt

            if params.guidance_scale:
                input_params["guidance_scale"] = params.guidance_scale

            if params.num_inference_steps:
                input_params["steps"] = params.num_inference_steps

            if params.strength is not None:
                input_params["strength"] = params.strength

            if params.seed is not None and params.seed >= 0:
                input_params["seed"] = params.seed

            # Extract any extra Kie-specific params (aspect_ratio, resolution, quality, etc.)
            self._extract_kie_params(params, input_params)

            # Add aspect ratio
            if params.target_width and params.target_height:
                if params.target_width == params.target_height:
                    input_params["aspect_ratio"] = "1:1"
                elif params.target_width > params.target_height:
                    ratio = params.target_width / params.target_height
                    if abs(ratio - 16 / 9) < 0.1:
                        input_params["aspect_ratio"] = "16:9"
                    elif abs(ratio - 4 / 3) < 0.1:
                        input_params["aspect_ratio"] = "4:3"
                else:
                    ratio = params.target_height / params.target_width
                    if abs(ratio - 16 / 9) < 0.1:
                        input_params["aspect_ratio"] = "9:16"
                    elif abs(ratio - 4 / 3) < 0.1:
                        input_params["aspect_ratio"] = "3:4"

            result = await self._execute_task(
                model=params.model.id,
                input_params=input_params,
                poll_interval=self._image_poll_interval,
                max_attempts=self._image_max_poll_attempts,
            )

            log.debug(f"Generated {len(result)} bytes of image data")
            return result

        except Exception as e:
            log.error(f"Kie.ai image-to-image generation failed: {e}")
            raise RuntimeError(f"Kie.ai image-to-image generation failed: {str(e)}") from e

    async def _upload_image(self, image_data: bytes) -> str:
        """Upload an image to Kie.ai and return the download URL."""
        upload_url = "https://kieai.redpandaai.co/api/file-stream-upload"
        filename = f"nodetool-{uuid.uuid4().hex}.png"

        headers = {"Authorization": f"Bearer {self.api_key}"}

        form = aiohttp.FormData()
        form.add_field("file", image_data, filename=filename, content_type="image/png")
        form.add_field("uploadPath", "images/user-uploads")
        form.add_field("fileName", filename)

        async with aiohttp.ClientSession() as session:
            async with session.post(upload_url, data=form, headers=headers) as response:
                response_data = await response.json()
                if "code" in response_data:
                    self._check_response_status(response_data)

                if response.status != 200 or not response_data.get("success"):
                    raise ValueError(f"Failed to upload image: {response.status} - {response_data}")

                download_url = response_data.get("data", {}).get("downloadUrl")
                if not download_url:
                    raise ValueError(f"No downloadUrl in upload response: {response_data}")
                return download_url

    # Video Generation Methods

    async def text_to_video(
        self,
        params: TextToVideoParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate a video from a text prompt using Kie.ai text-to-video models.

        Args:
            params: Text-to-video generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            node_id: Optional node ID for tracking

        Returns:
            Raw video bytes

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating video with Kie.ai model: {params.model.id}")

        if not params.prompt:
            raise ValueError("prompt must not be empty")

        try:
            # Build input parameters
            input_params: dict[str, Any] = {
                "prompt": params.prompt,
            }

            # Add optional parameters
            if params.negative_prompt:
                input_params["negative_prompt"] = params.negative_prompt

            if params.aspect_ratio:
                input_params["aspect_ratio"] = params.aspect_ratio

            if params.resolution:
                input_params["resolution"] = params.resolution

            if params.num_frames:
                # Different models use different parameter names
                if "sora" in params.model.id:
                    input_params["n_frames"] = params.num_frames
                else:
                    input_params["duration"] = str(params.num_frames // 24)  # Convert frames to seconds

            if params.guidance_scale:
                input_params["guidance_scale"] = params.guidance_scale

            if params.num_inference_steps:
                input_params["steps"] = params.num_inference_steps

            if params.seed is not None and params.seed >= 0:
                input_params["seed"] = params.seed

            # Extract any extra Kie-specific params (aspect_ratio, resolution, duration, remove_watermark, etc.)
            self._extract_kie_params(params, input_params)

            result = await self._execute_task(
                model=params.model.id,
                input_params=input_params,
                poll_interval=self._video_poll_interval,
                max_attempts=self._video_max_poll_attempts,
            )

            log.debug(f"Generated {len(result)} bytes of video data")
            return result

        except Exception as e:
            log.error(f"Kie.ai text-to-video generation failed: {e}")
            raise RuntimeError(f"Kie.ai text-to-video generation failed: {str(e)}") from e

    async def image_to_video(
        self,
        image: bytes,
        params: ImageToVideoParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate a video from an image using Kie.ai image-to-video models.

        Args:
            image: Input image as bytes
            params: Image-to-video generation parameters
            timeout_s: Optional timeout in seconds
            context: Processing context for asset handling
            node_id: Optional node ID for tracking

        Returns:
            Raw video bytes

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating video from image with Kie.ai model: {params.model.id}")

        try:
            # First, upload the image to get a URL
            image_url = await self._upload_image(image)

            # Build input parameters
            input_params: dict[str, Any] = {}

            # Different models use different field names for input image
            if "hailuo" in params.model.id:
                input_params["image_url"] = image_url
            elif "sora" in params.model.id:
                input_params["image_url"] = image_url
            elif "grok" in params.model.id:
                input_params["image"] = image_url
            else:
                input_params["image_urls"] = [image_url]

            # Add prompt if provided
            if params.prompt:
                input_params["prompt"] = params.prompt

            # Add optional parameters
            if params.negative_prompt:
                input_params["negative_prompt"] = params.negative_prompt

            if params.aspect_ratio:
                input_params["aspect_ratio"] = params.aspect_ratio

            if params.resolution:
                input_params["resolution"] = params.resolution

            if params.num_frames:
                if "sora" in params.model.id:
                    input_params["n_frames"] = params.num_frames
                else:
                    input_params["duration"] = str(params.num_frames // 24)

            if params.guidance_scale:
                input_params["guidance_scale"] = params.guidance_scale

            if params.num_inference_steps:
                input_params["steps"] = params.num_inference_steps

            if params.seed is not None and params.seed >= 0:
                input_params["seed"] = params.seed

            # Extract any extra Kie-specific params (aspect_ratio, resolution, duration, sound, mode, audio_url, etc.)
            self._extract_kie_params(params, input_params)

            result = await self._execute_task(
                model=params.model.id,
                input_params=input_params,
                poll_interval=self._video_poll_interval,
                max_attempts=self._video_max_poll_attempts,
            )

            log.debug(f"Generated {len(result)} bytes of video data")
            return result

        except Exception as e:
            log.error(f"Kie.ai image-to-video generation failed: {e}")
            raise RuntimeError(f"Kie.ai image-to-video generation failed: {str(e)}") from e
