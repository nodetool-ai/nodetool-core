"""
MiniMax provider implementation for chat completions, image generation,
video generation, and text-to-speech.

This module implements the ChatProvider interface for MiniMax models,
using their Anthropic-compatible API endpoint for chat, their
image generation API for text-to-image, their video generation API
for text-to-video and image-to-video, and their T2A API for text-to-speech.

MiniMax Anthropic API Documentation: https://platform.minimaxi.com/docs/api-reference/text-anthropic-api
MiniMax Image Generation API: https://platform.minimax.io/docs/guides/image-generation
MiniMax Video Generation API: https://platform.minimaxi.com/docs/api-reference/video-generation
MiniMax T2A API Documentation: https://platform.minimax.io/docs/api-reference/speech-t2a-intro
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING, Any, AsyncGenerator

import aiohttp
import anthropic
import numpy as np

if TYPE_CHECKING:
    from nodetool.providers.types import ImageToVideoParams, TextToImageParams, TextToVideoParams
    from nodetool.workflows.processing_context import ProcessingContext

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    ImageModel,
    LanguageModel,
    Provider,
    TTSModel,
    VideoModel,
)
from nodetool.providers.anthropic_provider import AnthropicProvider
from nodetool.providers.base import BaseProvider, register_provider

log = get_logger(__name__)

# MiniMax Anthropic-compatible API base URL
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"

# MiniMax Image Generation API base URL
MINIMAX_IMAGE_API_URL = "https://api.minimax.io/v1/image_generation"

# MiniMax Text-to-Audio (TTS) API base URL
MINIMAX_TTS_API_URL = "https://api.minimax.io/v1/t2a_v2"

# MiniMax Video Generation API base URL
MINIMAX_VIDEO_API_URL = "https://api.minimax.io/v1/video_generation"

# MiniMax Video Query API URL (polling endpoint)
MINIMAX_VIDEO_QUERY_API_URL = "https://api.minimax.io/v1/query/video_generation"

# MiniMax File Upload API base URL
MINIMAX_FILE_UPLOAD_URL = "https://api.minimax.io/v1/files/upload"

# MiniMax File Retrieve API URL (used to resolve file download URLs)
MINIMAX_FILE_RETRIEVE_URL = "https://api.minimax.io/v1/files/retrieve"

# Known MiniMax image models
MINIMAX_IMAGE_MODELS = [
    ImageModel(
        id="image-01",
        name="MiniMax Image-01",
        provider=Provider.MiniMax,
    ),
    ImageModel(
        id="image-01-live",
        name="MiniMax Image-01 Live",
        provider=Provider.MiniMax,
    ),
]

# Known MiniMax video models
MINIMAX_VIDEO_MODELS = [
    VideoModel(
        id="MiniMax-Hailuo-2.3",
        name="Hailuo 2.3",
        provider=Provider.MiniMax,
    ),
    VideoModel(
        id="MiniMax-Hailuo-2.3-Fast",
        name="Hailuo 2.3 Fast",
        provider=Provider.MiniMax,
    ),
    VideoModel(
        id="MiniMax-Hailuo-02",
        name="Hailuo 02",
        provider=Provider.MiniMax,
    ),
]

# Known MiniMax TTS models
# Based on: https://platform.minimax.io/docs/api-reference/speech-t2a-intro
MINIMAX_TTS_MODELS = [
    {
        "id": "speech-2.8-hd",
        "name": "MiniMax Speech 2.8 HD",
    },
    {
        "id": "speech-2.8-turbo",
        "name": "MiniMax Speech 2.8 Turbo",
    },
    {
        "id": "speech-2.6-hd",
        "name": "MiniMax Speech 2.6 HD",
    },
    {
        "id": "speech-2.6-turbo",
        "name": "MiniMax Speech 2.6 Turbo",
    },
]

# MiniMax TTS system voices
# Based on: https://platform.minimax.io/docs/api-reference/speech-t2a-intro
MINIMAX_TTS_VOICES = [
    # English voices
    "English_Graceful_Lady",
    "English_Insightful_Speaker",
    "English_radiant_girl",
    "English_Persuasive_Man",
    "English_Lucky_Robot",
    "English_expressive_narrator",
    # Chinese voices
    "Chinese (Mandarin)_Lyrical_Voice",
    "Chinese (Mandarin)_HK_Flight_Attendant",
    # Japanese voices
    "Japanese_Whisper_Belle",
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
        # Skip AnthropicProvider.__init__ which requires ANTHROPIC_API_KEY
        # Call BaseProvider.__init__ directly to initialize cost and secrets
        BaseProvider.__init__(self, secrets)
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
            "MiniMax-Text-01",
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

    async def get_available_video_models(self) -> list[VideoModel]:
        """Get available MiniMax video generation models.

        Returns a list of known MiniMax Hailuo video models for text-to-video
        and image-to-video generation.

        Returns:
            List of VideoModel instances for MiniMax
        """
        if not self.api_key:
            log.debug("No MiniMax API key configured, returning empty video model list")
            return []

        log.debug(f"Returning {len(MINIMAX_VIDEO_MODELS)} known MiniMax video models")
        return MINIMAX_VIDEO_MODELS

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
                "response_format": "base64",
            }

            # Add optional aspect_ratio if width/height suggest a specific ratio
            if params.width and params.height:
                aspect_ratio = self._calculate_aspect_ratio(params.width, params.height)
                if aspect_ratio:
                    payload["aspect_ratio"] = aspect_ratio

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
                if "data" in result and "image_base64" in result["data"]:
                    images = result["data"]["image_base64"]
                    if images and len(images) > 0:
                        image_bytes = base64.b64decode(images[0])
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

    async def get_available_tts_models(self) -> list[TTSModel]:
        """Get available MiniMax text-to-speech models.

        Returns a list of known MiniMax TTS models. MiniMax supports several
        speech synthesis models with various voices.

        Returns:
            List of TTSModel instances for MiniMax TTS
        """
        if not self.api_key:
            log.debug("No MiniMax API key configured, returning empty TTS model list")
            return []

        models: list[TTSModel] = []
        for config in MINIMAX_TTS_MODELS:
            models.append(
                TTSModel(
                    id=config["id"],
                    name=config["name"],
                    provider=Provider.MiniMax,
                    voices=MINIMAX_TTS_VOICES,
                )
            )

        log.debug(f"Returning {len(models)} MiniMax TTS models")
        return models

    async def text_to_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        speed: float = 1.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[np.ndarray[Any, np.dtype[np.int16]], None]:
        """Generate speech audio from text using MiniMax T2A API.

        Uses MiniMax's T2A (Text-to-Audio) API to generate speech from text.
        The audio is returned as int16 PCM samples at 24kHz mono.

        API Reference: https://platform.minimax.io/docs/api-reference/speech-t2a-intro

        Args:
            text: Input text to convert to speech (max 10,000 characters)
            model: Model identifier (e.g., "speech-2.8-hd", "speech-2.8-turbo")
            voice: Voice identifier (e.g., "English_Graceful_Lady")
            speed: Speech speed multiplier (0.5 to 2.0)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            **kwargs: Additional MiniMax parameters

        Yields:
            numpy.ndarray: Int16 audio chunks at 24kHz mono

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating speech for model: {model}, voice: {voice}, speed: {speed}")

        if not text:
            raise ValueError("text must not be empty")

        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required for text-to-speech generation")

        # Default voice if not specified
        voice = voice or "English_Graceful_Lady"

        # Clamp speed to MiniMax's supported range
        speed = max(0.5, min(2.0, speed))

        log.debug(f"Making TTS API call with model={model}, voice={voice}, speed={speed}")

        try:
            request_timeout = timeout_s if timeout_s and timeout_s > 0 else 120

            # Build the request payload according to MiniMax T2A API
            payload: dict[str, Any] = {
                "model": model,
                "text": text,
                "stream": False,
                "output_format": "hex",
                "voice_setting": {
                    "voice_id": voice,
                    "speed": speed,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {
                    "sample_rate": 24000,
                    "format": "pcm",
                    "channel": 1,
                },
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            timeout = aiohttp.ClientTimeout(total=request_timeout)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    MINIMAX_TTS_API_URL,
                    json=payload,
                    headers=headers,
                ) as response,
            ):
                if response.status != 200:
                    error_text = await response.text()
                    log.error(f"MiniMax TTS API error: {response.status} - {error_text}")
                    raise RuntimeError(
                        f"MiniMax text-to-speech generation failed with status {response.status}: {error_text}"
                    )

                result = await response.json()

                # Check for API errors in response
                base_resp = result.get("base_resp", {})
                status_code = base_resp.get("status_code", 0)
                if status_code != 0:
                    status_msg = base_resp.get("status_msg", "Unknown error")
                    log.error(f"MiniMax TTS API returned error: {status_code} - {status_msg}")
                    raise RuntimeError(
                        f"MiniMax text-to-speech generation failed: {status_msg} (code: {status_code})"
                    )

                # Extract audio data from response
                data = result.get("data", {})
                audio_hex = data.get("audio")

                if not audio_hex:
                    raise RuntimeError("No audio data returned in MiniMax TTS response")

                # Decode hex-encoded audio to bytes
                audio_bytes = bytes.fromhex(audio_hex)

                # Convert to int16 numpy array
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                log.debug(f"Generated audio, samples: {len(audio_array)}")
                self._log_api_response("text_to_speech", audio_samples=len(audio_array))

                yield audio_array

        except aiohttp.ClientError as e:
            log.error(f"MiniMax text-to-speech request failed: {e}")
            raise RuntimeError(f"MiniMax text-to-speech generation failed: {e}") from e
        except Exception as exc:
            log.error(f"MiniMax text-to-speech generation failed: {exc}")
            raise RuntimeError(f"MiniMax text-to-speech generation failed: {exc}") from exc

    async def _poll_video_task(
        self,
        task_id: str,
        timeout_s: int,
        poll_interval: int = 10,
    ) -> dict[str, Any]:
        """Poll a MiniMax video generation task until completion.

        Args:
            task_id: The task ID returned by the video generation request.
            timeout_s: Maximum time in seconds to wait for completion.
            poll_interval: Seconds between status polls.

        Returns:
            The completed task response dict.

        Raises:
            TimeoutError: If the task doesn't complete within timeout_s.
            RuntimeError: If the task fails.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        elapsed = 0
        status_url = MINIMAX_VIDEO_QUERY_API_URL

        while elapsed < timeout_s:
            async with (
                aiohttp.ClientSession() as session,
                session.get(status_url, headers=headers, params={"task_id": task_id}) as response,
            ):
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"MiniMax video task poll failed with status {response.status}: {error_text}"
                    )
                result = await response.json()

            base_resp = result.get("base_resp", {})
            status_code = base_resp.get("status_code", 0)
            if status_code != 0:
                status_msg = base_resp.get("status_msg", "Unknown error")
                raise RuntimeError(f"MiniMax video task poll failed: {status_msg} (code: {status_code})")

            task_data = result.get("data") if isinstance(result.get("data"), dict) else result
            status = str(task_data.get("status", "")).lower()
            if status in {"success", "succeeded"}:
                return task_data
            if status in {"failed", "fail", "error"}:
                raise RuntimeError(f"MiniMax video generation task failed: {task_data}")

            log.debug(f"Video task {task_id}: status={task_data.get('status')}, elapsed={elapsed}s")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"MiniMax video generation timed out after {timeout_s}s")

    async def _resolve_video_download_url(self, task_result: dict[str, Any]) -> str:
        """Resolve a downloadable URL from a completed MiniMax video task response."""
        video_url = task_result.get("video_url")
        if video_url:
            return str(video_url)

        file_info = task_result.get("file") if isinstance(task_result.get("file"), dict) else {}
        if file_info and file_info.get("download_url"):
            return str(file_info["download_url"])

        file_id = task_result.get("file_id") or file_info.get("file_id")
        if not file_id:
            raise RuntimeError("No video URL or file_id in completed task response")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        async with (
            aiohttp.ClientSession() as session,
            session.get(MINIMAX_FILE_RETRIEVE_URL, headers=headers, params={"file_id": file_id}) as response,
        ):
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"MiniMax file retrieve failed with status {response.status}: {error_text}")
            result = await response.json()

        base_resp = result.get("base_resp", {})
        status_code = base_resp.get("status_code", 0)
        if status_code != 0:
            status_msg = base_resp.get("status_msg", "Unknown error")
            raise RuntimeError(f"MiniMax file retrieve failed: {status_msg} (code: {status_code})")

        data = result.get("data") if isinstance(result.get("data"), dict) else result
        retrieved_file = data.get("file") if isinstance(data.get("file"), dict) else data
        download_url = retrieved_file.get("download_url") or retrieved_file.get("url")
        if not download_url:
            raise RuntimeError(f"No download URL returned from MiniMax file retrieve: {result}")
        return str(download_url)

    async def _upload_image(self, image: bytes) -> str:
        """Upload an image to MiniMax file service and return the file_id.

        Args:
            image: Raw image bytes to upload.

        Returns:
            The file_id string from MiniMax.

        Raises:
            RuntimeError: If the upload fails.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        form = aiohttp.FormData()
        form.add_field("file", image, filename="image.png", content_type="image/png")
        form.add_field("purpose", "video_generation")

        async with (
            aiohttp.ClientSession() as session,
            session.post(MINIMAX_FILE_UPLOAD_URL, data=form, headers=headers) as response,
        ):
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"MiniMax file upload failed with status {response.status}: {error_text}"
                )
            result = await response.json()

        file_id = result.get("file_id") or result.get("file", {}).get("file_id")
        if not file_id:
            raise RuntimeError(f"No file_id returned from MiniMax file upload: {result}")
        return file_id

    async def text_to_video(
        self,
        params: TextToVideoParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate a video from a text prompt using MiniMax Hailuo models.

        Submits a video generation task and polls until completion, then
        downloads the resulting video.

        API Reference: https://platform.minimaxi.com/docs/api-reference/video-generation

        Args:
            params: Text-to-video generation parameters including:
                - model: VideoModel with model ID (e.g., "MiniMax-Hailuo-2.3")
                - prompt: Text description of the desired video
                - aspect_ratio: Optional aspect ratio (e.g., "16:9")
            timeout_s: Optional timeout in seconds (default 600)
            context: Processing context (unused, reserved)
            node_id: Node ID for progress reporting (unused)

        Returns:
            Raw video bytes

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
            TimeoutError: If generation times out
        """
        if not params.prompt:
            raise ValueError("The input prompt cannot be empty.")

        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required for video generation.")

        model_id = params.model.id if params.model and params.model.id else "MiniMax-Hailuo-2.3"
        prompt = params.prompt.strip()

        log.debug(f"Generating video with MiniMax model: {model_id}")
        self._log_api_request("text_to_video", params=params)

        try:
            payload: dict[str, Any] = {
                "model": model_id,
                "prompt": prompt,
            }

            if params.aspect_ratio:
                payload["video_ratio"] = params.aspect_ratio

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with (
                aiohttp.ClientSession() as session,
                session.post(MINIMAX_VIDEO_API_URL, json=payload, headers=headers) as response,
            ):
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"MiniMax video generation failed with status {response.status}: {error_text}"
                    )
                result = await response.json()

            base_resp = result.get("base_resp", {})
            status_code = base_resp.get("status_code", 0)
            if status_code != 0:
                status_msg = base_resp.get("status_msg", "Unknown error")
                raise RuntimeError(f"MiniMax video generation failed: {status_msg} (code: {status_code})")

            task_data = result.get("data") if isinstance(result.get("data"), dict) else result
            task_id = task_data.get("task_id")
            if not task_id:
                raise RuntimeError(f"No task_id returned from MiniMax video generation: {result}")

            log.debug(f"Video generation task submitted: {task_id}")

            max_wait = timeout_s if timeout_s and timeout_s > 0 else 600
            completed = await self._poll_video_task(task_id, max_wait)
            video_url = await self._resolve_video_download_url(completed)

            async with (
                aiohttp.ClientSession() as session,
                session.get(video_url) as dl_resp,
            ):
                if dl_resp.status != 200:
                    error_text = await dl_resp.text()
                    raise RuntimeError(
                        f"MiniMax video download failed with status {dl_resp.status}: {error_text}"
                    )
                video_bytes = await dl_resp.read()

            log.debug(f"Generated video, size: {len(video_bytes)} bytes")
            self._log_api_response("text_to_video", video_bytes=len(video_bytes))
            return video_bytes

        except (TimeoutError, RuntimeError):
            raise
        except aiohttp.ClientError as e:
            log.error(f"MiniMax text-to-video request failed: {e}")
            raise RuntimeError(f"MiniMax text-to-video generation failed: {e}") from e
        except Exception as exc:
            log.error(f"MiniMax text-to-video generation failed: {exc}")
            raise RuntimeError(f"MiniMax text-to-video generation failed: {exc}") from exc

    async def image_to_video(
        self,
        image: bytes,
        params: ImageToVideoParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate a video from an input image using MiniMax Hailuo models.

        Uploads the image to MiniMax's file service, submits a video generation
        task with the first frame reference, and polls until completion.

        API Reference: https://platform.minimaxi.com/docs/api-reference/video-generation

        Args:
            image: Input image as bytes
            params: Image-to-video generation parameters including:
                - model: VideoModel with model ID (e.g., "MiniMax-Hailuo-2.3-Fast")
                - prompt: Optional text description to guide video generation
                - aspect_ratio: Optional aspect ratio (e.g., "16:9")
            timeout_s: Optional timeout in seconds (default 600)
            context: Processing context (unused, reserved)
            node_id: Node ID for progress reporting (unused)

        Returns:
            Raw video bytes

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
            TimeoutError: If generation times out
        """
        if not image:
            raise ValueError("Input image cannot be empty.")

        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required for video generation.")

        model_id = params.model.id if params.model and params.model.id else "MiniMax-Hailuo-2.3-Fast"
        prompt = (params.prompt or "Animate this image").strip()

        log.debug(f"Generating video from image with MiniMax model: {model_id}")
        self._log_api_request("image_to_video", params=params)

        try:
            image_base64 = base64.b64encode(image).decode("utf-8")

            payload: dict[str, Any] = {
                "model": model_id,
                "prompt": prompt,
                "first_frame_image": f"data:image/png;base64,{image_base64}",
            }

            if hasattr(params, "aspect_ratio") and params.aspect_ratio:
                payload["video_ratio"] = params.aspect_ratio

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with (
                aiohttp.ClientSession() as session,
                session.post(MINIMAX_VIDEO_API_URL, json=payload, headers=headers) as response,
            ):
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"MiniMax video generation failed with status {response.status}: {error_text}"
                    )
                result = await response.json()

            base_resp = result.get("base_resp", {})
            status_code = base_resp.get("status_code", 0)
            if status_code != 0:
                status_msg = base_resp.get("status_msg", "Unknown error")
                raise RuntimeError(f"MiniMax image-to-video generation failed: {status_msg} (code: {status_code})")

            task_data = result.get("data") if isinstance(result.get("data"), dict) else result
            task_id = task_data.get("task_id")
            if not task_id:
                raise RuntimeError(f"No task_id returned from MiniMax video generation: {result}")

            log.debug(f"Image-to-video generation task submitted: {task_id}")

            max_wait = timeout_s if timeout_s and timeout_s > 0 else 600
            completed = await self._poll_video_task(task_id, max_wait)
            video_url = await self._resolve_video_download_url(completed)

            async with (
                aiohttp.ClientSession() as session,
                session.get(video_url) as dl_resp,
            ):
                if dl_resp.status != 200:
                    error_text = await dl_resp.text()
                    raise RuntimeError(
                        f"MiniMax video download failed with status {dl_resp.status}: {error_text}"
                    )
                video_bytes = await dl_resp.read()

            log.debug(f"Generated video from image, size: {len(video_bytes)} bytes")
            self._log_api_response("image_to_video", video_bytes=len(video_bytes))
            return video_bytes

        except (TimeoutError, RuntimeError):
            raise
        except aiohttp.ClientError as e:
            log.error(f"MiniMax image-to-video request failed: {e}")
            raise RuntimeError(f"MiniMax image-to-video generation failed: {e}") from e
        except Exception as exc:
            log.error(f"MiniMax image-to-video generation failed: {exc}")
            raise RuntimeError(f"MiniMax image-to-video generation failed: {exc}") from exc
