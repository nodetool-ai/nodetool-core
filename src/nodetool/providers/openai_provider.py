"""
OpenAI provider implementation for chat completions.

This module implements the ChatProvider interface for OpenAI models,
handling message conversion, streaming, and tool integration.

"""

from __future__ import annotations

import base64
import inspect
import io
import math
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Dict, List, Sequence, cast

if TYPE_CHECKING:
    import httpx
from urllib.parse import unquote_to_bytes

import numpy as np
import openai
from openai import Omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
    Function,
)
from PIL import Image
from pydantic import BaseModel
from pydub import AudioSegment

if TYPE_CHECKING:
    from openai._types import NotGiven
    from openai.types import Video

    from nodetool.agents.tools.base import Tool
    from nodetool.metadata.types import MessageContent
    from nodetool.providers.types import (
        ImageToImageParams,
        ImageToVideoParams,
        TextToImageParams,
        TextToVideoParams,
    )
    from nodetool.workflows.processing_context import ProcessingContext

from nodetool.config.logging_config import get_logger
from nodetool.io.uri_utils import fetch_uri_bytes_and_mime
from nodetool.media.image.image_utils import image_data_to_base64_jpeg
from nodetool.metadata.types import (
    ASRModel,
    ImageModel,
    LanguageModel,
    Message,
    MessageAudioContent,
    MessageImageContent,
    MessageTextContent,
    Provider,
    ToolCall,
    TTSModel,
    VideoModel,
)
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.openai_prediction import calculate_chat_cost
from nodetool.runtime.resources import maybe_scope, require_scope
from nodetool.workflows.types import Chunk, NodeProgress

log = get_logger(__name__)


@register_provider(Provider.OpenAI)
class OpenAIProvider(BaseProvider):
    """OpenAI implementation of the ChatProvider interface.

    Handles conversion between internal message format and OpenAI's API format,
    streaming completions, and tool calling.

    Overview of OpenAI chat constructs used:

    1. Message Format:
       - Each message is a dict with "role" and "content" fields
       - Role can be: "system", "user", "assistant", or "tool"
       - Content contains the message text (string) or content blocks (multimodal)
       - Messages may include an optional "name" field

    2. Tool Calls:
       - When a model wants to call a tool, the response includes "tool_calls"
       - Each tool call has an "id" and a "function" with name and arguments
       - To respond, send a message with role "tool" including tool_call_id

    3. Response Structure:
       - ``response.choices[0].message`` holds the model response
       - ``response.usage`` contains token usage stats

    4. Flow with Tools:
       - Model returns tool_calls → App executes tools → App replies with role
         "tool" → Model continues using results.

    For details, see the OpenAI function calling guide.
    """

    has_code_interpreter: bool = False
    provider: Provider = Provider.OpenAI

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["OPENAI_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the OpenAI provider with client credentials.

        Reads ``OPENAI_API_KEY`` from environment and prepares usage tracking.
        """
        assert "OPENAI_API_KEY" in secrets, "OPENAI_API_KEY is required"
        self.api_key = secrets["OPENAI_API_KEY"]
        self.client = None
        self.cost = 0.0
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }
        log.debug("OpenAIProvider initialized. API key present: True")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``OPENAI_API_KEY`` if available; otherwise empty.
        """
        return {"OPENAI_API_KEY": self.api_key} if self.api_key else {}

    def get_client(
        self,
    ) -> openai.AsyncClient:
        """Create and return an OpenAI async client.

        Uses ResourceScope's HTTP client to ensure correct event loop binding.

        Returns:
            An initialized ``openai.AsyncClient`` with reasonable timeouts.
        """
        import httpx

        log.debug("Creating OpenAI async client")

        # Use ResourceScope's HTTP client if available, otherwise create a new one
        http_client = require_scope().get_http_client()

        client = openai.AsyncClient(
            api_key=self.api_key,
            http_client=http_client,
        )
        log.debug("OpenAI async client created successfully")
        return client

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        Most OpenAI models support function calling, with the notable exception
        of the O1 and O3 reasoning models which do not support tools.

        Args:
            model: Model identifier string.

        Returns:
            True if the model supports function calling, False otherwise.
        """
        log.debug(f"Checking tool support for model: {model}")

        # O1 and O3 series reasoning models do not support tools
        if model.startswith("o1") or model.startswith("o3"):
            log.debug(f"Model {model} is a reasoning model without tool support")
            return False

        # All other modern OpenAI models support tools (GPT-3.5-turbo, GPT-4, GPT-4o, GPT-5, etc.)
        log.debug(f"Model {model} supports tool calling")
        return True

    def structured_output(self) -> bool:
        """Check if provider supports structured JSON output natively.

        OpenAI supports structured output via response_format={"type": "json_schema", ...}.
        """
        return True

    async def get_available_language_models(self) -> List[LanguageModel]:
        """
        Get available OpenAI models.

        Fetches models dynamically from the OpenAI API if an API key is available.
        Returns an empty list if no API key is configured or if the fetch fails.

        Returns:
            List of LanguageModel instances for OpenAI
        """
        import aiohttp

        if not self.api_key:
            log.debug("No OpenAI API key configured, returning empty model list")
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=3)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                session.get("https://api.openai.com/v1/models") as response,
            ):
                if response.status != 200:
                    log.warning(f"Failed to fetch OpenAI models: HTTP {response.status}")
                    return []
                payload = await response.json()
                data = payload.get("data", [])

                models: List[LanguageModel] = []
                for item in data:
                    model_id = item.get("id")
                    if not model_id:
                        continue
                    models.append(
                        LanguageModel(
                            id=model_id,
                            name=model_id,
                            provider=Provider.OpenAI,
                        )
                    )
                log.debug(f"Fetched {len(models)} OpenAI models")
                return models
        except Exception as e:
            log.error(f"Error fetching OpenAI models: {e}")
            return []

    async def get_available_tts_models(self) -> List[TTSModel]:
        """
        Get available OpenAI TTS models.

        Returns TTS models with their supported voices.
        Returns an empty list if no API key is configured.

        Returns:
            List of TTSModel instances for OpenAI TTS
        """
        if not self.api_key:
            log.debug("No OpenAI API key configured, returning empty TTS model list")
            return []

        # OpenAI TTS models and their voices
        # Source: https://platform.openai.com/docs/guides/text-to-speech
        tts_models_config = [
            {
                "id": "tts-1",
                "name": "TTS 1",
                "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            },
            {
                "id": "tts-1-hd",
                "name": "TTS 1 HD",
                "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            },
        ]

        models: List[TTSModel] = []
        for config in tts_models_config:
            models.append(
                TTSModel(
                    id=config["id"],
                    name=config["name"],
                    provider=Provider.OpenAI,
                    voices=config["voices"],
                )
            )

        log.debug(f"Returning {len(models)} OpenAI TTS models")
        return models

    async def get_available_asr_models(self) -> List[ASRModel]:
        """
        Get available OpenAI ASR models.

        Returns ASR models (Whisper variants).
        Returns an empty list if no API key is configured.

        Returns:
            List of ASRModel instances for OpenAI ASR
        """
        if not self.api_key:
            log.debug("No OpenAI API key configured, returning empty ASR model list")
            return []

        # OpenAI ASR models
        # Source: https://platform.openai.com/docs/guides/speech-to-text
        asr_models_config = [
            {
                "id": "whisper-1",
                "name": "Whisper",
            },
        ]

        models: List[ASRModel] = []
        for config in asr_models_config:
            models.append(
                ASRModel(
                    id=config["id"],
                    name=config["name"],
                    provider=Provider.OpenAI,
                )
            )

        log.debug(f"Returning {len(models)} OpenAI ASR models")
        return models

    async def get_available_video_models(self) -> List[VideoModel]:
        """
        Get available OpenAI video generation models.

        Returns Sora video models only if OPENAI_API_KEY is configured.
        Source: https://platform.openai.com/docs/guides/video

        Returns:
            List of VideoModel instances for OpenAI Sora
        """
        if not self.api_key:
            log.debug("No OpenAI API key configured, returning empty video model list")
            return []

        models = [
            VideoModel(
                id="sora-2",
                name="Sora 2",
                provider=Provider.OpenAI,
                supported_tasks=["text_to_video"],
            ),
            VideoModel(
                id="sora-2-pro",
                name="Sora 2 Pro",
                provider=Provider.OpenAI,
                supported_tasks=["text_to_video"],
            ),
        ]

        log.debug(f"Returning {len(models)} OpenAI video models")
        return models

    async def get_available_image_models(self) -> List[ImageModel]:
        """
        Get available OpenAI image generation models.

        Returns DALL-E models for image generation.
        Returns an empty list if no API key is configured.

        Returns:
            List of ImageModel instances for OpenAI DALL-E
        """
        if not self.api_key:
            log.debug("No OpenAI API key configured, returning empty image model list")
            return []

        # OpenAI image generation models
        # Source: https://platform.openai.com/docs/guides/images
        image_models_config = [
            {
                "id": "gpt-image-1",
                "name": "GPT Image 1",
            },
            {
                "id": "dall-e-3",
                "name": "DALL-E 3 (legacy)",
            },
            {
                "id": "dall-e-2",
                "name": "DALL-E 2 (legacy)",
            },
        ]

        models: List[ImageModel] = []
        for config in image_models_config:
            model_id = config["id"]
            # Heuristic: GPT-Image-1 supports both generate and edit; DALL-E legacy considered text-to-image only
            tasks = ["text_to_image", "image_to_image"] if model_id == "gpt-image-1" else ["text_to_image"]
            models.append(
                ImageModel(
                    id=model_id,
                    name=config["name"],
                    provider=Provider.OpenAI,
                    supported_tasks=tasks,
                )
            )

        log.debug(f"Returning {len(models)} OpenAI image models")
        return models

    def _resolve_image_size(self, width: int | None, height: int | None) -> str | None:
        """Convert requested dimensions to OpenAI-supported image sizes.

        OpenAI DALL-E API supports: '1024x1024', '1024x1536', '1536x1024', 'auto'

        Args:
            width: Requested width
            height: Requested height

        Returns:
            OpenAI-compatible size string or None if no dimensions provided
        """
        if not width or not height:
            return None

        # OpenAI supported sizes
        supported_sizes = [
            (1024, 1024),
            (1024, 1536),
            (1536, 1024),
        ]

        # Find the closest supported size by area and aspect ratio
        target_area = width * height
        target_ratio = width / height

        def score_size(supported_w: int, supported_h: int) -> float:
            supported_area = supported_w * supported_h
            supported_ratio = supported_w / supported_h

            # Score based on area difference and aspect ratio difference
            area_score = abs(supported_area - target_area) / max(target_area, supported_area)
            ratio_score = abs(supported_ratio - target_ratio)

            return area_score * 0.7 + ratio_score * 0.3

        best_size = min(supported_sizes, key=lambda size: score_size(size[0], size[1]))
        return f"{best_size[0]}x{best_size[1]}"

    @staticmethod
    def _resolve_video_size(
        aspect_ratio: str | None,
        resolution: str | None,
    ) -> str | None:
        """Convert aspect ratio and resolution inputs into Sora size strings."""
        if not resolution:
            return None

        aspect_key = (aspect_ratio or "16:9").replace(" ", "")
        resolution_key = resolution.lower()

        supported_sizes: list[tuple[int, int]] = [
            (1280, 720),
            (1792, 1024),
            (720, 1280),
            (1024, 1792),
        ]

        def _format_size(size: tuple[int, int]) -> str:
            return f"{size[0]}x{size[1]}"

        def _closest_supported_size(width: int, height: int) -> str:
            target_ratio = width / height
            target_area = width * height

            def score(candidate: tuple[int, int]) -> float:
                cand_w, cand_h = candidate
                cand_ratio = cand_w / cand_h
                cand_area = cand_w * cand_h
                ratio_diff = abs(cand_ratio - target_ratio)
                area_diff = abs(cand_area - target_area) / max(target_area, cand_area)
                return ratio_diff * 5 + area_diff

            return _format_size(min(supported_sizes, key=score))

        size_presets: dict[str, dict[str, str]] = {
            "16:9": {
                "480p": "854x480",
                "720p": "1280x720",
                "1080p": "1920x1080",
            },
            "9:16": {
                "480p": "480x854",
                "720p": "720x1280",
                "1080p": "1080x1920",
            },
            "1:1": {
                "480p": "480x480",
                "720p": "720x720",
                "1080p": "1080x1080",
            },
            "4:3": {
                "480p": "640x480",
                "720p": "960x720",
                "1080p": "1440x1080",
            },
            "3:4": {
                "480p": "480x640",
                "720p": "720x960",
                "1080p": "1080x1440",
            },
        }

        preset_sizes = size_presets.get(aspect_key)
        if preset_sizes:
            size = preset_sizes.get(resolution_key)
            if size:
                width_str, height_str = size.split("x")
                return _closest_supported_size(int(width_str), int(height_str))

        # Fallback to computed size if preset not available
        if ":" in aspect_key:
            width_ratio_str, height_ratio_str = aspect_key.split(":", 1)
            width_ratio = float(width_ratio_str)
            height_ratio = float(height_ratio_str)
        else:
            # If no colon, assume square ratio
            width_ratio = height_ratio = 1.0

        resolution_value = int("".join(ch for ch in resolution_key if ch.isdigit()))
        if resolution_value <= 0 or width_ratio <= 0 or height_ratio <= 0:
            return None

        if width_ratio >= height_ratio:
            height = resolution_value
            width = height * (width_ratio / height_ratio)
        else:
            width = resolution_value
            height = width * (height_ratio / width_ratio)

        def _make_even(value: float) -> int:
            rounded = round(value)
            if rounded % 2 != 0:
                rounded += 1 if rounded > 0 else 0
            return max(2, rounded)

        computed_width = _make_even(width)
        computed_height = _make_even(height)
        return _closest_supported_size(computed_width, computed_height)

    @staticmethod
    def _snap_to_valid_video_dimensions(width: int, height: int) -> str:
        """Snap arbitrary dimensions to the closest valid OpenAI Sora video size.

        OpenAI Sora-2 supports only 2 specific video dimensions. This method finds the
        closest valid size while preserving aspect ratio as much as possible.

        Supported sizes for OpenAI Sora-2:
        - 1280x720 (16:9 landscape)
        - 720x1280 (9:16 portrait)

        Args:
            width: Desired width in pixels
            height: Desired height in pixels

        Returns:
            Size string in format "WIDTHxHEIGHT" snapped to valid dimensions
        """
        supported_sizes: list[tuple[int, int]] = [
            (1280, 720),  # 16:9 landscape
            (720, 1280),  # 9:16 portrait
        ]

        target_ratio = width / height
        target_area = width * height

        def score_size(candidate: tuple[int, int]) -> float:
            """Calculate how well a candidate size matches the target dimensions.

            Uses a weighted combination of aspect ratio difference and area difference.
            Aspect ratio is weighted more heavily to preserve the visual composition.
            """
            cand_w, cand_h = candidate
            cand_ratio = cand_w / cand_h
            cand_area = cand_w * cand_h

            # Aspect ratio difference (more important for visual preservation)
            ratio_diff = abs(cand_ratio - target_ratio)

            # Area difference (normalized to prevent dominating the score)
            area_diff = abs(cand_area - target_area) / max(target_area, cand_area)

            # Weight ratio more heavily (5x) to preserve aspect ratio
            return ratio_diff * 5 + area_diff

        best_size = min(supported_sizes, key=score_size)
        log.debug(
            f"Snapped dimensions {width}x{height} (ratio {target_ratio:.2f}) "
            f"to {best_size[0]}x{best_size[1]} (ratio {best_size[0] / best_size[1]:.2f})"
        )
        return f"{best_size[0]}x{best_size[1]}"

    @staticmethod
    def _extract_image_dimensions(image: bytes) -> tuple[int, int]:
        """Extract width and height from image bytes.

        Args:
            image: Image data as bytes

        Returns:
            Tuple of (width, height) in pixels

        Raises:
            ValueError: If image cannot be opened or dimensions cannot be extracted
        """
        try:
            with Image.open(io.BytesIO(image)) as img:
                width, height = img.size
                log.debug(f"Extracted image dimensions: {width}x{height}")
                return width, height
        except Exception as e:
            log.error(f"Failed to extract image dimensions: {e}")
            raise ValueError(f"Failed to extract image dimensions: {e}") from e

    @staticmethod
    def _resize_image(image: bytes, target_width: int, target_height: int) -> bytes:
        """Resize an image to exact target dimensions.

        Args:
            image: Original image data as bytes
            target_width: Target width in pixels
            target_height: Target height in pixels

        Returns:
            Resized image as bytes (PNG format)

        Raises:
            ValueError: If image cannot be resized
        """
        try:
            with Image.open(io.BytesIO(image)) as img:
                # Convert to RGB if needed (handle RGBA, grayscale, etc.)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                # Resize using high-quality LANCZOS resampling
                resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                # Convert back to bytes
                output = io.BytesIO()
                resized_img.save(output, format="PNG")
                result = output.getvalue()

                log.info(f"Resized image from {img.size[0]}x{img.size[1]} to {target_width}x{target_height}")
                return result
        except Exception as e:
            log.error(f"Failed to resize image: {e}")
            raise ValueError(f"Failed to resize image: {e}") from e

    @staticmethod
    def _seconds_from_params(
        params: TextToVideoParams | ImageToVideoParams,
    ) -> int | None:
        """Derive generation length in seconds from provided parameters."""
        num_frames = params.num_frames
        if not num_frames:
            return None

        # Assume 24 FPS if not specified by API. Clamp to Sora's supported buckets.
        estimated = math.ceil(num_frames / 24)
        # Sora currently accepts only 4, 8 or 12 second durations.
        if estimated < 4:
            return 4
        if estimated < 8:
            return 4
        if estimated < 12:
            return 8
        return 12

    async def uri_to_base64(self, uri: str) -> str:
        """Convert a URI to a base64-encoded ``data:`` URI string.

        If the URI points to audio, convert it to MP3 first for compatibility.

        Args:
            uri: Source URI. Supports standard URLs and ``data:`` URIs.

        Returns:
            A ``data:<mime>;base64,<data>`` string suitable for OpenAI APIs.
        """
        log.debug(f"Converting URI to base64: {uri[:50]}...")

        # Handle data URIs via normalizer to ensure audio normalization
        if uri.startswith("data:"):
            log.debug("Processing data URI directly via normalizer")
            return self._normalize_data_uri(uri)

        # Use shared utility for consistent fetching across providers
        mime_type, data_bytes = await fetch_uri_bytes_and_mime(uri)
        log.debug(f"Fetched bytes via utility. Mime: {mime_type}, length: {len(data_bytes)}")

        # Convert audio to mp3 if needed
        if mime_type.startswith("audio/") and mime_type != "audio/mpeg":
            log.debug("Converting audio to MP3 format")
            try:
                audio = AudioSegment.from_file(io.BytesIO(data_bytes))
                with io.BytesIO() as buffer:
                    audio.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                mime_type = "audio/mpeg"
                content_b64 = base64.b64encode(mp3_data).decode("utf-8")
                log.debug(f"Audio converted to MP3, new length: {len(mp3_data)}")
            except Exception as e:
                log.warning(f"Failed to convert audio URI {uri} to MP3: {e}. Using original content.")
                print(f"Warning: Failed to convert audio URI {uri} to MP3: {e}. Using original content.")
                content_b64 = base64.b64encode(data_bytes).decode("utf-8")
        else:
            log.debug("Encoding content to base64")
            content_b64 = base64.b64encode(data_bytes).decode("utf-8")

        result = f"data:{mime_type};base64,{content_b64}"
        log.debug(f"Created data URI with mime type: {mime_type}")
        return result

    def _normalize_data_uri(self, uri: str) -> str:
        """Normalize a data URI and convert audio to MP3 when necessary.

        Args:
            uri: A ``data:`` URI string.

        Returns:
            A normalized ``data:<mime>;base64,<base64data>`` string.
        """
        log.debug(f"Normalizing data URI: {uri[:50]}...")

        # Format: data:[<mediatype>][;base64],<data>
        try:
            header, data_part = uri.split(",", 1)
        except ValueError as e:
            log.error(f"Invalid data URI format: {uri[:64]}...")
            raise ValueError(f"Invalid data URI: {uri[:64]}...") from e

        is_base64 = ";base64" in header
        mime_type = "application/octet-stream"
        if header[5:]:  # after 'data:'
            mime_type = header[5:].split(";", 1)[0] or mime_type

        log.debug(f"Data URI mime type: {mime_type}, is_base64: {is_base64}")

        # Decode payload to bytes
        if is_base64:
            try:
                raw_bytes = base64.b64decode(data_part)
                log.debug(f"Decoded base64 data, length: {len(raw_bytes)}")
            except Exception as e:
                log.error(f"Failed to decode base64 data URI: {e}")
                raise ValueError(f"Failed to decode base64 data URI: {e}") from e
        else:
            # Percent-decoded textual payload → bytes
            raw_bytes = unquote_to_bytes(data_part)
            log.debug(f"Decoded percent-encoded data, length: {len(raw_bytes)}")

        # If audio and not mp3, convert to mp3; otherwise keep as-is
        if mime_type.startswith("audio/") and mime_type != "audio/mpeg":
            log.debug("Converting audio data to MP3 format")
            try:
                audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
                with io.BytesIO() as buffer:
                    audio.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                mime_type = "audio/mpeg"
                content_b64 = base64.b64encode(mp3_data).decode("utf-8")
                log.debug(f"Audio converted to MP3, new length: {len(mp3_data)}")
            except Exception as e:
                log.warning(f"Failed to convert data URI audio to MP3: {e}. Using original content.")
                print(f"Warning: Failed to convert data URI audio to MP3: {e}. Using original content.")
                content_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        else:
            log.debug("Encoding data to base64")
            content_b64 = base64.b64encode(raw_bytes).decode("utf-8")

        result = f"data:{mime_type};base64,{content_b64}"
        log.debug(f"Normalized data URI with mime type: {mime_type}")
        return result

    async def message_content_to_openai_content_part(self, content: MessageContent) -> ChatCompletionContentPartParam:
        """Convert a message content to an OpenAI content part.

        Args:
            content: Internal message content variant (text, image, audio).

        Returns:
            A content part dictionary per OpenAI's chat API specification.
        """
        log.debug(f"Converting message content type: {type(content)}")

        if isinstance(content, MessageTextContent):
            log.debug(f"Converting text content: {content.text[:50]}...")
            return {"type": "text", "text": content.text}
        elif isinstance(content, MessageAudioContent):
            log.debug("Converting audio content")
            if content.audio.uri:
                # uri_to_base64 now handles conversion and returns MP3 data URI
                data_uri = await self.uri_to_base64(content.audio.uri)
                # Extract base64 data part for OpenAI API
                base64_data = data_uri.split(",", 1)[1]
                log.debug(f"Audio URI processed, data length: {len(base64_data)}")
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "format": "mp3",
                        "data": base64_data,
                    },
                }
            else:
                log.debug("Converting raw audio data to MP3")
                # Convert raw bytes data to MP3 using pydub
                try:
                    audio = AudioSegment.from_file(io.BytesIO(content.audio.data))
                    with io.BytesIO() as buffer:
                        audio.export(buffer, format="mp3")
                        mp3_data = buffer.getvalue()
                    data = base64.b64encode(mp3_data).decode("utf-8")
                    log.debug(f"Audio converted to MP3, data length: {len(data)}")
                except Exception as e:
                    log.warning(f"Failed to convert raw audio data to MP3: {e}. Sending original data.")
                    print(f"Warning: Failed to convert raw audio data to MP3: {e}. Sending original data.")
                    # Fallback to sending original data if conversion fails
                    data = base64.b64encode(content.audio.data).decode("utf-8")

                return {
                    "type": "input_audio",
                    "input_audio": {
                        "format": "mp3",
                        "data": data,
                    },
                }
        elif isinstance(content, MessageImageContent):
            log.debug("Converting image content")
            if content.image.uri:
                # For images, use the original uri_to_base64 logic (implicitly called)
                image_url = await self.uri_to_base64(content.image.uri)
                log.debug(f"Image URI processed: {image_url[:50]}...")
                return {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            else:
                log.debug("Converting raw image data")
                # Normalize to JPEG base64 using shared helper
                data = image_data_to_base64_jpeg(content.image.data)
                image_url = f"data:image/jpeg;base64,{data}"
                log.debug(f"Raw image data processed, length: {len(data)}")
                return {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
        else:
            log.error(f"Unknown content type {content}")
            raise ValueError(f"Unknown content type {content}")

    async def convert_message(self, message: Message) -> ChatCompletionMessageParam:
        """Convert an internal message to OpenAI's message param format.

        Args:
            message: Internal ``Message`` instance.

        Returns:
            OpenAI chat message structure matching the input role/content.
        """
        import json

        log.debug(f"Converting message with role: {message.role}")

        if message.role == "tool":
            log.debug(f"Converting tool message, tool_call_id: {message.tool_call_id}")
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            elif isinstance(message.content, dict):
                content = json.dumps(message.content)
            elif isinstance(message.content, list):
                content = json.dumps([part.model_dump() for part in message.content])
            elif isinstance(message.content, str):
                content = message.content
            else:
                content = json.dumps(message.content)
            log.debug(f"Tool message content type: {type(message.content)}")
            assert message.tool_call_id is not None, "Tool call ID must not be None"
            return ChatCompletionToolMessageParam(
                role=message.role,
                content=content,
                tool_call_id=message.tool_call_id,
            )
        elif message.role == "system":
            log.debug("Converting system message")
            return ChatCompletionSystemMessageParam(role=message.role, content=str(message.content))
        elif message.role == "user":
            log.debug("Converting user message")
            assert message.content is not None, "User message content must not be None"
            if isinstance(message.content, str):
                content = message.content
                log.debug("User message has string content")
            elif message.content is not None:
                log.debug(f"Converting {len(message.content)} content parts")
                content = [await self.message_content_to_openai_content_part(c) for c in message.content]
            else:
                log.error(f"Unknown message content type {type(message.content)}")
                raise ValueError(f"Unknown message content type {type(message.content)}")
            return ChatCompletionUserMessageParam(role=message.role, content=content)
        elif message.role == "assistant":
            log.debug("Converting assistant message")
            tool_calls = [
                ChatCompletionMessageFunctionToolCallParam(
                    type="function",
                    id=tool_call.id,
                    function=Function(
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.args, default=self._default_serializer),
                    ),
                )
                for tool_call in message.tool_calls or []
            ]
            log.debug(f"Assistant message has {len(tool_calls)} tool calls")

            if isinstance(message.content, str):
                content = message.content
                log.debug("Assistant message has string content")
            elif message.content is not None:
                log.debug(f"Converting {len(message.content)} assistant content parts")
                content = [await self.message_content_to_openai_content_part(c) for c in message.content]
            else:
                content = None
                log.debug("Assistant message has no content")

            if len(tool_calls) == 0:
                log.debug("Returning assistant message without tool calls")
                return ChatCompletionAssistantMessageParam(
                    role=message.role,
                    content=content,  # type: ignore
                )
            else:
                log.debug("Returning assistant message with tool calls")
                return ChatCompletionAssistantMessageParam(
                    role=message.role,
                    content=content,  # type: ignore
                    tool_calls=tool_calls,  # type: ignore
                )
        else:
            log.error(f"Unknown message role: {message.role}")
            raise ValueError(f"Unknown message role {message.role}")

    def _default_serializer(self, obj: Any) -> dict:
        """Serialize Pydantic models to dict."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        raise TypeError("Type not serializable")

    def format_tools(self, tools: Sequence[Tool]) -> list[ChatCompletionMessageFunctionToolCallParam]:
        """Convert internal tools to OpenAI function/tool definitions.

        Args:
            tools: Iterable of tools to expose to the model.

        Returns:
            List of OpenAI-compatible tool specifications.
        """
        log.debug(f"Formatting {len(tools)} tools for OpenAI API")
        formatted_tools = []

        for tool in tools:
            log.debug(f"Formatting tool: {tool.name}")
            if tool.name == "code_interpreter":
                # Handle code_interpreter tool specially
                formatted_tools.append({"type": "code_interpreter"})
                log.debug("Added code_interpreter tool")
            else:
                # Handle regular function tools
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    }
                )
                log.debug(f"Added function tool: {tool.name}")

        log.debug(f"Formatted {len(formatted_tools)} tools total")
        return formatted_tools

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        json_schema: dict | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Stream assistant deltas and tool calls from OpenAI.

        Args:
            messages: Conversation history to send.
            model: Target OpenAI model.
            tools: Optional tool definitions to provide.
            max_tokens: Maximum tokens to generate.
            context_window: Maximum tokens considered for context.
            json_schema: Optional response schema.
            response_format: Optional structured output format.
            **kwargs: Additional OpenAI parameters such as temperature.

        Yields:
            Text ``Chunk`` items and ``ToolCall`` objects when the model
            requests tool execution.
        """
        import json

        log.debug(f"Starting streaming generation for model: {model}")
        log.debug(f"Streaming with {len(messages)} messages, {len(tools)} tools")

        if not messages:
            raise ValueError("messages must not be empty")

        # Convert system messages to user messages for O1/O3 models
        _kwargs: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if response_format is None:
            response_format = kwargs.get("response_format")
        if response_format is not None and json_schema is not None:
            raise ValueError("response_format and json_schema are mutually exclusive")
        if response_format is not None:
            _kwargs["response_format"] = response_format
        elif json_schema is not None:
            _kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        # Common sampling params if provided
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if key in kwargs and kwargs[key] is not None:
                _kwargs[key] = kwargs[key]
        log.debug(f"Initial kwargs: {_kwargs}")

        if kwargs.get("audio"):
            _kwargs["audio"] = kwargs.get("audio")
            _kwargs["modalities"] = ["text", "audio"]
            if not kwargs.get("audio"):
                _kwargs["audio"] = {
                    "voice": "alloy",
                    "format": "pcm16",
                }
            log.debug("Added audio modalities to request")

        if len(tools) > 0:
            _kwargs["tools"] = self.format_tools(tools)
            log.debug(f"Added {len(tools)} tools to request")

        if model.startswith("o"):
            log.debug("Converting system messages for O-series model")
            _kwargs.pop("temperature", None)
            converted_messages = []
            for msg in messages:
                if msg.role == "system":
                    log.debug("Converting system message to user message for O-series model")
                    converted_messages.append(
                        Message(
                            role="user",
                            content=f"Instructions: {msg.content}",
                            thread_id=msg.thread_id,
                        )
                    )
                else:
                    converted_messages.append(msg)
            messages = converted_messages
            log.debug(f"Converted {len(converted_messages)} messages for O-series model")

        self._log_api_request(
            "chat_stream",
            messages,
            **_kwargs,
        )

        log.debug(f"Converting {len(messages)} messages to OpenAI format")
        openai_messages = [await self.convert_message(m) for m in messages]
        log.debug("Making streaming API call to OpenAI")

        create_result = self.get_client().chat.completions.create(
            messages=openai_messages,
            **_kwargs,
        )
        if inspect.isawaitable(create_result):
            completion = await create_result
        else:
            completion = create_result
        log.debug("Streaming response initialized")
        delta_tool_calls = {}
        current_chunk = ""
        chunk_count = 0

        async for chunk in completion:
            chunk: ChatCompletionChunk = chunk
            chunk_count += 1

            # Track usage information (only available in the final chunk)
            if chunk.usage:
                log.debug("Processing usage statistics from chunk")
                self.usage["prompt_tokens"] += chunk.usage.prompt_tokens
                self.usage["completion_tokens"] += chunk.usage.completion_tokens
                self.usage["total_tokens"] += chunk.usage.total_tokens
                if chunk.usage.prompt_tokens_details and chunk.usage.prompt_tokens_details.cached_tokens:
                    self.usage["cached_prompt_tokens"] += chunk.usage.prompt_tokens_details.cached_tokens
                if chunk.usage.completion_tokens_details and chunk.usage.completion_tokens_details.reasoning_tokens:
                    self.usage["reasoning_tokens"] += chunk.usage.completion_tokens_details.reasoning_tokens
                log.debug(f"Updated usage stats: {self.usage}")

            if not chunk.choices:
                log.debug("Chunk has no choices, skipping")
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, "audio") and "data" in delta.audio:  # type: ignore
                log.debug("Yielding audio chunk")
                yield Chunk(
                    content=delta.audio["data"],  # type: ignore
                    content_type="audio",
                )

            # Process tool call deltas before checking finish_reason
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tc: dict[str, Any] | None = None
                    if tool_call.index in delta_tool_calls:
                        tc = delta_tool_calls[tool_call.index]
                    else:
                        tc = {"id": tool_call.id}
                        delta_tool_calls[tool_call.index] = tc
                    assert tc is not None, "Tool call must not be None"

                    if tool_call.id:
                        tc["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        tc["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        if "function" not in tc:
                            tc["function"] = {}
                        if "arguments" not in tc["function"]:
                            tc["function"]["arguments"] = ""
                        tc["function"]["arguments"] += tool_call.function.arguments

            if delta.content or chunk.choices[0].finish_reason == "stop":
                current_chunk += delta.content or ""
                finish_reason = chunk.choices[0].finish_reason
                log.debug(f"Content chunk - finish_reason: {finish_reason}, content length: {len(delta.content or '')}")

                if finish_reason == "stop":
                    log.debug("Final chunk received, logging response")
                    self._log_api_response(
                        "chat_stream",
                        Message(
                            role="assistant",
                            content=current_chunk,
                        ),
                    )

                content_to_yield = delta.content or ""
                yield Chunk(
                    content=content_to_yield,
                    done=finish_reason == "stop",
                )

            if chunk.choices[0].finish_reason == "tool_calls":
                log.debug("Processing tool calls completion")
                if delta_tool_calls:
                    log.debug(f"Yielding {len(delta_tool_calls)} tool calls")
                    for tc in delta_tool_calls.values():
                        assert tc is not None, "Tool call must not be None"
                        tool_call = ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            args=json.loads(tc["function"]["arguments"]),
                        )
                        self._log_tool_call(tool_call)
                        yield tool_call
                else:
                    log.error("No tool call found in delta_tool_calls")
                    raise ValueError("No tool call found")

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        json_schema: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        """Generate a non-streaming completion from OpenAI.

        Args:
            messages: The message history
            model: The model to use
            tools: Optional tools to provide to the model
            max_tokens: The maximum number of tokens to generate
            context_window: The maximum number of tokens to consider for the context
            response_format: The format of the response
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            A Message object containing the model's response
        """
        import json

        log.debug(f"Generating non-streaming message for model: {model}")
        log.debug(f"Non-streaming with {len(messages)} messages, {len(tools)} tools")

        if not messages:
            raise ValueError("messages must not be empty")

        request_kwargs: dict[str, Any] = {
            "max_completion_tokens": max_tokens,
        }
        if response_format is None:
            response_format = kwargs.get("response_format")
        if response_format is not None and json_schema is not None:
            raise ValueError("response_format and json_schema are mutually exclusive")
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        elif json_schema is not None:
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        # Common sampling params (pass-through if provided via caller)
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if presence_penalty is not None:
            request_kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            request_kwargs["frequency_penalty"] = frequency_penalty
        log.debug(f"Request kwargs: {request_kwargs}")

        # Convert system messages to user messages for O1/O3 models
        if model.startswith("o1") or model.startswith("o3"):
            log.debug("Converting system messages for O-series model")
            converted_messages = []
            for msg in messages:
                if msg.role == "system":
                    log.debug("Converting system message to user message")
                    converted_messages.append(
                        Message(
                            role="user",
                            content=f"Instructions: {msg.content}",
                            thread_id=msg.thread_id,
                        )
                    )
                else:
                    converted_messages.append(msg)
            messages = converted_messages
            log.debug(f"Converted {len(converted_messages)} messages for O-series model")

        self._log_api_request("chat", messages, **request_kwargs)

        if len(tools) > 0:
            request_kwargs["tools"] = self.format_tools(tools)
            log.debug(f"Added {len(tools)} tools to request")

        log.debug(f"Converting {len(messages)} messages to OpenAI format")
        openai_messages = [await self.convert_message(m) for m in messages]
        log.debug("Making non-streaming API call to OpenAI")

        # Make non-streaming call to OpenAI
        try:
            create_result = self.get_client().chat.completions.create(
                model=model,
                messages=openai_messages,
                stream=False,
                **request_kwargs,
            )
            if inspect.isawaitable(create_result):
                completion = await create_result
            else:
                completion = create_result
        except openai.OpenAIError as exc:
            raise self._as_httpx_status_error(exc) from exc
        log.debug("Received response from OpenAI API")

        # Debug log the raw response for structured output debugging
        if completion.choices:
            choice = completion.choices[0]
            log.info("Response finish_reason: %s", choice.finish_reason)
            log.info("Response message content length: %d", len(choice.message.content or ""))
            if choice.message.content:
                log.debug("Response message content (first 500 chars): %s", choice.message.content[:500])
            else:
                log.warning("Response message content is None!")
            if choice.message.refusal:
                log.warning("Response message refusal: %s", choice.message.refusal)

        # Update usage stats
        if completion.usage:
            log.debug("Processing usage statistics")
            self.usage["prompt_tokens"] += completion.usage.prompt_tokens
            self.usage["completion_tokens"] += completion.usage.completion_tokens
            self.usage["total_tokens"] += completion.usage.total_tokens
            cost = await calculate_chat_cost(
                model,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )
            self.cost += cost
            log.debug(f"Updated usage: {self.usage}, cost: {cost}")

        choice = completion.choices[0]
        response_message = choice.message
        log.debug(f"Response content length: {len(response_message.content or '')}")

        def try_parse_args(args: Any) -> Any:
            try:
                return json.loads(args)
            except Exception:
                log.warning(f"Error parsing tool call arguments: {args}")
                print(f"Warning: Error parsing tool call arguments: {args}")
                return {}

        # Create tool calls if present
        tool_calls = None
        if response_message.tool_calls:
            log.debug(f"Processing {len(response_message.tool_calls)} tool calls")
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,  # type: ignore
                    args=try_parse_args(tool_call.function.arguments),  # type: ignore
                )
                for tool_call in response_message.tool_calls
            ]
        else:
            log.debug("Response contains no tool calls")

        message = Message(
            role="assistant",
            content=response_message.content,
            tool_calls=tool_calls,
        )

        self._log_api_response("chat", message)
        log.debug("Returning generated message")

        return message

    @staticmethod
    def _as_httpx_status_error(exc: openai.OpenAIError) -> httpx.HTTPStatusError:
        """Normalize OpenAI SDK exceptions to `httpx.HTTPStatusError`.

        Provider tests and shared error handling expect HTTPStatusError semantics.
        """
        import httpx

        maybe_response = getattr(exc, "response", None)
        status_code = getattr(maybe_response, "status_code", None) or getattr(exc, "status_code", 500)

        request = getattr(maybe_response, "request", None)
        if not isinstance(request, httpx.Request):
            request = httpx.Request(
                "POST",
                "https://api.openai.com/v1/chat/completions",
            )

        response = maybe_response if isinstance(maybe_response, httpx.Response) else None
        if response is None:
            response = httpx.Response(status_code=int(status_code), request=request)

        return httpx.HTTPStatusError(str(exc), request=request, response=response)

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate an image from a text prompt using OpenAI's Images API."""
        if not params.prompt:
            raise ValueError("The input prompt cannot be empty.")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for image generation.")

        model_id = params.model.id
        if not model_id:
            raise ValueError("A text-to-image model with a valid id must be specified for image generation.")

        prompt = params.prompt.strip()
        if params.negative_prompt:
            prompt = f"{prompt}\n\nDo not include: {params.negative_prompt.strip()}"

        size = None
        if params.width and params.height:
            if params.width <= 0 or params.height <= 0:
                raise ValueError("width and height must be positive integers.")
            size = self._resolve_image_size(int(params.width), int(params.height))

        try:
            request_timeout = timeout_s if timeout_s and timeout_s > 0 else 120
            client = self.get_client()

            response = await client.images.generate(
                model=model_id,
                prompt=prompt,
                n=1,
                size=size if size else Omit,  # type: ignore
                timeout=request_timeout,
            )

            data = response.data or []
            if len(data) == 0:
                raise RuntimeError("OpenAI image generation returned no data.")

            image_entry = data[0]
            image_bytes: bytes | None = None
            b64_data = image_entry.b64_json
            if b64_data:
                image_bytes = base64.b64decode(b64_data)
            else:
                image_url = image_entry.url
                if image_url:
                    _, image_bytes = await fetch_uri_bytes_and_mime(image_url)

            if not image_bytes:
                raise RuntimeError("OpenAI image generation returned no image bytes.")

            return image_bytes

        except openai.APIStatusError as api_error:
            log.error(
                "OpenAI text-to-image generation failed (status=%s): %s",
                api_error.status_code,
                api_error.message,
            )
            raise RuntimeError(
                f"OpenAI text-to-image generation failed with status {api_error.status_code}: {api_error.message}"
            ) from api_error
        except Exception as exc:
            log.error(f"OpenAI text-to-image generation failed: {exc}")
            raise RuntimeError(f"OpenAI text-to-image generation failed: {exc}") from exc

    async def image_to_image(
        self,
        image: bytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
    ) -> bytes:
        """Transform an image based on a prompt using OpenAI's image editing API."""
        if not image:
            raise ValueError("image must not be empty.")

        if not params.prompt:
            raise ValueError("The input prompt cannot be empty.")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for image editing.")

        model_id = params.model.id
        if not model_id:
            raise ValueError("An image-to-image model with a valid id must be specified for image editing.")

        prompt = params.prompt.strip()
        if params.negative_prompt:
            prompt = f"{prompt}\n\nDo not include: {params.negative_prompt.strip()}"

        size = None
        if params.target_width and params.target_height:
            if params.target_width <= 0 or params.target_height <= 0:
                raise ValueError("target_width and target_height must be positive integers.")
            size = self._resolve_image_size(int(params.target_width), int(params.target_height))

        try:
            request_timeout = timeout_s if timeout_s and timeout_s > 0 else 120
            client = self.get_client()
            response = await client.images.edit(
                model=model_id,
                prompt=prompt,
                image=("image.png", image, "image/png"),
                size=size if size else Omit,  # type: ignore
                timeout=request_timeout,
            )

            data_list = response.data or []
            if len(data_list) == 0:
                raise RuntimeError("OpenAI image editing returned no data.")

            entry = data_list[0]
            image_bytes: bytes | None = None
            b64_data = entry.b64_json
            if b64_data:
                image_bytes = base64.b64decode(b64_data)
            else:
                image_url = entry.url
                if image_url:
                    _, image_bytes = await fetch_uri_bytes_and_mime(image_url)

            if not image_bytes:
                raise RuntimeError("OpenAI image editing returned no image bytes.")

            return image_bytes

        except openai.APIStatusError as api_error:
            log.error(
                "OpenAI image editing failed (status=%s): %s",
                api_error.status_code,
                api_error.message,
            )
            raise RuntimeError(
                f"OpenAI image editing failed with status {api_error.status_code}: {api_error.message}"
            ) from api_error
        except Exception as exc:
            log.error(f"OpenAI image editing failed: {exc}")
            raise RuntimeError(f"OpenAI image editing failed: {exc}") from exc

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
        """Generate speech audio from text using OpenAI TTS with streaming.

        Uses OpenAI's streaming TTS API to yield audio chunks as they're generated,
        enabling real-time playback.

        Args:
            text: Input text to convert to speech
            model: Model identifier (e.g., "tts-1", "tts-1-hd", "gpt-4o-mini-tts")
            voice: Voice identifier (e.g., "alloy", "echo", "fable", "onyx", "nova", "shimmer")
            speed: Speech speed multiplier (0.25 to 4.0)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            **kwargs: Additional OpenAI parameters

        Yields:
            numpy.ndarray: Int16 audio chunks at 24kHz mono

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails
        """
        log.debug(f"Generating streaming speech for model: {model}, voice: {voice}, speed: {speed}")

        if not text:
            raise ValueError("text must not be empty")

        # Default voice to "alloy" if not specified
        voice = voice or "alloy"

        # Clamp speed to OpenAI's supported range
        speed = max(0.25, min(4.0, speed))
        log.debug(f"Making streaming TTS API call with model={model}, voice={voice}, speed={speed}")

        try:
            # Use streaming response
            async with self.get_client().audio.speech.with_streaming_response.create(
                model=model,
                input=text,
                voice=voice,  # type: ignore
                speed=speed,
                response_format="pcm",
            ) as response:
                log.debug("TTS streaming API call started")

                # Collect all chunks first (OpenAI sends complete MP3)
                async for chunk in response.iter_bytes(chunk_size=4096):
                    yield np.frombuffer(chunk, dtype=np.int16)

                log.debug("TTS streaming completed")

            self._log_api_response("text_to_speech")

        except Exception as e:
            log.error(f"OpenAI TTS streaming failed: {e}")
            raise RuntimeError(f"OpenAI TTS generation failed: {str(e)}") from e

    async def text_to_video(
        self,
        params: TextToVideoParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate a video from a text prompt using OpenAI Sora models.

        Args:
            params: Text-to-video generation parameters including:
                - prompt: Text description of the video
                - model: VideoModel with Sora model ID
                - aspect_ratio: Optional aspect ratio (e.g., "16:9")
                - resolution: Optional resolution (e.g., "1080p")
                - num_frames: Optional frame count (used to estimate duration)
            timeout_s: Optional timeout in seconds (defaults to 10 minutes)
            context: Processing context for asset handling (unused, reserved)

        Returns:
            Raw video bytes (MP4 format)

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails or the API returns an error
        """
        import asyncio

        if not params.prompt:
            raise ValueError("The input prompt cannot be empty.")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for video generation")

        if not params.model.id:
            raise ValueError("A video model with a valid id must be specified for text-to-video generation.")

        log.debug(f"Starting OpenAI video generation with model: {params.model.id}")

        size = self._resolve_video_size(
            params.aspect_ratio,
            params.resolution,
        )
        seconds = self._seconds_from_params(params)

        if params.negative_prompt:
            log.debug("negative_prompt provided but not currently supported by the OpenAI video API; ignoring.")

        self._log_api_request("text_to_video", params=params)

        request_timeout = timeout_s if timeout_s and timeout_s > 0 else 600
        client = self.get_client()

        try:
            video = await self._create_video_job(
                client=client,
                model_id=params.model.id,
                prompt=params.prompt,
                size=size if size else "1024x1024",
                seconds=seconds if seconds else 4,
                timeout=request_timeout,
            )
            log.debug("Submitted video generation request via OpenAI client.")
        except openai.APIStatusError as api_error:
            log.error(
                "OpenAI video generation failed (status=%s): %s",
                api_error.status_code,
                api_error.message,
            )
            raise RuntimeError(
                f"OpenAI video generation failed with status {api_error.status_code}: {api_error.message}"
            ) from api_error
        except Exception as exc:
            log.error(f"OpenAI video generation failed: {exc}")
            raise RuntimeError(f"OpenAI video generation failed: {exc}") from exc

        maximum_wait = request_timeout
        poll_interval = max(2, min(10, maximum_wait)) if maximum_wait else 10
        if not video.id:
            log.error(f"OpenAI video create response missing id: {video}")
            raise RuntimeError("OpenAI video create response did not contain a video id")

        log.debug(f"Video job {video.id} created with initial status '{video.status}' and progress {video.progress}")

        elapsed = 0
        while video.status in ("queued", "in_progress"):
            if maximum_wait and elapsed >= maximum_wait:
                raise TimeoutError(f"Video generation timed out after {maximum_wait} seconds")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            video = await client.videos.retrieve(
                video_id=video.id,
                timeout=request_timeout,
            )

            log.debug(f"Video job {video.id} status update: {video.status} (progress={video.progress})")
            if "node_id" in kwargs and context is not None and video.progress is not None:
                context.post_message(
                    NodeProgress(
                        node_id=kwargs["node_id"],
                        progress=video.progress,  # type: ignore
                        total=100,  # type: ignore
                    )
                )

            if video.status == "failed":
                break

        if video.status != "completed":
            message = video.error or f"Video generation ended with status '{video.status}'"
            raise RuntimeError(message)

        video_bytes = await self._download_video_content(
            client=client,
            video_id=video.id,
            timeout=request_timeout,
        )

        if not video_bytes:
            raise RuntimeError("OpenAI video download returned no data")

        log.debug(f"Downloaded {len(video_bytes)} bytes for video job {video.id}")

        self._log_api_response("text_to_video", video_bytes=len(video_bytes))
        return video_bytes

    async def _create_video_job(
        self,
        client: openai.AsyncClient,
        model_id: str,
        prompt: str,
        size: str | NotGiven,
        seconds: int,
        timeout: float,
    ) -> Video:
        log.debug(f"Submitting video generation request: {model_id}, {prompt}, {size}, {seconds}")
        return await client.videos.create(
            model=model_id,  # type: ignore
            prompt=prompt,
            size=size if size else Omit,  # type: ignore
            seconds=str(seconds) if seconds else Omit,  # type: ignore
            timeout=timeout,
        )

    async def _download_video_content(
        self,
        client: openai.AsyncClient,
        video_id: str,
        timeout: float,
    ) -> bytes:
        return await client.get(
            f"/videos/{video_id}/content",
            cast_to=bytes,
            options={  # type: ignore
                "timeout": timeout,
                "params": {"variant": "video"},
            },
        )

    async def image_to_video(
        self,
        image: bytes,
        params: ImageToVideoParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate a video from an input image using OpenAI Sora models.

        Automatically matches the output video dimensions to the input image dimensions,
        snapping to the closest valid OpenAI Sora video size while preserving aspect ratio.

        Args:
            image: Input image as bytes
            params: Image-to-video generation parameters including:
                - model: VideoModel with Sora model ID
                - prompt: Optional text description to guide video generation
                - negative_prompt: Elements to exclude from generation
                - num_frames: Optional frame count (used to estimate duration)
                Note: aspect_ratio and resolution are ignored; dimensions are extracted from the image
            timeout_s: Optional timeout in seconds (defaults to 10 minutes)
            context: Processing context for asset handling (unused, reserved)

        Returns:
            Raw video bytes (MP4 format)

        Raises:
            ValueError: If required parameters are missing or image dimensions cannot be extracted
            RuntimeError: If generation fails or the API returns an error
        """
        import asyncio
        import imghdr

        if not image:
            raise ValueError("The input image cannot be empty.")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for image-to-video generation")

        if not params.model.id:
            raise ValueError("A video model with a valid id must be specified for image-to-video generation.")

        log.debug(f"Starting OpenAI image-to-video generation with model: {params.model.id}")

        # Extract dimensions from input image and snap to valid OpenAI sizes
        try:
            img_width, img_height = self._extract_image_dimensions(image)
            size = self._snap_to_valid_video_dimensions(img_width, img_height)
            log.info(f"Using image dimensions {img_width}x{img_height}, snapped to valid video size: {size}")

            # Resize image to match the snapped video dimensions
            target_width, target_height = map(int, size.split("x"))
            if img_width != target_width or img_height != target_height:
                log.info(f"Resizing image to match video dimensions: {size}")
                image = self._resize_image(image, target_width, target_height)
            else:
                log.debug("Image already matches target dimensions, no resize needed")

        except ValueError as e:
            log.error(f"Failed to extract/resize image dimensions: {e}")
            raise ValueError(f"Could not prepare image for video generation: {e}") from e

        seconds = self._seconds_from_params(params)

        if params.negative_prompt:
            log.debug("negative_prompt provided but not currently supported by the OpenAI video API; ignoring.")

        self._log_api_request("image_to_video", params=params)

        request_timeout = timeout_s if timeout_s and timeout_s > 0 else 600
        client = self.get_client()

        # Determine image format for proper MIME type
        image_format = imghdr.what(None, h=image) or "png"
        mime_type = f"image/{image_format}"

        try:
            # Create video job with input_reference parameter using image-derived dimensions
            video = await self._create_video_job_with_image(
                client=client,
                model_id=params.model.id,
                image=image,
                mime_type=mime_type,
                prompt=params.prompt,
                size=size,  # Use the size derived from image dimensions
                seconds=seconds if seconds else 4,
                timeout=request_timeout,
            )
            log.debug("Submitted image-to-video generation request via OpenAI client.")
        except openai.APIStatusError as api_error:
            log.error(
                "OpenAI image-to-video generation failed (status=%s): %s",
                api_error.status_code,
                api_error.message,
            )
            raise RuntimeError(
                f"OpenAI image-to-video generation failed with status {api_error.status_code}: {api_error.message}"
            ) from api_error
        except Exception as exc:
            log.error(f"OpenAI image-to-video generation failed: {exc}")
            raise RuntimeError(f"OpenAI image-to-video generation failed: {exc}") from exc

        maximum_wait = request_timeout
        poll_interval = max(2, min(10, maximum_wait)) if maximum_wait else 10
        if not video.id:
            log.error(f"OpenAI video create response missing id: {video}")
            raise RuntimeError("OpenAI video create response did not contain a video id")

        log.debug(f"Video job {video.id} created with initial status '{video.status}' and progress {video.progress}")

        elapsed = 0
        while video.status in ("queued", "in_progress"):
            if maximum_wait and elapsed >= maximum_wait:
                raise TimeoutError(f"Image-to-video generation timed out after {maximum_wait} seconds")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            video = await client.videos.retrieve(
                video_id=video.id,
                timeout=request_timeout,
            )

            log.debug(f"Video job {video.id} status update: {video.status} (progress={video.progress})")
            if "node_id" in kwargs and context is not None and video.progress is not None:
                context.post_message(
                    NodeProgress(
                        node_id=kwargs["node_id"],
                        progress=video.progress,  # type: ignore
                        total=100,  # type: ignore
                    )
                )

            if video.status == "failed":
                break

        if video.status != "completed":
            message = video.error or f"Image-to-video generation ended with status '{video.status}'"
            raise RuntimeError(message)

        video_bytes = await self._download_video_content(
            client=client,
            video_id=video.id,
            timeout=request_timeout,
        )

        if not video_bytes:
            raise RuntimeError("OpenAI image-to-video download returned no data")

        log.debug(f"Downloaded {len(video_bytes)} bytes for video job {video.id}")

        self._log_api_response("image_to_video", video_bytes=len(video_bytes))
        return video_bytes

    async def _create_video_job_with_image(
        self,
        client: openai.AsyncClient,
        model_id: str,
        image: bytes,
        mime_type: str,
        prompt: str | None,
        size: str | NotGiven,
        seconds: int,
        timeout: float,
    ) -> Video:
        """Create a video generation job with an input image.

        Args:
            client: OpenAI async client
            model_id: Model ID for video generation
            image: Input image bytes
            mime_type: MIME type of the image
            prompt: Optional text prompt to guide generation
            size: Video size specification
            seconds: Duration in seconds
            timeout: Request timeout

        Returns:
            Video object with job details
        """
        log.debug(
            f"Submitting image-to-video generation request: {model_id}, prompt={prompt}, size={size}, seconds={seconds}"
        )

        # Determine filename from mime_type
        ext = mime_type.split("/")[-1]
        filename = f"input_image.{ext}"

        # Create the request payload with input_reference
        return await client.videos.create(
            model=model_id,  # type: ignore
            prompt=prompt or "",
            input_reference=(filename, image, mime_type),  # type: ignore
            size=size if size else Omit,  # type: ignore
            seconds=str(seconds) if seconds else Omit,  # type: ignore
            timeout=timeout,
        )

    async def automatic_speech_recognition(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio to text using OpenAI's Whisper ASR.

        Uses OpenAI's Whisper API to transcribe audio to text. Supports various
        audio formats including mp3, mp4, mpeg, mpga, m4a, wav, and webm.

        Args:
            audio: Input audio as bytes
            model: Model identifier (e.g., "whisper-1")
            language: Optional ISO-639-1 language code (e.g., "en", "es") to improve accuracy
            prompt: Optional text to guide the model's style or continue a previous segment
            temperature: Sampling temperature between 0 and 1 (default 0)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            **kwargs: Additional OpenAI parameters

        Returns:
            str: Transcribed text from the audio

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If transcription fails
        """
        log.debug(f"Transcribing audio with model: {model}, language: {language}, temperature: {temperature}")

        if not audio:
            raise ValueError("audio must not be empty")

        # Clamp temperature to OpenAI's supported range
        temperature = max(0.0, min(1.0, temperature))

        try:
            # Create a file-like object from bytes
            # OpenAI requires a file name with extension for format detection
            from io import BytesIO

            audio_file = BytesIO(audio)
            audio_file.name = "audio.mp3"  # Default to mp3, OpenAI will detect format

            # Build API parameters
            api_params: dict[str, Any] = {
                "file": audio_file,
                "model": model,
                "temperature": temperature,
            }

            if language:
                api_params["language"] = language

            if prompt:
                api_params["prompt"] = prompt

            log.debug(f"Making ASR API call with model={model}")

            # Call OpenAI Whisper API
            client = self.get_client()
            transcription = await client.audio.transcriptions.create(**api_params)

            result_text = transcription.text
            log.debug(f"ASR transcription completed, length: {len(result_text)}")

            self._log_api_response("automatic_speech_recognition")

            return result_text

        except Exception as e:
            log.error(f"OpenAI ASR transcription failed: {e}")
            raise RuntimeError(f"OpenAI ASR transcription failed: {str(e)}") from e

    def get_usage(self) -> dict:
        """Return the current accumulated token usage statistics.

        Returns:
            A shallow copy of the usage counters collected so far.
        """
        log.debug(f"Getting usage stats: {self.usage}")
        return self.usage.copy()

    def reset_usage(self) -> None:
        """Reset the usage counters to zero."""
        log.debug("Resetting usage counters")
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }
        self.cost = 0.0

    def is_context_length_error(self, error: Exception) -> bool:
        """Detect whether an exception represents a context window error.

        Args:
            error: Exception to inspect.

        Returns:
            True if the error message suggests a context length violation.
        """
        msg = str(error).lower()
        is_context_error = "context length" in msg or "maximum context" in msg
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error
