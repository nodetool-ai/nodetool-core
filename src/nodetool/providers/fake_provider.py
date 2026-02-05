"""
Fake provider implementation for easy testing.

This module provides a simplified testing provider that can return configurable
responses without requiring predefined message sequences. Ideal for unit tests.

Supports all provider capabilities:
- Text generation (generate_message, generate_messages)
- Image generation (text_to_image, image_to_image) - generates valid PNG using Pillow
- Audio generation (text_to_speech) - generates valid audio using pydub
- Speech recognition (automatic_speech_recognition) - returns configurable text
- Video generation (text_to_video, image_to_video) - generates valid MP4 using ffmpeg
- Embeddings (generate_embedding) - returns configurable embedding vectors

Example usage:

    # Simple text response
    provider = create_simple_fake_provider("Test response")

    # Streaming response
    provider = create_streaming_fake_provider("Hello world", chunk_size=5)

    # Tool calling
    tool_calls = [create_fake_tool_call("search", {"query": "test"})]
    provider = create_tool_calling_fake_provider(tool_calls)

    # Custom logic based on input
    def smart_response(messages, model):
        if "math" in str(messages):
            return "42"
        return "I don't know"

    provider = FakeProvider(custom_response_fn=smart_response)

    # Image generation
    params = TextToImageParams(model=image_model, prompt="test")
    image_bytes = await provider.text_to_image(params)

    # Audio generation
    async for audio_chunk in provider.text_to_speech("Hello", "fake-tts"):
        # audio_chunk is numpy int16 array at 24kHz

    # Use in tests
    with patch("module.get_provider", return_value=provider):
        # test code here
"""

import io
import subprocess
import tempfile
import uuid
from typing import Any, AsyncGenerator, Callable, Sequence

import numpy as np
from PIL import Image
from pydub import AudioSegment

from nodetool.metadata.types import (
    ASRModel,
    EmbeddingModel,
    ImageModel,
    LanguageModel,
    Message,
    MessageTextContent,
    Provider,
    ToolCall,
    TTSModel,
    VideoModel,
)
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.types import (
    ImageToImageParams,
    ImageToVideoParams,
    TextToImageParams,
    TextToVideoParams,
)
from nodetool.workflows.types import Chunk


def _generate_fake_image(
    width: int = 512,
    height: int = 512,
    color: tuple[int, int, int] = (100, 100, 100),
) -> bytes:
    """Generate a valid PNG image with the specified dimensions and color.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: RGB color tuple for the image fill

    Returns:
        PNG image bytes
    """
    img = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return bytes(buffer.read())


def _generate_fake_audio(
    duration_ms: int = 1000,
    sample_rate: int = 24000,
    channels: int = 1,
) -> AudioSegment:
    """Generate a silent audio segment with the specified duration.

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz
        channels: Number of audio channels

    Returns:
        AudioSegment with silence
    """
    return AudioSegment.silent(
        duration=duration_ms,
        frame_rate=sample_rate,
    ).set_channels(channels)


# Resolution mapping for video generation
_RESOLUTION_MAP: dict[str, tuple[int, int]] = {
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


def _parse_resolution(resolution: str | None, default_width: int = 512, default_height: int = 512) -> tuple[int, int]:
    """Parse resolution string to width/height tuple.

    Args:
        resolution: Resolution string (e.g., "720p", "1080p") or None
        default_width: Default width if resolution is not specified
        default_height: Default height if resolution is not specified

    Returns:
        Tuple of (width, height)
    """
    if resolution and resolution in _RESOLUTION_MAP:
        return _RESOLUTION_MAP[resolution]
    return (default_width, default_height)


def _generate_fake_video(
    width: int = 512,
    height: int = 512,
    duration_s: float = 1.0,
    fps: int = 24,
    color: tuple[int, int, int] = (100, 100, 100),
) -> bytes:
    """Generate a valid MP4 video with the specified dimensions.

    Uses ffmpeg to generate a solid color video.

    Args:
        width: Video width in pixels
        height: Video height in pixels
        duration_s: Duration in seconds
        fps: Frames per second
        color: RGB color tuple for the video fill

    Returns:
        MP4 video bytes
    """
    # Convert RGB to ffmpeg color format (hex)
    color_hex = f"0x{color[0]:02x}{color[1]:02x}{color[2]:02x}"

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f",
            "lavfi",
            "-i",
            f"color=c={color_hex}:s={width}x{height}:r={fps}:d={duration_s}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-t",
            str(duration_s),
            output_path,
        ]

        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )

        with open(output_path, "rb") as f:
            return bytes(f.read())

    finally:
        import os

        if os.path.exists(output_path):
            os.remove(output_path)


@register_provider(Provider.Fake)
class FakeProvider(BaseProvider):
    """
    A simplified fake provider for testing that implements all provider capabilities.

    Unlike MockProvider which requires predefined responses, FakeProvider allows
    configuring simple text responses, tool calls, or custom response functions
    on the fly. It also supports all media generation capabilities:

    - Text generation: Returns configurable text or tool calls
    - Image generation: Generates valid PNG images using Pillow
    - Audio generation: Generates valid audio using pydub
    - Speech recognition: Returns configurable transcription text
    - Video generation: Generates valid MP4 videos using ffmpeg
    - Embeddings: Returns configurable embedding vectors

    Perfect for unit tests that need predictable behavior without external API calls.
    """

    provider_name: str = "fake"

    def __init__(
        self,
        text_response: str = "Hello, this is a fake response!",
        tool_calls: list[ToolCall] | None = None,
        should_stream: bool = True,
        chunk_size: int = 10,
        custom_response_fn: (Callable[[Sequence[Message], str], str | list[ToolCall]] | None) = None,
        secrets: dict[str, str] | None = None,
        # Media generation settings
        image_color: tuple[int, int, int] = (100, 100, 100),
        audio_duration_ms: int = 1000,
        video_duration_s: float = 1.0,
        video_fps: int = 24,
        video_color: tuple[int, int, int] = (100, 100, 100),
        asr_response: str = "This is a fake transcription.",
        embedding_dimensions: int = 1536,
    ):
        """
        Initialize the FakeProvider.

        Args:
            text_response: Default text to return (if no custom_response_fn)
            tool_calls: Optional list of tool calls to return instead of text
            should_stream: Whether to stream response in chunks or return all at once
            chunk_size: Number of characters per chunk when streaming text
            custom_response_fn: Optional function that takes (messages, model) and returns
                               either a string or list[ToolCall]
            secrets: API secrets (not used by FakeProvider, but required by BaseProvider)
            image_color: RGB color tuple for generated images
            audio_duration_ms: Duration of generated audio in milliseconds
            video_duration_s: Duration of generated videos in seconds
            video_fps: Frames per second for generated videos
            video_color: RGB color tuple for generated videos
            asr_response: Text to return for automatic speech recognition
            embedding_dimensions: Dimensions for generated embedding vectors
        """
        super().__init__(secrets=secrets)
        self.text_response = text_response
        self.tool_calls = tool_calls or []
        self.should_stream = should_stream
        self.chunk_size = chunk_size
        self.custom_response_fn = custom_response_fn
        self.call_count = 0
        self.last_messages: Sequence[Message] | None = None
        self.last_model: str | None = None
        self.last_tools: Sequence[Any] = []
        self.last_kwargs: dict[str, Any] = {}

        # Media generation settings
        self.image_color = image_color
        self.audio_duration_ms = audio_duration_ms
        self.video_duration_s = video_duration_s
        self.video_fps = video_fps
        self.video_color = video_color
        self.asr_response = asr_response
        self.embedding_dimensions = embedding_dimensions

        # Track media generation calls
        self.image_generation_count = 0
        self.audio_generation_count = 0
        self.video_generation_count = 0
        self.asr_count = 0
        self.embedding_count = 0

    def get_response(self, messages: Sequence[Message], model: str) -> str | list[ToolCall]:
        """Get the response to return (text or tool calls)."""
        if self.custom_response_fn:
            return self.custom_response_fn(messages, model)
        elif self.tool_calls:
            return self.tool_calls
        else:
            return self.text_response

    def reset_call_count(self) -> None:
        """Reset the call count to 0."""
        self.call_count = 0

    def reset_all_counts(self) -> None:
        """Reset all call counts to 0."""
        self.call_count = 0
        self.image_generation_count = 0
        self.audio_generation_count = 0
        self.video_generation_count = 0
        self.asr_count = 0
        self.embedding_count = 0

    # ==================== Model Discovery ====================

    async def get_available_language_models(self) -> list[LanguageModel]:
        """Return fake language models for testing."""
        return [
            LanguageModel(
                id="fake-model-v1",
                name="Fake Model v1",
                provider=Provider.Fake,
            ),
            LanguageModel(
                id="fake-model-v2",
                name="Fake Model v2",
                provider=Provider.Fake,
            ),
            LanguageModel(
                id="fake-fast-model",
                name="Fake Fast Model",
                provider=Provider.Fake,
            ),
        ]

    async def get_available_image_models(self) -> list[ImageModel]:
        """Return fake image models for testing."""
        return [
            ImageModel(
                id="fake-image-model",
                name="Fake Image Model",
                provider=Provider.Fake,
                supported_tasks=["text_to_image", "image_to_image"],
            ),
            ImageModel(
                id="fake-image-hd",
                name="Fake HD Image Model",
                provider=Provider.Fake,
                supported_tasks=["text_to_image", "image_to_image"],
            ),
        ]

    async def get_available_tts_models(self) -> list[TTSModel]:
        """Return fake text-to-speech models for testing."""
        return [
            TTSModel(
                id="fake-tts",
                name="Fake TTS Model",
                provider=Provider.Fake,
                voices=["default", "female", "male"],
                selected_voice="default",
            ),
            TTSModel(
                id="fake-tts-hd",
                name="Fake HD TTS Model",
                provider=Provider.Fake,
                voices=["alloy", "echo", "nova"],
                selected_voice="alloy",
            ),
        ]

    async def get_available_asr_models(self) -> list[ASRModel]:
        """Return fake automatic speech recognition models for testing."""
        return [
            ASRModel(
                id="fake-asr",
                name="Fake ASR Model",
                provider=Provider.Fake,
            ),
            ASRModel(
                id="fake-whisper",
                name="Fake Whisper Model",
                provider=Provider.Fake,
            ),
        ]

    async def get_available_video_models(self) -> list[VideoModel]:
        """Return fake video models for testing."""
        return [
            VideoModel(
                id="fake-video-model",
                name="Fake Video Model",
                provider=Provider.Fake,
                supported_tasks=["text_to_video", "image_to_video"],
            ),
            VideoModel(
                id="fake-video-hd",
                name="Fake HD Video Model",
                provider=Provider.Fake,
                supported_tasks=["text_to_video", "image_to_video"],
            ),
        ]

    async def get_available_embedding_models(self) -> list[EmbeddingModel]:
        """Return fake embedding models for testing."""
        return [
            EmbeddingModel(
                id="fake-embedding",
                name="Fake Embedding Model",
                provider=Provider.Fake,
                dimensions=self.embedding_dimensions,
            ),
            EmbeddingModel(
                id="fake-embedding-small",
                name="Fake Small Embedding Model",
                provider=Provider.Fake,
                dimensions=512,
            ),
        ]

    # ==================== Text Generation ====================

    async def generate_message(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> Message:
        """
        Generate a single message response.

        Returns a Message containing either text content or tool calls
        based on the provider configuration.
        """
        self.call_count += 1
        self.last_messages = messages
        self.last_model = model
        self.last_tools = tools
        self.last_kwargs = kwargs

        response = self.get_response(messages, model)

        if isinstance(response, list):  # Tool calls
            return Message(
                role="assistant",
                content=[],
                tool_calls=response,
            )
        else:  # Text response
            return Message(role="assistant", content=[MessageTextContent(text=response)])

    async def generate_messages(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """
        Generate streaming message responses.

        Yields either Chunk objects (for text) or ToolCall objects,
        optionally breaking text into smaller chunks for streaming simulation.
        """
        self.call_count += 1
        self.last_messages = messages
        self.last_model = model
        self.last_tools = tools
        self.last_kwargs = kwargs

        response = self.get_response(messages, model)

        if isinstance(response, list):  # Tool calls
            for tool_call in response:
                yield tool_call
        else:  # Text response
            if self.should_stream and len(response) > self.chunk_size:
                # Break into chunks
                for i in range(0, len(response), self.chunk_size):
                    chunk_text = response[i : i + self.chunk_size]
                    is_done = i + self.chunk_size >= len(response)
                    yield Chunk(content=chunk_text, done=is_done, content_type="text")
            else:
                # Return as single chunk
                yield Chunk(content=response, done=True, content_type="text")

    # ==================== Image Generation ====================

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate a fake image from a text prompt.

        Creates a valid PNG image with the specified dimensions filled with
        a configurable solid color.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds (ignored)
            context: Optional processing context (ignored)
            node_id: Optional node ID for progress tracking (ignored)

        Returns:
            Raw PNG image bytes
        """
        self.image_generation_count += 1
        width = params.width or 512
        height = params.height or 512
        return _generate_fake_image(width, height, self.image_color)

    async def image_to_image(
        self,
        image: bytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Transform an image (returns a fake generated image).

        Creates a valid PNG image with the specified dimensions.
        The input image is ignored and a new image is generated.

        Args:
            image: Input image as bytes (ignored for fake generation)
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds (ignored)
            context: Optional processing context (ignored)
            node_id: Optional node ID for progress tracking (ignored)

        Returns:
            Raw PNG image bytes
        """
        self.image_generation_count += 1
        # Use target dimensions if specified, otherwise use defaults
        width = params.target_width or 512
        height = params.target_height or 512
        return _generate_fake_image(width, height, self.image_color)

    # ==================== Audio Generation ====================

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
        """Generate speech audio from text as a streaming generator.

        Creates silent audio chunks at 24kHz mono as int16 arrays.

        Args:
            text: Input text to convert to speech (used to calculate duration)
            model: Model identifier for TTS (ignored)
            voice: Voice identifier/name (ignored)
            speed: Speech speed multiplier (ignored)
            timeout_s: Optional timeout in seconds (ignored)
            context: Optional processing context (ignored)
            **kwargs: Additional parameters (ignored)

        Yields:
            numpy.ndarray: Int16 audio chunks at 24kHz mono
        """
        self.audio_generation_count += 1

        # Generate silent audio based on configured duration
        audio = _generate_fake_audio(
            duration_ms=self.audio_duration_ms,
            sample_rate=24000,
            channels=1,
        )

        # Convert to numpy int16 array and yield in chunks
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)

        # Yield in chunks (simulate streaming)
        chunk_size = 4096
        for i in range(0, len(samples), chunk_size):
            yield samples[i : i + chunk_size]

    # ==================== Speech Recognition ====================

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
        """Transcribe audio to text (returns configured fake transcription).

        Args:
            audio: Input audio as bytes (ignored for fake generation)
            model: Model identifier for ASR (ignored)
            language: Optional language code (ignored)
            prompt: Optional guiding prompt (ignored)
            temperature: Sampling temperature (ignored)
            timeout_s: Optional timeout in seconds (ignored)
            context: Optional processing context (ignored)
            **kwargs: Additional parameters (ignored)

        Returns:
            str: Configured fake transcription text
        """
        self.asr_count += 1
        return self.asr_response

    # ==================== Video Generation ====================

    async def text_to_video(
        self,
        params: TextToVideoParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
    ) -> bytes:
        """Generate a fake video from a text prompt.

        Creates a valid MP4 video with the specified dimensions filled with
        a configurable solid color using ffmpeg.

        Args:
            params: Text-to-video generation parameters
            timeout_s: Optional timeout in seconds (ignored)
            context: Optional processing context (ignored)
            node_id: Optional node ID for progress tracking (ignored)

        Returns:
            Raw MP4 video bytes
        """
        self.video_generation_count += 1
        width, height = _parse_resolution(params.resolution)

        return _generate_fake_video(
            width=width,
            height=height,
            duration_s=self.video_duration_s,
            fps=self.video_fps,
            color=self.video_color,
        )

    async def image_to_video(
        self,
        image: bytes,
        params: ImageToVideoParams,
        timeout_s: int | None = None,
        context: Any = None,
        node_id: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate a fake video from an input image.

        Creates a valid MP4 video. The input image is ignored and
        a solid color video is generated.

        Args:
            image: Input image as bytes (ignored for fake generation)
            params: Image-to-video generation parameters
            timeout_s: Optional timeout in seconds (ignored)
            context: Optional processing context (ignored)
            node_id: Optional node ID for progress tracking (ignored)
            **kwargs: Additional parameters (ignored)

        Returns:
            Raw MP4 video bytes
        """
        self.video_generation_count += 1
        width, height = _parse_resolution(params.resolution)

        return _generate_fake_video(
            width=width,
            height=height,
            duration_s=self.video_duration_s,
            fps=self.video_fps,
            color=self.video_color,
        )

    # ==================== Embeddings ====================

    async def generate_embedding(
        self,
        text: str | list[str],
        model: str,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate fake embedding vectors.

        Returns deterministic embedding vectors based on the input text hash.
        Each text gets a unique but reproducible embedding.

        Args:
            text: Single text string or list of text strings to embed
            model: Model identifier (ignored)
            **kwargs: Additional parameters (ignored)

        Returns:
            List of embedding vectors, one for each input text.
            Each embedding has `embedding_dimensions` floats.
        """
        self.embedding_count += 1

        # Normalize input to list
        texts = [text] if isinstance(text, str) else text

        embeddings = []
        for t in texts:
            # Generate a deterministic embedding based on text hash
            # This ensures the same text always gets the same embedding
            text_hash = hash(t)
            np.random.seed(abs(text_hash) % (2**31 - 1))
            embedding = np.random.randn(self.embedding_dimensions).tolist()
            embeddings.append(embedding)

        return embeddings


def create_fake_tool_call(
    name: str,
    args: dict[str, Any] | None = None,
    call_id: str | None = None,
) -> ToolCall:
    """
    Convenience function to create a ToolCall for testing.

    Args:
        name: Name of the tool
        args: Arguments dictionary (defaults to empty dict)
        call_id: Tool call ID (generates random UUID if not provided)

    Returns:
        ToolCall object ready for use with FakeProvider
    """
    return ToolCall(
        id=call_id or str(uuid.uuid4()),
        name=name,
        args=args or {},
    )


def create_simple_fake_provider(response_text: str = "Test response") -> FakeProvider:
    """
    Create a FakeProvider with a simple text response.

    Args:
        response_text: The text to return

    Returns:
        Configured FakeProvider
    """
    return FakeProvider(text_response=response_text, should_stream=False)


def create_streaming_fake_provider(
    response_text: str = "This is a streaming test response",
    chunk_size: int = 5,
) -> FakeProvider:
    """
    Create a FakeProvider that streams responses in chunks.

    Args:
        response_text: The text to stream
        chunk_size: Number of characters per chunk

    Returns:
        Configured FakeProvider that streams responses
    """
    return FakeProvider(
        text_response=response_text,
        should_stream=True,
        chunk_size=chunk_size,
    )


def create_tool_calling_fake_provider(tool_calls: list[ToolCall]) -> FakeProvider:
    """
    Create a FakeProvider that returns tool calls.

    Args:
        tool_calls: List of ToolCall objects to return

    Returns:
        Configured FakeProvider that returns tool calls
    """
    return FakeProvider(tool_calls=tool_calls)
