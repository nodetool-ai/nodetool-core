"""
Request schemas for Provider Capabilities API.

This module defines request models for chat completions, image generation,
audio synthesis/transcription, and video generation endpoints. It maximizes
reuse of existing types from nodetool.metadata.types and nodetool.providers.types.
"""

from typing import Any

from pydantic import BaseModel, Field

from nodetool.api.schemas.common import APIMetadata

# === Existing Types (DO NOT DUPLICATE) ===
from nodetool.metadata.types import (
    ASRModel,
    FunctionDefinition,
    ImageModel,
    # Model types
    LanguageModel,
    # Message types
    Message,
    # Provider enum
    Provider,
    # Tool types
    ToolCall,
    TTSModel,
    VideoModel,
)

# Generation parameter types
from nodetool.providers.types import (
    ImageToImageParams,
    ImageToVideoParams,
    TextToImageParams,
    TextToVideoParams,
)


class ChatCompletionRequest(BaseModel):
    """Request for creating a chat completion.

    Uses existing Message and ToolCall types from nodetool.metadata.types.
    """

    provider: Provider = Field(description="AI provider to use for generation")
    model: str = Field(description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')")
    messages: list[Message] = Field(description="Conversation history as Message objects")
    tools: list[FunctionDefinition] | None = Field(
        default=None, description="Available tools/functions for the model to use"
    )
    max_tokens: int = Field(default=8192, ge=1, description="Maximum tokens to generate")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature (0.0 = deterministic)"
    )
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling probability")
    response_format: dict[str, Any] | None = Field(
        default=None, description="Response format specification (e.g., JSON schema)"
    )
    metadata: APIMetadata | None = Field(default=None, description="Optional request metadata")


class ImageGenerateRequest(BaseModel):
    """Request for generating an image from text.

    Wraps TextToImageParams from nodetool.providers.types.
    """

    provider: Provider = Field(description="AI provider to use for generation")
    params: TextToImageParams = Field(description="Text-to-image generation parameters")
    timeout_s: int | None = Field(default=None, ge=1, description="Request timeout in seconds")
    metadata: APIMetadata | None = Field(default=None, description="Optional request metadata")


class ImageTransformRequest(BaseModel):
    """Request for transforming an image based on a prompt.

    Wraps ImageToImageParams from nodetool.providers.types.
    """

    provider: Provider = Field(description="AI provider to use for transformation")
    image_uri: str = Field(description="URI of the source image (data:, file://, http(s)://, asset://)")
    params: ImageToImageParams = Field(description="Image-to-image transformation parameters")
    timeout_s: int | None = Field(default=None, ge=1, description="Request timeout in seconds")
    metadata: APIMetadata | None = Field(default=None, description="Optional request metadata")


class AudioSynthesizeRequest(BaseModel):
    """Request for synthesizing speech from text (TTS).

    Uses TTSModel from nodetool.metadata.types.
    """

    provider: Provider = Field(description="AI provider to use for synthesis")
    model: TTSModel = Field(description="TTS model to use")
    text: str = Field(description="Text to synthesize into speech")
    voice: str | None = Field(default=None, description="Voice identifier (provider-specific)")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier")
    timeout_s: int | None = Field(default=None, ge=1, description="Request timeout in seconds")
    metadata: APIMetadata | None = Field(default=None, description="Optional request metadata")


class AudioTranscribeRequest(BaseModel):
    """Request for transcribing audio to text (ASR).

    Uses ASRModel from nodetool.metadata.types.
    """

    provider: Provider = Field(description="AI provider to use for transcription")
    model: ASRModel = Field(description="ASR model to use")
    audio_uri: str = Field(description="URI of the audio file (data:, file://, http(s)://, asset://)")
    language: str | None = Field(default=None, description="ISO-639-1 language code hint")
    prompt: str | None = Field(default=None, description="Optional prompt to guide transcription")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    timeout_s: int | None = Field(default=None, ge=1, description="Request timeout in seconds")
    metadata: APIMetadata | None = Field(default=None, description="Optional request metadata")


class VideoGenerateRequest(BaseModel):
    """Request for generating video from text.

    Wraps TextToVideoParams from nodetool.providers.types.
    """

    provider: Provider = Field(description="AI provider to use for generation")
    params: TextToVideoParams = Field(description="Text-to-video generation parameters")
    timeout_s: int | None = Field(default=None, ge=1, description="Request timeout in seconds")
    metadata: APIMetadata | None = Field(default=None, description="Optional request metadata")


class VideoTransformRequest(BaseModel):
    """Request for generating video from an image.

    Wraps ImageToVideoParams from nodetool.providers.types.
    """

    provider: Provider = Field(description="AI provider to use for generation")
    image_uri: str = Field(description="URI of the source image (data:, file://, http(s)://, asset://)")
    params: ImageToVideoParams = Field(description="Image-to-video generation parameters")
    timeout_s: int | None = Field(default=None, ge=1, description="Request timeout in seconds")
    metadata: APIMetadata | None = Field(default=None, description="Optional request metadata")
