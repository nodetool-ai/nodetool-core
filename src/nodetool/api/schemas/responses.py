"""
Response schemas for Provider Capabilities API.

This module defines response models for chat completions, image generation,
audio transcription, and video generation endpoints. It maximizes reuse of
existing types from nodetool.metadata.types.
"""

import time
from typing import Any

from pydantic import BaseModel, Field

from nodetool.api.schemas.common import PaginationInfo, UsageInfo

# === Existing Types (DO NOT DUPLICATE) ===
from nodetool.metadata.types import (
    ASRModel,
    AudioRef,
    # Streaming types
    Chunk,
    ImageModel,
    # Asset types
    ImageRef,
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
    VideoRef,
)

# Provider system
from nodetool.providers.base import ProviderCapability


class ProviderInfo(BaseModel):
    """Information about a registered provider."""

    provider: Provider = Field(description="Provider enum value")
    name: str = Field(description="Human-readable provider name")
    capabilities: list[ProviderCapability] = Field(description="List of supported capabilities")
    is_available: bool = Field(default=True, description="Whether the provider is currently available")


class ProviderListResponse(BaseModel):
    """Response containing list of available providers."""

    providers: list[ProviderInfo] = Field(description="List of available providers")


class ModelListResponse(BaseModel):
    """Response containing list of available models."""

    models: list[LanguageModel | ImageModel | VideoModel | TTSModel | ASRModel] = Field(
        description="List of available models"
    )
    pagination: PaginationInfo | None = Field(default=None, description="Pagination information")


class ChatCompletionResponse(BaseModel):
    """Response from a chat completion request.

    Uses existing Message type from nodetool.metadata.types.
    """

    id: str = Field(description="Unique identifier for this completion")
    provider: Provider = Field(description="Provider that generated the completion")
    model: str = Field(description="Model that generated the completion")
    created: int = Field(default_factory=lambda: int(time.time()), description="Unix timestamp of creation")
    message: Message = Field(description="Generated message response")
    usage: UsageInfo = Field(default_factory=UsageInfo, description="Token usage information")
    cost: float = Field(default=0.0, ge=0.0, description="Cost in credits for this completion")


class ChatCompletionChunkEvent(BaseModel):
    """Streaming event for chat completion chunks.

    Uses existing Chunk type from nodetool.metadata.types.
    """

    event: str = Field(default="chunk", description="Event type identifier")
    chunk: Chunk = Field(description="Content chunk from the stream")


class ChatCompletionToolCallEvent(BaseModel):
    """Streaming event for tool calls during chat completion.

    Uses existing ToolCall type from nodetool.metadata.types.
    """

    event: str = Field(default="tool_call", description="Event type identifier")
    tool_call: ToolCall = Field(description="Tool call from the model")


class ImageGenerateResponse(BaseModel):
    """Response from an image generation request."""

    id: str = Field(description="Unique identifier for this generation")
    provider: Provider = Field(description="Provider that generated the image")
    model: str = Field(description="Model that generated the image")
    created: int = Field(default_factory=lambda: int(time.time()), description="Unix timestamp of creation")
    image: ImageRef = Field(description="Reference to the generated image")
    cost: float = Field(default=0.0, ge=0.0, description="Cost in credits for this generation")


class ImageTransformResponse(BaseModel):
    """Response from an image transformation request."""

    id: str = Field(description="Unique identifier for this transformation")
    provider: Provider = Field(description="Provider that transformed the image")
    model: str = Field(description="Model that transformed the image")
    created: int = Field(default_factory=lambda: int(time.time()), description="Unix timestamp of creation")
    image: ImageRef = Field(description="Reference to the transformed image")
    cost: float = Field(default=0.0, ge=0.0, description="Cost in credits for this transformation")


class AudioTranscribeResponse(BaseModel):
    """Response from an audio transcription request."""

    id: str = Field(description="Unique identifier for this transcription")
    provider: Provider = Field(description="Provider that transcribed the audio")
    model: str = Field(description="Model that transcribed the audio")
    created: int = Field(default_factory=lambda: int(time.time()), description="Unix timestamp of creation")
    text: str = Field(description="Transcribed text content")
    language: str | None = Field(default=None, description="Detected language code")
    duration: float | None = Field(default=None, description="Audio duration in seconds")
    cost: float = Field(default=0.0, ge=0.0, description="Cost in credits for this transcription")


class VideoGenerateResponse(BaseModel):
    """Response from a video generation request."""

    id: str = Field(description="Unique identifier for this generation")
    provider: Provider = Field(description="Provider that generated the video")
    model: str = Field(description="Model that generated the video")
    created: int = Field(default_factory=lambda: int(time.time()), description="Unix timestamp of creation")
    video: VideoRef = Field(description="Reference to the generated video")
    duration: float | None = Field(default=None, description="Video duration in seconds")
    cost: float = Field(default=0.0, ge=0.0, description="Cost in credits for this generation")


class VideoTransformResponse(BaseModel):
    """Response from a video transformation (image-to-video) request."""

    id: str = Field(description="Unique identifier for this transformation")
    provider: Provider = Field(description="Provider that transformed the image to video")
    model: str = Field(description="Model that generated the video")
    created: int = Field(default_factory=lambda: int(time.time()), description="Unix timestamp of creation")
    video: VideoRef = Field(description="Reference to the generated video")
    duration: float | None = Field(default=None, description="Video duration in seconds")
    cost: float = Field(default=0.0, ge=0.0, description="Cost in credits for this transformation")
