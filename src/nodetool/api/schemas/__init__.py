"""
API Schemas for Provider Capabilities API.

This module contains Pydantic models for request and response schemas,
maximizing reuse of existing types from nodetool.metadata.types and
nodetool.providers.types.
"""

from nodetool.api.schemas.common import (
    APIMetadata,
    ErrorDetail,
    ErrorResponse,
    PaginationInfo,
    UsageInfo,
)
from nodetool.api.schemas.requests import (
    AudioSynthesizeRequest,
    AudioTranscribeRequest,
    ChatCompletionRequest,
    ImageGenerateRequest,
    ImageTransformRequest,
    VideoGenerateRequest,
    VideoTransformRequest,
)
from nodetool.api.schemas.responses import (
    AudioTranscribeResponse,
    ChatCompletionChunkEvent,
    ChatCompletionResponse,
    ChatCompletionToolCallEvent,
    ImageGenerateResponse,
    ImageTransformResponse,
    ModelListResponse,
    ProviderInfo,
    ProviderListResponse,
    VideoGenerateResponse,
    VideoTransformResponse,
)

__all__ = [
    "APIMetadata",
    "AudioSynthesizeRequest",
    "AudioTranscribeRequest",
    "AudioTranscribeResponse",
    "ChatCompletionChunkEvent",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionToolCallEvent",
    "ErrorDetail",
    "ErrorResponse",
    "ImageGenerateRequest",
    "ImageGenerateResponse",
    "ImageTransformRequest",
    "ImageTransformResponse",
    "ModelListResponse",
    "PaginationInfo",
    "ProviderInfo",
    "ProviderListResponse",
    "UsageInfo",
    "VideoGenerateRequest",
    "VideoGenerateResponse",
    "VideoTransformRequest",
    "VideoTransformResponse",
]
