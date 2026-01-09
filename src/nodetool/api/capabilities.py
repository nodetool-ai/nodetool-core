"""
Provider Capabilities API endpoints.

This module exposes all provider capabilities (text generation, image generation,
speech synthesis, video generation, etc.) as RESTful endpoints. All request and
response models are defined as Pydantic models, maximizing reuse of existing types
from nodetool.metadata.types and nodetool.providers.types.
"""

import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from nodetool.api.schemas.common import APIMetadata, ErrorResponse, PaginationInfo, UsageInfo
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
from nodetool.api.utils import current_user
from nodetool.io.uri_utils import fetch_uri_bytes_and_mime
from nodetool.metadata.types import (
    ASRModel,
    AudioRef,
    Chunk,
    ImageModel,
    ImageRef,
    LanguageModel,
    Message,
    Provider,
    ToolCall,
    TTSModel,
    VideoModel,
    VideoRef,
)
from nodetool.providers import get_provider, import_providers
from nodetool.providers.base import ProviderCapability
from nodetool.providers.types import ImageToImageParams, ImageToVideoParams, TextToImageParams, TextToVideoParams

router = APIRouter(prefix="/api/capabilities", tags=["capabilities"])


def _generate_id() -> str:
    return f"{uuid.uuid4().hex[:12]}"


async def _get_provider_info_list(user: str) -> list[ProviderInfo]:
    """Get information about all available providers including their capabilities."""
    import_providers()

    from nodetool.providers.base import _PROVIDER_REGISTRY, get_registered_provider
    from nodetool.security.secret_helper import get_secrets_batch

    provider_enums = list[Provider](_PROVIDER_REGISTRY.keys())
    all_required_secrets = set()
    provider_secret_map = {}

    for provider_enum in provider_enums:
        provider_cls, kwargs = get_registered_provider(provider_enum)
        required_secrets = provider_cls.required_secrets()
        provider_secret_map[provider_enum] = (provider_cls, kwargs, required_secrets)
        all_required_secrets.update(required_secrets)

    if all_required_secrets:
        secrets_dict = await get_secrets_batch(list(all_required_secrets), user)
    else:
        secrets_dict = {}

    providers_info = []
    for provider_enum, (provider_cls, kwargs, required_secrets) in provider_secret_map.items():
        provider_secrets = {}
        for secret in required_secrets:
            secret_value = secrets_dict.get(secret)
            if secret_value:
                provider_secrets[secret] = secret_value

        if len(required_secrets) > 0 and len(provider_secrets) == 0:
            continue

        try:
            import inspect

            init_signature = inspect.signature(provider_cls.__init__)
            init_params = list(init_signature.parameters.keys())

            if "secrets" in init_params:
                provider = provider_cls(secrets=provider_secrets, **kwargs)
            else:
                provider = provider_cls(**kwargs)

            capabilities = provider.get_capabilities()

            providers_info.append(
                ProviderInfo(
                    provider=provider_enum,
                    name=provider_enum.value.title(),
                    capabilities=list(capabilities),
                    is_available=True,
                )
            )
        except Exception:
            continue

    return providers_info


@router.get(
    "/providers",
    response_model=ProviderListResponse,
    summary="List all providers",
    description="Get a list of all available AI providers with their capabilities.",
)
async def list_providers_endpoint(user: str = Depends(current_user)) -> ProviderListResponse:
    """List all available providers with their capabilities."""
    providers = await _get_provider_info_list(user)
    return ProviderListResponse(providers=providers)


@router.get(
    "/providers/{provider}/models",
    response_model=ModelListResponse,
    summary="List models for a provider",
    description="Get all available models for a specific provider.",
)
async def list_models_endpoint(
    provider: Provider,
    page: int = 1,
    per_page: int = 20,
    user: str = Depends(current_user),
) -> ModelListResponse:
    """List all models available from a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    models = await provider_instance.get_available_models()

    total = len(models)
    total_pages = (total + per_page - 1) // per_page if total > 0 else 0
    start = (page - 1) * per_page
    end = start + per_page
    paginated_models = models[start:end]

    pagination = PaginationInfo(
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )

    return ModelListResponse(models=paginated_models, pagination=pagination)


@router.get(
    "/providers/{provider}/models/language",
    response_model=list[LanguageModel],
    summary="List language models",
    description="Get all available language models from a specific provider.",
)
async def list_language_models_endpoint(
    provider: Provider,
    user: str = Depends(current_user),
) -> list[LanguageModel]:
    """List language models for a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return await provider_instance.get_available_language_models()


@router.get(
    "/providers/{provider}/models/image",
    response_model=list[ImageModel],
    summary="List image models",
    description="Get all available image models from a specific provider.",
)
async def list_image_models_endpoint(
    provider: Provider,
    user: str = Depends(current_user),
) -> list[ImageModel]:
    """List image models for a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return await provider_instance.get_available_image_models()


@router.get(
    "/providers/{provider}/models/tts",
    response_model=list[TTSModel],
    summary="List TTS models",
    description="Get all available text-to-speech models from a specific provider.",
)
async def list_tts_models_endpoint(
    provider: Provider,
    user: str = Depends(current_user),
) -> list[TTSModel]:
    """List TTS models for a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return await provider_instance.get_available_tts_models()


@router.get(
    "/providers/{provider}/models/asr",
    response_model=list[ASRModel],
    summary="List ASR models",
    description="Get all available automatic speech recognition models from a specific provider.",
)
async def list_asr_models_endpoint(
    provider: Provider,
    user: str = Depends(current_user),
) -> list[ASRModel]:
    """List ASR models for a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return await provider_instance.get_available_asr_models()


@router.get(
    "/providers/{provider}/models/video",
    response_model=list[VideoModel],
    summary="List video models",
    description="Get all available video models from a specific provider.",
)
async def list_video_models_endpoint(
    provider: Provider,
    user: str = Depends(current_user),
) -> list[VideoModel]:
    """List video models for a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return await provider_instance.get_available_video_models()


@router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    summary="Create chat completion",
    description="Generate a single chat completion using the specified provider and model.",
)
async def create_completion_endpoint(
    request: ChatCompletionRequest,
    user: str = Depends(current_user),
) -> ChatCompletionResponse:
    """Create a chat completion."""
    try:
        provider_instance = await get_provider(request.provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    capabilities = provider_instance.get_capabilities()
    if ProviderCapability.GENERATE_MESSAGE not in capabilities:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {request.provider.value} does not support chat completions",
        )

    try:
        message = await provider_instance.generate_message(
            messages=request.messages,
            model=request.model,
            tools=request.tools,
            max_tokens=request.max_tokens,
            response_format=request.response_format,
        )

        usage = UsageInfo(
            prompt_tokens=getattr(message, "usage_prompt_tokens", 0),
            completion_tokens=getattr(message, "usage_completion_tokens", 0),
            total_tokens=getattr(message, "usage_total_tokens", 0),
        )

        return ChatCompletionResponse(
            id=_generate_id(),
            provider=request.provider,
            model=request.model,
            message=message,
            usage=usage,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/completions/stream",
    summary="Create streaming chat completion",
    description="Generate a streaming chat completion using the specified provider and model.",
)
async def create_streaming_completion_endpoint(
    request: ChatCompletionRequest,
    user: str = Depends(current_user),
) -> StreamingResponse:
    """Create a streaming chat completion."""
    try:
        provider_instance = await get_provider(request.provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    capabilities = provider_instance.get_capabilities()
    if ProviderCapability.GENERATE_MESSAGES not in capabilities:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {request.provider.value} does not support streaming completions",
        )

    async def generate_stream() -> AsyncIterator[str]:
        try:
            async for chunk in provider_instance.generate_messages(
                messages=request.messages,
                model=request.model,
                tools=request.tools,
                max_tokens=request.max_tokens,
                response_format=request.response_format,
            ):
                if isinstance(chunk, Chunk):
                    event = ChatCompletionChunkEvent(chunk=chunk)
                    yield f"data: {event.model_dump_json()}\n\n"
                elif isinstance(chunk, ToolCall):
                    event = ChatCompletionToolCallEvent(tool_call=chunk)
                    yield f"data: {event.model_dump_json()}\n\n"
                elif hasattr(chunk, "model_dump"):
                    yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {ErrorResponse(error='stream_error', message=str(e)).model_dump_json()}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@router.post(
    "/images/generate",
    response_model=ImageGenerateResponse,
    summary="Generate image from text",
    description="Generate an image from a text prompt using the specified provider.",
)
async def generate_image_endpoint(
    request: ImageGenerateRequest,
    user: str = Depends(current_user),
) -> ImageGenerateResponse:
    """Generate an image from text."""
    try:
        provider_instance = await get_provider(request.provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    capabilities = provider_instance.get_capabilities()
    if ProviderCapability.TEXT_TO_IMAGE not in capabilities:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {request.provider.value} does not support text-to-image generation",
        )

    try:
        image_bytes = await provider_instance.text_to_image(
            params=request.params,
            timeout_s=request.timeout_s,
        )

        import base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = "image/png"
        if "jpeg" in request.params.model.id.lower() or "jpg" in request.params.model.id.lower():
            mime_type = "image/jpeg"

        image_ref = ImageRef(uri=f"data:{mime_type};base64,{image_b64}")

        return ImageGenerateResponse(
            id=_generate_id(),
            provider=request.provider,
            model=request.params.model.id,
            image=image_ref,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/images/transform",
    response_model=ImageTransformResponse,
    summary="Transform image",
    description="Transform an image based on a text prompt using the specified provider.",
)
async def transform_image_endpoint(
    request: ImageTransformRequest,
    user: str = Depends(current_user),
) -> ImageTransformResponse:
    """Transform an image."""
    try:
        provider_instance = await get_provider(request.provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    capabilities = provider_instance.get_capabilities()
    if ProviderCapability.IMAGE_TO_IMAGE not in capabilities:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {request.provider.value} does not support image-to-image transformation",
        )

    try:
        _, image_bytes = await fetch_uri_bytes_and_mime(request.image_uri)

        result_bytes = await provider_instance.image_to_image(
            image=image_bytes,
            params=request.params,
            timeout_s=request.timeout_s,
        )

        import base64
        image_b64 = base64.b64encode(result_bytes).decode("utf-8")

        image_ref = ImageRef(uri=f"data:image/png;base64,{image_b64}")

        return ImageTransformResponse(
            id=_generate_id(),
            provider=request.provider,
            model=request.params.model.id,
            image=image_ref,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/audio/synthesize",
    summary="Synthesize speech from text",
    description="Generate speech audio from text using the specified TTS provider.",
)
async def synthesize_audio_endpoint(
    request: AudioSynthesizeRequest,
    user: str = Depends(current_user),
) -> StreamingResponse:
    """Synthesize speech from text."""
    try:
        provider_instance = await get_provider(request.provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    capabilities = provider_instance.get_capabilities()
    if ProviderCapability.TEXT_TO_SPEECH not in capabilities:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {request.provider.value} does not support text-to-speech",
        )

    async def generate_audio() -> AsyncIterator[bytes]:
        try:
            async for chunk in provider_instance.text_to_speech(
                text=request.text,
                model=request.model.id,
                voice=request.voice,
                speed=request.speed,
                timeout_s=request.timeout_s,
            ):
                yield chunk.tobytes()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(generate_audio(), media_type="audio/wav")


@router.post(
    "/audio/transcribe",
    response_model=AudioTranscribeResponse,
    summary="Transcribe audio to text",
    description="Transcribe audio to text using the specified ASR provider.",
)
async def transcribe_audio_endpoint(
    request: AudioTranscribeRequest,
    user: str = Depends(current_user),
) -> AudioTranscribeResponse:
    """Transcribe audio to text."""
    try:
        provider_instance = await get_provider(request.provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    capabilities = provider_instance.get_capabilities()
    if ProviderCapability.AUTOMATIC_SPEECH_RECOGNITION not in capabilities:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {request.provider.value} does not support automatic speech recognition",
        )

    try:
        _, audio_bytes = await fetch_uri_bytes_and_mime(request.audio_uri)

        text = await provider_instance.automatic_speech_recognition(
            audio=audio_bytes,
            model=request.model.id,
            language=request.language,
            prompt=request.prompt,
            temperature=request.temperature,
            timeout_s=request.timeout_s,
        )

        return AudioTranscribeResponse(
            id=_generate_id(),
            provider=request.provider,
            model=request.model.id,
            text=text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/video/generate",
    response_model=VideoGenerateResponse,
    summary="Generate video from text",
    description="Generate a video from a text prompt using the specified provider.",
)
async def generate_video_endpoint(
    request: VideoGenerateRequest,
    user: str = Depends(current_user),
) -> VideoGenerateResponse:
    """Generate a video from text."""
    try:
        provider_instance = await get_provider(request.provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    capabilities = provider_instance.get_capabilities()
    if ProviderCapability.TEXT_TO_VIDEO not in capabilities:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {request.provider.value} does not support text-to-video generation",
        )

    try:
        video_bytes = await provider_instance.text_to_video(
            params=request.params,
            timeout_s=request.timeout_s,
        )

        import base64
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")

        video_ref = VideoRef(uri=f"data:video/mp4;base64,{video_b64}")

        return VideoGenerateResponse(
            id=_generate_id(),
            provider=request.provider,
            model=request.params.model.id,
            video=video_ref,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/video/transform",
    response_model=VideoTransformResponse,
    summary="Generate video from image",
    description="Generate a video from an image using the specified provider.",
)
async def transform_video_endpoint(
    request: VideoTransformRequest,
    user: str = Depends(current_user),
) -> VideoTransformResponse:
    """Generate video from image."""
    try:
        provider_instance = await get_provider(request.provider, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    capabilities = provider_instance.get_capabilities()
    if ProviderCapability.IMAGE_TO_VIDEO not in capabilities:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {request.provider.value} does not support image-to-video generation",
        )

    try:
        _, image_bytes = await fetch_uri_bytes_and_mime(request.image_uri)

        video_bytes = await provider_instance.image_to_video(
            image=image_bytes,
            params=request.params,
            timeout_s=request.timeout_s,
        )

        import base64
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")

        video_ref = VideoRef(uri=f"data:video/mp4;base64,{video_b64}")

        return VideoTransformResponse(
            id=_generate_id(),
            provider=request.provider,
            model=request.params.model.id,
            video=video_ref,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
