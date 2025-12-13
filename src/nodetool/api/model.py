#!/usr/bin/env python

import asyncio
import os
from typing import Any, Callable, ClassVar, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from huggingface_hub import HfApi, try_to_load_from_cache
from huggingface_hub.hf_api import RepoFile
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.chat.ollama_service import (
    delete_ollama_model as _delete_ollama_model,
)
from nodetool.chat.ollama_service import (
    get_ollama_model_info,
    get_ollama_models,
    get_ollama_models_unified,
    stream_ollama_model_pull,
)
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.hf_auth import get_hf_token
from nodetool.integrations.huggingface.hf_cache import filter_repo_paths, has_cached_files
from nodetool.integrations.huggingface.huggingface_file import (
    HFFileInfo,
    HFFileRequest,
    get_huggingface_file_infos_async,
)
from nodetool.integrations.huggingface.huggingface_models import (
    delete_cached_hf_model,
    get_models_by_hf_type,
    read_cached_hf_models,
    search_cached_hf_models,
)
from nodetool.metadata.types import (
    ASRModel,
    ImageModel,
    LanguageModel,
    LlamaModel,
    ModelFile,
    Provider,
    TTSModel,
    VideoModel,
    comfy_model_to_folder,
)
from nodetool.ml.models.language_models import get_all_language_models
from nodetool.providers import get_provider, import_providers
from nodetool.providers.base import _PROVIDER_REGISTRY, get_registered_provider
from nodetool.types.model import CachedRepo, RepoPath, UnifiedModel
from nodetool.workflows.recommended_models import (
    get_recommended_asr_models,
    get_recommended_image_models,
    get_recommended_image_to_image_models,
    get_recommended_image_to_video_models,
    get_recommended_language_embedding_models,
    get_recommended_language_models,
    get_recommended_language_text_generation_models,
    get_recommended_models,
    get_recommended_text_to_image_models,
    get_recommended_text_to_video_models,
    get_recommended_tts_models,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])


def dedupe_models(models: list[UnifiedModel]) -> list[UnifiedModel]:
    seen_ids = set()
    deduped_models = []
    for model in models:
        model_id = (model.repo_id, model.path or "")
        if model_id not in seen_ids:
            seen_ids.add(model_id)
            deduped_models.append(model)
    return deduped_models



async def _mark_downloaded_status(models: list[UnifiedModel]) -> list[UnifiedModel]:
    """
    Check local cache for each model and update its 'downloaded' status.
    Uses asyncio.gather for parallel checking.
    """
    async def check_model(model: UnifiedModel):
        if model.repo_id:
            # Run blocking file check in thread pool
            is_downloaded = await asyncio.to_thread(has_cached_files, model.repo_id)
            if is_downloaded:
                model.downloaded = True

    await asyncio.gather(*(check_model(m) for m in models))
    return models


# Exported functions for direct use (e.g., by MCP server)
async def get_all_models(_user: str) -> list[UnifiedModel]:
    """Get all available models of all types."""
    reco_models = [
        model
        for model_list in get_recommended_models().values()
        for model in model_list
    ]
    # gguf_models = await load_gguf_models_from_file()
    # mlx_models = await load_mlx_models_from_file()

    # Run in parallel, catching exceptions to handle them specifically
    results = await asyncio.gather(
        read_cached_hf_models(), get_ollama_models_unified(), return_exceptions=True
    )
    hf_models = results[0]
    ollama_models_unified = results[1]

    if isinstance(hf_models, Exception):
        raise hf_models

    if isinstance(ollama_models_unified, Exception):
        e = ollama_models_unified
        # Check if it looks like a connection error
        if "connect" in str(e).lower() or "refused" in str(e).lower():
             raise HTTPException(
                status_code=503,
                detail="ConnectionError: Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download"
             ) from e
        raise e

    # order matters: cached models should be first to have correct downloaded status
    all_models = hf_models + ollama_models_unified + reco_models

    # Ensure recommended models also get their status checked (in case they are downloaded but not in hf_models for some reason,
    # or if we want to be double sure)
    # However, hf_models comes from read_cached_hf_models which scans the cache.
    # dedupe_models keeps the first occurrence.
    # If a model is in hf_models, it is downloaded.
    # If it is ONLY in reco_models, it might be downloaded or not.
    # Let's check status for the final list to be safe, or at least for the reco part.
    # But dedupe_models logic:
    # 1. hf_models (downloaded=True)
    # 2. ollama (downloaded=True)
    # 3. reco (downloaded=False usually)
    # If a model is in 1 and 3, 1 wins -> downloaded=True.
    # If a model is only in 3, it means it's NOT in 1, so it's NOT downloaded.
    # So get_all_models logic seems correct without extra check, assuming read_cached_hf_models is accurate.

    return dedupe_models(all_models)


async def recommended_models(_user: str) -> list[UnifiedModel]:
    """Get recommended models."""
    models = [
        model
        for model_list in get_recommended_models().values()
        for model in model_list
    ]
    models = [model for model in models if model is not None]
    return await _mark_downloaded_status(models)


async def get_language_models(user: str = "1") -> list[LanguageModel]:
    """Get all available language models."""
    return await get_all_language_models(user)


class ProviderInfo(BaseModel):
    """Information about a provider including its key and capabilities."""

    provider: Provider
    capabilities: list[str]

    class Config:
        json_encoders: ClassVar[Dict[type, Callable[[Any], Any]]] = {
            Provider: lambda v: v.value,
        }


async def get_providers_info(user: str) -> list[ProviderInfo]:
    """
    Get information about all available providers including their keys and capabilities.

    This function iterates through the provider registry and creates ProviderInfo
    objects for each provider that can be initialized (has required secrets).
    """
    import_providers()

    # Get providers from the registry
    from nodetool.metadata.types import Provider as ProviderEnum
    from nodetool.security.secret_helper import get_secrets_batch

    provider_enums = list[ProviderEnum](_PROVIDER_REGISTRY.keys())

    # Collect all required secrets across all providers
    all_required_secrets = set()
    provider_secret_map = {}
    for provider_enum in provider_enums:
        provider_cls, kwargs = get_registered_provider(provider_enum)
        required_secrets = provider_cls.required_secrets()
        provider_secret_map[provider_enum] = (provider_cls, kwargs, required_secrets)
        all_required_secrets.update(required_secrets)

    # Batch fetch all secrets in one query
    if all_required_secrets:
        secrets_dict = await get_secrets_batch(list(all_required_secrets), user)
    else:
        secrets_dict = {}

    # Build provider info list
    providers_info = []
    for provider_enum, (
        provider_cls,
        kwargs,
        required_secrets,
    ) in provider_secret_map.items():
        # Collect this provider's secrets
        provider_secrets = {}
        for secret in required_secrets:
            secret_value = secrets_dict.get(secret)
            if secret_value:
                provider_secrets[secret] = secret_value

        # Skip provider if required secrets are missing
        if len(required_secrets) > 0 and len(provider_secrets) == 0:
            log.debug(
                f"Skipping provider {provider_enum.value}: missing required secrets {required_secrets}"
            )
            continue

        # Initialize provider to get capabilities
        try:
            # Some providers (like MLX) don't accept secrets parameter
            # Check if __init__ accepts secrets parameter
            import inspect

            init_signature = inspect.signature(provider_cls.__init__)
            init_params = list(init_signature.parameters.keys())

            if "secrets" in init_params:
                provider = provider_cls(secrets=provider_secrets, **kwargs)
            else:
                # Provider doesn't accept secrets, initialize without it
                provider = provider_cls(**kwargs)

            capabilities = provider.get_capabilities()
            capabilities_list = [cap.value for cap in capabilities]

            providers_info.append(
                ProviderInfo(
                    provider=provider_enum,
                    capabilities=capabilities_list,
                )
            )
        except Exception as e:
            log.warning(
                f"Failed to initialize provider {provider_enum.value}: {e}",
                exc_info=True,
            )
            continue

    return providers_info


@router.get("/providers")
async def get_providers_endpoint(
    user: str = Depends(current_user),
) -> list[ProviderInfo]:
    """
    Get all available providers with their keys and capabilities.
    """
    return await get_providers_info(user)


@router.get("/recommended")
async def recommended_models_endpoint(
    user: str = Depends(current_user),
) -> list[UnifiedModel]:
    return await recommended_models(user)


@router.get("/recommended/image")
async def recommended_image_models_endpoint(
    _user: str = Depends(current_user),
) -> list[UnifiedModel]:
    # Determine platform on the server; do not accept client override
    models = get_recommended_image_models()
    return await _mark_downloaded_status(models)


@router.get("/recommended/image/text-to-image")
async def recommended_text_to_image_models_endpoint(
    _user: str = Depends(current_user),
) -> list[UnifiedModel]:
    models = get_recommended_text_to_image_models()
    return await _mark_downloaded_status(models)


@router.get("/recommended/image/image-to-image")
async def recommended_image_to_image_models_endpoint(
    _user: str = Depends(current_user),
) -> list[UnifiedModel]:
    models = get_recommended_image_to_image_models()
    return await _mark_downloaded_status(models)


@router.get("/recommended/language")
async def recommended_language_models_endpoint(
    _user: str = Depends(current_user),
) -> list[UnifiedModel]:
    models = get_recommended_language_models()
    return await _mark_downloaded_status(models)


@router.get("/recommended/language/text-generation")
async def recommended_language_text_generation_models_endpoint(
    _user: str = Depends(current_user),
) -> list[UnifiedModel]:
    models = get_recommended_language_text_generation_models()
    return await _mark_downloaded_status(models)


@router.get("/recommended/language/embedding")
async def recommended_language_embedding_models_endpoint(
    user: str = Depends(current_user),
) -> list[UnifiedModel]:
    models = get_recommended_language_embedding_models()
    return await _mark_downloaded_status(models)


@router.get("/recommended/asr")
async def recommended_asr_models_endpoint(
    user: str = Depends(current_user),
) -> list[UnifiedModel]:
    models = get_recommended_asr_models()
    return await _mark_downloaded_status(models)


@router.get("/recommended/tts")
async def recommended_tts_models_endpoint(
    user: str = Depends(current_user),
) -> list[UnifiedModel]:
    models = get_recommended_tts_models()
    return await _mark_downloaded_status(models)


@router.get("/recommended/video/text-to-video")
async def recommended_text_to_video_models_endpoint(
    user: str = Depends(current_user),
) -> list[UnifiedModel]:
    models = get_recommended_text_to_video_models()
    return await _mark_downloaded_status(models)


@router.get("/recommended/video/image-to-video")
async def recommended_image_to_video_models_endpoint(
    user: str = Depends(current_user),
) -> list[UnifiedModel]:
    models = get_recommended_image_to_video_models()
    return await _mark_downloaded_status(models)


@router.get("/all")
async def get_all_models_endpoint(
    user: str = Depends(current_user),
) -> list[UnifiedModel]:
    return await get_all_models(user)


@router.get("/huggingface")
async def get_huggingface_models(
    user: str = Depends(current_user),
) -> list[UnifiedModel]:
    return await read_cached_hf_models()


@router.get("/huggingface/search")
async def search_huggingface_models_endpoint(
    repo_pattern: list[str] | None = Query(None),
    filename_pattern: list[str] | None = Query(None),
    user: str = Depends(current_user),
) -> list[UnifiedModel]:
    """
    Search cached Hugging Face repos/files using offline cache data.
    """
    if Environment.is_production():
        log.warning("Cannot search Hugging Face models in production")
        return []

    return await search_cached_hf_models(
        repo_patterns=repo_pattern,
        filename_patterns=filename_pattern,
    )


@router.get("/huggingface/type/{model_type}")
async def get_huggingface_models_by_type_endpoint(
    model_type: str,
    user: str = Depends(current_user),
) -> list[UnifiedModel]:
    """
    Return cached Hugging Face models matching an hf.* type using server-side heuristics.
    """
    return await get_models_by_hf_type(model_type)


@router.delete("/huggingface")
async def delete_huggingface_model(repo_id: str) -> bool:
    if Environment.is_production():
        log.warning("Cannot delete models in production")
        return False
    return await delete_cached_hf_model(repo_id)


@router.get("/ollama")
async def get_ollama_models_endpoint(
    user: str = Depends(current_user),
) -> list[LlamaModel]:
    try:
        return await get_ollama_models()
    except Exception as e:
        if "connect" in str(e).lower() or "refused" in str(e).lower():
             raise HTTPException(
                status_code=503,
                detail="ConnectionError: Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download"
             ) from e
        raise e


@router.delete("/ollama")
async def delete_ollama_model_endpoint(model_name: str) -> bool:
    if Environment.is_production():
        log.warning("Cannot delete ollama models in production")
        return False
    return await _delete_ollama_model(model_name)


async def get_language_models_by_provider(
    provider: Provider, user: str
) -> list[LanguageModel]:
    """Get language models for a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
        models = await provider_instance.get_available_language_models()
        log.debug(
            f"Successfully retrieved {len(models)} language models from provider {provider.value}"
        )
        return models
    except ValueError as e:
        log.warning(
            f"Provider {provider.value} not available: {e}. "
            "This may be expected if the provider package is not installed."
        )
        return []
    except Exception as e:
        log.error(
            f"Error getting language models from {provider.value}: {e}",
            exc_info=True,
        )
        return []


async def get_image_models_by_provider(
    provider: Provider, user: str
) -> list[ImageModel]:
    """Get image models for a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
        models = await provider_instance.get_available_image_models()
        log.debug(
            f"Successfully retrieved {len(models)} image models from provider {provider.value}"
        )
        return models
    except ValueError as e:
        log.warning(
            f"Provider {provider.value} not available: {e}. "
            "This may be expected if the provider package is not installed."
        )
        # For MLX provider, try to discover models from cache even if provider isn't installed
        if provider == Provider.MLX:
            try:
                from nodetool.integrations.huggingface.huggingface_models import (
                    get_mlx_image_models_from_hf_cache,
                )

                log.info(
                    "MLX provider not available, attempting to discover MLX image models (mflux) from HF cache directly"
                )
                models = await get_mlx_image_models_from_hf_cache()
                log.info(
                    f"Discovered {len(models)} MLX image models from HuggingFace cache"
                )
                return models
            except Exception as cache_error:
                log.debug(
                    f"Failed to discover MLX image models from cache: {cache_error}",
                    exc_info=True,
                )
        return []
    except Exception as e:
        log.error(
            f"Error getting image models from {provider.value}: {e}",
            exc_info=True,
        )
        # For MLX provider, try to discover models from cache even on error
        if provider == Provider.MLX:
            try:
                from nodetool.integrations.huggingface.huggingface_models import (
                    get_mlx_image_models_from_hf_cache,
                )

                log.info(
                    "Error occurred with MLX provider, attempting to discover MLX image models (mflux) from HF cache as fallback"
                )
                models = await get_mlx_image_models_from_hf_cache()
                log.info(
                    f"Discovered {len(models)} MLX image models from HuggingFace cache"
                )
                return models
            except Exception as cache_error:
                log.warning(
                    f"Failed to discover MLX image models from cache as fallback: {cache_error}",
                    exc_info=True,
                )
        return []


async def get_tts_models_by_provider(provider: Provider, user: str) -> list[TTSModel]:
    """Get TTS models for a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
        return await provider_instance.get_available_tts_models()
    except ValueError as e:
        log.warning(f"Provider {provider.value} not available: {e}")
        return []
    except Exception as e:
        log.error(f"Error getting TTS models from {provider.value}: {e}")
        return []


async def get_asr_models_by_provider(provider: Provider, user: str) -> list[ASRModel]:
    """Get ASR models for a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
        return await provider_instance.get_available_asr_models()
    except ValueError as e:
        log.warning(f"Provider {provider.value} not available: {e}")
        return []
    except Exception as e:
        log.error(f"Error getting ASR models from {provider.value}: {e}")
        return []


async def get_video_models_by_provider(
    provider: Provider, user: str
) -> list[VideoModel]:
    """Get video models for a specific provider."""
    try:
        provider_instance = await get_provider(provider, user)
        return await provider_instance.get_available_video_models()
    except ValueError as e:
        log.warning(f"Provider {provider.value} not available: {e}")
        return []
    except Exception as e:
        log.error(f"Error getting video models from {provider.value}: {e}")
        return []


@router.get("/llm/{provider}")
async def get_language_models_endpoint(
    provider: Provider,
    user: str = Depends(current_user),
) -> list[LanguageModel]:
    """
    Get all available language models from a specific provider.
    """
    return await get_language_models_by_provider(provider, user)


@router.get("/image/{provider}")
async def get_image_models_endpoint(
    provider: Provider,
    user: str = Depends(current_user),
) -> list[ImageModel]:
    """
    Get all available image generation models from a specific provider.
    """
    return await get_image_models_by_provider(provider, user)


@router.get("/tts/{provider}")
async def get_tts_models_endpoint(
    provider: Provider,
    user: str = Depends(current_user),
) -> list[TTSModel]:
    """
    Get all available text-to-speech models from a specific provider.
    """
    return await get_tts_models_by_provider(provider, user)


@router.get("/asr/{provider}")
async def get_asr_models_endpoint(
    provider: Provider,
    user: str = Depends(current_user),
) -> list[ASRModel]:
    """
    Get all available automatic speech recognition models from a specific provider.
    """
    return await get_asr_models_by_provider(provider, user)


@router.get("/video/{provider}")
async def get_video_models_endpoint(
    provider: Provider,
    user: str = Depends(current_user),
) -> list[VideoModel]:
    """
    Get all available video generation models from a specific provider.
    """
    return await get_video_models_by_provider(provider, user)


@router.get("/ollama_model_info")
async def get_ollama_model_info_endpoint(
    model_name: str, user: str = Depends(current_user)
) -> dict | None:
    return await get_ollama_model_info(model_name)


@router.post("/huggingface/try_cache_files")
async def try_cache_files(
    paths: list[RepoPath],
    user: str = Depends(current_user),
) -> list[RepoPath]:
    def check_path(path: RepoPath) -> bool:
        return try_to_load_from_cache(path.repo_id, path.path) is not None

    # Offload blocking cache checks to a thread to avoid blocking the loop
    results = await asyncio.gather(*(asyncio.to_thread(check_path, p) for p in paths))
    return [
        RepoPath(repo_id=p.repo_id, path=p.path, downloaded=downloaded)
        for p, downloaded in zip(paths, results, strict=False)
    ]


@router.post("/huggingface/try_cache_repos")
async def try_cache_repos(
    repos: list[str],
    user: str = Depends(current_user),
) -> list[CachedRepo]:
    def check_repo(repo_id: str) -> bool:
        return has_cached_files(repo_id)

    # Offload blocking cache checks to a thread
    results = await asyncio.gather(*(asyncio.to_thread(check_repo, r) for r in repos))
    return [
        CachedRepo(repo_id=repo_id, downloaded=downloaded)
        for repo_id, downloaded in zip(repos, results, strict=False)
    ]


class HFCacheCheckRequest(BaseModel):
    """
    Request payload to check Hugging Face cache presence for matched files.

    allow_pattern and ignore_pattern accept a single pattern or a list of patterns.
    Patterns are Unix shell-style wildcards (fnmatch).
    """

    repo_id: str
    allow_pattern: str | list[str] | None = None
    ignore_pattern: str | list[str] | None = None


class HFCacheCheckResponse(BaseModel):
    repo_id: str
    all_present: bool
    total_files: int
    missing: list[str]


@router.post("/huggingface/check_cache")
async def check_huggingface_cache(
    body: HFCacheCheckRequest, user: str = Depends(current_user)
) -> HFCacheCheckResponse:
    """
    Check if all files in a Hugging Face repo that match allow/ignore patterns
    exist in the local HF cache.

    Returns a concise status including whether all are present and which are missing.
    """

    # Use HF token if available (for gated models)
    token = await get_hf_token(user)
    api = HfApi(token=token) if token else HfApi()

    # List repo files in a worker thread to avoid blocking the event loop
    items = await asyncio.to_thread(
        api.list_repo_tree, body.repo_id, recursive=True
    )
    files = [f for f in items if isinstance(f, RepoFile)]
    filtered_files = filter_repo_paths(files, body.allow_pattern, body.ignore_pattern)

    # Check cache presence concurrently (offload to threads)
    def is_cached(file: RepoFile) -> bool:
        try:
            cache_path = try_to_load_from_cache(body.repo_id, file.path)
            return cache_path is not None and os.path.exists(cache_path)
        except Exception:
            return False

    results = await asyncio.gather(
        *(asyncio.to_thread(is_cached, f) for f in filtered_files)
    )

    missing = [
        f.path for f, ok in zip(filtered_files, results, strict=False) if not ok
    ]
    return HFCacheCheckResponse(
        repo_id=body.repo_id,
        all_present=len(missing) == 0,
        total_files=len(filtered_files),
        missing=missing,
    )


if not Environment.is_production():

    @router.post("/pull_ollama_model")
    async def pull_ollama_model(model_name: str, user: str = Depends(current_user)):
        # Preflight: attempt a lightweight call to detect if Ollama is reachable
        try:
            await get_ollama_models()
        except Exception as e:
            api_url = Environment.get("OLLAMA_API_URL")
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unavailable",
                    "message": (
                        f"Cannot connect to Ollama at {api_url!s}. "
                        "Make sure Ollama is running. Try: 'ollama serve' or set OLLAMA_API_URL."
                    ),
                    "error": str(e),
                },
            ) from e

        # If reachable, start the streaming response
        return StreamingResponse(
            stream_ollama_model_pull(model_name), media_type="application/json"
        )

    @router.post("/huggingface/file_info")
    async def get_huggingface_file_info(
        requests: list[HFFileRequest],
        user: str = Depends(current_user),
    ) -> list[HFFileInfo]:
        # Use async wrapper to avoid blocking the loop
        return await get_huggingface_file_infos_async(requests)

    # @router.get("/{model_type}")
    # async def index(
    #     model_type: str, user: str = Depends(current_user)
    # ) -> list[ModelFile]:
    #     folder = comfy_model_to_folder(model_type)
    #     try:
    #         # needs comfyui installed
    #         import folder_paths  # type: ignore

    #         files = folder_paths.get_filename_list(folder)
    #     except Exception as e:
    #         log.error(f"Error getting files for {folder}: {e}")
    #         files = []

    #     return [ModelFile(type=folder, name=file) for file in files]
