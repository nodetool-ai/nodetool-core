#!/usr/bin/env python

from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from nodetool.integrations.huggingface.huggingface_cache import has_cached_files
from nodetool.integrations.huggingface.huggingface_file import (
    HFFileInfo,
    HFFileRequest,
    get_huggingface_file_infos_async,
)
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.ml.models.language_models import get_all_language_models
from nodetool.ml.models.image_models import get_all_image_models
from nodetool.ml.models.tts_models import get_all_tts_models
from nodetool.ml.models.asr_models import get_all_asr_models
from nodetool.ml.models.video_models import get_all_video_models
from nodetool.metadata.types import (
    LanguageModel,
    ImageModel,
    TTSModel,
    ASRModel,
    VideoModel,
    ModelFile,
    LlamaModel,
    comfy_model_to_folder,
    Provider,
)
from nodetool.providers import get_provider
from nodetool.providers.base import ProviderCapability, _PROVIDER_REGISTRY, get_registered_provider
from nodetool.providers import import_providers
from pydantic import BaseModel
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE
from nodetool.api.utils import current_user
from fastapi import APIRouter, Depends, Query
from nodetool.integrations.huggingface.huggingface_models import (
    delete_cached_hf_model,
    load_gguf_models_from_file,
    load_mlx_models_from_file,
    read_cached_hf_models,
)
from nodetool.types.model import CachedRepo, RepoPath, UnifiedModel
from nodetool.workflows.recommended_models import get_recommended_models
from nodetool.chat.ollama_service import (
    get_ollama_models,
    get_ollama_models_unified,
    get_ollama_model_info,
    stream_ollama_model_pull,
    delete_ollama_model as _delete_ollama_model,
)
from pathlib import Path
import asyncio
from nodetool.io.file_explorer import (
    get_ollama_models_dir as common_get_ollama_models_dir,
    open_in_explorer as common_open_in_explorer,
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


# Exported functions for direct use (e.g., by MCP server)
async def get_all_models(user: str) -> list[UnifiedModel]:
    """Get all available models of all types."""
    reco_models = [
        model
        for model_list in get_recommended_models().values()
        for model in model_list
    ]
    gguf_models = await load_gguf_models_from_file()
    mlx_models = await load_mlx_models_from_file()
    hf_models = await read_cached_hf_models()
    ollama_models_unified = await get_ollama_models_unified()

    # order matters: cached models should be first to have correct downloaded status
    all_models = (
        hf_models + ollama_models_unified + reco_models + gguf_models + mlx_models
    )
    return dedupe_models(all_models)


async def recommended_models(user: str) -> list[UnifiedModel]:
    """Get recommended models."""
    models = [
        model
        for model_list in get_recommended_models().values()
        for model in model_list
    ]
    gguf_models = await load_gguf_models_from_file()
    mlx_models = await load_mlx_models_from_file()
    models.extend(gguf_models)
    models.extend(mlx_models)
    return [model for model in models if model is not None]


async def get_language_models(user: str = "1") -> list[LanguageModel]:
    """Get all available language models."""
    return await get_all_language_models(user)


class ProviderInfo(BaseModel):
    """Information about a provider including its key and capabilities."""
    provider: Provider
    capabilities: list[str]

    class Config:
        json_encoders = {
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
    for provider_enum, (provider_cls, kwargs, required_secrets) in provider_secret_map.items():
        # Collect this provider's secrets
        provider_secrets = {}
        for secret in required_secrets:
            secret_value = secrets_dict.get(secret)
            if secret_value:
                provider_secrets[secret] = secret_value
        
        # Skip provider if required secrets are missing
        if len(required_secrets) > 0 and len(provider_secrets) == 0:
            log.debug(f"Skipping provider {provider_enum.value}: missing required secrets {required_secrets}")
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
            log.warning(f"Failed to initialize provider {provider_enum.value}: {e}", exc_info=True)
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


@router.delete("/huggingface")
async def delete_huggingface_model(repo_id: str) -> bool:
    if Environment.is_production():
        log.warning("Cannot delete models in production")
        return False
    return delete_cached_hf_model(repo_id)


@router.get("/ollama")
async def get_ollama_models_endpoint(
    user: str = Depends(current_user),
) -> list[LlamaModel]:
    return await get_ollama_models()


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
        # For MLX provider, try to discover models from cache even if provider isn't installed
        if provider == Provider.MLX:
            try:
                from nodetool.integrations.huggingface.huggingface_models import (
                    get_mlx_language_models_from_hf_cache,
                )
                log.info(
                    "MLX provider not available, attempting to discover MLX models from HF cache directly"
                )
                models = await get_mlx_language_models_from_hf_cache()
                log.info(f"Discovered {len(models)} MLX models from HuggingFace cache")
                return models
            except Exception as cache_error:
                log.debug(
                    f"Failed to discover MLX models from cache: {cache_error}",
                    exc_info=True,
                )
        return []
    except Exception as e:
        log.error(
            f"Error getting language models from {provider.value}: {e}",
            exc_info=True,
        )
        # For MLX provider, try to discover models from cache even on error
        if provider == Provider.MLX:
            try:
                from nodetool.integrations.huggingface.huggingface_models import (
                    get_mlx_language_models_from_hf_cache,
                )
                log.info(
                    "Error occurred with MLX provider, attempting to discover MLX models from HF cache as fallback"
                )
                models = await get_mlx_language_models_from_hf_cache()
                log.info(f"Discovered {len(models)} MLX models from HuggingFace cache")
                return models
            except Exception as cache_error:
                log.warning(
                    f"Failed to discover MLX models from cache as fallback: {cache_error}",
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
                log.info(f"Discovered {len(models)} MLX image models from HuggingFace cache")
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
                log.info(f"Discovered {len(models)} MLX image models from HuggingFace cache")
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
        for p, downloaded in zip(paths, results)
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
        for repo_id, downloaded in zip(repos, results)
    ]


if not Environment.is_production():

    @router.get("/ollama_base_path")
    async def get_ollama_base_path_endpoint(user: str = Depends(current_user)) -> dict:
        """Retrieves the Ollama models directory path.

        The path is determined by the `_get_ollama_models_dir` helper function, which
        includes OS-specific lookup and caching.

        Args:
            user (str): The current user, injected by FastAPI dependency.

        Returns:
            dict: A dictionary containing the path if found (e.g., {"path": "/path/to/ollama/models"}),
                  or an error message if not found (e.g., {"status": "error", "message": "..."}).
        """
        ollama_path = common_get_ollama_models_dir()
        if ollama_path:
            return {"path": str(ollama_path)}
        else:
            # _get_ollama_models_dir already logs the specific error.
            return {
                "status": "error",
                "message": "Could not determine Ollama models path. Please check server logs for details.",
            }

    @router.get("/huggingface_base_path")
    async def get_huggingface_base_path_endpoint(
        user: str = Depends(current_user),
    ) -> dict:
        """Retrieves the Hugging Face cache directory path.

        The path is determined from the HF_HUB_CACHE constant which points to the
        root of the Hugging Face cache directory.

        Args:
            user (str): The current user, injected by FastAPI dependency.

        Returns:
            dict: A dictionary containing the path if found (e.g., {"path": "/path/to/hf/cache"}),
                  or an error message if not found (e.g., {"status": "error", "message": "..."}).
        """
        try:
            hf_cache_path = Path(HF_HUB_CACHE).resolve()
            if hf_cache_path.exists() and hf_cache_path.is_dir():
                return {"path": str(hf_cache_path)}
            else:
                log.warning(
                    f"Hugging Face cache directory {hf_cache_path} does not exist or is not a directory."
                )
                return {
                    "status": "error",
                    "message": f"Hugging Face cache directory does not exist: {hf_cache_path}",
                }
        except Exception as e:
            log.error(f"Error determining Hugging Face cache directory: {e}")
            return {
                "status": "error",
                "message": "Could not determine Hugging Face cache path. Please check server logs for details.",
            }

    @router.post("/pull_ollama_model")
    async def pull_ollama_model(model_name: str, user: str = Depends(current_user)):
        # Preflight: attempt a lightweight call to detect if Ollama is reachable
        try:
            await get_ollama_models()
        except Exception as e:  # noqa: BLE001
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
            )

        # If reachable, start the streaming response
        return StreamingResponse(
            stream_ollama_model_pull(model_name), media_type="application/json"
        )

    @router.post("/open_in_explorer")
    async def open_in_explorer(
        path: str = Query(...), user: str = Depends(current_user)
    ):
        return await common_open_in_explorer(path)

    @router.post("/huggingface/file_info")
    async def get_huggingface_file_info(
        requests: list[HFFileRequest],
        user: str = Depends(current_user),
    ) -> list[HFFileInfo]:
        # Use async wrapper to avoid blocking the loop
        return await get_huggingface_file_infos_async(requests)

    @router.get("/{model_type}")
    async def index(
        model_type: str, user: str = Depends(current_user)
    ) -> list[ModelFile]:
        folder = comfy_model_to_folder(model_type)
        try:
            # needs comfyui installed
            import folder_paths  # type: ignore

            files = folder_paths.get_filename_list(folder)
        except Exception as e:
            log.error(f"Error getting files for {folder}: {e}")
            files = []

        return [ModelFile(type=folder, name=file) for file in files]
