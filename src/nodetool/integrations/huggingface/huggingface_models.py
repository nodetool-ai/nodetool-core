"""
Hugging Face model management utilities for interacting with cached models and the Hugging Face API.

This module provides functionality to:
- Fetch and cache model information from the Hugging Face API (using custom disk cache)
- Fetch README files leveraging the built-in huggingface_hub cache system
- Read and manage locally cached Hugging Face models
- Determine model types based on model information and recommended models
- Delete models from the local cache

The module uses a hybrid caching approach:
- Model info: Custom disk-based cache for API metadata to persist between sessions
- README files: Leverages huggingface_hub's built-in cache system for efficiency
"""

import asyncio
import json
import os
import shutil
from fnmatch import fnmatch
from pathlib import Path
from typing import List

from huggingface_hub import CacheNotFound, HfApi, ModelInfo, scan_cache_dir
from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.hf_fast_cache import HfFastCache
from nodetool.metadata.types import (
    CLASSNAME_TO_MODEL_TYPE,
    HuggingFaceModel,
    ImageModel,
    LanguageModel,
    Provider,
)
from nodetool.ml.models.model_cache import ModelCache
from nodetool.runtime.resources import maybe_scope
from nodetool.security.secret_helper import get_secret
from nodetool.types.model import CachedFileInfo, UnifiedModel
from nodetool.workflows.recommended_models import get_recommended_models

SINGLE_FILE_DIFFUSION_EXTENSIONS = (
    ".safetensors",
    ".ckpt",
    ".bin",
    ".pt",
    ".pth",
)

SINGLE_FILE_DIFFUSION_TAGS = {
    "diffusers",
    "diffusers:stablediffusionpipeline",
    "diffusers:stablediffusionxlpipeline",
    "diffusers:stablediffusion3pipeline",
    "diffusion-single-file",
    "stable-diffusion",
    "flux",
}

log = get_logger(__name__)


async def get_hf_token(user_id: str | None = None) -> str | None:
    """Get HF_TOKEN from environment variables or database secrets (async).

    Args:
        user_id: Optional user ID. If not provided, will try to get from ResourceScope if available.

    Returns:
        HF_TOKEN if available, None otherwise.
    """

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    if user_id:
        return await get_secret("HF_TOKEN", user_id)
    return None


# Model info cache instance - 24 hour TTL for model metadata
MODEL_INFO_CACHE = ModelCache("model_info")
MODEL_INFO_CACHE_TTL = 30 * 24 * 3600  # 30 days in seconds
# Backwards compatibility alias for legacy references in tests
_model_info_cache = MODEL_INFO_CACHE

# Fast HF cache view for local snapshot lookups.
HF_FAST_CACHE = HfFastCache()

# GGUF_MODELS_FILE = Path(__file__).parent / "gguf_models.json"
# MLX_MODELS_FILE = Path(__file__).parent / "mlx_models.json"


def size_on_disk(
    model_info: ModelInfo,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> int:
    """Calculate the total size of files matching the given patterns.

    Args:
        model_info: ModelInfo object containing siblings list
        allow_patterns: List of patterns to allow (Unix shell-style wildcards)
        ignore_patterns: List of patterns to ignore (Unix shell-style wildcards)

    Returns:
        Total size in bytes of matching files
    """
    siblings = model_info.siblings or []
    total_size = 0

    for sib in siblings:
        if sib.size is None:
            continue

        if not sib.rfilename:
            continue

        if allow_patterns is not None and not any(
            fnmatch(sib.rfilename, pattern) for pattern in allow_patterns
        ):
            continue

        if ignore_patterns is not None and any(
            fnmatch(sib.rfilename, pattern) for pattern in ignore_patterns
        ):
            continue

        total_size += sib.size

    return total_size


def has_model_index(model_info: ModelInfo) -> bool:
    return any(
        sib.rfilename == "model_index.json" for sib in (model_info.siblings or [])
    )


def _is_single_file_diffusion_weight(file_name: str) -> bool:
    """
    Heuristically detect raw checkpoint files (e.g. Stable Diffusion .safetensors)
    that live at the repo root inside the HF cache.

    Excludes standard model weight files that are part of multi-file repos:
    - model.safetensors
    - pytorch_model.bin
    - model.bin
    - model.pt
    - model.pth
    """
    normalized = file_name.replace("\\", "/")
    if "/" in normalized:
        return False
    lower = normalized.lower()

    # Must have a supported extension
    if not lower.endswith(SINGLE_FILE_DIFFUSION_EXTENSIONS):
        return False

    # Exclude standard model weight filenames that are part of multi-file repos
    standard_weight_names = {
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "model.pt",
        "model.pth",
    }
    return lower not in standard_weight_names


def _repo_supports_diffusion_checkpoint(model_info: ModelInfo | None) -> bool:
    """Return True if the repo advertises a compatible diffusion checkpoint."""
    if model_info is None:
        return False
    if model_info.author in ("lllyasviel", "bdsqlsz"):
        return True
    if not model_info.tags:
        return False
    tags = {tag.lower() for tag in model_info.tags}
    return any(tag in tags for tag in SINGLE_FILE_DIFFUSION_TAGS) or any(
        tag in tags for tag in model_info.tags
    )


async def unified_model(
    model: HuggingFaceModel,
    model_info: ModelInfo | None = None,
    size: int | None = None,
    user_id: str | None = None,
) -> UnifiedModel | None:
    if model_info is None or model_info.siblings is None:
        # Use HF_TOKEN from secrets if available for gated model downloads
        token = await get_hf_token(user_id)
        if token:
            log.debug(
                f"unified_model: Fetching model info for {model.repo_id} with HF_TOKEN (token length: {len(token)} chars)"
            )
            api = HfApi(token=token)
        else:
            log.debug(
                f"unified_model: Fetching model info for {model.repo_id} without HF_TOKEN - gated models may not be accessible"
            )
            api = HfApi()
        # Run blocking HfApi call in thread executor
        model_info = await asyncio.get_event_loop().run_in_executor(
            None, lambda: api.model_info(model.repo_id, files_metadata=True)
        )

    # After this point, model_info is guaranteed to be not None
    if model_info is None:
        return None

    model_id = (
        f"{model.repo_id}:{model.path}" if model.path is not None else model.repo_id
    )

    if size is None:
        if model.path:
            # For single-file models, only get the size of the specific file
            size = next(
                (
                    sib.size
                    for sib in (model_info.siblings or [])
                    if sib.rfilename == model.path
                ),
                None,
            )
            # Don't fall back to entire repo size for single-file models
            # If the file size isn't found, keep it as None
        else:
            # For multi-file models without a specific path, use total repo size
            # Respect allow_patterns and ignore_patterns when calculating size
            size = size_on_disk(
                model_info,
                allow_patterns=model.allow_patterns,
                ignore_patterns=model.ignore_patterns,
            )
    return UnifiedModel(
        id=model_id,
        repo_id=model.repo_id,
        path=model.path,
        type=model.type,
        name=model.repo_id,
        cache_path=None,
        allow_patterns=model.allow_patterns,
        ignore_patterns=model.ignore_patterns,
        description=None,
        readme=None,
        size_on_disk=size,
        downloaded=False,
        pipeline_tag=model_info.pipeline_tag,
        tags=model_info.tags,
        has_model_index=has_model_index(model_info),
        downloads=model_info.downloads,
        likes=model_info.likes,
        trending_score=model_info.trending_score,
    )


async def fetch_model_readme(model_id: str) -> str | None:
    """
    Fetches the readme from the Hugging Face hub cache or downloads it
    using the huggingface_hub library, leveraging the built-in cache system.

    Args:
        model_id (str): The ID of the model to fetch.

    Returns:
        str: The readme content, or None if not found.
    """
    from huggingface_hub import (
        _CACHED_NO_EXIST,
        hf_hub_download,
        try_to_load_from_cache,
    )

    # First, try to load from the HF hub cache
    cached_path = try_to_load_from_cache(repo_id=model_id, filename="README.md")

    if isinstance(cached_path, str):
        # File exists in cache, read and return it
        try:
            with open(cached_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            log.debug(f"Failed to read cached README for {model_id}: {e}")
    elif cached_path is _CACHED_NO_EXIST:
        # Non-existence is cached, return None immediately
        return None

    # File not in cache, try to download it
    try:
        # Use HF_TOKEN from secrets if available for gated model downloads
        # Note: user_id would need to be passed from caller context
        token = await get_hf_token()
        if token:
            log.debug(
                f"fetch_model_readme: Downloading README for {model_id} with HF_TOKEN (token length: {len(token)} chars)"
            )
        else:
            log.debug(
                f"fetch_model_readme: Downloading README for {model_id} without HF_TOKEN - gated models may not be accessible"
            )
        readme_path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: hf_hub_download(
                repo_id=model_id, filename="README.md", repo_type="model", token=token
            ),
        )
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        log.debug(f"Failed to download README for {model_id}: {e}")
        return None


async def fetch_model_info(model_id: str) -> ModelInfo | None:
    """
    Fetches model info from the cache or Hugging Face API.
    Uses nodetool's disk-based cache with 24-hour TTL for model metadata.

    Args:
        model_id (str): The ID of the model to fetch.

    Returns:
        ModelInfo: The model info, or None if not found.
    """
    cache_key = f"model_info:{model_id}"
    cached_result = MODEL_INFO_CACHE.get(cache_key)
    if cached_result is not None:
        log.debug(f"Cache hit for model info: {model_id}")
        return cached_result

    # Use HF_TOKEN from secrets if available for gated model downloads
    # Note: user_id would need to be passed from caller context
    token = await get_hf_token()
    if token:
        log.debug(
            f"fetch_model_info: Fetching model info for {model_id} with HF_TOKEN (token length: {len(token)} chars)"
        )
        api = HfApi(token=token)
    else:
        log.debug(
            f"fetch_model_info: Fetching model info for {model_id} without HF_TOKEN - gated models may not be accessible"
        )
        api = HfApi()

    model_info: ModelInfo = await asyncio.get_event_loop().run_in_executor(
        None, lambda: api.model_info(model_id, files_metadata=True)
    )

    # # Store in cache for future use
    MODEL_INFO_CACHE.set(cache_key, model_info, MODEL_INFO_CACHE_TTL)
    log.debug(f"Cached model info for: {model_id}")

    return model_info


def model_type_from_model_info(
    recommended_models: dict[str, list[UnifiedModel]],
    repo_id: str,
    model_info: ModelInfo | None,
) -> str | None:
    recommended = recommended_models.get(repo_id, [])
    if len(recommended) == 1:
        return recommended[0].type
    if model_info is None:
        return None
    if (
        model_info.config
        and "diffusers" in model_info.config
        and "_class_name" in model_info.config["diffusers"]
    ):
        return CLASSNAME_TO_MODEL_TYPE.get(
            model_info.config["diffusers"]["_class_name"],
            None,  # type: ignore[no-any-return]
        )
    if model_info.pipeline_tag:
        name = model_info.pipeline_tag.replace("-", "_")
        return f"hf.{name}"
    if model_info.tags:
        if "mlx" in model_info.tags:
            return "mlx"
        if "gguf" in model_info.tags:
            return "llama_cpp"
    return None


async def read_cached_hf_files(
    pipeline_tag: str | None = None,
    library_name: str | None = None,
    tags: list[str] | None = None,
) -> List[CachedFileInfo]:
    """
    Reads all models from the Hugging Face cache.

    Returns:
        List[CachedFileInfo]: A list of CachedFileInfo objects found in the cache.
    """
    tags = [tag.lower() for tag in (tags or [])]

    # Create cache key based on tags filter
    cache_key_parts = []
    if pipeline_tag:
        cache_key_parts.append(pipeline_tag)
    if library_name:
        cache_key_parts.append(library_name)
    if tags:
        cache_key_parts.extend(tags)

    # Check cache first
    # cache_key = ":".join(cache_key_parts)
    # cached_result = MODEL_INFO_CACHE.get(cache_key)
    # if cached_result is not None:
    #     log.debug(f"Returning {len(cached_result)} cached HF files from cache")
    #     return cached_result

    # Offload scanning HF cache to a thread (filesystem heavy)
    try:
        cache_info = await asyncio.to_thread(scan_cache_dir)
    except CacheNotFound:
        log.debug("Hugging Face cache directory not found; returning empty list")
        # Don't cache non-existence - allow retry
        return []

    model_repos = [repo for repo in cache_info.repos if repo.repo_type == "model"]
    log.debug(f"Scanning {len(model_repos)} HF repos for files with tags={tags}")

    cached_files = []

    model_infos = await asyncio.gather(
        *[fetch_model_info(repo.repo_id) for repo in model_repos],
        return_exceptions=True,  # Don't fail entire operation if one model fails
    )

    for repo, model_info in zip(model_repos, model_infos, strict=False):
        # Handle exceptions from individual fetch_model_info calls
        if isinstance(model_info, BaseException):
            log.debug(f"Failed to fetch model info for {repo.repo_id}: {model_info}")
            continue

        # Get cached files from all revisions
        if model_info is None:
            continue
        if pipeline_tag and model_info.pipeline_tag != pipeline_tag:
            continue
        if library_name and model_info.library_name != library_name:
            continue
        if tags and not all(tag.lower() in (model_info.tags or []) for tag in tags):
            continue
        for revision in repo.revisions:
            for file_info in revision.files:
                cached_files.append(
                    CachedFileInfo(
                        repo_id=repo.repo_id,
                        file_name=file_info.file_name,
                        size_on_disk=file_info.size_on_disk,
                        model_info=model_info,
                    )
                )

    # Cache for 1 hour (3600 seconds) - even partial results
    # MODEL_INFO_CACHE.set(cache_key, cached_files, ttl=MODEL_INFO_CACHE_TTL)

    return cached_files


async def read_cached_hf_models() -> List[UnifiedModel]:
    """
    Reads all models from the Hugging Face cache.
    Results are cached for 1 hour to avoid repeated filesystem scanning.

    Returns:
        List[UnifiedModel]: A list of UnifiedModel objects found in the cache.
    """
    cache_key = "cached_hf_models:all"

    # Check cache first
    cached_result = MODEL_INFO_CACHE.get(cache_key)
    if cached_result is not None:
        log.info(
            f"✓ CACHE HIT: Returning {len(cached_result)} cached HF models (skipping scan)"
        )
        return cached_result

    log.info("✗ CACHE MISS: Scanning HF cache directory for models")

    # Offload scanning HF cache to a thread (filesystem heavy)
    try:
        cache_info = await asyncio.to_thread(scan_cache_dir)
    except CacheNotFound:
        log.debug("Hugging Face cache directory not found; returning empty model list")
        # Don't cache non-existence - allow retry
        return []

    model_repos = [repo for repo in cache_info.repos if repo.repo_type == "model"]
    log.debug(f"Fetching info for {len(model_repos)} cached HF models")

    recommended_models = get_recommended_models()
    models: list[UnifiedModel] = []

    try:
        model_infos = await asyncio.gather(
            *[fetch_model_info(repo.repo_id) for repo in model_repos],
            return_exceptions=True,  # Don't fail entire operation if one model fails
        )

        for repo, model_info in zip(model_repos, model_infos, strict=False):
            # Handle exceptions from individual fetch_model_info calls
            if isinstance(model_info, BaseException):
                log.debug(
                    f"Failed to fetch model info for {repo.repo_id}: {model_info}"
                )
                # Still create a basic model entry without the extra metadata
                model_info = None

            models.append(
                UnifiedModel(
                    id=repo.repo_id,
                    type=model_type_from_model_info(
                        recommended_models, repo.repo_id, model_info
                    ),
                    name=repo.repo_id,
                    cache_path=str(repo.repo_path),
                    allow_patterns=None,
                    ignore_patterns=None,
                    description=None,
                    readme=None,
                    downloaded=repo.repo_path is not None,
                    pipeline_tag=model_info.pipeline_tag if model_info else None,
                    tags=model_info.tags if model_info else None,
                    has_model_index=(
                        has_model_index(model_info) if model_info else False
                    ),
                    repo_id=repo.repo_id,
                    path=None,
                    size_on_disk=repo.size_on_disk,
                    downloads=model_info.downloads if model_info else None,
                    likes=model_info.likes if model_info else None,
                    trending_score=model_info.trending_score if model_info else None,
                )
            )

        # Cache for 1 hour (3600 seconds) - even partial results
        log.info(
            f"💾 Attempting to cache {len(models)} HF models with key: {cache_key}"
        )
        MODEL_INFO_CACHE.set(cache_key, models, ttl=3600)
        log.info(f"✓ Successfully cached {len(models)} HF models (TTL: 1 hour)")

    except Exception as e:
        log.error(f"✗ Error processing cached HF models: {e}", exc_info=True)
        # Return what we have, don't cache on error

    return models


async def get_llamacpp_language_models_from_hf_cache() -> List[LanguageModel]:
    """
    Return LanguageModel entries for cached Hugging Face repos containing GGUF files
    that look suitable for llama.cpp.

    Heuristics:
    - File ends with .gguf (case-insensitive)
    - Each GGUF file yields a LanguageModel with id "<repo_id>:<filename>"

    Returns:
        List[LanguageModel]: Llama.cpp-compatible models discovered in the HF cache
    """
    cached = await read_cached_hf_files(
        "text-generation", "transformers", tags=["gguf"]
    )
    results: list[LanguageModel] = []

    for f in cached:
        fname = f.file_name
        if not fname:
            continue
        if not fname.lower().endswith(".gguf"):
            continue
        model_id = f"{f.repo_id}:{fname}"
        display = f"{f.repo_id.split('/')[-1]} • {fname}"
        results.append(
            LanguageModel(
                id=model_id,
                name=display,
                path=fname,
                provider=Provider.LlamaCpp,
            )
        )

    # Sort for stability: by repo then filename
    results.sort(key=lambda m: (m.id.split(":", 1)[0], m.id))
    return results


async def get_vllm_language_models_from_hf_cache() -> List[LanguageModel]:
    """Return LanguageModel entries tagged as vLLM in cached metadata files."""
    cached = await read_cached_hf_files(
        "text-generation", "transformers", tags=["vllm"]
    )
    seen_repos: set[str] = set()
    results: list[LanguageModel] = []

    SUPPORTED_WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth")

    for f in cached:
        # Skip cache entries without an actual weight filename to point at
        if not f.file_name:
            continue

        lower_name = f.file_name.lower()
        if not lower_name.endswith(SUPPORTED_WEIGHT_EXTENSIONS):
            continue

        # We only need a single listing per repo, so collapse duplicates
        if f.repo_id in seen_repos:
            continue
        seen_repos.add(f.repo_id)

        repo_display = f.repo_id.split("/")[-1]
        results.append(
            LanguageModel(
                id=f.repo_id,
                name=repo_display,
                provider=Provider.VLLM,
            )
        )
    return results


async def get_mlx_language_models_from_hf_cache() -> List[LanguageModel]:
    """
    Return LanguageModel entries for cached Hugging Face repos that look suitable
    for MLX runtime (Apple Silicon).

    Each qualifying repo yields a LanguageModel with id "<repo_id>" (no file suffix),
    because MLX loaders typically resolve the correct shard/quantization internally.

    Returns:
        List[LanguageModel]: MLX-compatible models discovered in the HF cache
    """
    cached = await read_cached_hf_files("text-generation", "mlx")
    result: dict[str, LanguageModel] = {}

    for model in cached:
        # read_cached_hf_files already filtered by tags, so all models here have either "mlx" or "mflux" tag
        display = model.repo_id.split("/")[-1]
        result[model.repo_id] = LanguageModel(
            id=model.repo_id,
            name=display,
            provider=Provider.MLX,
        )

    return list(result.values())


async def get_text_to_image_models_from_hf_cache() -> List[ImageModel]:
    """
    Return ImageModel entries for cached Hugging Face repos that are text-to-image models,
    including single-file checkpoints stored at the repo root (e.g. Stable Diffusion safetensors).
    """
    cached = await read_cached_hf_files()
    result: dict[str, ImageModel] = {}
    repos_with_single_files: set[str] = set()

    for model in cached:
        fname = model.file_name
        display = model.repo_id.split("/")[-1]
        lower_name = fname.lower()
        if lower_name.endswith(".gguf"):
            model_id = f"{model.repo_id}:{fname}"
            repos_with_single_files.add(model.repo_id)
            result.pop(model.repo_id, None)
            result[model_id] = ImageModel(
                id=model.repo_id,
                name=display,
                path=fname,
                provider=Provider.HuggingFace,
                supported_tasks=["text_to_image"],
            )
            continue

        if _is_single_file_diffusion_weight(fname):
            model_info = model.model_info
            # Include single-file checkpoints even if repo has model_index.json
            # Prefer single-file versions over multi-file when available
            if _repo_supports_diffusion_checkpoint(model_info):
                model_id = f"{model.repo_id}:{fname}"
                repos_with_single_files.add(model.repo_id)
                # Remove multi-file entry if it exists - prefer single-file
                result.pop(model.repo_id, None)
                result[model_id] = ImageModel(
                    id=model.repo_id,
                    name=display,
                    path=fname,
                    provider=Provider.HuggingFace,
                    supported_tasks=["text_to_image"],
                )
                continue

        # Skip multi-file entry if repo has single files (prefer single-file versions)
        if model.repo_id in repos_with_single_files:
            continue

        result[model.repo_id] = ImageModel(
            id=model.repo_id,
            name=display,
            provider=Provider.HuggingFace,
            supported_tasks=["text_to_image"],
        )

    return list(result.values())


async def get_image_to_image_models_from_hf_cache() -> List[ImageModel]:
    """
    Return ImageModel entries for cached Hugging Face repos that are image-to-image models,
    including single-file checkpoints stored at the repo root.
    """
    cached = await read_cached_hf_files()
    result: dict[str, ImageModel] = {}
    repos_with_single_files: set[str] = set()
    repo_info_cache: dict[str, ModelInfo | None] = {}

    for model in cached:
        fname = model.file_name
        display = model.repo_id.split("/")[-1]
        lower_name = fname.lower()
        if lower_name.endswith(".gguf"):
            model_id = f"{model.repo_id}:{fname}"
            repos_with_single_files.add(model.repo_id)
            result.pop(model.repo_id, None)
            result[model_id] = ImageModel(
                id=model.repo_id,
                name=display,
                path=fname,
                provider=Provider.HuggingFace,
                supported_tasks=["image_to_image"],
            )
            continue

        if _is_single_file_diffusion_weight(fname):
            model_info = model.model_info
            # Include single-file checkpoints even if repo has model_index.json
            # Prefer single-file versions over multi-file when available
            if _repo_supports_diffusion_checkpoint(model_info):
                model_id = f"{model.repo_id}:{fname}"
                repos_with_single_files.add(model.repo_id)
                # Remove multi-file entry if it exists - prefer single-file
                result.pop(model.repo_id, None)
                result[model_id] = ImageModel(
                    id=model.repo_id,
                    name=display,
                    path=fname,
                    provider=Provider.HuggingFace,
                    supported_tasks=["image_to_image"],
                )
                continue

        # Skip multi-file entry if repo has single files (prefer single-file versions)
        if model.repo_id in repos_with_single_files:
            continue

        result[model.repo_id] = ImageModel(
            id=model.repo_id,
            name=display,
            provider=Provider.HuggingFace,
            supported_tasks=["image_to_image"],
        )

    return list(result.values())


async def get_mlx_image_models_from_hf_cache() -> List[ImageModel]:
    """
    Return ImageModel entries for cached Hugging Face repos that are mflux models
    (MLX-compatible image generation models).

    Returns:
        List[ImageModel]: MLX-compatible image models (mflux) discovered in the HF cache
    """
    # Search for models with "mflux" tag - these are MLX-compatible image generation models
    cached = await read_cached_hf_files("text-to-image", None, tags=["mflux"])
    result: dict[str, ImageModel] = {}

    for model in cached:
        # read_cached_hf_files already filtered by mflux tag
        display = model.repo_id.split("/")[-1]
        result[model.repo_id] = ImageModel(
            id=model.repo_id,
            name=display,
            provider=Provider.MLX,
        )

    return list(result.values())


async def _fetch_models_by_author(
    user_id: str | None = None, **kwargs
) -> list[ModelInfo]:
    """Fetch models list from HF API for a given author using HFAPI.

    Returns raw model dicts from the public API.
    """
    # Use HF_TOKEN from secrets if available for gated model downloads
    token = await get_hf_token(user_id)
    author = kwargs.get("author", "unknown")
    if token:
        log.debug(
            f"_fetch_models_by_author: Fetching models for author {author} with HF_TOKEN (token length: {len(token)} chars)"
        )
        api = HfApi(token=token)
    else:
        log.debug(
            f"_fetch_models_by_author: Fetching models for author {author} without HF_TOKEN - gated models may not be accessible"
        )
        api = HfApi()
    # Run the blocking call in a thread executor
    models = await asyncio.get_event_loop().run_in_executor(
        None, lambda: api.list_models(**kwargs)
    )
    return list(models)


async def get_gguf_language_models_from_authors(
    authors: list[str],
    limit: int = 200,
    sort: str = "downloads",
    tags: str = "gguf",
) -> List[UnifiedModel]:
    """
    Fetch all HF repos authored by the given authors that include GGUF files/tags.

    Heuristic: filter API results to those with a "gguf" tag, then for each
    author select the top 30 repos sorted by likes.

    Args:
        authors: List of HF author/org names (e.g., ["unsloth", "ggml-org"]).

    Returns:
        List[HuggingFaceModel]: One entry per matching repo.
    """
    # Fetch authors concurrently
    # Note: user_id would need to be passed from caller context
    results = await asyncio.gather(
        *(
            _fetch_models_by_author(
                user_id=None,
                author=a,
                limit=limit,
                sort=sort,
                tags=tags,
            )
            for a in authors
        )
    )
    repos = [item for sublist in results for item in sublist]
    model_infos = await asyncio.gather(*[fetch_model_info(repo.id) for repo in repos])

    # Collect all unified_model tasks
    tasks: list[tuple[HuggingFaceModel, ModelInfo, int | None]] = []
    seen_file: set[str] = set()
    for info in model_infos:
        if info is None:
            continue
        sibs = info.siblings or []
        for sib in sibs:
            fname = getattr(sib, "rfilename", None)
            if not isinstance(fname, str) or not fname.lower().endswith(".gguf"):
                continue
            if fname in seen_file:
                continue
            seen_file.add(fname)
            tasks.append(
                (
                    HuggingFaceModel(type="llama_cpp", repo_id=info.id, path=fname),
                    info,
                    sib.size,
                )
            )

    # Execute all unified_model calls in parallel
    entries = await asyncio.gather(
        *[unified_model(model, info, size) for model, info, size in tasks]
    )

    # Sort for stability: repo then filename
    entries = [entry for entry in entries if entry is not None]
    return entries


async def get_mlx_language_models_from_authors(
    authors: list[str],
    limit: int = 200,
    sort: str = "trending_score",
    tags: str = "mlx",
) -> List[UnifiedModel]:
    """
    Fetch MLX-friendly repos authored by the given authors/orgs and return
    one LanguageModel per repo id.

    Heuristics:
    - Prefer orgs like "mlx-community" via the authors parameter
    - Filter API results to those with a tag containing "mlx"
    - Per author, take the top 30 repos sorted by likes

    Args:
        authors: List of HF author/org names (e.g., ["mlx-community"]).

    Returns:
        List[HuggingFaceModel]: One entry per qualifying repo.
    """
    # Fetch authors concurrently
    # Note: user_id would need to be passed from caller context
    results = await asyncio.gather(
        *(
            _fetch_models_by_author(
                user_id=None, author=a, limit=limit, sort=sort, tags=tags
            )
            for a in authors
        )
    )
    model_infos = [item for sublist in results for item in sublist]

    # Execute all unified_model calls in parallel
    entries = await asyncio.gather(
        *[
            unified_model(HuggingFaceModel(type="mlx", repo_id=info.id), info)
            for info in model_infos
        ]
    )

    # Stable order
    return [entry for entry in entries if entry is not None]


def delete_cached_hf_model(model_id: str) -> bool:
    """
    Deletes a model from the Hugging Face cache and the disk cache.

    Args:
        model_id (str): The ID of the model to delete.
    """
    # Use HfFastCache to resolve the repo root without walking the entire cache.
    repo_root = HF_FAST_CACHE.repo_root(model_id, repo_type="model")
    if not repo_root:
        return False

    if not os.path.exists(repo_root):
        return False

    shutil.rmtree(repo_root)

    # Purge all HuggingFace caches after successful deletion
    log.info("Purging HuggingFace model caches after model deletion")
    MODEL_INFO_CACHE.delete_pattern("cached_hf_*")
    HF_FAST_CACHE.invalidate(model_id, repo_type="model")
    return True


# GGUF_AUTHORS = [
# "unsloth",
# "ggml-org",
# "LiquidAI",
# "gabriellarson",
# "openbmb",
# "zai-org",
# "vikhyatk",
# "01-ai",
# "BAAI",
# "Lin-Chen",
# "mtgv",
# "lm-sys",
# "NousResearch",
# ]
# MLX_AUTHORS = ["mlx-community"]


# async def save_gguf_models_to_file() -> None:
#     models = await get_gguf_language_models_from_authors(
#         GGUF_AUTHORS, limit=500, sort="downloads", tags="gguf"
#     )
#     with open(GGUF_MODELS_FILE, "w") as f:
#         json.dump(
#             [model.model_dump() for model in models if model is not None], f, indent=2
#         )


# async def save_mlx_models_to_file() -> None:
#     models = await get_mlx_language_models_from_authors(
#         MLX_AUTHORS, limit=1000, sort="downloads", tags="mlx"
#     )
#     with open(MLX_MODELS_FILE, "w") as f:
#         json.dump([model.model_dump() for model in models], f, indent=2)


# async def load_gguf_models_from_file() -> List[UnifiedModel]:
#     async with aiofiles.open(GGUF_MODELS_FILE, "r") as f:
#         content = await f.read()
#         return [UnifiedModel(**model) for model in json.loads(content)]


# async def load_mlx_models_from_file() -> List[UnifiedModel]:
#     async with aiofiles.open(MLX_MODELS_FILE, "r") as f:
#         content = await f.read()
#         return [UnifiedModel(**model) for model in json.loads(content)]


if __name__ == "__main__":

    async def main():
        cached = await get_text_to_image_models_from_hf_cache()
        for model in cached:
            print(model.id)

    asyncio.run(main())
