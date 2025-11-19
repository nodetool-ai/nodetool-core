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
from typing import Callable, List, Sequence

from huggingface_hub import HfApi, ModelInfo
from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.hf_fast_cache import (
    DEFAULT_MODEL_INFO_CACHE_TTL,
    HfFastCache,
)
from nodetool.metadata.types import (
    CLASSNAME_TO_MODEL_TYPE,
    HuggingFaceModel,
    ImageModel,
    LanguageModel,
    Provider,
)
from nodetool.runtime.resources import maybe_scope
from nodetool.security.secret_helper import get_secret
from nodetool.types.model import UnifiedModel
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

CACHED_HF_MODELS_CACHE_KEY = "cached_hf_models"
CACHED_HF_MODELS_TTL = 3600  # 1 hour

CACHED_HF_MODELS_CACHE_KEY = "cached_hf_models"
CACHED_HF_MODELS_TTL = 3600  # 1 hour

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
    cached_result = HF_FAST_CACHE.model_info_cache.get(cache_key)
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
    HF_FAST_CACHE.model_info_cache.set(
        cache_key,
        model_info,
        DEFAULT_MODEL_INFO_CACHE_TTL,
    )
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




def _get_file_size(file_path: Path) -> int:
    """Get file size, handling symlinks."""
    try:
        if file_path.is_symlink():
            # Resolve symlink to get actual file size
            resolved = file_path.resolve(strict=False)
            if resolved.exists():
                return resolved.stat().st_size
        elif file_path.exists():
            return file_path.stat().st_size
    except OSError:
        pass
    return 0


async def _build_cached_repo_entry(
    repo_id: str,
    repo_dir: Path,
    model_info: ModelInfo | None,
    recommended_models: dict[str, list[UnifiedModel]],
) -> tuple[UnifiedModel, list[tuple[str, int]]]:
    """Build the UnifiedModel entry and collect file metadata for a cached repo."""
    repo_root = await HF_FAST_CACHE.repo_root(repo_id, repo_type="model")
    snapshot_dir = await HF_FAST_CACHE.active_snapshot_dir(repo_id, repo_type="model")

    file_entries: list[tuple[str, int]] = []
    size_on_disk = 0

    if snapshot_dir:
        snapshot_path = Path(snapshot_dir)
        try:
            file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
        except Exception as exc:  # pragma: no cover - defensive
            log.debug(f"Failed to list files for {repo_id}: {exc}")
            file_list = []

        for file_name in file_list:
            file_path = snapshot_path / file_name
            file_size = _get_file_size(file_path)
            size_on_disk += file_size
            file_entries.append((file_name, file_size))

    repo_model = UnifiedModel(
        id=repo_id,
        type=model_type_from_model_info(
            recommended_models,
            repo_id,
            model_info,
        ),
        name=repo_id,
        cache_path=str(repo_root) if repo_root else str(repo_dir),
        allow_patterns=None,
        ignore_patterns=None,
        description=None,
        readme=None,
        downloaded=repo_root is not None or repo_dir.exists(),
        pipeline_tag=model_info.pipeline_tag if model_info else None,
        tags=model_info.tags if model_info else None,
        has_model_index=has_model_index(model_info) if model_info else False,
        repo_id=repo_id,
        path=None,
        size_on_disk=size_on_disk,
        downloads=model_info.downloads if model_info else None,
        likes=model_info.likes if model_info else None,
        trending_score=model_info.trending_score if model_info else None,
    )

    return repo_model, file_entries


async def read_cached_hf_models() -> List[UnifiedModel]:
    """
    Reads all models from the Hugging Face cache using HfFastCache for efficient lookups.
    Results are cached for 1 hour to avoid repeated filesystem scanning.

    Returns:
        List[UnifiedModel]: A list of UnifiedModel objects found in the cache.
    """

    cached_models = HF_FAST_CACHE.model_info_cache.get(CACHED_HF_MODELS_CACHE_KEY)
    if cached_models is not None:
        return cached_models

    try:
        # Discover repos by listing cache directory (lightweight)
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:  # pragma: no cover - defensive guard
        log.warning(f"Failed to discover cached HF repos: {exc}")
        return []

    recommended_models = get_recommended_models()
    models: list[UnifiedModel] = []

    model_infos = await asyncio.gather(
        *[fetch_model_info(repo_id) for repo_id, _ in repo_list],
        return_exceptions=True,  # Don't fail entire operation if one model fails
    )

    for (repo_id, repo_dir), model_info in zip(repo_list, model_infos, strict=False):
        # Handle exceptions from individual fetch_model_info calls
        if isinstance(model_info, BaseException):
            log.debug(f"Failed to fetch model info for {repo_id}: {model_info}")
            # Still create a basic model entry without the extra metadata
            model_info = None

        repo_model, _ = await _build_cached_repo_entry(
            repo_id,
            repo_dir,
            model_info,
            recommended_models,
        )
        models.append(repo_model)

    HF_FAST_CACHE.model_info_cache.set(
        CACHED_HF_MODELS_CACHE_KEY,
        models,
        CACHED_HF_MODELS_TTL,
    )

    return models


def _normalize_patterns(values: Sequence[str] | None, *, lower: bool = False) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        if value is None:
            continue
        trimmed = value.strip()
        if not trimmed:
            continue
        normalized.append(trimmed.lower() if lower else trimmed)
    return normalized


def _matches_any_pattern(value: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch(value, pattern) for pattern in patterns)


def _repo_tags_match_patterns(repo_tags: list[str], patterns: list[str]) -> bool:
    if not patterns:
        return True
    if not repo_tags:
        return False
    for pattern in patterns:
        if not any(fnmatch(tag, pattern) for tag in repo_tags):
            return False
    return True


async def search_cached_hf_models(
    repo_patterns: Sequence[str] | None = None,
    filename_patterns: Sequence[str] | None = None,
    pipeline_tags: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    authors: Sequence[str] | None = None,
    library_name: str | None = None,
) -> List[UnifiedModel]:
    """
    Search the Hugging Face cache by repo metadata and optional filename patterns.
    Returns matching repo entries and (optionally) file-level entries.
    """

    try:
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:  # pragma: no cover - defensive guard
        log.warning(f"Failed to discover cached HF repos: {exc}")
        return []

    if not repo_list:
        return []

    repo_pattern_list = _normalize_patterns(repo_patterns)
    filename_pattern_list = _normalize_patterns(filename_patterns)
    pipeline_tag_patterns = _normalize_patterns(pipeline_tags, lower=True)
    tag_patterns = _normalize_patterns(tags, lower=True)
    author_patterns = _normalize_patterns(authors, lower=True)
    library_pattern = (
        library_name.strip().lower() if library_name and library_name.strip() else None
    )

    recommended_models = get_recommended_models()
    results: list[UnifiedModel] = []
    requires_metadata = any(
        [pipeline_tag_patterns, tag_patterns, author_patterns, library_pattern]
    )

    model_infos = await asyncio.gather(
        *[fetch_model_info(repo_id) for repo_id, _ in repo_list],
        return_exceptions=True,
    )

    for (repo_id, repo_dir), model_info in zip(repo_list, model_infos, strict=False):
        info: ModelInfo | None
        if isinstance(model_info, BaseException):
            log.debug(f"Failed to fetch model info for {repo_id}: {model_info}")
            info = None
        else:
            info = model_info

        if repo_pattern_list and not _matches_any_pattern(repo_id, repo_pattern_list):
            continue

        if requires_metadata and info is None:
            continue

        if info:
            pipeline_value = (info.pipeline_tag or "").lower()
            if pipeline_tag_patterns and not _matches_any_pattern(
                pipeline_value, pipeline_tag_patterns
            ):
                continue

            repo_tags = [tag.lower() for tag in (info.tags or [])]
            if not _repo_tags_match_patterns(repo_tags, tag_patterns):
                continue

            author_value = (info.author or "").lower()
            if author_patterns and not _matches_any_pattern(
                author_value, author_patterns
            ):
                continue

            library_value = (getattr(info, "library_name", "") or "").lower()
            if library_pattern and not fnmatch(library_value, library_pattern):
                continue

        repo_model, file_entries = await _build_cached_repo_entry(
            repo_id,
            repo_dir,
            info,
            recommended_models,
        )
        results.append(repo_model)

        if filename_pattern_list and file_entries:
            for relative_name, file_size in file_entries:
                if not _matches_any_pattern(relative_name, filename_pattern_list):
                    continue

                file_model = UnifiedModel(
                    id=f"{repo_id}:{relative_name}",
                    type=repo_model.type,
                    name=f"{repo_id}/{relative_name}",
                    repo_id=repo_id,
                    path=relative_name,
                    cache_path=repo_model.cache_path,
                    allow_patterns=None,
                    ignore_patterns=None,
                    description=None,
                    readme=None,
                    size_on_disk=file_size,
                    downloaded=repo_model.downloaded,
                    pipeline_tag=repo_model.pipeline_tag,
                    tags=repo_model.tags,
                    has_model_index=repo_model.has_model_index,
                    downloads=repo_model.downloads,
                    likes=repo_model.likes,
                    trending_score=repo_model.trending_score,
                )
                results.append(file_model)

    return results


async def _filter_repos_by_metadata(
    pipeline_tag: str | None = None,
    library_name: str | None = None,
    tags: list[str] | None = None,
    predicate: Callable[[ModelInfo], bool] | None = None,
) -> list[tuple[str, Path, ModelInfo]]:
    """Return repo tuples filtered by metadata before any file scans."""
    tags = [tag.lower() for tag in (tags or [])]

    repo_list = await HF_FAST_CACHE.discover_repos("model")
    model_infos = await asyncio.gather(
        *[fetch_model_info(repo_id) for repo_id, _ in repo_list],
        return_exceptions=True,
    )

    filtered: list[tuple[str, Path, ModelInfo]] = []
    for (repo_id, repo_dir), model_info in zip(repo_list, model_infos, strict=False):
        if isinstance(model_info, BaseException) or model_info is None:
            continue
        if pipeline_tag and model_info.pipeline_tag != pipeline_tag:
            continue
        if library_name and model_info.library_name != library_name:
            continue
        if tags and not all(tag in (model_info.tags or []) for tag in tags):
            continue
        if predicate and not predicate(model_info):
            continue
        filtered.append((repo_id, repo_dir, model_info))
    return filtered


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
    filtered_repos = await _filter_repos_by_metadata(
        pipeline_tag="text-generation", library_name="transformers", tags=["gguf"]
    )
    results: list[LanguageModel] = []

    for repo_id, _repo_dir, model_info in filtered_repos:
        snapshot_dir = await HF_FAST_CACHE.active_snapshot_dir(repo_id, repo_type="model")
        if not snapshot_dir:
            continue
        file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
        for fname in file_list:
            if not fname.lower().endswith(".gguf"):
                continue
            model_id = f"{repo_id}:{fname}"
            display = f"{repo_id.split('/')[-1]} • {fname}"
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
    filtered_repos = await _filter_repos_by_metadata(
        pipeline_tag="text-generation", library_name="transformers", tags=["vllm"]
    )
    seen_repos: set[str] = set()
    results: list[LanguageModel] = []

    SUPPORTED_WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth")

    for repo_id, _repo_dir, _info in filtered_repos:
        if repo_id in seen_repos:
            continue
        file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
        if any(fname.lower().endswith(SUPPORTED_WEIGHT_EXTENSIONS) for fname in file_list):
            seen_repos.add(repo_id)
            repo_display = repo_id.split("/")[-1]
            results.append(
                LanguageModel(
                    id=repo_id,
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
    filtered_repos = await _filter_repos_by_metadata(
        pipeline_tag="text-generation", library_name="mlx"
    )
    result: dict[str, LanguageModel] = {}

    for repo_id, _repo_dir, _info in filtered_repos:
        display = repo_id.split("/")[-1]
        result[repo_id] = LanguageModel(
            id=repo_id,
            name=display,
            provider=Provider.MLX,
        )

    return list(result.values())


async def get_text_to_image_models_from_hf_cache() -> List[ImageModel]:
    """
    Return ImageModel entries for cached Hugging Face repos that are text-to-image models,
    including single-file checkpoints stored at the repo root (e.g. Stable Diffusion safetensors).
    """
    def _is_image_repo(info: ModelInfo) -> bool:
        if info.pipeline_tag in {"text-to-image", "image-to-image"}:
            return True
        return _repo_supports_diffusion_checkpoint(info)

    filtered_repos = await _filter_repos_by_metadata(predicate=_is_image_repo)
    result: dict[str, ImageModel] = {}
    repos_with_single_files: set[str] = set()

    for repo_id, _repo_dir, model_info in filtered_repos:
        snapshot_dir = await HF_FAST_CACHE.active_snapshot_dir(repo_id, repo_type="model")
        if not snapshot_dir:
            continue
        file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
        for fname in file_list:
            display = repo_id.split("/")[-1]
        lower_name = fname.lower()
        if lower_name.endswith(".gguf"):
            model_id = f"{repo_id}:{fname}"
            repos_with_single_files.add(repo_id)
            result.pop(repo_id, None)
            result[model_id] = ImageModel(
                id=repo_id,
                name=display,
                path=fname,
                provider=Provider.HuggingFace,
                supported_tasks=["text_to_image"],
            )
            continue

        if _is_single_file_diffusion_weight(fname):
            # Include single-file checkpoints even if repo has model_index.json
            # Prefer single-file versions over multi-file when available
            if _repo_supports_diffusion_checkpoint(model_info):
                model_id = f"{repo_id}:{fname}"
                repos_with_single_files.add(repo_id)
                # Remove multi-file entry if it exists - prefer single-file
                result.pop(repo_id, None)
                result[model_id] = ImageModel(
                    id=repo_id,
                    name=display,
                    path=fname,
                    provider=Provider.HuggingFace,
                    supported_tasks=["text_to_image"],
                )
                continue

        # Skip multi-file entry if repo has single files (prefer single-file versions)
        if repo_id in repos_with_single_files:
            continue

        result[repo_id] = ImageModel(
            id=repo_id,
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
    def _is_image_repo(info: ModelInfo) -> bool:
        if info.pipeline_tag in {"image-to-image", "text-to-image"}:
            return True
        return _repo_supports_diffusion_checkpoint(info)

    filtered_repos = await _filter_repos_by_metadata(predicate=_is_image_repo)
    result: dict[str, ImageModel] = {}
    repos_with_single_files: set[str] = set()

    for repo_id, _repo_dir, model_info in filtered_repos:
        snapshot_dir = await HF_FAST_CACHE.active_snapshot_dir(repo_id, repo_type="model")
        if not snapshot_dir:
            continue
        file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
        for fname in file_list:
            display = repo_id.split("/")[-1]
        lower_name = fname.lower()
        if lower_name.endswith(".gguf"):
            model_id = f"{repo_id}:{fname}"
            repos_with_single_files.add(repo_id)
            result.pop(repo_id, None)
            result[model_id] = ImageModel(
                id=repo_id,
                name=display,
                path=fname,
                provider=Provider.HuggingFace,
                supported_tasks=["image_to_image"],
            )
            continue

        if _is_single_file_diffusion_weight(fname):
            # Include single-file checkpoints even if repo has model_index.json
            # Prefer single-file versions over multi-file when available
            if _repo_supports_diffusion_checkpoint(model_info):
                model_id = f"{repo_id}:{fname}"
                repos_with_single_files.add(repo_id)
                # Remove multi-file entry if it exists - prefer single-file
                result.pop(repo_id, None)
                result[model_id] = ImageModel(
                    id=repo_id,
                    name=display,
                    path=fname,
                    provider=Provider.HuggingFace,
                    supported_tasks=["image_to_image"],
                )
                continue

        # Skip multi-file entry if repo has single files (prefer single-file versions)
        if repo_id in repos_with_single_files:
            continue

        result[repo_id] = ImageModel(
            id=repo_id,
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
    filtered_repos = await _filter_repos_by_metadata(
        pipeline_tag="text-to-image", tags=["mflux"]
    )
    result: dict[str, ImageModel] = {}

    for repo_id, _repo_dir, _info in filtered_repos:
        display = repo_id.split("/")[-1]
        result[repo_id] = ImageModel(
            id=repo_id,
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


async def delete_cached_hf_model(model_id: str) -> bool:
    """
    Deletes a model from the Hugging Face cache and the disk cache.

    Args:
        model_id (str): The ID of the model to delete.
    """
    # Use HfFastCache to resolve the repo root without walking the entire cache.
    repo_root = await HF_FAST_CACHE.repo_root(model_id, repo_type="model")
    if not repo_root:
        return False

    if not await asyncio.to_thread(os.path.exists, repo_root):
        return False

    shutil.rmtree(repo_root)

    # Purge all HuggingFace caches after successful deletion
    log.info("Purging HuggingFace model caches after model deletion")
    HF_FAST_CACHE.model_info_cache.delete_pattern("cached_hf_*")
    await HF_FAST_CACHE.invalidate(model_id, repo_type="model")
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
        cached = await get_image_to_image_models_from_hf_cache()
        for model in cached:
            if "IP-Adapter" in model.id:
                print(model.path)

    asyncio.run(main())
