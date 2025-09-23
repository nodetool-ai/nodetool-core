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
import aiofiles
from dataclasses import asdict
from datetime import datetime
from nodetool.types.model import CachedFileInfo, UnifiedModel
from huggingface_hub import scan_cache_dir, HfApi, ModelInfo, try_to_load_from_cache
from typing import Any, List
import os
import shutil
import json
from pathlib import Path
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    CLASSNAME_TO_MODEL_TYPE,
    HuggingFaceModel,
    LanguageModel,
    Provider,
)
from nodetool.workflows.recommended_models import get_recommended_models

log = get_logger(__name__)

# Cache configuration
CACHE_VERSION = "1.0"
CACHE_EXPIRY_DAYS = int(os.environ.get("NODETOOL_CACHE_EXPIRY_DAYS", "7"))


def size_on_disk(model_info: ModelInfo) -> int:
    return sum(sib.size for sib in (model_info.siblings or []) if sib.size is not None)


def has_model_index(model_info: ModelInfo) -> bool:
    return any(
        sib.rfilename == "model_index.json" for sib in (model_info.siblings or [])
    )


def unified_model(
    model: HuggingFaceModel,
    model_info: ModelInfo | None = None,
    size: int | None = None,
) -> UnifiedModel | None:
    if model_info is None or model_info.siblings is None:
        try:
            model_info = HfApi().model_info(model.repo_id, files_metadata=True)
        except Exception as e:
            log.debug(f"Failed to fetch model info for {model.repo_id}: {e}")
            return None
    if model_info is None:
        return None
    # cache_path = try_to_load_from_cache(
    #     model.repo_id, model.path if model.path is not None else "config.json"
    return UnifiedModel(
        id=model.repo_id,
        repo_id=model.repo_id,
        path=model.path,
        type=model.type,
        name=model.repo_id,
        cache_path=None,
        allow_patterns=model.allow_patterns,
        ignore_patterns=model.ignore_patterns,
        description=None,
        readme=None,
        size_on_disk=size or size_on_disk(model_info),
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
        try_to_load_from_cache,
        hf_hub_download,
        _CACHED_NO_EXIST,
    )

    # First, try to load from the HF hub cache
    cached_path = try_to_load_from_cache(repo_id=model_id, filename="README.md")

    if isinstance(cached_path, str):
        # File exists in cache, read and return it
        try:
            with open(cached_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            log.debug(f"Failed to read cached README for {model_id}: {e}")
    elif cached_path is _CACHED_NO_EXIST:
        # Non-existence is cached, return None immediately
        return None

    # File not in cache, try to download it
    try:
        readme_path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: hf_hub_download(
                repo_id=model_id, filename="README.md", repo_type="model"
            ),
        )
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        log.debug(f"Failed to download README for {model_id}: {e}")
        return None


async def fetch_model_info(model_id: str) -> ModelInfo | None:
    """
    Fetches model info from the Hugging Face API or cache
    using the HfApi client
    https://huggingface.co/api/models/{model_id}

    Args:
        model_id (str): The ID of the model to fetch.

    Returns:
        ModelInfo: The model info.
    """
    # Use HfApi to fetch model info
    api = HfApi()
    try:
        model_info: ModelInfo = await asyncio.get_event_loop().run_in_executor(
            None, lambda: api.model_info(model_id, files_metadata=True)
        )
    except Exception as e:
        log.debug(f"Failed to fetch model info for {model_id}: {e}")
        return None

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
            model_info.config["diffusers"]["_class_name"], None  # type: ignore[no-any-return]
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


async def read_cached_hf_files() -> List[CachedFileInfo]:
    """
    Reads all models from the Hugging Face cache.

    Returns:
        List[CachedFileInfo]: A list of CachedFileInfo objects found in the cache.
    """
    # Offload scanning HF cache to a thread (filesystem heavy)
    cache_info = await asyncio.to_thread(scan_cache_dir)
    model_repos = [repo for repo in cache_info.repos if repo.repo_type == "model"]

    for repo in model_repos:
        # Get cached files from all revisions
        cached_files = []
        for revision in repo.revisions:
            for file_info in revision.files:
                cached_files.append(
                    CachedFileInfo(
                        repo_id=repo.repo_id,
                        file_name=file_info.file_name,
                        size_on_disk=file_info.size_on_disk,
                    )
                )
    return cached_files


async def read_cached_hf_models() -> List[UnifiedModel]:
    """
    Reads all models from the Hugging Face cache.

    Returns:
        List[UnifiedModel]: A list of UnifiedModel objects found in the cache.
    """
    # Offload scanning HF cache to a thread (filesystem heavy)
    cache_info = await asyncio.to_thread(scan_cache_dir)
    model_repos = [repo for repo in cache_info.repos if repo.repo_type == "model"]
    recommended_models = get_recommended_models()
    model_infos = await asyncio.gather(
        *[fetch_model_info(repo.repo_id) for repo in model_repos]
    )

    models: list[UnifiedModel] = []
    for repo, model_info in zip(model_repos, model_infos):
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
                has_model_index=has_model_index(model_info) if model_info else False,
                repo_id=repo.repo_id,
                path=None,
                size_on_disk=repo.size_on_disk,
                downloads=model_info.downloads if model_info else None,
                likes=model_info.likes if model_info else None,
                trending_score=model_info.trending_score if model_info else None,
            )
        )
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
    cached = await read_cached_hf_files()
    seen: set[str] = set()
    results: list[LanguageModel] = []

    for f in cached:
        fname = f.file_name
        if not fname:
            continue
        if not fname.lower().endswith(".gguf"):
            continue
        model_id = f"{f.repo_id}:{fname}"
        if model_id in seen:
            continue
        seen.add(model_id)
        display = f"{f.repo_id.split('/')[-1]} • {fname}"
        results.append(
            LanguageModel(
                id=model_id,
                name=display,
                provider=Provider.LlamaCpp,
            )
        )

    # Sort for stability: by repo then filename
    results.sort(key=lambda m: (m.id.split(":", 1)[0], m.id))
    return results


async def get_mlx_language_models_from_hf_cache() -> List[LanguageModel]:
    """
    Return LanguageModel entries for cached Hugging Face repos that look suitable
    for MLX runtime (Apple Silicon). We use heuristics:

    - Prefer repos under the "mlx-community" org
    - Additionally include repos that have tags containing "mlx" in their model info

    Each qualifying repo yields a LanguageModel with id "<repo_id>" (no file suffix),
    because MLX loaders typically resolve the correct shard/quantization internally.

    Returns:
        List[LanguageModel]: MLX-compatible models discovered in the HF cache
    """
    cached = await read_cached_hf_models()
    results: list[LanguageModel] = []
    result: dict[str, LanguageModel] = {}

    for model in cached:
        has_mlx_tag = any("mlx" in t for t in model.tags or [])

        if not has_mlx_tag or model.repo_id is None:
            continue

        display = model.repo_id.split("/")[-1]
        result[model.repo_id] = LanguageModel(
            id=model.repo_id,
            name=display,
            provider=Provider.MLX,
        )

    return list(result.values())


async def _fetch_models_by_author(**kwargs) -> list[ModelInfo]:
    """Fetch models list from HF API for a given author using HFAPI.

    Returns raw model dicts from the public API.
    """
    api = HfApi()
    # Run the blocking call in a thread executor
    models = await asyncio.get_event_loop().run_in_executor(
        None, lambda: api.list_models(**kwargs)
    )
    return list(models)


async def get_gguf_language_models_from_authors(
    authors: list[str],
) -> List[UnifiedModel | None]:
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
    results = await asyncio.gather(
        *(
            _fetch_models_by_author(
                author=a,
                limit=200,
                sort="trending_score",
                tags="gguf",
            )
            for a in authors
        )
    )
    repos = [item for sublist in results for item in sublist]
    model_infos = await asyncio.gather(*[fetch_model_info(repo.id) for repo in repos])
    entries: list[UnifiedModel | None] = []
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
            entries.append(
                unified_model(
                    HuggingFaceModel(type="llama_cpp", repo_id=info.id, path=fname),
                    info,
                    sib.size,
                )
            )

    # Sort for stability: repo then filename
    entries = [entry for entry in entries if entry is not None]
    return entries


async def get_mlx_language_models_from_authors(
    authors: list[str],
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
    results = await asyncio.gather(
        *(
            _fetch_models_by_author(
                author=a, limit=200, sort="trending_score", tags="mlx"
            )
            for a in authors
        )
    )
    model_infos = [item for sublist in results for item in sublist]

    # Produce one LanguageModel per repo id
    entries: list[UnifiedModel | None] = []
    for info in model_infos:
        entries.append(
            unified_model(HuggingFaceModel(type="mlx", repo_id=info.id), info)
        )

    # Stable order
    return [entry for entry in entries if entry is not None]


def delete_cached_hf_model(model_id: str) -> bool:
    """
    Deletes a model from the Hugging Face cache and the disk cache.

    Args:
        model_id (str): The ID of the model to delete.
    """
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_type == "model" and repo.repo_id == model_id:
            if os.path.exists(repo.repo_path):
                shutil.rmtree(repo.repo_path)
                return True
    return False


GGUF_AUTHORS = [
    "unsloth",
    "ggml-org",
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
]
MLX_AUTHORS = ["mlx-community"]


async def save_gguf_language_models_to_file() -> None:
    models = await get_gguf_language_models_from_authors(GGUF_AUTHORS)
    current_dir = Path(__file__).parent
    cache_file = current_dir / "gguf_language_models.json"
    with open(cache_file, "w") as f:
        json.dump(
            [model.model_dump() for model in models if model is not None], f, indent=2
        )


async def save_mlx_language_models_to_file() -> None:
    models = await get_mlx_language_models_from_authors(MLX_AUTHORS)
    current_dir = Path(__file__).parent
    cache_file = current_dir / "mlx_language_models.json"
    with open(cache_file, "w") as f:
        json.dump([model.model_dump() for model in models], f, indent=2)


async def load_gguf_language_models_from_file() -> List[UnifiedModel]:
    current_dir = Path(__file__).parent
    cache_file = current_dir / "gguf_language_models.json"
    async with aiofiles.open(cache_file, "r") as f:
        content = await f.read()
        return [UnifiedModel(**model) for model in json.loads(content)]


async def load_mlx_language_models_from_file() -> List[UnifiedModel]:
    current_dir = Path(__file__).parent
    cache_file = current_dir / "mlx_language_models.json"
    async with aiofiles.open(cache_file, "r") as f:
        content = await f.read()
        return [UnifiedModel(**model) for model in json.loads(content)]


if __name__ == "__main__":
    asyncio.run(save_gguf_language_models_to_file())
    asyncio.run(save_mlx_language_models_to_file())
