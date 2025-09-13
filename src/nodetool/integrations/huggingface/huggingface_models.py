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
from datetime import datetime
import httpx
from pydantic import Field
from huggingface_hub import scan_cache_dir
from typing import Any, List, Optional
from pydantic import BaseModel
import os
import shutil
import json
import hashlib
from pathlib import Path
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import CLASSNAME_TO_MODEL_TYPE, HuggingFaceModel
from nodetool.workflows.base_node import get_recommended_models

log = get_logger(__name__)

# Cache configuration
CACHE_VERSION = "1.0"
CACHE_EXPIRY_DAYS = int(os.environ.get("NODETOOL_CACHE_EXPIRY_DAYS", "7"))


def get_model_info_cache_directory() -> Path:
    """
    Get system-specific cache directory for nodetool's custom model info cache.

    Returns:
        Path: System-specific cache directory
    """
    try:
        # Try to use platformdirs if available
        import platformdirs

        cache_dir = platformdirs.user_cache_dir("nodetool", "nodetool")
    except ImportError:
        # Fallback to manual platform detection
        if os.name == "nt":  # Windows
            cache_dir = os.path.expandvars(r"%LOCALAPPDATA%\nodetool\Cache")
        elif os.name == "posix":
            if os.uname().sysname == "Darwin":  # macOS
                cache_dir = os.path.expanduser("~/Library/Caches/nodetool")
            else:  # Linux and other Unix-like
                cache_dir = os.path.expanduser("~/.cache/nodetool")
        else:
            # Fallback for unknown systems
            cache_dir = os.path.expanduser("~/.nodetool_cache")

    cache_path = Path(cache_dir) / "model_info_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_cache_file_path(model_id: str, cache_type: str = "model_info") -> Path:
    """
    Get the cache file path for a specific model's metadata.

    Args:
        model_id (str): The model ID
        cache_type (str): Type of cache (defaults to 'model_info')

    Returns:
        Path: Cache file path
    """
    # Create a safe filename from model_id
    safe_model_id = hashlib.md5(model_id.encode()).hexdigest()
    cache_dir = get_model_info_cache_directory()
    return cache_dir / f"{safe_model_id}_{cache_type}.json"


def is_cache_valid(cache_file: Path) -> bool:
    """
    Check if cache file is valid (exists and not expired).

    Args:
        cache_file (Path): Path to cache file

    Returns:
        bool: True if cache is valid
    """
    if not cache_file.exists():
        return False

    try:
        # Check if file is older than CACHE_EXPIRY_DAYS
        file_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        return file_age < (CACHE_EXPIRY_DAYS * 24 * 3600)
    except Exception:
        return False


def read_cache_file(cache_file: Path) -> dict[str, Any] | None:
    """
    Read and parse cache file.

    Args:
        cache_file (Path): Path to cache file

    Returns:
        dict | None: Cached data or None if invalid
    """
    try:
        if not is_cache_valid(cache_file):
            return None

        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify cache version
        if data.get("version") != CACHE_VERSION:
            return None

        return data.get("data")  # type: ignore[no-any-return]
    except Exception as e:
        log.debug(f"Failed to read cache file {cache_file}: {e}")
        return None


def write_cache_file(cache_file: Path, data: Any) -> None:
    """
    Write data to cache file with size verification.

    Args:
        cache_file (Path): Path to cache file
        data (Any): Data to cache
    """
    try:
        cache_data = {
            "version": CACHE_VERSION,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }

        # Ensure directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON string first to check size
        json_str = json.dumps(cache_data, indent=2, default=str)
        expected_size = len(json_str.encode("utf-8"))

        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(json_str)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        # Verify file was written completely
        actual_size = cache_file.stat().st_size
        if actual_size != expected_size:
            log.warning(
                f"Cache file size mismatch for {cache_file}: expected {expected_size}, got {actual_size}"
            )
            cache_file.unlink()  # Remove potentially corrupted file

    except Exception as e:
        log.debug(f"Failed to write cache file {cache_file}: {e}")


def delete_cache_file(model_id: str, cache_type: str = "model_info") -> None:
    """
    Delete cache file for a specific model's metadata.

    Args:
        model_id (str): The model ID
        cache_type (str): Type of cache (defaults to 'model_info')
    """
    try:
        cache_file = get_cache_file_path(model_id, cache_type)
        if cache_file.exists():
            cache_file.unlink()
    except Exception as e:
        log.debug(f"Failed to delete cache file for {model_id}: {e}")


def cleanup_expired_cache() -> int:
    """
    Clean up expired cache files.

    Returns:
        int: Number of files removed
    """
    removed_count = 0
    try:
        cache_dir = get_model_info_cache_directory()
        if not cache_dir.exists():
            return 0

        for cache_file in cache_dir.glob("*.json"):
            if not is_cache_valid(cache_file):
                try:
                    cache_file.unlink()
                    removed_count += 1
                    log.debug(f"Removed expired cache file: {cache_file}")
                except Exception as e:
                    log.debug(f"Failed to remove expired cache file {cache_file}: {e}")

    except Exception as e:
        log.debug(f"Failed to cleanup expired cache: {e}")

    return removed_count


class Sibling(BaseModel):
    rfilename: str


class ModelInfo(BaseModel):
    _id: str
    id: str
    modelId: str
    author: str
    sha: str
    lastModified: datetime
    private: bool
    disabled: bool
    gated: bool | str
    pipeline_tag: str | None = None
    tags: List[str]
    downloads: int
    library_name: str | None = None
    likes: int
    the_model_index: Optional[Any] = Field(None, alias="model-index")
    config: dict | None = None
    cardData: dict | None = None
    siblings: List[Sibling] | None = None
    spaces: List[str] | None = None
    createdAt: datetime


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
    using httpx
    https://huggingface.co/api/models/{model_id}

    Args:
        model_id (str): The ID of the model to fetch.

    Returns:
        ModelInfo: The model info.
    """
    cache_file = get_cache_file_path(model_id, "model_info")
    cached_data = read_cache_file(cache_file)
    if cached_data is not None:
        try:
            # Reconstruct ModelInfo from cached dict
            return ModelInfo(**cached_data) if isinstance(cached_data, dict) else None
        except Exception as e:
            log.debug(f"Failed to deserialize cached model info for {model_id}: {e}")
            # Invalid cache data, continue to fetch from API

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"https://huggingface.co/api/models/{model_id}")
        except httpx.ConnectError:
            log.info("Huggingface not reachable")
            return None

        if response.status_code != 200:
            write_cache_file(cache_file, None)
            return None

        response_data = response.json()
        model_info = ModelInfo(**response_data)

        # Cache the model info as dict for JSON serialization
        write_cache_file(cache_file, response_data)
        return model_info


class CachedFileInfo(BaseModel):
    file_name: str
    size_on_disk: int


class CachedModel(BaseModel):
    repo_id: str
    repo_type: str
    path: str
    size_on_disk: int
    has_model_index: bool = False
    the_model_type: Optional[str] = None
    the_model_info: ModelInfo | None = None
    readme: str | None = None
    cached_files: List[CachedFileInfo] = []


def model_type_from_model_info(
    recommended_models: dict[str, list[HuggingFaceModel]],
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
    return None


async def read_cached_hf_models(
    load_model_info: bool = True,
) -> List[CachedModel]:
    """
    Reads all models from the Hugging Face cache.

    Returns:
        List[CachedModel]: A list of CachedModel objects found in the cache.
    """
    # Offload scanning HF cache to a thread (filesystem heavy)
    cache_info = await asyncio.to_thread(scan_cache_dir)
    model_repos = [repo for repo in cache_info.repos if repo.repo_type == "model"]
    recommended_models = get_recommended_models()
    if load_model_info:
        model_infos = await asyncio.gather(
            *[fetch_model_info(repo.repo_id) for repo in model_repos]
        )
    else:
        model_infos = [None] * len(model_repos)

    def has_model_index(model_info: ModelInfo | None) -> bool:
        if model_info is None:
            return False
        if model_info.siblings is None:
            return False
        for sibling in model_info.siblings:
            if sibling.rfilename == "model_index.json":
                return True
        return False

    models = []
    for repo, model_info in zip(model_repos, model_infos):
        # Get cached files from all revisions
        cached_files = []
        for revision in repo.revisions:
            for file_info in revision.files:
                cached_files.append(CachedFileInfo(
                    file_name=file_info.file_name,
                    size_on_disk=file_info.size_on_disk,
                ))
        
        models.append(CachedModel(
            repo_id=repo.repo_id,
            repo_type=repo.repo_type,
            path=str(repo.repo_path),
            size_on_disk=repo.size_on_disk,
            has_model_index=has_model_index(model_info),
            the_model_info=model_info,
            the_model_type=model_type_from_model_info(
                recommended_models, repo.repo_id, model_info
            ),
            cached_files=cached_files,
        ))
    return models


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
                # Remove model info from our custom disk cache
                # README files are handled by HF hub cache (deleted above)
                delete_cache_file(model_id, "model_info")
                return True
    return False


if __name__ == "__main__":

    async def main() -> None:
        models = await read_cached_hf_models()
        for model in models:
            if model.the_model_info is not None:
                print(model.repo_id, model.the_model_info.tags)

    asyncio.run(main())
