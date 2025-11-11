"""
Admin Operations Module for RunPod Handlers

This module provides shared admin operations for model management across
RunPod workflow and chat handlers. All operations support streaming where applicable.

Operations include:
- Hugging Face model downloads (with streaming progress)
- Ollama model downloads (with streaming progress)
- Cache management and scanning
- Model deletion

Usage:
    from nodetool.deploy.admin_operations import handle_admin_operation

    result = await handle_admin_operation(job_input)
    # For streaming operations, iterate over the async generator:
    async for chunk in handle_admin_operation(job_input):
        yield chunk
"""

import os
import asyncio
from typing import AsyncGenerator
from huggingface_hub import (
    hf_hub_download,
    scan_cache_dir,
    HfApi,
    try_to_load_from_cache,
)
from huggingface_hub.hf_api import RepoFile
from nodetool.integrations.huggingface.huggingface_models import delete_cached_hf_model
from nodetool.config.logging_config import get_logger
from nodetool.chat.ollama_service import get_ollama_client
from nodetool.integrations.huggingface.huggingface_cache import filter_repo_paths
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
from nodetool.security.secret_helper import get_secret_sync

logger = get_logger(__name__)


def get_hf_token() -> str | None:
    """Get HF_TOKEN from environment variables or secrets.
    
    Returns:
        HF_TOKEN if available, None otherwise.
    """
    token = get_secret_sync("HF_TOKEN")
    if token:
        # Check if it came from environment or database
        if os.environ.get("HF_TOKEN"):
            logger.debug("HF_TOKEN found in environment variables (admin_operations)")
        else:
            logger.debug("HF_TOKEN found in database secrets (admin_operations)")
        return token
    
    # Fallback to direct environment check
    token = os.environ.get("HF_TOKEN")
    if token:
        logger.debug("HF_TOKEN found in environment variables - direct check (admin_operations)")
    else:
        logger.debug("HF_TOKEN not found in environment or database secrets (admin_operations)")
    return token


class AdminDownloadManager:
    """Download manager for admin operations that yields progress updates without WebSocket dependency"""

    def __init__(self):
        # Use HF_TOKEN from secrets if available for gated model downloads
        token = get_hf_token()
        if token:
            logger.debug(f"AdminDownloadManager initialized with HF_TOKEN (token length: {len(token)} chars)")
            self.api = HfApi(token=token)
        else:
            logger.debug("AdminDownloadManager initialized without HF_TOKEN - gated models may not be accessible")
            self.api = HfApi()
        self.process_pool = ThreadPoolExecutor(max_workers=4)
        self.manager = Manager()
        self.token = token

    async def download_with_progress(
        self,
        repo_id: str,
        cache_dir: str = "/app/.cache/huggingface/hub",
        file_path: str | None = None,
        ignore_patterns: list | None = None,
        allow_patterns: list | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Download HuggingFace model with detailed progress updates"""

        try:
            logger.info(f"Starting HF model download with progress: {repo_id}")

            # Send initial status
            yield {
                "status": "starting",
                "repo_id": repo_id,
                "message": f"Starting download of {repo_id}",
            }

            if file_path:
                # Single file download
                yield {
                    "status": "progress",
                    "repo_id": repo_id,
                    "message": f"Downloading single file: {file_path}",
                    "current_file": file_path,
                }
                # Use HF_TOKEN from secrets if available for gated model downloads
                if self.token:
                    logger.debug(f"AdminDownloadManager: Downloading single file {repo_id}/{file_path} with HF_TOKEN (token length: {len(self.token)} chars)")
                else:
                    logger.debug(f"AdminDownloadManager: Downloading single file {repo_id}/{file_path} without HF_TOKEN - gated models may not be accessible")
                local_path = hf_hub_download(repo_id, file_path, cache_dir=cache_dir, token=self.token)
                yield {
                    "status": "completed",
                    "repo_id": repo_id,
                    "local_path": local_path,
                    "message": f"Successfully downloaded {repo_id}/{file_path}",
                }
                return

            # Repository download with progress tracking
            yield {
                "status": "progress",
                "repo_id": repo_id,
                "message": "Fetching file list...",
            }

            # Get file list
            files = self.api.list_repo_tree(repo_id, recursive=True)
            files = [file for file in files if isinstance(file, RepoFile)]
            files = filter_repo_paths(files, allow_patterns, ignore_patterns)

            # Filter out cached files
            files_to_download = []
            cached_files = []
            for file in files:
                cache_path = try_to_load_from_cache(repo_id, file.path)
                if cache_path is None or not os.path.exists(cache_path):
                    files_to_download.append(file)
                else:
                    cached_files.append(file.path)

            total_files = len(files_to_download)
            total_size = sum(file.size for file in files_to_download)

            yield {
                "status": "progress",
                "repo_id": repo_id,
                "message": f"Found {total_files} files to download, {len(cached_files)} already cached",
                "total_files": total_files,
                "total_size": total_size,
                "cached_files": len(cached_files),
            }

            if total_files == 0:
                yield {
                    "status": "completed",
                    "repo_id": repo_id,
                    "message": f"All files already cached for {repo_id}",
                    "total_files": 0,
                    "cached_files": len(cached_files),
                }
                return

            # Download files with progress
            downloaded_files = []
            downloaded_size = 0

            for i, file in enumerate(files_to_download):
                yield {
                    "status": "progress",
                    "repo_id": repo_id,
                    "message": f"Downloading {file.path}",
                    "current_file": file.path,
                    "file_progress": i + 1,
                    "total_files": total_files,
                    "downloaded_size": downloaded_size,
                    "total_size": total_size,
                }

                try:
                    # Download individual file - use HF_TOKEN from secrets if available for gated model downloads
                    if self.token:
                        logger.debug(f"AdminDownloadManager: Downloading file {repo_id}/{file.path} with HF_TOKEN (token length: {len(self.token)} chars)")
                    else:
                        logger.debug(f"AdminDownloadManager: Downloading file {repo_id}/{file.path} without HF_TOKEN - gated models may not be accessible")
                    local_path = hf_hub_download(
                        repo_id=repo_id, filename=file.path, cache_dir=cache_dir, token=self.token
                    )
                    downloaded_files.append(file.path)
                    downloaded_size += file.size

                    yield {
                        "status": "progress",
                        "repo_id": repo_id,
                        "message": f"Downloaded {file.path}",
                        "current_file": file.path,
                        "file_progress": i + 1,
                        "total_files": total_files,
                        "downloaded_files": len(downloaded_files),
                        "downloaded_size": downloaded_size,
                        "total_size": total_size,
                    }

                except Exception as e:
                    logger.error(f"Error downloading file {file.path}: {e}")
                    yield {
                        "status": "progress",
                        "repo_id": repo_id,
                        "message": f"Error downloading {file.path}: {str(e)}",
                        "current_file": file.path,
                        "error_file": file.path,
                    }

            # Final completion message
            yield {
                "status": "completed",
                "repo_id": repo_id,
                "message": f"Successfully downloaded {len(downloaded_files)}/{total_files} files for {repo_id}",
                "downloaded_files": len(downloaded_files),
                "total_files": total_files,
                "total_size": total_size,
                "downloaded_size": downloaded_size,
            }

        except Exception as e:
            logger.error(f"Error in HF model download {repo_id}: {e}")
            yield {
                "status": "error",
                "repo_id": repo_id,
                "error": str(e),
                "message": f"Error downloading {repo_id}: {str(e)}",
            }


def convert_file_info(file_info):
    """Convert HuggingFace file info to serializable dict"""
    return {
        "file_name": file_info.file_name,
        "size_on_disk": file_info.size_on_disk,
        "file_path": str(file_info.file_path),
        "blob_path": str(file_info.blob_path),
    }


def convert_revision_info(revision_info):
    """Convert HuggingFace revision info to serializable dict"""
    return {
        "commit_hash": revision_info.commit_hash,
        "size_on_disk": revision_info.size_on_disk,
        "snapshot_path": str(revision_info.snapshot_path),
        "files": [convert_file_info(f) for f in revision_info.files],
    }


def convert_repo_info(repo_info):
    """Convert HuggingFace repo info to serializable dict"""
    return {
        "repo_id": repo_info.repo_id,
        "repo_type": repo_info.repo_type,
        "repo_path": str(repo_info.repo_path),
        "size_on_disk": repo_info.size_on_disk,
        "nb_files": repo_info.nb_files,
        "revisions": [convert_revision_info(r) for r in repo_info.revisions],
    }


def convert_cache_info(cache_info):
    """Convert HuggingFace cache info to serializable dict"""
    return {
        "size_on_disk": cache_info.size_on_disk,
        "repos": [convert_repo_info(r) for r in cache_info.repos],
        "warnings": [str(w) for w in cache_info.warnings],
    }


async def stream_ollama_model_pull(model_name: str) -> AsyncGenerator[dict, None]:
    """
    Stream Ollama model download progress.

    Args:
        model_name (str): Name of the Ollama model to download

    Yields:
        str: JSON-encoded progress updates
    """
    try:
        ollama = get_ollama_client()
        logger.info(f"Starting Ollama model pull: {model_name}")

        # Send initial status
        yield {
            "status": "starting",
            "model": model_name,
            "message": f"Starting download of {model_name}",
        }

        # Some clients return an async generator directly; others return a coroutine
        res = ollama.pull(model_name, stream=True)
        # If coroutine, await to get the async iterator
        if asyncio.iscoroutine(res):  # type: ignore
            res = await res  # type: ignore

        # If callable (async generator factory), call it
        if callable(res):  # type: ignore
            res = res()  # type: ignore

        async for chunk in res:  # type: ignore
            data = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
            yield data

        yield {
            "status": "completed",
            "model": model_name,
            "message": f"Successfully downloaded {model_name}",
        }

    except Exception as e:
        logger.error(f"Error pulling Ollama model {model_name}: {e}")
        yield {"status": "error", "model": model_name, "error": str(e)}


async def stream_hf_model_download(
    repo_id: str,
    cache_dir: str = "/app/.cache/huggingface/hub",
    file_path: str | None = None,
    ignore_patterns: list | None = None,
    allow_patterns: list | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Stream Hugging Face model download progress using AdminDownloadManager.

    Args:
        repo_id (str): HuggingFace repository ID
        cache_dir (str): Cache directory path
        file_path (str, optional): Specific file to download
        ignore_patterns (list, optional): Patterns to ignore
        allow_patterns (list, optional): Patterns to allow

    Yields:
        str: JSON-encoded progress updates
    """
    download_manager = AdminDownloadManager()
    async for progress_update in download_manager.download_with_progress(
        repo_id=repo_id,
        cache_dir=cache_dir,
        file_path=file_path,
        ignore_patterns=ignore_patterns,
        allow_patterns=allow_patterns,
    ):
        yield progress_update


async def download_hf_model(
    repo_id: str,
    cache_dir: str = "/app/.cache/huggingface/hub",
    file_path: str | None = None,
    ignore_patterns: list | None = None,
    allow_patterns: list | None = None,
    stream: bool = True,
) -> AsyncGenerator[dict, None]:
    """Download HuggingFace model with optional streaming."""
    if not repo_id:
        raise ValueError("repo_id is required for HuggingFace download")

    if stream:
        async for chunk in stream_hf_model_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            file_path=file_path,
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns,
        ):
            yield chunk
    else:
        # Non-streaming download - still use the download manager but just return final result
        download_manager = AdminDownloadManager()
        final_result = None
        async for progress_update in download_manager.download_with_progress(
            repo_id=repo_id,
            cache_dir=cache_dir,
            file_path=file_path,
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns,
        ):
            final_result = progress_update

        if final_result:
            yield final_result


async def download_ollama_model(
    model_name: str, stream: bool = True
) -> AsyncGenerator[dict, None]:
    """Download Ollama model with optional streaming."""
    if not model_name:
        raise ValueError("model_name is required for Ollama download")

    if stream:
        async for chunk in stream_ollama_model_pull(model_name):
            yield chunk
    else:
        # Non-streaming download
        try:
            ollama = get_ollama_client()
            await ollama.pull(model_name)
            yield {
                "status": "completed",
                "model": model_name,
                "message": f"Successfully downloaded {model_name}",
            }
        except Exception as e:
            logger.error(f"Error downloading Ollama model {model_name}: {e}")
            yield {"status": "error", "model": model_name, "error": str(e)}


async def scan_hf_cache() -> AsyncGenerator[dict, None]:
    """Scan HuggingFace cache directory."""
    try:
        cache_info = scan_cache_dir()
        yield {"status": "completed", "cache_info": convert_cache_info(cache_info)}
    except Exception as e:
        logger.error(f"Error scanning cache: {e}")
        yield {"status": "error", "error": str(e)}


async def delete_hf_model(repo_id: str) -> AsyncGenerator[dict, None]:
    """Delete HuggingFace model from cache."""
    if not repo_id:
        raise ValueError("repo_id is required for HuggingFace model deletion")

    try:
        delete_cached_hf_model(repo_id)
        yield {
            "status": "completed",
            "repo_id": repo_id,
            "message": f"Successfully deleted {repo_id}",
        }
    except Exception as e:
        logger.error(f"Error deleting HF model {repo_id}: {e}")
        yield {"status": "error", "repo_id": repo_id, "error": str(e)}


async def calculate_cache_size(
    cache_dir: str = "/app/.cache/huggingface/hub",
) -> AsyncGenerator[dict, None]:
    """Calculate total cache size."""
    try:
        total_size = 0
        if os.path.exists(cache_dir):
            for dirpath, dirnames, filenames in os.walk(cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)

        size_gb = total_size / (1024**3)
        yield {
            "success": True,
            "cache_dir": cache_dir,
            "total_size_bytes": total_size,
            "size_gb": round(size_gb, 2),
        }
    except Exception as e:
        logger.error(f"Error calculating cache size: {e}")
        yield {"status": "error", "cache_dir": cache_dir, "error": str(e)}
