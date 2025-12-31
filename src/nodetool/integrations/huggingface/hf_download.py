"""
Hugging Face Download Management Module

This module provides functionality for downloading files from Hugging Face repositories,
with support for multi-process downloads, progress tracking, and cancellation.
"""

import asyncio
import os
import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable, Literal

import httpx
from fastapi import WebSocket
from huggingface_hub import (
    _CACHED_NO_EXIST,
    HfApi,
    try_to_load_from_cache,
)
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile

from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface import async_downloader, hf_auth, hf_cache
from nodetool.ml.models.model_cache import ModelCache

log = get_logger(__name__)


def filter_repo_paths(*args, **kwargs):
    """Back-compat wrapper for tests and callers patching this symbol."""
    return hf_cache.filter_repo_paths(*args, **kwargs)


async def async_hf_download(*args, **kwargs):
    """Back-compat wrapper for tests and callers patching this symbol."""
    return await async_downloader.async_hf_download(*args, **kwargs)


@dataclass
class DownloadState:
    """Tracks the state of an individual download."""

    repo_id: str
    task: asyncio.Task | None = None
    monitor_task: asyncio.Task | None = None
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    downloaded_bytes: int = 0
    total_bytes: int = 0
    status: Literal["idle", "progress", "start", "error", "completed", "cancelled"] = "idle"
    downloaded_files: list[str] = field(default_factory=list)
    current_files: list[str] = field(default_factory=list)
    total_files: int = 0
    error_message: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class DownloadManager:
    """Manages concurrent downloads from Hugging Face repositories with WebSocket progress tracking."""

    active_websockets: set[WebSocket]
    downloads: dict[str, DownloadState]

    def __init__(self, token: str | None = None):
        """Initialize DownloadManager.

        Args:
            token: Optional HF_TOKEN. If not provided, will be fetched async when needed.
        """
        self.token = token
        if token:
            log.debug(f"DownloadManager initialized with HF_TOKEN (length: {len(token)} chars)")
            self.api = HfApi(token=token)
        else:
            log.debug("DownloadManager initialized without HF_TOKEN - will fetch async when needed")
            self.api = HfApi()
        self.logger = get_logger(__name__)
        self.downloads = {}
        self.active_websockets = set()
        self.model_cache = ModelCache("model_info")
        self._token_initialized = token is not None

    @classmethod
    async def create(cls, user_id: str | None = None):
        """Create DownloadManager with async token initialization.

        Args:
            user_id: Optional user ID for database secret lookup.
        """
        log.debug(f"DownloadManager.create: Creating DownloadManager with user_id={user_id}")
        token = await hf_auth.get_hf_token(user_id)
        log.debug(f"DownloadManager.create: Retrieved token for user_id={user_id}, token_present={token is not None}")
        return cls(token=token)

    def add_websocket(self, websocket: WebSocket):
        """Add a WebSocket connection to receive updates."""
        self.active_websockets.add(websocket)
        self.logger.debug(f"WebSocket added. Active connections: {len(self.active_websockets)}")

    def remove_websocket(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_websockets:
            self.active_websockets.remove(websocket)
            self.logger.debug(f"WebSocket removed. Active connections: {len(self.active_websockets)}")

    async def sync_state(self, websocket: WebSocket):
        """Send current state of all downloads to a specific WebSocket."""
        for _repo_id, state in self.downloads.items():
            await self.send_update(state.repo_id, None, specific_websocket=websocket)

    async def start_download(
        self,
        repo_id: str,
        path: str | None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        user_id: str | None = None,
        cache_dir: str | None = None,
    ):
        id = repo_id if path is None else f"{repo_id}/{path}"

        self.logger.info(f"start_download: Request for {id} with user_id={user_id}")

        if id in self.downloads:
            state = self.downloads[id]
            if state.status not in ["completed", "error", "cancelled"]:
                self.logger.warning(f"Download already in progress for: {id}")
                # Broadcast that it's already running
                await self.send_update(repo_id, path)
                return

        self.logger.info(f"Starting download task for: {id}")
        download_state = DownloadState(repo_id=repo_id)
        self.downloads[id] = download_state

        # Create background task for the download
        task = asyncio.create_task(
            self._download_task(repo_id, path, allow_patterns, ignore_patterns, user_id, cache_dir)
        )
        download_state.task = task

        # Start monitoring task
        download_state.monitor_task = asyncio.create_task(self.monitor_progress(repo_id, path))

    async def _download_task(
        self,
        repo_id: str,
        path: str | None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        user_id: str | None = None,
        cache_dir: str | None = None,
    ):
        id = repo_id if path is None else f"{repo_id}/{path}"
        download_state = self.downloads[id]

        try:
            await self.download_huggingface_repo(
                repo_id=repo_id,
                path=path,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_id=user_id,
                cache_dir=cache_dir,
            )
        except asyncio.CancelledError:
            self.logger.info(f"Download task cancelled: {id}")
            download_state.status = "cancelled"
            await self.send_update(repo_id, path)
        except Exception as e:
            self.logger.error(f"Error in download {id}: {e}")
            self.logger.error(traceback.format_exc())
            download_state.status = "error"
            download_state.error_message = str(e)
            # Ensure final update is sent
            await self.send_update(repo_id, path)
        finally:
            self.logger.info(f"Download process finished: {id}")
            # Ensure one last update if completed
            if download_state.status == "completed":
                await self.send_update(repo_id, path)

            # We don't delete from self.downloads immediately so user can see completion status
            # It will be cleaned up on next start_download or manually if we implement cleanup

    async def cancel_download(self, id: str):
        """Cancel an ongoing download."""
        if id not in self.downloads:
            self.logger.warning(f"Cancel requested for non-existent download: {id}")
            return

        self.logger.info(f"Cancelling download for: {id}")
        self.downloads[id].cancel.set()
        self.logger.debug(f"Set cancel event for {id}")
        self.downloads[id].status = "cancelled"
        await self.send_update(self.downloads[id].repo_id, None)  # Force update

    async def monitor_progress(self, repo_id: str, path: str | None):
        """Monitor progress and send updates periodically."""
        id = repo_id if path is None else f"{repo_id}/{path}"
        while True:
            if id not in self.downloads:
                break
            state = self.downloads[id]
            if state.status in ["completed", "error", "cancelled"]:
                break

            # If we have bytes, we are in progress
            if state.downloaded_bytes > 0 and state.status == "idle":
                state.status = "progress"

            await self.send_update(repo_id, path)
            await asyncio.sleep(0.1)

    async def send_update(self, repo_id: str, path: str | None = None, specific_websocket: WebSocket | None = None):
        """Send an update to WebSocket clients."""
        id = repo_id if path is None else f"{repo_id}/{path}"
        if id not in self.downloads:
            return
        state = self.downloads[id]

        # Create update dict
        update = {
            "status": state.status,
            "repo_id": state.repo_id,
            "path": path,
            "downloaded_bytes": state.downloaded_bytes,
            "total_bytes": state.total_bytes,
            "downloaded_files": len(state.downloaded_files),
            "current_files": state.current_files,
            "total_files": state.total_files,
        }
        if state.error_message:
            update["error"] = state.error_message

        targets = [specific_websocket] if specific_websocket else self.active_websockets

        for ws in targets:
            try:
                await ws.send_json(update)
            except Exception as e:
                self.logger.warning(f"Failed to send websocket update: {e}")
                # We might want to remove dead sockets here, but let the endpoint handle disconnects

    async def download_huggingface_repo(
        self,
        repo_id: str,
        path: str | None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        user_id: str | None = None,
        cache_dir: str | None = None,
    ):
        """Download files from a Hugging Face repository.

        Args:
            repo_id: HuggingFace repository ID
            path: Optional specific file path within the repo
            allow_patterns: Optional file patterns to include
            ignore_patterns: Optional file patterns to exclude
            user_id: Optional user ID for authentication
            cache_dir: Optional cache directory. If provided, downloads will go there
                instead of the default HF cache. Use for llama_cpp_model types.
        """
        id = repo_id if path is None else f"{repo_id}/{path}"
        state = self.downloads[id]

        # Ensure token is initialized
        if not self._token_initialized:
            self.token = await hf_auth.get_hf_token(user_id)
            if self.token:
                if isinstance(self.api, HfApi):
                    self.api = HfApi(token=self.token)
                self._token_initialized = True
            else:
                self.logger.warning(
                    f"No token found for user_id={user_id} after initialization attempt. Gated models will fail."
                )

        self.logger.info(f"Fetching file list for repo: {repo_id}")
        raw_files = self.api.list_repo_tree(repo_id, recursive=True)
        files = [file for file in raw_files if isinstance(file, RepoFile) or getattr(file, "type", None) == "file"]
        files = filter_repo_paths(files, allow_patterns, ignore_patterns)

        # Filter out files that already exist in the cache
        files_to_download = []
        for file in files:
            cache_path = try_to_load_from_cache(repo_id, file.path)

            if cache_path is None or cache_path is _CACHED_NO_EXIST:
                files_to_download.append(file)
                continue

            if not isinstance(cache_path, str | os.PathLike):
                self.logger.warning("Unexpected cache entry type for %s: %s", file.path, type(cache_path))
                files_to_download.append(file)
                continue

            if not os.path.exists(cache_path):
                files_to_download.append(file)
            else:
                state.downloaded_files.append(file.path)

        state.total_files = len(files_to_download)
        state.total_bytes = sum(getattr(file, "size", 0) for file in files_to_download)
        self.logger.info(
            f"download_huggingface_repo: Processing {len(files)} total files for {repo_id}. "
            f"Already cached: {len(files) - len(files_to_download)}. "
            f"To download: {state.total_files} files, {state.total_bytes} bytes."
        )

        # Initial update
        await self.send_update(repo_id, path)

        asyncio.get_running_loop()
        tasks = []
        self.logger.debug(
            f"download_huggingface_repo: Starting download of {len(files_to_download)} files for {repo_id} (user_id={user_id})"
        )

        def on_progress(delta: int, total: int | None):
            with state.lock:
                state.downloaded_bytes += delta
                if state.status == "idle":
                    state.status = "progress"

        async def run_single_download(file_path: str):
            # Use llama.cpp-specific download for llama_cpp_model type
            if cache_dir is not None:
                from nodetool.integrations.huggingface.llama_cpp_download import (
                    download_llama_cpp_model,
                )

                log.info(f"Downloading {repo_id}/{file_path} to {cache_dir}")
                local_path = await download_llama_cpp_model(
                    repo_id=repo_id,
                    filename=file_path,
                    token=self.token,
                    progress_callback=on_progress,
                    cancel_event=state.cancel,
                )
            else:
                log.info(f"Downloading {repo_id}/{file_path}")
                local_path = await async_hf_download(
                    repo_id=repo_id,
                    filename=file_path,
                    token=self.token,
                    progress_callback=on_progress,
                    cancel_event=state.cancel,
                )
            return file_path, local_path

        for file in files_to_download:
            if state.cancel.is_set():
                self.logger.info("Download cancelled before queuing task")
                break
            state.current_files.append(file.path)
            self.logger.debug(
                f"download_huggingface_repo: Queuing download of {file.path} for {repo_id} (user_id={user_id})"
            )
            tasks.append(run_single_download(file.path))

        # Use return_exceptions=True to catch exceptions from individual tasks
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions from individual tasks.
        errors: list[Exception] = []
        for result in completed_tasks:
            if isinstance(result, EntryNotFoundError):
                state.status = "error"
                state.error_message = str(result)
                self.logger.error(f"Download task failed with 404: {result}")
                return
            if isinstance(result, asyncio.CancelledError):
                self.logger.info(f"Download task for {id} received cancellation")
                continue
            if isinstance(result, Exception):
                errors.append(result)

        if any(isinstance(result, asyncio.CancelledError) for result in completed_tasks):
            state.status = "cancelled"
            self.logger.info(f"Download cancelled for repo: {repo_id}")
            await self.send_update(repo_id, path)
            return

        if errors:
            self.logger.error(f"Download task failed: {errors[0]}")
            raise errors[0]

        # Process successful downloads
        for result in completed_tasks:
            if isinstance(result, tuple):
                filename, local_path = result
                if local_path:
                    state.downloaded_files.append(filename)
                    self.logger.debug(f"Downloaded file: {filename}")

        state.status = "completed" if not state.cancel.is_set() else "cancelled"
        self.logger.info(f"Download {state.status} for repo: {repo_id}")

        if state.status == "completed":
            self.logger.info("Purging HuggingFace model caches after successful download")
            self.model_cache.delete_pattern("cached_hf_*")


# Singleton management
_download_managers: dict[str, DownloadManager] = {}
_manager_lock = asyncio.Lock()


async def get_download_manager(user_id: str) -> DownloadManager:
    """Get or create a singleton DownloadManager for a specific user."""
    async with _manager_lock:
        if user_id not in _download_managers:
            log.info(f"Creating new DownloadManager for user_id={user_id}")
            _download_managers[user_id] = await DownloadManager.create(user_id)
        return _download_managers[user_id]
