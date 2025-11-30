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
from typing import Literal, Callable

from fastapi import WebSocket
from huggingface_hub import (
    HfApi,
    _CACHED_NO_EXIST,
    try_to_load_from_cache,
)
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile

from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface import hf_auth
from nodetool.integrations.huggingface.hf_cache import filter_repo_paths
from nodetool.integrations.huggingface.async_downloader import async_hf_download
from nodetool.ml.models.model_cache import ModelCache

log = get_logger(__name__)


@dataclass
class DownloadState:
    """Tracks the state of an individual download."""
    repo_id: str
    websocket: WebSocket
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    downloaded_bytes: int = 0
    total_bytes: int = 0
    status: Literal["idle", "progress", "start", "error", "completed", "cancelled"] = (
        "idle"
    )
    downloaded_files: list[str] = field(default_factory=list)
    current_files: list[str] = field(default_factory=list)
    total_files: int = 0
    error_message: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)





class DownloadManager:
    """Manages concurrent downloads from Hugging Face repositories with WebSocket progress tracking."""

    websocket: WebSocket
    cancel: asyncio.Event
    downloaded_bytes: int = 0
    total_bytes: int = 0
    repo_id: str = ""
    status: (
        Literal["idle"]
        | Literal["progress"]
        | Literal["start"]
        | Literal["error"]
        | Literal["completed"]
        | Literal["cancelled"]
    ) = "idle"

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
        self.downloads: dict[str, DownloadState] = {}
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

    async def start_download(
        self,
        repo_id: str,
        path: str | None,
        websocket: WebSocket,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        user_id: str | None = None,
    ):
        id = repo_id if path is None else f"{repo_id}/{path}"

        self.logger.info(f"start_download: Starting download for {id} with user_id={user_id}")

        if id in self.downloads:
            self.logger.warning(f"Download already in progress for: {id}")
            await websocket.send_json(
                {"status": "error", "message": "Download already in progress"}
            )
            return

        self.logger.info(f"Starting download for: {id}")
        download_state = DownloadState(repo_id=repo_id, websocket=websocket)
        self.downloads[id] = download_state

        # Start monitoring task
        monitor_task = asyncio.create_task(self.monitor_progress(repo_id, path))

        try:
            await self.download_huggingface_repo(
                repo_id=repo_id,
                path=path,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_id=user_id,
            )
        except Exception as e:
            self.logger.error(f"Error in download {id}: {e}")
            self.logger.error(traceback.format_exc())
            download_state.status = "error"
            download_state.error_message = str(e)
            # Ensure final update is sent
            await self.send_update(repo_id, path)
            raise  # Re-raise to bubble up to the endpoint
        finally:
            self.logger.info(f"Download process finished: {id}")
            # Stop monitoring
            if not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Ensure one last update if completed
            if download_state.status == "completed":
                await self.send_update(repo_id, path)
                
            del self.downloads[id]

    async def cancel_download(self, id: str):
        """Cancel an ongoing download."""
        if id not in self.downloads:
            return

        self.logger.info(f"Cancelling download for: {id}")
        self.downloads[id].cancel.set()
        self.downloads[id].status = "cancelled"

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

    async def send_update(self, repo_id: str, path: str | None = None):
        """Send an update to the WebSocket client."""
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
            
        try:
            await state.websocket.send_json(update)
        except Exception as e:
             self.logger.warning(f"Failed to send websocket update: {e}")

    async def download_huggingface_repo(
        self,
        repo_id: str,
        path: str | None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        user_id: str | None = None,
    ):
        """Download files from a Hugging Face repository."""
        id = repo_id if path is None else f"{repo_id}/{path}"
        state = self.downloads[id]

        self.logger.debug(f"download_huggingface_repo: Starting download for {repo_id} with user_id={user_id}")

        # Ensure token is initialized
        if not self._token_initialized:
            self.logger.debug(f"download_huggingface_repo: Token not initialized, fetching with user_id={user_id}")
            self.token = await hf_auth.get_hf_token(user_id)
            if self.token:
                if isinstance(self.api, HfApi):
                    self.api = HfApi(token=self.token)
                self._token_initialized = True
                self.logger.debug(f"download_huggingface_repo: Token initialized for user_id={user_id} (token length: {len(self.token)} chars)")
            else:
                self.logger.debug(f"download_huggingface_repo: No token found for user_id={user_id}")

        # Log HF_TOKEN presence for debugging
        if self.token:
            self.logger.debug(f"download_huggingface_repo: Starting download for {repo_id} with HF_TOKEN (token length: {len(self.token)} chars, user_id={user_id})")
        else:
            self.logger.debug(f"download_huggingface_repo: Starting download for {repo_id} without HF_TOKEN - gated models may not be accessible (user_id={user_id})")

        self.logger.info(f"Fetching file list for repo: {repo_id} (user_id={user_id})")
        raw_files = self.api.list_repo_tree(repo_id, recursive=True)
        files = [
            file
            for file in raw_files
            if isinstance(file, RepoFile) or getattr(file, "type", None) == "file"
        ]
        files = filter_repo_paths(files, allow_patterns, ignore_patterns)

        # Filter out files that already exist in the cache
        files_to_download = []
        for file in files:
            cache_path = try_to_load_from_cache(repo_id, file.path)

            if cache_path is None or cache_path is _CACHED_NO_EXIST:
                files_to_download.append(file)
                continue

            if not isinstance(cache_path, (str, os.PathLike)):
                self.logger.warning(
                    "Unexpected cache entry type for %s: %s", file.path, type(cache_path)
                )
                files_to_download.append(file)
                continue

            if not os.path.exists(cache_path):
                files_to_download.append(file)
            else:
                state.downloaded_files.append(file.path)

        state.total_files = len(files_to_download)
        state.total_bytes = sum(getattr(file, "size", 0) for file in files_to_download)
        self.logger.info(
            f"Total files to download: {state.total_files}, Total size: {state.total_bytes} bytes"
        )
        
        # Initial update
        await self.send_update(repo_id, path)

        loop = asyncio.get_running_loop()
        tasks = []
        self.logger.debug(f"download_huggingface_repo: Starting download of {len(files_to_download)} files for {repo_id} (user_id={user_id})")

        def on_progress(delta: int, total: int | None):
            with state.lock:
                state.downloaded_bytes += delta
                if state.status == "idle":
                    state.status = "progress"

        async def run_single_download(file_path: str):
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
                self.logger.info("Download cancelled")
                break
            state.current_files.append(file.path)
            self.logger.debug(f"download_huggingface_repo: Queuing download of {file.path} for {repo_id} (user_id={user_id})")
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
            if isinstance(result, Exception):
                errors.append(result)

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
            self.logger.info(
                "Purging HuggingFace model caches after successful download"
            )
            self.model_cache.delete_pattern("cached_hf_*")
