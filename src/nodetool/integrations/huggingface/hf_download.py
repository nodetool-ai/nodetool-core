"""
Hugging Face Download Management Module

This module provides functionality for downloading files from Hugging Face repositories,
with support for multi-process downloads, progress tracking, and cancellation.
"""

import asyncio
import importlib
import os
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Manager
from queue import Empty, Queue
from typing import Literal

import huggingface_hub.file_download
from fastapi import WebSocket
from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
from huggingface_hub.hf_api import RepoFile

from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface import hf_auth
from nodetool.integrations.huggingface.hf_cache import filter_repo_paths
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


# This will be used to send progress updates to the client
# It will be set only in the sub processes
parent_queue = None


def download_file(repo_id: str, filename: str, queue: Queue, token: str | None = None):
    """Download a file from HuggingFace Hub.

    Note: This function runs in a separate process, so it cannot be async.
    The token must be passed in from the parent process.
    """
    global parent_queue
    parent_queue = queue

    # Token should be passed from parent process (cannot call async here)
    if token:
        log.debug(f"download_file: Downloading {repo_id}/{filename} with HF_TOKEN (token length: {len(token)} chars)")
    else:
        log.debug(f"download_file: Downloading {repo_id}/{filename} without HF_TOKEN - gated models may not be accessible")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
    )
    return filename, local_path


class CustomTqdm(huggingface_hub.file_download.tqdm):  # type: ignore
    """Custom progress bar that sends updates through a queue for WebSocket integration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if parent_queue and "initial" in kwargs:
            parent_queue.put({"n": kwargs["initial"]})

    def update(self, n=1):
        if n and parent_queue:
            parent_queue.put({"n": n})
        super().update(n)


# Replace the tqdm used by huggingface_hub
huggingface_hub.file_download.tqdm = CustomTqdm  # type: ignore

# `huggingface_hub.utils.tqdm` is a class exported at package import time which
# also needs to be replaced so that `_get_progress_bar_context` uses our custom
# implementation. Importing the submodule through `importlib` bypasses the alias
# defined in `huggingface_hub.utils` and gives access to the actual module
# object.
tqdm_module = importlib.import_module("huggingface_hub.utils.tqdm")
tqdm_module.tqdm = CustomTqdm  # type: ignore
huggingface_hub.utils.tqdm = CustomTqdm  # type: ignore


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
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        self.manager = Manager()
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

        queue = self.manager.Queue()

        # Start the progress updates in a separate thread
        progress_thread = threading.Thread(
            target=self.run_progress_updates, args=(repo_id, path, queue)
        )
        progress_thread.start()

        try:
            await self.download_huggingface_repo(
                repo_id=repo_id,
                path=path,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                queue=queue,
                user_id=user_id,
            )
        except Exception as e:
            self.logger.error(f"Error in download {id}: {e}")
            self.logger.error(traceback.format_exc())
            download_state.status = "error"
            download_state.error_message = str(e)
            await self.send_update(repo_id, path)
            raise  # Re-raise to bubble up to the endpoint
        finally:
            self.logger.info(f"Download process finished: {id}")
            queue.put(None)  # Signal to stop the progress thread
            progress_thread.join()
            del self.downloads[id]

    async def cancel_download(self, id: str):
        """Cancel an ongoing download."""
        if id not in self.downloads:
            return

        self.logger.info(f"Cancelling download for: {id}")
        self.downloads[id].cancel.set()
        self.downloads[id].status = "cancelled"

    def run_progress_updates(self, repo_id: str, path: str | None, queue: Queue):
        """Run progress updates in a separate thread."""
        asyncio.run(self.send_progress_updates(repo_id, path, queue))

    async def send_update(self, repo_id: str, path: str | None = None):
        """Send an update to the WebSocket client."""
        id = repo_id if path is None else f"{repo_id}/{path}"
        state = self.downloads[id]
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
        await state.websocket.send_json(update)

    async def send_progress_updates(self, repo_id: str, path: str | None, queue: Queue):
        """Process progress updates from the download queue."""
        id = repo_id if path is None else f"{repo_id}/{path}"
        while True:
            try:
                message = queue.get(timeout=0.1)
                if message is None:
                    break
                state = self.downloads[id]
                state.status = "progress"
                state.downloaded_bytes += message["n"]
                await self.send_update(repo_id, path)
            except Empty:
                pass
            except TimeoutError:
                pass
            except EOFError:
                pass
            except BrokenPipeError:
                pass
            except Exception:
                import traceback

                traceback.print_exc()

    async def download_huggingface_repo(
        self,
        repo_id: str,
        path: str | None,
        queue: Queue,
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
                    # Only recreate the API client when we're still using the default implementation.
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
            if isinstance(file, RepoFile) or hasattr(file, "path")
        ]
        files = filter_repo_paths(files, allow_patterns, ignore_patterns)

        # Filter out files that already exist in the cache
        files_to_download = []
        for file in files:
            cache_path = try_to_load_from_cache(repo_id, file.path)
            if cache_path is None or not os.path.exists(cache_path):
                files_to_download.append(file)
            else:
                state.downloaded_files.append(file.path)

        state.total_files = len(files_to_download)
        state.total_bytes = sum(getattr(file, "size", 0) for file in files_to_download)
        self.logger.info(
            f"Total files to download: {state.total_files}, Total size: {state.total_bytes} bytes"
        )
        await self.send_update(repo_id, path)

        loop = asyncio.get_running_loop()
        tasks = []
        self.logger.debug(f"download_huggingface_repo: Starting download of {len(files_to_download)} files for {repo_id} (user_id={user_id})")

        async def run_single_download(file_path: str):
            return await loop.run_in_executor(
                self.process_pool,
                download_file,
                repo_id,
                file_path,
                queue,
                self.token,  # Pass token for gated model downloads
            )

        for file in files_to_download:
            if state.cancel.is_set():
                self.logger.info("Download cancelled")
                break
            state.current_files.append(file.path)
            await self.send_update(repo_id, path)
            self.logger.debug(f"download_huggingface_repo: Queuing download of {file.path} for {repo_id} (user_id={user_id})")
            tasks.append(run_single_download(file.path))

        # Use return_exceptions=True to catch exceptions from individual tasks
        # but still allow us to process successful downloads
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions and raise the first one found
        # This ensures errors like GatedRepoError bubble up properly
        for result in completed_tasks:
            if isinstance(result, Exception):
                # Log the error for debugging
                self.logger.error(f"Download task failed: {result}")
                raise result

        # Process successful downloads
        for result in completed_tasks:
            if isinstance(result, tuple):
                filename, local_path = result
                if local_path:
                    state.downloaded_files.append(filename)
                    self.logger.debug(f"Downloaded file: {filename}")

        state.status = "completed" if not state.cancel.is_set() else "cancelled"
        self.logger.info(f"Download {state.status} for repo: {repo_id}")

        # Purge all HuggingFace caches when download completes successfully
        if state.status == "completed":
            self.logger.info(
                "Purging HuggingFace model caches after successful download"
            )
            self.model_cache.delete_pattern("cached_hf_*")

        await self.send_update(repo_id, path)
