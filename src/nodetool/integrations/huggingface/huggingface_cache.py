"""
Hugging Face Cache Management Module

This module provides functionality for managing and downloading files from Hugging Face repositories,
with support for WebSocket-based progress tracking and caching. Key features include:

- Checking for cached repository files
- Downloading repository files with pattern filtering
- Real-time download progress tracking via WebSockets
- Multi-process download management
- Download cancellation support

The module uses a custom progress bar implementation that sends updates through a WebSocket
connection, allowing clients to monitor download progress in real-time.

Main Components:
- DownloadState: Tracks the state of individual downloads
- DownloadManager: Handles concurrent downloads and WebSocket communications
- CustomTqdm: Modified progress bar for WebSocket integration
"""

import asyncio
from dataclasses import dataclass, field
from fnmatch import fnmatch
import importlib
from nodetool.config.logging_config import get_logger
import traceback
from typing import Literal
from fastapi import WebSocket
from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
from huggingface_hub.hf_api import RepoFile
import huggingface_hub.file_download
from multiprocessing import Manager
from queue import Empty, Queue
from concurrent.futures import ProcessPoolExecutor
import threading
import os
from huggingface_hub import constants
from nodetool.ml.models.model_cache import ModelCache
from nodetool.security.secret_helper import get_secret
from nodetool.runtime.resources import maybe_scope

log = get_logger(__name__)


async def get_hf_token(user_id: str | None = None) -> str | None:
    """Get HF_TOKEN from environment variables or database secrets (async).
    
    Args:
        user_id: Optional user ID. If not provided, will try to get from ResourceScope if available.
    
    Returns:
        HF_TOKEN if available, None otherwise.
    """
    log.debug(f"get_hf_token: Looking up HF_TOKEN for user_id={user_id}")
    
    # 1. Check environment variable first (highest priority)
    token = os.environ.get("HF_TOKEN")
    if token:
        log.debug(f"get_hf_token: HF_TOKEN found in environment variables (user_id={user_id} was provided but env takes priority)")
        return token
    
    # 2. Try to get from database if user_id is available
    if user_id is None:
        log.debug("get_hf_token: No user_id provided, checking ResourceScope")
        # Try to get user_id from ResourceScope if available
        try:
            scope = maybe_scope()
            # Note: ResourceScope doesn't store user_id directly
            # In real usage, user_id would come from authentication context
        except Exception:
            pass
    
    if user_id:
        log.debug(f"get_hf_token: Attempting to retrieve HF_TOKEN from database for user_id={user_id}")
        try:
            token = await get_secret("HF_TOKEN", user_id)
            if token:
                log.debug(f"get_hf_token: HF_TOKEN found in database secrets for user_id={user_id}")
                return token
            else:
                log.debug(f"get_hf_token: HF_TOKEN not found in database for user_id={user_id}")
        except Exception as e:
            log.debug(f"get_hf_token: Failed to get HF_TOKEN from database for user_id={user_id}: {e}")
    else:
        log.debug("get_hf_token: No user_id available, skipping database lookup")
    
    log.debug(f"get_hf_token: HF_TOKEN not found in environment or database secrets (user_id={user_id})")
    return None


def has_cached_files(
    repo_id: str,
) -> bool:
    """Check if any files from the specified repo exist in the local HF cache.

    Args:
        repo_id (`str`):
            A namespace (user or an organization) name and a repo name separated
            by a `/`.

    Returns:
        `bool`: `True` if any files from the repo exist in cache, `False` otherwise.
    ```
    """
    cache_dir = constants.HF_HUB_CACHE

    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"models--{object_id}")

    # Check if repo folder exists and contains any snapshots
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return False

    # Check if any snapshot contains files
    for revision in os.listdir(snapshots_dir):
        revision_path = os.path.join(snapshots_dir, revision)
        if os.path.isdir(revision_path) and any(os.scandir(revision_path)):
            return True

    return False


@dataclass
class DownloadState:
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


async def get_repo_size(
    repo_id: str,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    user_id: str | None = None,
) -> int:
    """
    Get the total size of files in a Hugging Face repository that match the given patterns.

    Args:
        repo_id (str): The ID of the Hugging Face repository.
        allow_patterns (list[str] | None): List of patterns to allow.
        ignore_patterns (list[str] | None): List of patterns to ignore.
        user_id (str | None): Optional user ID for database secret lookup.

    Returns:
        int: Total size of matching files in bytes.
    """
    log.debug(f"get_repo_size: Getting repo size for {repo_id} with user_id={user_id}")
    # Use HF_TOKEN from secrets if available for gated model downloads
    token = await get_hf_token(user_id)
    if token:
        log.debug(f"get_repo_size: Using HF_TOKEN for repo {repo_id} (token length: {len(token)} chars, user_id={user_id})")
        api = HfApi(token=token)
    else:
        log.debug(f"get_repo_size: No HF_TOKEN available for repo {repo_id} - gated models may not be accessible (user_id={user_id})")
        api = HfApi()
    files = api.list_repo_tree(repo_id, recursive=True)
    files = [file for file in files if isinstance(file, RepoFile)]
    filtered_files = filter_repo_paths(files, allow_patterns, ignore_patterns)

    total_size = sum(file.size for file in filtered_files)
    return total_size


def filter_repo_paths(
    items: list[RepoFile],
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> list[RepoFile]:
    """Filter repo objects based on an allowlist and a denylist.

    Patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Args:
        items (list[RepoFile]):
            List of items to filter.
        allow_patterns (`str` or `List[str]`, *optional*):
            Patterns constituting the allowlist. If provided, item paths must match at
            least one pattern from the allowlist.
        ignore_patterns (`str` or `List[str]`, *optional*):
            Patterns constituting the denylist. If provided, item paths must not match
            any patterns from the denylist.

    Returns:
        Filtered list of paths

    ```
    """
    if isinstance(allow_patterns, str):
        allow_patterns = [allow_patterns]

    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]

    filtered_paths = []
    for file in items:
        path = file.path
        # Skip if there's an allowlist and path doesn't match any
        if allow_patterns is not None and not any(
            fnmatch(path, r) for r in allow_patterns
        ):
            continue

        # Skip if there's a denylist and path matches any
        if ignore_patterns is not None and any(
            fnmatch(path, r) for r in ignore_patterns
        ):
            continue

        filtered_paths.append(file)

    return filtered_paths


class DownloadManager:
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
        token = await get_hf_token(user_id)
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
        if id not in self.downloads:
            return

        self.logger.info(f"Cancelling download for: {id}")
        self.downloads[id].cancel.set()
        self.downloads[id].status = "cancelled"

    def run_progress_updates(self, repo_id: str, path: str | None, queue: Queue):
        asyncio.run(self.send_progress_updates(repo_id, path, queue))

    async def send_update(self, repo_id: str, path: str | None = None):
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
        id = repo_id if path is None else f"{repo_id}/{path}"
        state = self.downloads[id]

        self.logger.debug(f"download_huggingface_repo: Starting download for {repo_id} with user_id={user_id}")
        
        # Ensure token is initialized
        if not self._token_initialized:
            self.logger.debug(f"download_huggingface_repo: Token not initialized, fetching with user_id={user_id}")
            self.token = await get_hf_token(user_id)
            if self.token:
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
        files = self.api.list_repo_tree(repo_id, recursive=True)
        files = [file for file in files if isinstance(file, RepoFile)]
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
        state.total_bytes = sum(file.size for file in files_to_download)
        self.logger.info(
            f"Total files to download: {state.total_files}, Total size: {state.total_bytes} bytes"
        )
        await self.send_update(repo_id, path)

        loop = asyncio.get_running_loop()
        tasks = []
        self.logger.debug(f"download_huggingface_repo: Starting download of {len(files_to_download)} files for {repo_id} (user_id={user_id})")
        for file in files_to_download:
            if state.cancel.is_set():
                self.logger.info("Download cancelled")
                break
            state.current_files.append(file.path)
            await self.send_update(repo_id, path)
            self.logger.debug(f"download_huggingface_repo: Queuing download of {file.path} for {repo_id} (user_id={user_id})")
            task = loop.run_in_executor(
                self.process_pool,
                download_file,
                repo_id,
                file.path,
                queue,
                self.token,  # Pass token for gated model downloads
            )
            tasks.append(task)

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


async def huggingface_download_endpoint(websocket: WebSocket):
    """WebSocket endpoint for HuggingFace model downloads with authentication."""
    from nodetool.runtime.resources import get_static_auth_provider, get_user_auth_provider
    from nodetool.config.environment import Environment
    
    # Get auth providers
    static_provider = get_static_auth_provider()
    user_provider = get_user_auth_provider()
    enforce_auth = Environment.enforce_auth()
    
    # Authenticate websocket
    if not enforce_auth:
        # In local mode, fallback to user_id "1" if no auth is provided
        user_id = getattr(static_provider, 'user_id', None) or "1"
        token = None
    else:
        token = static_provider.extract_token_from_ws(
            websocket.headers, websocket.query_params
        )
        if not token:
            await websocket.close(code=1008, reason="Missing authentication")
            log.warning("HF download WebSocket connection rejected: Missing authentication header")
            return
        
        static_result = await static_provider.verify_token(token)
        if static_result.ok and static_result.user_id:
            user_id = static_result.user_id
        elif Environment.get_auth_provider_kind() == "supabase" and user_provider:
            user_result = await user_provider.verify_token(token)
            if user_result.ok and user_result.user_id:
                user_id = user_result.user_id
            else:
                await websocket.close(code=1008, reason="Invalid authentication")
                log.warning("HF download WebSocket connection rejected: Invalid token")
                return
        else:
            await websocket.close(code=1008, reason="Invalid authentication")
            log.warning("HF download WebSocket connection rejected: Invalid token")
            return
    
    # Ensure user_id is set (fallback to "1" for local mode)
    if not user_id:
        user_id = "1"
    
    log.info(f"huggingface_download_endpoint: Authenticated websocket connection with user_id={user_id}")
    
    # Create download manager with user_id for database secret lookup
    download_manager = await DownloadManager.create(user_id=user_id)
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")
            repo_id = data.get("repo_id")
            path = data.get("path")
            allow_patterns = data.get("allow_patterns")
            ignore_patterns = data.get("ignore_patterns")

            if command == "start_download":
                log.info(f"huggingface_download_endpoint: Received start_download command for {repo_id}/{path} (user_id={user_id})")
                print(f"Starting download for {repo_id}/{path} (user_id={user_id})")
                try:
                    await download_manager.start_download(
                        repo_id=repo_id,
                        path=path,
                        websocket=websocket,
                        allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
                        user_id=user_id,
                    )
                except Exception as e:
                    # Error should already be sent by start_download, but send a final error message
                    # in case the WebSocket update failed
                    await websocket.send_json(
                        {
                            "status": "error",
                            "error": str(e),
                            "repo_id": repo_id,
                            "path": path,
                        }
                    )
                    raise  # Re-raise to be caught by outer handler
            elif command == "cancel_download":
                await download_manager.cancel_download(data.get("id"))
            else:
                await websocket.send_json(
                    {"status": "error", "message": "Unknown command"}
                )
    except Exception as e:
        print(f"WebSocket error: {e}")
        # print stacktrace
        import traceback

        traceback.print_exc()
    finally:
        await websocket.close()
