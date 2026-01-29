"""
Hugging Face Cache Utilities Module

This module provides utilities for checking and managing Hugging Face cache,
including cache existence checks, file filtering, and repository size calculation.
"""

import os
from fnmatch import fnmatch

from huggingface_hub import HfApi, constants
from huggingface_hub.hf_api import RepoFile

from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.hf_auth import get_hf_token

log = get_logger(__name__)


def has_cached_files(repo_id: str) -> bool:
    """Check if any files from the specified repo exist in the local HF cache.

    Args:
        repo_id (`str`):
            A namespace (user or an organization) name and a repo name separated
            by a `/`.

    Returns:
        `bool`: `True` if any files from the repo exist in cache, `False` otherwise.
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


def filter_repo_paths(
    items: list[RepoFile],
    allow_patterns: str | list[str] | None = None,
    ignore_patterns: str | list[str] | None = None,
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
    """
    if isinstance(allow_patterns, str):
        allow_patterns = [allow_patterns]

    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]

    filtered_paths = []
    for file in items:
        path = file.path
        # Skip if there's an allowlist and path doesn't match any
        if allow_patterns is not None and not any(fnmatch(path, r) for r in allow_patterns):
            continue

        # Skip if there's a denylist and path matches any
        if ignore_patterns is not None and any(fnmatch(path, r) for r in ignore_patterns):
            continue

        filtered_paths.append(file)

    return filtered_paths


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
        log.debug(
            f"get_repo_size: Using HF_TOKEN for repo {repo_id} (token length: {len(token)} chars, user_id={user_id})"
        )
        api = HfApi(token=token)
    else:
        log.debug(
            f"get_repo_size: No HF_TOKEN available for repo {repo_id} - gated models may not be accessible (user_id={user_id})"
        )
        api = HfApi()
    files = api.list_repo_tree(repo_id, recursive=True)
    files = [file for file in files if isinstance(file, RepoFile)]
    filtered_files = filter_repo_paths(files, allow_patterns, ignore_patterns)

    total_size = sum(file.size for file in filtered_files)
    return total_size
