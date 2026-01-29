"""
Path utilities for workspace and file path resolution.
"""

import os

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


def resolve_workspace_path(workspace_dir: str | None, path: str) -> str:
    """
    Resolve a path relative to the workspace directory.
    Handles paths starting with '/workspace/', 'workspace/', or absolute paths
    by interpreting them relative to the `workspace_dir`.

    Args:
        workspace_dir: The absolute path to the workspace directory.
        path: The path to resolve, which can be:
            - Prefixed with '/workspace/' (e.g., '/workspace/output/file.txt')
            - Prefixed with 'workspace/' (e.g., 'workspace/output/file.txt')
            - An absolute path (e.g., '/input/data.csv') - treated relative to workspace root
            - A relative path (e.g., 'output/file.txt')

    Returns:
        The absolute path in the actual filesystem.

    Raises:
        PermissionError: If workspace_dir is None (no workspace assigned).
        ValueError: If workspace_dir is empty string.
    """
    if workspace_dir is None:
        raise PermissionError(
            "No workspace is assigned. File operations require a user-defined workspace. "
            "Please configure a workspace before performing disk I/O operations."
        )
    if not workspace_dir:
        raise ValueError("Workspace directory is required")

    relative_path: str
    # Normalize path separators for consistent checks
    normalized_path = path.replace("\\", "/")

    # Handle paths with /workspace/ prefix
    if normalized_path.startswith("/workspace/"):
        relative_path = normalized_path[len("/workspace/") :]
    # Handle paths with workspace/ prefix (without leading slash)
    elif normalized_path.startswith("workspace/"):
        relative_path = normalized_path[len("workspace/") :]
    # Handle absolute paths by stripping leading slash and treating as relative to workspace
    elif os.path.isabs(normalized_path):
        # On Windows, isabs('/') is False. Check explicitly.
        if normalized_path.startswith("/"):
            relative_path = normalized_path[1:]
        else:
            # For Windows absolute paths (e.g., C:\...), we still want to join them relative to workspace?
            # This behaviour might need clarification. Assuming here they are treated as relative for consistency.
            # If absolute paths outside workspace should be allowed, this needs change.
            log.warning(f"Treating absolute path '{path}' as relative to workspace root '{workspace_dir}'.")
            # Attempt to get path relative to drive root
            _drive, path_part = os.path.splitdrive(normalized_path)
            relative_path = path_part.lstrip("\\/")  # Strip leading slashes from the part after drive
    # Handle relative paths
    else:
        relative_path = normalized_path

    # Prevent path traversal attempts (e.g., ../../etc/passwd)
    # Join the workspace directory with the potentially cleaned relative path
    abs_path = os.path.abspath(os.path.join(workspace_dir, relative_path))

    # Final check: ensure the resolved path is still within the workspace directory
    # Use commonprefix for robustness across OS
    common_prefix = os.path.commonprefix([os.path.abspath(workspace_dir), abs_path])
    if os.path.abspath(workspace_dir) != common_prefix:
        log.error(
            f"Resolved path '{abs_path}' is outside the workspace directory '{workspace_dir}'. Original path: '{path}'"
        )
        # Option 1: Raise an error
        raise ValueError(f"Resolved path '{abs_path}' is outside the workspace directory.")
        # Option 2: Return a default safe path or the workspace root (less ideal)
        # return workspace_dir

    return abs_path
