#!/usr/bin/env python

import asyncio
import os
from datetime import UTC, datetime

import aiofiles
import aiofiles.os
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/files", tags=["files"])


class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    is_dir: bool
    modified_at: str


class WorkspaceInfo(BaseModel):
    workspace_id: str
    workflow_id: str | None
    path: str
    size: int
    file_count: int
    created_at: str
    modified_at: str


async def get_file_info(path: str) -> FileInfo:
    """Helper function to get file information"""
    try:
        stat = await aiofiles.os.stat(path)
        is_dir = await asyncio.to_thread(os.path.isdir, path)
        return FileInfo(
            name=os.path.basename(path),
            path=path,
            size=stat.st_size,
            is_dir=is_dir,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        )
    except Exception as e:
        log.error(f"Error getting file info for {path}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"File not found: {path}") from e


def ensure_within_root(root: str, path: str, error_message: str) -> str:
    """
    Ensure the given path is contained within root, normalizing for case-insensitive filesystems.
    """
    normalized_root = os.path.normcase(os.path.realpath(root))
    normalized_path = os.path.normcase(os.path.realpath(path))
    root_prefix = normalized_root if normalized_root.endswith(os.sep) else normalized_root + os.sep
    if normalized_path != normalized_root and not normalized_path.startswith(root_prefix):
        raise HTTPException(status_code=403, detail=error_message)
    return normalized_path


@router.get("/list")
async def list_files(path: str = ".", __user: str = Depends(current_user)) -> list[FileInfo]:
    """
    List files and directories in the specified path, excluding hidden files (starting with dot)
    """
    try:
        # Special handling for cross-platform base paths
        # "~" means: on Windows -> list all available drive roots; on POSIX -> user's home directory
        if path == "~":
            if os.name == "nt":
                # Probe all possible drive letters concurrently
                roots = [f"{letter}:\\" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
                exists_flags = await asyncio.gather(*[asyncio.to_thread(os.path.exists, root) for root in roots])

                existing_roots = [root for root, ok in zip(roots, exists_flags, strict=False) if ok]

                async def get_drive_mtime(root: str) -> float | None:
                    try:
                        st = await aiofiles.os.stat(root)
                        return st.st_mtime
                    except Exception:
                        return None

                # Fetch mtimes concurrently for existing roots
                mtimes = await asyncio.gather(*[get_drive_mtime(root) for root in existing_roots])

                files: list[FileInfo] = []
                now_iso = datetime.now(UTC).isoformat()
                for root, mtime in zip(existing_roots, mtimes, strict=False):
                    modified = datetime.fromtimestamp(mtime, tz=UTC).isoformat() if mtime is not None else now_iso

                    files.append(
                        FileInfo(
                            name=root,
                            path=root,
                            size=0,
                            is_dir=True,
                            modified_at=modified,
                        )
                    )
                return files
            else:
                # Expand to home directory on POSIX systems
                path = os.path.expanduser("~")
        else:
            # Expand user (~) if included in other paths
            path = os.path.expanduser(path)

        # Validate and normalize path
        abs_path = path
        if not _is_safe_path(abs_path):
            raise HTTPException(status_code=403, detail="Access to this path is forbidden")

        exists = await asyncio.to_thread(os.path.exists, abs_path)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")

        files = []
        entries = await aiofiles.os.listdir(abs_path)
        for entry in entries:
            # Skip files/directories that start with a dot
            if entry.startswith("."):
                continue

            entry_path = os.path.join(abs_path, entry)
            try:
                file_info = await get_file_info(entry_path)
                files.append(file_info)
            except Exception as e:
                log.warning(f"Skipping {entry_path}: {str(e)}")
                continue

        return files
    except HTTPException:
        # Re-raise HTTPExceptions (like from validate_path)
        raise
    except Exception as e:
        log.error(f"Error listing files in {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/info")
async def get_file(path: str, __user: str = Depends(current_user)) -> FileInfo:
    """
    Get information about a specific file or directory
    """
    try:
        if not _is_safe_path(path):
            raise HTTPException(status_code=403, detail="Access to this path is forbidden")
        exists = await asyncio.to_thread(os.path.exists, path)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")
        return await get_file_info(path)
    except HTTPException:
        # Re-raise HTTPExceptions (like from validate_path)
        raise
    except Exception as e:
        log.error(f"Error getting file info for {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


SENSITIVE_PATHS = {
    "/etc",
    "/root",
    "/home",
    "/var",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/opt",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/run",
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
}

# Define safe roots (whitelist) where file access is permitted
SAFE_ROOTS = [
    os.path.abspath(os.getcwd()),  # Current working directory
    os.path.abspath(os.path.expanduser("~")),  # User home directory
]


def _is_safe_path(path: str) -> bool:
    """
    Check if the path is safe for access.

    Security checks:
    1. Resolves symlinks to prevent traversal attacks.
    2. Enforces whitelist (CWD or Home) OR ensures not in blacklist (SENSITIVE_PATHS).
    3. Blocks hidden files.
    """
    try:
        # Get absolute path and resolved path (following symlinks)
        abs_path = os.path.abspath(path)
        real_path = os.path.realpath(path)

        # Check all paths involved (original abs and resolved real)
        paths_to_check = {abs_path, real_path}

        is_in_safe_root = False
        for safe_root in SAFE_ROOTS:
            if real_path == safe_root or real_path.startswith(safe_root + os.sep):
                is_in_safe_root = True
                break

        for p in paths_to_check:
            # Check sensitive system paths (Blacklist)
            for sensitive in SENSITIVE_PATHS:
                sensitive_abs = os.path.abspath(sensitive)
                if p == sensitive_abs or p.startswith(sensitive_abs + os.sep):
                    # If we are inside a safe root (e.g. /home/user), ignore the broader sensitive path (e.g. /home)
                    if is_in_safe_root:
                         pass
                    else:
                        return False

            # Check for hidden files or directories (starting with .)
            parts = p.split(os.sep)
            if any(part.startswith(".") for part in parts if part):
                return False

        return is_in_safe_root

    except Exception as e:
        log.error(f"Error checking path safety: {e}")
        return False


@router.get("/download/{path:path}")
async def download_file(path: str, __user: str = Depends(current_user)):
    """
    Download a file from the specified path.

    Security Note: This endpoint restricts downloads to prevent access to sensitive
    system paths like /etc, /root, /proc, etc.
    """
    try:
        if not _is_safe_path(path):
            raise HTTPException(status_code=403, detail="Access to this path is forbidden")

        abs_path = path
        exists = await asyncio.to_thread(os.path.exists, abs_path)
        is_dir = await asyncio.to_thread(os.path.isdir, abs_path)
        if not exists or is_dir:
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        async def file_iterator():
            chunk_size = 8192  # 8KB chunks
            async with aiofiles.open(abs_path, "rb") as f:
                while chunk := await f.read(chunk_size):
                    yield chunk

        return StreamingResponse(
            file_iterator(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{os.path.basename(path)}"'},
        )
    except HTTPException:
        # Re-raise HTTPExceptions (like from validate_path)
        raise
    except Exception as e:
        log.error(f"Error downloading file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/upload/{path:path}")
async def upload_file(path: str, file: UploadFile, __user: str = Depends(current_user)):
    """
    Upload a file to the specified path
    """
    try:
        if not _is_safe_path(path):
            raise HTTPException(status_code=403, detail="Access to this path is forbidden")

        abs_path = path
        await asyncio.to_thread(os.makedirs, os.path.dirname(abs_path), exist_ok=True)

        # Read and write in chunks to handle large files
        async with aiofiles.open(abs_path, "wb") as f:
            while chunk := await file.read(8192):
                await f.write(chunk)

        return await get_file_info(abs_path)
    except HTTPException:
        # Re-raise HTTPExceptions (like from validate_path)
        raise
    except Exception as e:
        log.error(f"Error uploading file to {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Workspace-specific endpoints


async def get_workspace_info_from_path(workspace_path: str, workspace_id: str) -> WorkspaceInfo:
    """Helper function to get workspace information"""
    try:
        stat = await asyncio.to_thread(os.stat, workspace_path)

        # Extract workflow_id from workspace_id (format: workflow_{workflow_id})
        workflow_id = None
        if workspace_id.startswith("workflow_"):
            workflow_id = workspace_id[9:]  # Remove "workflow_" prefix

        # Count files recursively in a thread to avoid blocking the event loop
        def walk(path: str) -> tuple[int, int]:
            total_size = 0
            file_count = 0
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        entry_path = os.path.join(root, name)
                        entry_stat = os.stat(entry_path)
                        total_size += entry_stat.st_size
                        file_count += 1
                    except OSError:
                        continue
            return total_size, file_count

        total_size, file_count = await asyncio.to_thread(walk, workspace_path)

        return WorkspaceInfo(
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            path=workspace_path,
            size=total_size,
            file_count=file_count,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat(),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        )
    except Exception as e:
        log.error(f"Error getting workspace info for {workspace_path}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_id}") from e
