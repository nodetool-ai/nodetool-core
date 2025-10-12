#!/usr/bin/env python

import os
import asyncio
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import aiofiles
import aiofiles.os
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
            modified_at=datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        )
    except Exception as e:
        log.error(f"Error getting file info for {path}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"File not found: {path}")


@router.get("/list")
async def list_files(
    path: str = ".", user: str = Depends(current_user)
) -> List[FileInfo]:
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
                exists_flags = await asyncio.gather(
                    *[asyncio.to_thread(os.path.exists, root) for root in roots]
                )

                existing_roots = [root for root, ok in zip(roots, exists_flags) if ok]

                async def get_drive_mtime(root: str) -> float | None:
                    try:
                        st = await aiofiles.os.stat(root)
                        return st.st_mtime
                    except Exception:
                        return None

                # Fetch mtimes concurrently for existing roots
                mtimes = await asyncio.gather(
                    *[get_drive_mtime(root) for root in existing_roots]
                )

                files: List[FileInfo] = []
                now_iso = datetime.now(timezone.utc).isoformat()
                for root, mtime in zip(existing_roots, mtimes):
                    modified = (
                        datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                        if mtime is not None
                        else now_iso
                    )

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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_file(path: str, user: str = Depends(current_user)) -> FileInfo:
    """
    Get information about a specific file or directory
    """
    try:
        exists = await asyncio.to_thread(os.path.exists, path)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")
        return await get_file_info(path)
    except HTTPException:
        # Re-raise HTTPExceptions (like from validate_path)
        raise
    except Exception as e:
        log.error(f"Error getting file info for {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{path:path}")
async def download_file(path: str, user: str = Depends(current_user)):
    """
    Download a file from the specified path
    """
    try:
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
            headers={
                "Content-Disposition": f'attachment; filename="{os.path.basename(path)}"'
            },
        )
    except HTTPException:
        # Re-raise HTTPExceptions (like from validate_path)
        raise
    except Exception as e:
        log.error(f"Error downloading file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/{path:path}")
async def upload_file(path: str, file: UploadFile, user: str = Depends(current_user)):
    """
    Upload a file to the specified path
    """
    try:
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
        raise HTTPException(status_code=500, detail=str(e))


# Workspace-specific endpoints


def get_workspace_root() -> str:
    """Get the root directory for all workspaces"""
    return os.path.expanduser("~/.nodetool-workspaces")


async def get_workspace_info_from_path(
    workspace_path: str, workspace_id: str
) -> WorkspaceInfo:
    """Helper function to get workspace information"""
    try:
        stat = await aiofiles.os.stat(workspace_path)

        # Extract workflow_id from workspace_id (format: workflow_{workflow_id})
        workflow_id = None
        if workspace_id.startswith("workflow_"):
            workflow_id = workspace_id[9:]  # Remove "workflow_" prefix

        # Count files recursively
        file_count = 0
        total_size = 0

        async def count_files(path: str):
            nonlocal file_count, total_size
            try:
                entries = await aiofiles.os.listdir(path)
                for entry in entries:
                    entry_path = os.path.join(path, entry)
                    try:
                        entry_stat = await aiofiles.os.stat(entry_path)
                        is_dir = await asyncio.to_thread(os.path.isdir, entry_path)
                        if is_dir:
                            await count_files(entry_path)
                        else:
                            file_count += 1
                            total_size += entry_stat.st_size
                    except Exception:
                        pass
            except Exception:
                pass

        await count_files(workspace_path)

        return WorkspaceInfo(
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            path=workspace_path,
            size=total_size,
            file_count=file_count,
            created_at=datetime.fromtimestamp(
                stat.st_ctime, tz=timezone.utc
            ).isoformat(),
            modified_at=datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        )
    except Exception as e:
        log.error(f"Error getting workspace info for {workspace_path}: {str(e)}")
        raise HTTPException(
            status_code=404, detail=f"Workspace not found: {workspace_id}"
        )


@router.get("/workspaces")
async def list_workspaces(user: str = Depends(current_user)) -> List[WorkspaceInfo]:
    """
    List all workspaces in ~/.nodetool-workspaces
    """
    try:
        workspace_root = get_workspace_root()

        # Create workspace root if it doesn't exist
        await asyncio.to_thread(os.makedirs, workspace_root, exist_ok=True)

        workspaces = []
        entries = await aiofiles.os.listdir(workspace_root)

        for entry in entries:
            # Skip hidden directories
            if entry.startswith("."):
                continue

            entry_path = os.path.join(workspace_root, entry)
            is_dir = await asyncio.to_thread(os.path.isdir, entry_path)

            if is_dir:
                try:
                    workspace_info = await get_workspace_info_from_path(
                        entry_path, entry
                    )
                    workspaces.append(workspace_info)
                except Exception as e:
                    log.warning(f"Skipping workspace {entry}: {str(e)}")
                    continue

        # Sort by modified_at descending (most recent first)
        workspaces.sort(key=lambda w: w.modified_at, reverse=True)

        return workspaces
    except Exception as e:
        log.error(f"Error listing workspaces: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces/{workspace_id}/info")
async def get_workspace_info(
    workspace_id: str, user: str = Depends(current_user)
) -> WorkspaceInfo:
    """
    Get information about a specific workspace
    """
    try:
        workspace_root = get_workspace_root()
        workspace_path = os.path.join(workspace_root, workspace_id)

        exists = await asyncio.to_thread(os.path.exists, workspace_path)
        if not exists:
            raise HTTPException(
                status_code=404, detail=f"Workspace not found: {workspace_id}"
            )

        return await get_workspace_info_from_path(workspace_path, workspace_id)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting workspace info for {workspace_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces/{workspace_id}/list")
async def list_workspace_files(
    workspace_id: str, path: str = ".", user: str = Depends(current_user)
) -> List[FileInfo]:
    """
    List files and directories in a workspace
    """
    try:
        workspace_root = get_workspace_root()
        workspace_path = os.path.join(workspace_root, workspace_id)

        # Construct full path within workspace
        if path == ".":
            full_path = workspace_path
        else:
            full_path = os.path.join(workspace_path, path)

        resolved_path = os.path.realpath(full_path)

        log.info(f"Resolved path: {resolved_path}")

        # Ensure path is within workspace (security check)
        if not resolved_path.startswith(workspace_path):
            raise HTTPException(
                status_code=403, detail="Access denied: path outside workspace"
            )

        log.info(f"Listing workspace files at {resolved_path}")

        files = []

        entries = await aiofiles.os.listdir(resolved_path)
        for entry in entries:
            # Include hidden files in workspace view (unlike general file listing)
            entry_path = os.path.join(resolved_path, entry)
            log.info(f"Listing workspace file {entry_path}")
            try:
                file_info = await get_file_info(entry_path)
                files.append(file_info)
            except Exception as e:
                log.warning(f"Skipping {entry_path}: {str(e)}")
                continue

        return files
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error listing workspace files for {workspace_id}/{path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces/{workspace_id}/download/{file_path:path}")
async def download_workspace_file(
    workspace_id: str, file_path: str, user: str = Depends(current_user)
):
    """
    Download a file from a workspace
    """
    try:
        workspace_root = get_workspace_root()
        workspace_path = os.path.join(workspace_root, workspace_id)

        exists = await asyncio.to_thread(os.path.exists, workspace_path)
        if not exists:
            raise HTTPException(
                status_code=404, detail=f"Workspace not found: {workspace_id}"
            )

        # Construct full file path
        full_path = os.path.join(workspace_path, file_path)

        # Ensure path is within workspace (security check)
        full_path = os.path.abspath(full_path)
        if not full_path.startswith(workspace_path):
            raise HTTPException(
                status_code=403, detail="Access denied: path outside workspace"
            )

        exists = await asyncio.to_thread(os.path.exists, full_path)
        is_dir = await asyncio.to_thread(os.path.isdir, full_path)
        if not exists or is_dir:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        async def file_iterator():
            chunk_size = 8192  # 8KB chunks
            async with aiofiles.open(full_path, "rb") as f:
                while chunk := await f.read(chunk_size):
                    yield chunk

        return StreamingResponse(
            file_iterator(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{os.path.basename(file_path)}"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error downloading file {workspace_id}/{file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workspaces/{workspace_id}/upload/{file_path:path}")
async def upload_workspace_file(
    workspace_id: str,
    file_path: str,
    file: UploadFile,
    user: str = Depends(current_user),
):
    """
    Upload a file to a workspace
    """
    try:
        workspace_root = get_workspace_root()
        workspace_path = os.path.join(workspace_root, workspace_id)

        # Create workspace if it doesn't exist
        await asyncio.to_thread(os.makedirs, workspace_path, exist_ok=True)

        # Construct full file path
        full_path = os.path.join(workspace_path, file_path)

        # Ensure path is within workspace (security check)
        full_path = os.path.abspath(full_path)
        if not full_path.startswith(workspace_path):
            raise HTTPException(
                status_code=403, detail="Access denied: path outside workspace"
            )

        # Create parent directories if needed
        await asyncio.to_thread(os.makedirs, os.path.dirname(full_path), exist_ok=True)

        # Read and write in chunks to handle large files
        async with aiofiles.open(full_path, "wb") as f:
            while chunk := await file.read(8192):
                await f.write(chunk)

        return await get_file_info(full_path)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error uploading file to {workspace_id}/{file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
