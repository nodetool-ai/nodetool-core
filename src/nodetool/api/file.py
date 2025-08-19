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
from nodetool.common.environment import Environment

log = Environment.get_logger()
router = APIRouter(prefix="/api/files", tags=["files"])

# Base directory for file operations - restrict access to user's home directory
# or a specific workspace directory
def get_base_directory():
    """Get the base directory for file operations."""
    # Check for environment variable override first
    if "FILE_API_BASE_DIR" in os.environ:
        return os.environ["FILE_API_BASE_DIR"]
    # In test mode, use temp directory
    if Environment.is_test():
        return "/tmp/nodetool_test_files"
    # Otherwise use home directory
    return os.path.expanduser("~")


def validate_path(path: str) -> str:
    """
    Validate that the resolved path is within the allowed base directory.
    Prevents path traversal attacks.
    
    Args:
        path: The path to validate
        
    Returns:
        The absolute path if valid
        
    Raises:
        HTTPException: If the path is outside the allowed directory
    """
    base_directory = get_base_directory()
    
    # Expand user home directory and resolve to absolute path
    expanded_path = os.path.expanduser(path)
    abs_path = os.path.abspath(expanded_path)
    
    # Ensure the resolved path is within the base directory
    try:
        # os.path.commonpath will raise ValueError if paths are on different drives (Windows)
        common_path = os.path.commonpath([base_directory, abs_path])
        if common_path != base_directory:
            raise HTTPException(
                status_code=403,
                detail="Access denied: Path is outside allowed directory"
            )
    except ValueError:
        # Paths are on different drives
        raise HTTPException(
            status_code=403,
            detail="Access denied: Path is outside allowed directory"
        )
    
    return abs_path


class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    is_dir: bool
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
        # Validate and normalize path
        abs_path = validate_path(path)
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
        abs_path = validate_path(path)
        exists = await asyncio.to_thread(os.path.exists, abs_path)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")
        return await get_file_info(abs_path)
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
        abs_path = validate_path(path)
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
        abs_path = validate_path(path)
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
