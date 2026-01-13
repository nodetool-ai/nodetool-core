#!/usr/bin/env python

"""
API endpoints for workspace management.

This module provides REST API endpoints for managing user-defined workspaces:
- GET /api/workspaces - List all workspaces for the current user
- POST /api/workspaces - Create a new workspace
- GET /api/workspaces/{id} - Get a specific workspace
- DELETE /api/workspaces/{id} - Delete a workspace
- GET /api/workspaces/workflow/{workflow_id}/files - List files in a workflow's workspace
- GET /api/workspaces/workflow/{workflow_id}/download/{file_path} - Download a file from a workflow's workspace
- POST /api/workspaces/workflow/{workflow_id}/upload/{file_path} - Upload a file to a workflow's workspace
"""

import asyncio
import os
from datetime import UTC, datetime
from typing import List, Optional

import aiofiles
import aiofiles.os
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.models.workspace import Workspace as WorkspaceModel

log = get_logger(__name__)
router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


class WorkspaceResponse(BaseModel):
    """Response model for a workspace."""

    id: str
    user_id: str
    name: str
    path: str
    is_default: bool
    is_accessible: bool = Field(description="Whether the path exists and is writable")
    created_at: str
    updated_at: str


class WorkspaceCreateRequest(BaseModel):
    """Request model for creating a workspace."""

    name: str = Field(..., description="Display name for the workspace")
    path: str = Field(..., description="Absolute path to the workspace directory")
    is_default: bool = Field(default=False, description="Set as default workspace")


class WorkspaceUpdateRequest(BaseModel):
    """Request model for updating a workspace."""

    name: Optional[str] = Field(None, description="Display name for the workspace")
    path: Optional[str] = Field(None, description="Absolute path to the workspace directory")
    is_default: Optional[bool] = Field(None, description="Set as default workspace")


class WorkspaceListResponse(BaseModel):
    """Response model for listing workspaces."""

    workspaces: List[WorkspaceResponse]
    next_cursor: str = ""


def workspace_to_response(workspace: WorkspaceModel) -> WorkspaceResponse:
    """Convert a Workspace model to a response object."""
    return WorkspaceResponse(
        id=workspace.id,
        user_id=workspace.user_id,
        name=workspace.name,
        path=workspace.path,
        is_default=workspace.is_default,
        is_accessible=workspace.is_accessible(),
        created_at=workspace.created_at.isoformat(),
        updated_at=workspace.updated_at.isoformat(),
    )


@router.get("", response_model=WorkspaceListResponse)
async def list_workspaces(
    limit: int = 100,
    cursor: Optional[str] = None,
    user: str = Depends(current_user),
) -> WorkspaceListResponse:
    """
    List all configured workspaces for the current user.

    Returns a list of workspaces with a calculated `is_accessible` boolean
    indicating whether the path exists and is writable.
    """
    try:
        workspaces, next_cursor = await WorkspaceModel.paginate(
            user_id=user,
            limit=limit,
            start_key=cursor,
        )
        return WorkspaceListResponse(
            workspaces=[workspace_to_response(ws) for ws in workspaces],
            next_cursor=next_cursor,
        )
    except Exception as e:
        log.error(f"Error listing workspaces: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("", response_model=WorkspaceResponse, status_code=201)
async def create_workspace(
    request: WorkspaceCreateRequest,
    user: str = Depends(current_user),
) -> WorkspaceResponse:
    """
    Create a new workspace.

    Validation:
    - Rejects relative paths
    - Rejects paths that are not writable by the NodeTool service user
    - If `is_default` is true, sets all other user workspaces to `is_default = false`
    """
    # Validate path is absolute
    if not os.path.isabs(request.path):
        raise HTTPException(
            status_code=400,
            detail=f"Workspace path must be absolute. Got: {request.path}",
        )

    # Validate path exists
    if not os.path.exists(request.path):
        raise HTTPException(
            status_code=400,
            detail=f"Workspace path does not exist: {request.path}",
        )

    # Validate path is a directory
    if not os.path.isdir(request.path):
        raise HTTPException(
            status_code=400,
            detail=f"Workspace path is not a directory: {request.path}",
        )

    # Validate path is writable
    if not os.access(request.path, os.W_OK):
        raise HTTPException(
            status_code=400,
            detail=f"Workspace path is not writable: {request.path}",
        )

    try:
        workspace = await WorkspaceModel.create(
            user_id=user,
            name=request.name,
            path=request.path,
            is_default=request.is_default,
        )
        return workspace_to_response(workspace)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.error(f"Error creating workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: str,
    user: str = Depends(current_user),
) -> WorkspaceResponse:
    """
    Get a specific workspace by ID.
    """
    workspace = await WorkspaceModel.find(user, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_id}")
    return workspace_to_response(workspace)


@router.put("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: str,
    request: WorkspaceUpdateRequest,
    user: str = Depends(current_user),
) -> WorkspaceResponse:
    """
    Update a workspace.
    """
    workspace = await WorkspaceModel.find(user, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_id}")

    # Validate new path if provided
    if request.path is not None:
        if not os.path.isabs(request.path):
            raise HTTPException(
                status_code=400,
                detail=f"Workspace path must be absolute. Got: {request.path}",
            )
        if not os.path.isdir(request.path):
            raise HTTPException(
                status_code=400,
                detail=f"Workspace path does not exist or is not a directory: {request.path}",
            )
        if not os.access(request.path, os.W_OK):
            raise HTTPException(
                status_code=400,
                detail=f"Workspace path is not writable: {request.path}",
            )
        workspace.path = request.path

    if request.name is not None:
        workspace.name = request.name

    if request.is_default is not None:
        if request.is_default and not workspace.is_default:
            # Unset other defaults first
            await WorkspaceModel._unset_other_defaults(user)
        workspace.is_default = request.is_default

    try:
        await workspace.save()
        return workspace_to_response(workspace)
    except Exception as e:
        log.error(f"Error updating workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{workspace_id}", status_code=204)
async def delete_workspace(
    workspace_id: str,
    user: str = Depends(current_user),
) -> None:
    """
    Delete a workspace.

    Returns 400 Bad Request if any workflows are currently linked to this workspace ID.
    """
    workspace = await WorkspaceModel.find(user, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_id}")

    # Check if any workflows are linked to this workspace
    if await WorkspaceModel.has_linked_workflows(workspace_id):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete workspace: workflows are linked to this workspace. "
            "Please unlink or delete those workflows first.",
        )

    try:
        await workspace.delete()
    except Exception as e:
        log.error(f"Error deleting workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/default", response_model=Optional[WorkspaceResponse])
async def get_default_workspace(
    user: str = Depends(current_user),
) -> Optional[WorkspaceResponse]:
    """
    Get the default workspace for the current user.
    """
    workspace = await WorkspaceModel.get_default(user)
    if workspace:
        return workspace_to_response(workspace)
    return None


# --- File listing functionality ---


class FileInfo(BaseModel):
    """Information about a file or directory."""

    name: str
    path: str
    size: int
    is_dir: bool
    modified_at: str


async def get_file_info(path: str) -> FileInfo:
    """Helper function to get file information."""
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
    normalized_root = os.path.normcase(os.path.abspath(root))
    normalized_path = os.path.normcase(os.path.abspath(path))
    root_prefix = (
        normalized_root if normalized_root.endswith(os.sep) else normalized_root + os.sep
    )
    if normalized_path != normalized_root and not normalized_path.startswith(root_prefix):
        raise HTTPException(status_code=403, detail=error_message)
    return normalized_path


@router.get("/workflow/{workflow_id}/files", response_model=List[FileInfo])
async def list_workflow_files(
    workflow_id: str,
    path: str = ".",
    user: str = Depends(current_user),
) -> List[FileInfo]:
    """
    List files and directories in the workspace associated with a workflow.

    This endpoint resolves the workspace path by looking up the workflow's
    associated workspace_id and using the workspace's configured path.

    Args:
        workflow_id: The ID of the workflow whose workspace files to list.
        path: Relative path within the workspace (default: "." for root).
        user: The authenticated user.

    Returns:
        List of FileInfo objects for files and directories in the workspace.

    Raises:
        404: If the workflow or workspace is not found.
        403: If the requested path is outside the workspace root.
    """
    try:
        # Find the workflow
        workflow = await WorkflowModel.find(user, workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")

        # Check if workflow has a workspace_id
        if not workflow.workspace_id:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow '{workflow_id}' does not have an associated workspace",
            )

        # Get the workspace to resolve the actual path
        workspace = await WorkspaceModel.find(user, workflow.workspace_id)
        if not workspace:
            raise HTTPException(
                status_code=404,
                detail=f"Workspace not found: {workflow.workspace_id}",
            )

        workspace_path = os.path.abspath(workspace.path)

        # Construct full path within workspace
        full_path = workspace_path if path == "." else os.path.join(workspace_path, path)

        resolved_path = ensure_within_root(
            workspace_path,
            full_path,
            "Access denied: path outside workspace",
        )

        # Check if path exists
        exists = await asyncio.to_thread(os.path.exists, resolved_path)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")

        files = []
        entries = await aiofiles.os.listdir(resolved_path)
        for entry in entries:
            # Include hidden files in workspace view
            entry_path = os.path.join(resolved_path, entry)
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
        log.error(f"Error listing workspace files for workflow {workflow_id}/{path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def get_workspace_path_for_workflow(user: str, workflow_id: str) -> str:
    """
    Helper function to resolve the workspace path for a given workflow.

    Args:
        user: The authenticated user.
        workflow_id: The ID of the workflow.

    Returns:
        The absolute path to the workspace directory.

    Raises:
        HTTPException: If the workflow or workspace is not found.
    """
    workflow = await WorkflowModel.find(user, workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")

    if not workflow.workspace_id:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' does not have an associated workspace",
        )

    workspace = await WorkspaceModel.find(user, workflow.workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=404,
            detail=f"Workspace not found: {workflow.workspace_id}",
        )

    return os.path.abspath(workspace.path)


@router.get("/workflow/{workflow_id}/download/{file_path:path}")
async def download_workflow_file(
    workflow_id: str,
    file_path: str,
    user: str = Depends(current_user),
):
    """
    Download a file from a workflow's workspace.

    Args:
        workflow_id: The ID of the workflow whose workspace contains the file.
        file_path: Relative path to the file within the workspace.
        user: The authenticated user.

    Returns:
        StreamingResponse with the file contents.

    Raises:
        404: If the workflow, workspace, or file is not found.
        403: If the requested path is outside the workspace root.
    """
    try:
        workspace_path = await get_workspace_path_for_workflow(user, workflow_id)

        # Construct full file path
        full_path = ensure_within_root(
            workspace_path,
            os.path.join(workspace_path, file_path),
            "Access denied: path outside workspace",
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
            headers={"Content-Disposition": f'attachment; filename="{os.path.basename(file_path)}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error downloading file from workflow {workflow_id}/{file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/workflow/{workflow_id}/upload/{file_path:path}")
async def upload_workflow_file(
    workflow_id: str,
    file_path: str,
    file: UploadFile,
    user: str = Depends(current_user),
):
    """
    Upload a file to a workflow's workspace.

    Args:
        workflow_id: The ID of the workflow whose workspace to upload to.
        file_path: Relative path where the file should be saved within the workspace.
        file: The file to upload.
        user: The authenticated user.

    Returns:
        FileInfo for the uploaded file.

    Raises:
        404: If the workflow or workspace is not found.
        403: If the requested path is outside the workspace root.
    """
    try:
        workspace_path = await get_workspace_path_for_workflow(user, workflow_id)

        # Construct full file path
        full_path = ensure_within_root(
            workspace_path,
            os.path.join(workspace_path, file_path),
            "Access denied: path outside workspace",
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
        log.error(f"Error uploading file to workflow {workflow_id}/{file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e
