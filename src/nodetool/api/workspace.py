#!/usr/bin/env python

"""
API endpoints for workspace management.

This module provides REST API endpoints for managing user-defined workspaces:
- GET /api/workspaces - List all workspaces for the current user
- POST /api/workspaces - Create a new workspace
- GET /api/workspaces/{id} - Get a specific workspace
- DELETE /api/workspaces/{id} - Delete a workspace
"""

import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger
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
