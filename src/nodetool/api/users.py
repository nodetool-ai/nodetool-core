"""
User management API endpoints.

This module provides REST endpoints for managing users via API.
Works with all deployment types: Docker, Root, GCP, RunPod.
"""

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.security.admin_auth import is_admin_user, require_admin
from nodetool.security.user_manager import CreateUserResult, UserManager

router = APIRouter(prefix="/api/users", tags=["users"])


class AddUserRequest(BaseModel):
    """Request model for adding a user."""

    username: str
    role: str = "user"


class ResetTokenRequest(BaseModel):
    """Request model for resetting a token."""

    username: str


@router.get("/")
async def list_users(request: Request) -> dict:
    """List all users (tokens masked).

    Requires admin role for multi_user auth.
    For other auth providers, returns empty list or error.
    """
    user_id = getattr(request.state, "user_id", None)

    # Only multi_user auth provider supports user listing
    from nodetool.config.environment import Environment

    if Environment.get_auth_provider_kind() != "multi_user":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User management not available for current auth provider",
        )

    # Check if user has admin role
    if user_id is None or not is_admin_user(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    manager = UserManager()
    users = manager.list_users()

    # Return masked version (no plaintext tokens)
    return {
        "users": [
            {
                "username": username,
                "user_id": user.user_id,
                "role": user.role,
                "token_hash": user.token_hash[:16] + "...",
                "created_at": user.created_at,
            }
            for username, user in users.items()
        ]
    }


@router.get("/{username}")
async def get_user(username: str, request: Request) -> dict:
    """Get specific user information (token masked).

    Requires admin role for multi_user auth.
    """
    user_id = getattr(request.state, "user_id", None)

    from nodetool.config.environment import Environment

    if Environment.get_auth_provider_kind() != "multi_user":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User management not available for current auth provider",
        )

    if not is_admin_user(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    manager = UserManager()
    user = manager.get_user(username)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found",
        )

    return {
        "username": username,
        "user_id": user.user_id,
        "role": user.role,
        "token_hash": user.token_hash[:16] + "...",
        "created_at": user.created_at,
    }


@router.post("/")
async def add_user(req: AddUserRequest, request: Request) -> dict:
    """Add a new user.

    Returns plaintext token (only shown on creation).
    Requires admin role for multi_user auth.
    """
    user_id = getattr(request.state, "user_id", None)

    from nodetool.config.environment import Environment

    if Environment.get_auth_provider_kind() != "multi_user":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User management not available for current auth provider",
        )

    if user_id is None or not is_admin_user(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    manager = UserManager()

    try:
        result = manager.add_user(req.username, req.role)
        return {
            "username": result.username,
            "user_id": result.user_id,
            "role": result.role,
            "token": result.token,
            "created_at": result.created_at,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/reset-token")
async def reset_token(req: ResetTokenRequest, request: Request) -> dict:
    """Generate new bearer token for a user.

    Returns plaintext token.
    Requires admin role for multi_user auth.
    """
    user_id = getattr(request.state, "user_id", None)

    from nodetool.config.environment import Environment

    if Environment.get_auth_provider_kind() != "multi_user":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User management not available for current auth provider",
        )

    if user_id is None or not is_admin_user(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    manager = UserManager()

    try:
        result = manager.reset_token(req.username)
        return {
            "username": result.username,
            "user_id": result.user_id,
            "role": result.role,
            "token": result.token,
            "created_at": result.created_at,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.delete("/{username}")
async def delete_user(username: str, request: Request) -> dict:
    """Remove a user.

    Requires admin role for multi_user auth.
    """
    user_id = getattr(request.state, "user_id", None)

    from nodetool.config.environment import Environment

    if Environment.get_auth_provider_kind() != "multi_user":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User management not available for current auth provider",
        )

    if user_id is None or not is_admin_user(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    manager = UserManager()

    try:
        manager.remove_user(username)
        return {"message": f"User '{username}' removed successfully"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
