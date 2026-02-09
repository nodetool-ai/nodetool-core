"""
Admin token authentication for production admin endpoints.

In production, sensitive admin operations (/admin/*) require either:
1. ADMIN_TOKEN header (X-Admin-Token) - for admin-only access
2. SERVER_AUTH_TOKEN (Authorization: Bearer) - standard auth still required

This provides defense-in-depth: users need both valid user auth AND admin token
for sensitive operations.

For multi_user auth, admin role is enforced via require_admin() dependency.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

if TYPE_CHECKING:
    from fastapi import Request

log = get_logger(__name__)

# Paths that require admin token in production
ADMIN_TOKEN_REQUIRED_PATHS = [
    "/admin/models/",
    "/admin/cache/",
    "/admin/db/",
    "/admin/collections/",
    "/admin/storage/",
    "/admin/assets/",
]

# Paths that are always public (no auth needed)
PUBLIC_PATHS = ["/health", "/ping"]


def get_admin_token() -> str | None:
    """Get admin token from environment."""
    return os.environ.get("ADMIN_TOKEN")


def requires_admin_token(path: str) -> bool:
    """Check if path requires admin token."""
    return any(path.startswith(p) for p in ADMIN_TOKEN_REQUIRED_PATHS)


def create_admin_auth_middleware(
    enforce_in_production: bool = True,
) -> Callable[[Request, Callable], Awaitable]:
    """
    Create middleware that enforces admin token for sensitive endpoints.

    This middleware runs AFTER the regular auth middleware, so request.state.user_id
    is already set. It adds an additional check for admin operations.
    """

    async def middleware(request: Request, call_next):
        path = request.url.path

        # Skip for non-admin paths
        if not requires_admin_token(path):
            return await call_next(request)

        # Skip enforcement in development
        if not Environment.is_production() and not enforce_in_production:
            return await call_next(request)

        admin_token = get_admin_token()

        # If no admin token configured, allow (with warning logged at startup)
        if not admin_token:
            return await call_next(request)

        # Check for admin token header
        provided_token = request.headers.get("X-Admin-Token")

        if not provided_token:
            return JSONResponse(
                status_code=403,
                content={"detail": "Admin token required. Use X-Admin-Token header."},
            )

        if provided_token != admin_token:
            log.warning(f"Invalid admin token attempt for {path}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid admin token."},
            )

        return await call_next(request)

    return middleware


def is_admin_user(user_id: str) -> bool:
    """Check if user has admin role.

    Only works with multi_user auth provider. Other providers
    return False (no admin concept).

    Args:
        user_id: User ID from auth middleware

    Returns:
        True if user is admin, False otherwise
    """
    auth_provider = Environment.get_auth_provider_kind()

    # Only multi_user has admin roles
    if auth_provider != "multi_user":
        return False

    # Get user info from provider
    from nodetool.runtime.resources import get_user_auth_provider

    provider = get_user_auth_provider()

    if provider is None:
        return False

    # Check if provider has multi_user capability
    from nodetool.security.providers.multi_user import MultiUserAuthProvider

    if not isinstance(provider, MultiUserAuthProvider):
        return False

    user = provider.get_user(user_id)
    if user is None:
        return False

    return user.role == "admin"


async def require_admin(request: Request) -> None:
    """Dependency that enforces admin access via multi_user role.

    For multi_user auth provider:
        - Requires user to have admin role
        - Raises HTTPException 403 if user is not admin

    For other auth providers:
        - Allows access (X-Admin-Token middleware handles authorization)

    Usage:
        @router.post("/admin/collections")
        async def create_collection(..., _: None = Depends(require_admin)):
            # User is guaranteed to be admin here (for multi_user auth)
            pass
    """
    user_id = getattr(request.state, "user_id", None)

    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    # Only enforce admin role for multi_user auth provider
    auth_provider = Environment.get_auth_provider_kind()
    if auth_provider == "multi_user":
        if not is_admin_user(user_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    # For other auth providers, X-Admin-Token middleware handles authorization
