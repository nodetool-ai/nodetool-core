"""
Admin token authentication for production admin endpoints.

In production, sensitive admin operations (/admin/*) require either:
1. ADMIN_TOKEN header (X-Admin-Token) - for admin-only access
2. WORKER_AUTH_TOKEN (Authorization: Bearer) - standard auth still required

This provides defense-in-depth: users need both valid user auth AND admin token
for sensitive operations.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Awaitable, Callable

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
