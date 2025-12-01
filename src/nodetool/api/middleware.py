"""FastAPI middleware for ResourceScope integration.

Provides per-request resource isolation for API endpoints using ResourceScope.
Each request gets its own database connections from shared pools.
"""

from typing import Callable, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from nodetool.config.logging_config import get_logger
from nodetool.runtime.resources import ResourceScope

log = get_logger(__name__)


class ResourceScopeMiddleware(BaseHTTPMiddleware):
    """Middleware to provide ResourceScope for each API request.

    This middleware:
    1. Creates a ResourceScope for each request
    2. Binds it to contextvars so it's accessible throughout the request
    3. Automatically cleans up resources on request completion

    ResourceScope provides:
    - Per-request database connections from shared pools
    - Per-request adapter memoization
    - Automatic isolation between concurrent requests
    - Deterministic resource cleanup on exit

    Configuration:
    - Exempt paths can be configured to skip ResourceScope creation
    """

    def __init__(
        self,
        app: Callable,
        exempt_paths: Optional[set[str]] = None,
    ):
        """Initialize the middleware.

        Args:
            app: The FastAPI application
            exempt_paths: Set of paths that should skip ResourceScope
        """
        super().__init__(app)
        self.exempt_paths = exempt_paths or {
            "/health",
            "/ping",
            "/docs",
            "/openapi.json",
            "/redoc",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request with ResourceScope binding.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            The response
        """
        # Skip ResourceScope for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        try:
            # Create and use ResourceScope for the request
            # ResourceScope auto-detects database type and uses shared pools
            async with ResourceScope():
                return await call_next(request)
        except Exception as e:
            log.error(
                f"Error in ResourceScope middleware for {request.url.path}: {e}",
                exc_info=True,
            )
            # Let the exception propagate so normal handlers run and avoid double execution
            raise
