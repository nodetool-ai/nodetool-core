"""FastAPI middleware for ResourceScope integration.

Provides per-request resource isolation for API endpoints using ResourceScope.
Each request gets its own database connections from shared pools.

Note: API call tracing is handled automatically by OpenTelemetry auto-instrumentation.
"""

from collections.abc import Callable
from typing import Optional

from starlette.types import ASGIApp, Receive, Scope, Send

from nodetool.config.logging_config import get_logger
from nodetool.runtime.resources import ResourceScope

log = get_logger(__name__)


class ResourceScopeMiddleware:
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
        app: ASGIApp,
        exempt_paths: Optional[set[str]] = None,
    ):
        """Initialize the middleware.

        Args:
            app: The FastAPI application
            exempt_paths: Set of paths that should skip ResourceScope
        """
        self.app = app
        self.exempt_paths = exempt_paths or {
            "/health",
            "/ping",
            "/docs",
            "/openapi.json",
            "/redoc",
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process ASGI requests with ResourceScope binding."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        try:
            async with ResourceScope():
                await self.app(scope, receive, send)
        except Exception as e:
            log.error(
                f"Error in ResourceScope middleware for {path}: {e}",
                exc_info=True,
            )
            raise
