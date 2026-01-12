"""FastAPI middleware for ResourceScope integration and observability.

Provides per-request resource isolation for API endpoints using ResourceScope.
Each request gets its own database connections from shared pools.

Also provides API call tracing via OpenTelemetry-compatible spans for observability.
"""

from typing import Callable, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from nodetool.config.logging_config import get_logger
from nodetool.observability.tracing import (
    is_tracing_enabled,
    set_response_attributes,
    trace_api_call,
)
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


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware to provide distributed tracing for API requests.

    This middleware:
    1. Creates a trace span for each request
    2. Records HTTP method, path, status code, and timing
    3. Propagates trace context to downstream services

    Configuration:
    - Exempt paths can be configured to skip tracing
    """

    def __init__(
        self,
        app: Callable,
        exempt_paths: Optional[set[str]] = None,
    ):
        """Initialize the tracing middleware.

        Args:
            app: The FastAPI application
            exempt_paths: Set of paths that should skip tracing
        """
        super().__init__(app)
        self.exempt_paths = exempt_paths or {
            "/health",
            "/ping",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/favicon.ico",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request with tracing.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            The response
        """
        # Skip tracing for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Skip if tracing is disabled
        if not is_tracing_enabled():
            return await call_next(request)

        # Extract user ID from request state if available
        user_id = getattr(request.state, "user_id", None) if hasattr(request, "state") else None

        async with trace_api_call(
            method=request.method,
            path=request.url.path,
            user_id=user_id,
        ) as span:
            # Add request attributes
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.host", request.url.hostname or "")
            span.set_attribute("http.scheme", request.url.scheme)

            # Add query parameters count (not values for privacy)
            if request.query_params:
                span.set_attribute("http.query_params_count", len(request.query_params))

            try:
                response = await call_next(request)

                # Record response attributes
                set_response_attributes(
                    span,
                    status_code=response.status_code,
                    content_length=int(response.headers.get("content-length", 0)) or None,
                )

                return response

            except Exception as e:
                # Record exception details
                span.record_exception(e)
                raise
