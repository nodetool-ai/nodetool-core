"""
Webhook Trigger Node
====================

This module provides a webhook trigger that starts an HTTP server to receive
webhook requests. Each incoming request is emitted as an event.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from nodetool.nodes.triggers.base import TriggerEvent, TriggerNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class WebhookTrigger(TriggerNode):
    """
    Trigger node that starts an HTTP server to receive webhook requests.

    Each incoming HTTP request is emitted as an event containing:
    - The request body (parsed as JSON if applicable)
    - Request headers
    - Query parameters
    - HTTP method

    This trigger is useful for:
    - Receiving notifications from external services
    - Building API endpoints that trigger workflows
    - Integration with third-party webhook providers

    Example:
        Set up a webhook trigger on port 8080 at /webhook:
        - External services can POST to http://localhost:8080/webhook
        - Each request triggers the workflow with the request data
    """

    port: int = Field(
        default=8080,
        description="Port to listen on for webhook requests",
        ge=1,
        le=65535,
    )
    path: str = Field(
        default="/webhook",
        description="URL path to listen on",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host address to bind to",
    )
    methods: list[str] = Field(
        default=["POST"],
        description="HTTP methods to accept",
    )
    secret: str = Field(
        default="",
        description="Optional secret for validating requests (checks X-Webhook-Secret header)",
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._server: asyncio.Server | None = None
        self._server_task: asyncio.Task | None = None

    async def setup_trigger(self, context: ProcessingContext) -> None:
        """Start the HTTP server for receiving webhooks."""
        log.info(
            f"Setting up webhook trigger on {self.host}:{self.port}{self.path}"
        )

        # Use aiohttp to create a simple HTTP server
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp is required for WebhookTrigger. "
                "Install it with: pip install aiohttp"
            )

        app = web.Application()

        async def handle_webhook(request: web.Request) -> web.Response:
            """Handle incoming webhook requests."""
            # Check method
            if request.method not in self.methods:
                return web.Response(
                    status=405,
                    text=f"Method {request.method} not allowed",
                )

            # Check secret if configured
            if self.secret:
                provided_secret = request.headers.get("X-Webhook-Secret", "")
                if provided_secret != self.secret:
                    return web.Response(status=401, text="Invalid secret")

            # Parse body
            body: Any = None
            content_type = request.content_type
            try:
                if content_type == "application/json":
                    body = await request.json()
                elif content_type in (
                    "application/x-www-form-urlencoded",
                    "multipart/form-data",
                ):
                    body = dict(await request.post())
                else:
                    body = await request.text()
            except Exception as e:
                log.warning(f"Failed to parse request body: {e}")
                body = await request.text()

            # Build event
            event: TriggerEvent = {
                "data": {
                    "body": body,
                    "headers": dict(request.headers),
                    "query": dict(request.query),
                    "method": request.method,
                    "path": request.path,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": f"{request.remote}",
                "event_type": "webhook",
            }

            # Push to queue
            self.push_event(event)

            return web.Response(
                status=200,
                text=json.dumps({"status": "accepted"}),
                content_type="application/json",
            )

        # Register handler for all methods at the configured path
        for method in self.methods:
            app.router.add_route(method, self.path, handle_webhook)

        # Also handle root path if different from configured path
        if self.path != "/":
            for method in self.methods:
                app.router.add_route(method, "/", handle_webhook)

        # Start the server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)

        try:
            await site.start()
            log.info(f"Webhook server started on {self.host}:{self.port}")
        except OSError as e:
            raise RuntimeError(
                f"Failed to start webhook server on {self.host}:{self.port}: {e}"
            )

        # Store references for cleanup
        self._runner = runner
        self._site = site

    async def wait_for_event(self, context: ProcessingContext) -> TriggerEvent | None:
        """Wait for the next webhook request."""
        return await self.get_event_from_queue()

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """Stop the HTTP server."""
        log.info("Cleaning up webhook trigger")

        if hasattr(self, "_site") and self._site:
            try:
                await self._site.stop()
            except Exception as e:
                log.warning(f"Error stopping site: {e}")

        if hasattr(self, "_runner") and self._runner:
            try:
                await self._runner.cleanup()
            except Exception as e:
                log.warning(f"Error cleaning up runner: {e}")
