"""
Async Docker reverse proxy server using FastAPI.

Provides on-demand container startup, HTTP/HTTPS proxying with streaming,
and Let's Encrypt ACME support.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, Response, StreamingResponse

from nodetool.proxy.config import ProxyConfig, ServiceConfig
from nodetool.proxy.docker_manager import DockerManager
from nodetool.proxy.filters import filter_request_headers, filter_response_headers

log = logging.getLogger(__name__)

ACME_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_acme_token_path(token: str, acme_webroot: str) -> Path | None:
    """
    Validate an ACME token and return a safe file path if valid.

    Args:
        token: The ACME challenge token from the URL.
        acme_webroot: The configured ACME webroot directory.

    Returns:
        A validated Path object pointing to the challenge file, or None if invalid.
    """
    # Validate token format - only alphanumeric, underscore, and hyphen allowed
    if not token or not ACME_TOKEN_RE.match(token):
        return None

    # Ensure token contains no path separators (additional safety check)
    if "/" in token or "\\" in token or ".." in token:
        return None

    try:
        acme_root = Path(acme_webroot).resolve(strict=False)
        # Redundant check for defense-in-depth and to help static analyzers
        # verify the token is safe before use in path construction
        safe_token = "".join(c for c in token if c.isalnum() or c in "_-")
        if safe_token != token:
            return None
        challenge_path = (acme_root / safe_token).resolve(strict=False)
    except (OSError, ValueError):
        return None

    # Verify path is within acme_root (prevents any path escape)
    if not challenge_path.is_relative_to(acme_root):
        return None

    return challenge_path


class AsyncReverseProxy:
    """Async reverse proxy with Docker container management."""

    def __init__(self, config: ProxyConfig):
        """
        Initialize the reverse proxy.

        Args:
            config: ProxyConfig instance with global and service settings.
        """
        self.config = config
        self.docker_manager = DockerManager(
            idle_timeout=config.global_.idle_timeout,
            network_name=config.global_.docker_network,
            connect_mode=config.global_.connect_mode,
        )

        # Precompute longest-prefix route index
        self.services_by_name: Dict[str, ServiceConfig] = {s.name: s for s in config.services}
        self.prefix_index: List[Tuple[str, ServiceConfig]] = sorted(
            [(s.path, s) for s in config.services],
            key=lambda x: (-len(x[0]), x[1].name),  # Sort by length desc, then name
        )

        # HTTP client for upstream requests
        self.httpx_client: Optional[httpx.AsyncClient] = None

    async def startup(self):
        """Initialize the proxy (async startup)."""
        # Create async HTTP client
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
        self.httpx_client = httpx.AsyncClient(timeout=timeout, limits=limits, http2=False)

        # Initialize Docker manager
        await self.docker_manager.initialize()

        # Register all services
        for service in self.config.services:
            self.docker_manager.register_service(service.name)

        log.info(
            "Async proxy started, managing %d services",
            len(self.config.services),
        )

    async def shutdown(self):
        """Shutdown the proxy and clean up resources."""
        if self.httpx_client:
            await self.httpx_client.aclose()

        await self.docker_manager.shutdown()
        log.info("Async proxy shutdown")

    def match_service(self, incoming_path: str) -> Tuple[Optional[ServiceConfig], Optional[str]]:
        """
        Match incoming path to service using longest-prefix matching.

        Args:
            incoming_path: Request path (must start with /).

        Returns:
            Tuple of (service_config, stripped_path) or (None, None) if no match.
        """
        for prefix, service in self.prefix_index:
            if prefix == "/":
                # Root match - everything goes here if no other match
                return service, incoming_path

            if incoming_path == prefix or incoming_path.startswith(prefix + "/"):
                # Matched prefix
                stripped = incoming_path[len(prefix) :] or "/"
                return service, stripped

        return None, None

    async def handle_proxy_request(self, request: Request, full_path: str) -> StreamingResponse:
        """
        Handle a proxied HTTP request.

        Args:
            request: FastAPI Request object.
            full_path: Request path (without leading slash from FastAPI).

        Returns:
            StreamingResponse with upstream response.

        Raises:
            HTTPException: If service not found or upstream error occurs.
        """
        incoming_path = "/" + (full_path or "")
        service, stripped_path = self.match_service(incoming_path)

        if not service:
            raise HTTPException(status_code=404, detail="No backend for path")

        name = service.name
        runtime = self.docker_manager.runtime[name]

        # Serialize cold-starts per service using lock
        async with runtime.lock:
            # Ensure container is running
            host_port = await self.docker_manager.ensure_running(service)
            runtime.host_port = host_port

        # Update last access time
        runtime.last_access = time.time()

        # Build upstream URL based on connection mode
        if self.config.global_.connect_mode == "docker_dns":
            upstream_host = service.name
            upstream_port = service.internal_port
        else:
            upstream_host = "127.0.0.1"
            upstream_port = host_port

        target_url = f"http://{upstream_host}:{upstream_port}{stripped_path}"
        if request.url.query:
            target_url += f"?{request.url.query}"

        log.debug(f"Proxying {request.method} {incoming_path} -> {target_url}")

        # Filter request headers
        upstream_headers = filter_request_headers(dict(request.headers))

        # Add service auth token if configured
        if service.auth_token:
            upstream_headers["Authorization"] = f"Bearer {service.auth_token}"

        # Stream request body to upstream
        async def request_body_iter():
            async for chunk in request.stream():
                if chunk:
                    yield chunk

        assert self.httpx_client is not None

        try:
            async with self.httpx_client.stream(
                method=request.method,
                url=target_url,
                headers=upstream_headers,
                content=request_body_iter(),
            ) as upstream_response:
                # Filter response headers
                response_headers = filter_response_headers(dict(upstream_response.headers))

                # For HEAD requests, don't stream body
                if request.method.upper() == "HEAD":

                    async def empty_generator():
                        if False:
                            yield b""  # Never executes

                    return StreamingResponse(  # type: ignore[return-value]
                        empty_generator(),
                        status_code=upstream_response.status_code,
                        headers=response_headers,
                    )

                content = await upstream_response.aread()
                return Response(  # type: ignore[return-value]
                    content,
                    status_code=upstream_response.status_code,
                    headers=response_headers,
                )

        except httpx.RequestError as e:
            log.error(f"Upstream error for {incoming_path}: {e}")
            raise HTTPException(status_code=502, detail=f"Upstream error: {e!s}") from e

    async def handle_status(self) -> StreamingResponse:
        """
        Get status of all services.

        Returns:
            StreamingResponse with JSON status report.
        """

        async def build_status_report():
            """Build complete status report for all services."""
            report = []

            async def get_service_status(service: ServiceConfig):
                """Get status for a single service."""
                name = service.name
                runtime = self.docker_manager.runtime[name]
                status_info = await self.docker_manager.get_container_status(name)

                # Extract host port from port map if available
                port_map = status_info.get("port_map", {})
                internal_key = f"{ServiceConfig.INTERNAL_PORT}/tcp"
                host_port = None

                if port_map.get(internal_key):
                    host_port = int(port_map[internal_key][0]["HostPort"])
                elif self.config.global_.connect_mode == "docker_dns":
                    host_port = ServiceConfig.INTERNAL_PORT

                return {
                    "name": name,
                    "path": service.path,
                    "status": status_info.get("status", "unknown"),
                    "host_port": host_port,
                    "last_access_epoch": runtime.last_access or None,
                }

            # Gather status for all services concurrently
            tasks = [get_service_status(svc) for svc in self.config.services]
            report = await asyncio.gather(*tasks)

            return json.dumps(report, indent=2).encode("utf-8")

        payload = await build_status_report()

        async def stream_json():
            """Stream JSON payload."""
            yield payload

        return StreamingResponse(stream_json(), media_type="application/json")

    async def handle_acme_challenge(self, token: str) -> PlainTextResponse:
        """
        Serve ACME challenge token.

        Args:
            token: Challenge token.

        Returns:
            PlainTextResponse with challenge file content or 404.
        """
        challenge_path = validate_acme_token_path(token, self.config.global_.acme_webroot)
        if challenge_path is None:
            log.warning(f"Invalid ACME token format: {token}")
            return PlainTextResponse("Invalid token", status_code=400)

        if not challenge_path.is_file():
            return PlainTextResponse("Not found", status_code=404)

        try:

            async def stream_file():
                async with aiofiles.open(challenge_path, "rb") as f:
                    while True:
                        chunk = await f.read(8192)
                        if not chunk:
                            break
                        yield chunk

            return StreamingResponse(stream_file(), media_type="text/plain")  # type: ignore[return-value]
        except OSError as e:
            log.error(f"Failed to read ACME challenge {token}: {e}")
            return PlainTextResponse("Error reading file", status_code=500)


def create_proxy_app(config: ProxyConfig) -> FastAPI:
    """
    Create and configure the FastAPI proxy application.

    Args:
        config: ProxyConfig instance.

    Returns:
        Configured FastAPI app ready to run.
    """
    app = FastAPI(
        title="Async Docker Reverse Proxy",
        description="On-demand Docker container proxy with Let's Encrypt support",
    )

    proxy = AsyncReverseProxy(config)

    @app.on_event("startup")
    async def startup():
        await proxy.startup()
        # Ensure ACME webroot exists
        acme_root = Path(config.global_.acme_webroot)
        acme_root.mkdir(parents=True, exist_ok=True)

    @app.on_event("shutdown")
    async def shutdown():
        await proxy.shutdown()

    # ---- Auth dependency ----
    async def require_bearer_auth(request: Request):
        """Dependency to check Bearer token authentication."""
        auth_header = request.headers.get("authorization", "")
        expected = f"Bearer {config.global_.bearer_token}"

        if auth_header != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    # ---- ACME challenge endpoint (no auth) ----
    @app.get("/.well-known/acme-challenge/{token}", include_in_schema=False)
    async def acme_challenge(token: str):
        """Serve ACME challenge tokens for Let's Encrypt."""
        return await proxy.handle_acme_challenge(token)

    # ---- Status endpoint (auth required) ----
    @app.get("/status", dependencies=[Depends(require_bearer_auth)])
    async def status():
        """Get status of all managed services."""
        return await proxy.handle_status()

    # ---- Container health endpoint (no auth) ----
    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return PlainTextResponse("ok", status_code=200)

    # ---- Catch-all proxy endpoint (auth required) ----
    HTTP_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

    @app.api_route(
        "/{full_path:path}",
        methods=HTTP_METHODS,
        dependencies=[Depends(require_bearer_auth)],
        include_in_schema=False,
    )
    async def proxy_request(full_path: str, request: Request):
        """Proxy HTTP requests to managed services."""
        return await proxy.handle_proxy_request(request, full_path)

    return app


def create_acme_only_app(config: ProxyConfig) -> FastAPI:
    """
    Create a minimal FastAPI app for serving ACME challenges and HTTP-to-HTTPS redirects.

    This app does not initialize the Docker manager; it only serves files from the ACME webroot.
    """
    app = FastAPI(
        title="ACME Challenge Server",
        description="Serves HTTP-01 challenges for certificate issuance",
    )

    acme_root = Path(config.global_.acme_webroot)
    acme_root.mkdir(parents=True, exist_ok=True)

    @app.get("/.well-known/acme-challenge/{token}", include_in_schema=False)
    async def acme_only(token: str):
        challenge_path = validate_acme_token_path(token, config.global_.acme_webroot)
        if challenge_path is None:
            return PlainTextResponse("Invalid token", status_code=400)

        if not challenge_path.is_file():
            return PlainTextResponse("Not found", status_code=404)

        async def stream_file():
            async with aiofiles.open(challenge_path, "rb") as fh:
                while True:
                    chunk = await fh.read(8192)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(stream_file(), media_type="text/plain")

    @app.get("/{path:path}", include_in_schema=False)
    async def redirect_all(path: str, request: Request):
        if not config.global_.http_redirect_to_https:
            return PlainTextResponse("Use HTTPS", status_code=426)

        host = request.headers.get("host", config.global_.domain)
        if host in {"testserver", "localhost", "127.0.0.1"}:
            host = config.global_.domain
        stripped = path.lstrip("/")
        target = f"https://{host}/{stripped}" if stripped else f"https://{host}/"
        return RedirectResponse(url=target, status_code=308)

    return app


async def run_proxy_app(
    config: ProxyConfig,
    host: str = "0.0.0.0",
    port: int = 443,
    use_tls: bool = True,
) -> None:
    """
    Run the proxy app using uvicorn.

    Args:
        config: ProxyConfig instance.
        host: Bind host.
        port: Bind port.
        use_tls: Whether to use TLS.
    """
    import uvicorn

    app = create_proxy_app(config)

    if use_tls:
        if not config.global_.tls_certfile or not config.global_.tls_keyfile:
            raise ValueError("TLS requested but tls_certfile or tls_keyfile not configured")

        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_certfile=config.global_.tls_certfile,
            ssl_keyfile=config.global_.tls_keyfile,
            log_level=config.global_.log_level.lower(),
        )
    else:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=config.global_.log_level.lower(),
        )


async def run_proxy_daemon(config: ProxyConfig) -> None:
    """
    Run ACME (HTTP) and proxy (HTTPS) servers concurrently when TLS is configured.

    Falls back to single HTTP server when TLS material is not present.
    """
    import uvicorn

    use_tls = bool(config.global_.tls_certfile and config.global_.tls_keyfile)

    if not use_tls:
        proxy_app = create_proxy_app(config)
        proxy_cfg = uvicorn.Config(
            proxy_app,
            host="0.0.0.0",
            port=config.global_.listen_http,
            log_level=config.global_.log_level.lower(),
            loop="asyncio",
        )
        proxy_server = uvicorn.Server(proxy_cfg)
        proxy_server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        await proxy_server.serve()
        return

    acme_app = create_acme_only_app(config)
    proxy_app = create_proxy_app(config)

    acme_cfg = uvicorn.Config(
        acme_app,
        host="0.0.0.0",
        port=config.global_.listen_http,
        log_level=config.global_.log_level.lower(),
        access_log=False,
        loop="asyncio",
    )
    proxy_cfg = uvicorn.Config(
        proxy_app,
        host="0.0.0.0",
        port=config.global_.listen_https,
        ssl_certfile=config.global_.tls_certfile,
        ssl_keyfile=config.global_.tls_keyfile,
        log_level=config.global_.log_level.lower(),
        loop="asyncio",
    )

    acme_server = uvicorn.Server(acme_cfg)
    proxy_server = uvicorn.Server(proxy_cfg)

    # Prevent conflicting signal handlers when running both servers in one process
    acme_server.install_signal_handlers = lambda: None  # type: ignore[assignment]
    proxy_server.install_signal_handlers = lambda: None  # type: ignore[assignment]

    await asyncio.gather(acme_server.serve(), proxy_server.serve())
