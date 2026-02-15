from __future__ import annotations

import asyncio
import datetime
import logging
import mimetypes
import os
import platform
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, ClassVar

from fastapi import APIRouter, FastAPI, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from uvicorn import run as uvicorn

from nodetool.config.env_guard import RUNNING_PYTEST
from nodetool.config.environment import Environment
from nodetool.config.logging_config import configure_logging, get_logger

_windows_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
if platform.system() == "Windows" and _windows_policy is not None:
    asyncio.set_event_loop_policy(_windows_policy())

# FIX: Windows: mimetypes.guess_type() returns None for some files
# See:
# - https://github.com/encode/starlette/issues/829
# - https://github.com/pallets/flask/issues/1045
#
# The Python mimetypes module on Windows pulls values from the registry.
# If mimetypes.guess_type() returns None for some files, it may indicate
# that the Windows registry is corrupted or missing entries.
#
# This issue affects Windows systems and may cause problems with
# file type detection in web frameworks like Starlette or Flask.
#
# Let's add the missing mime types to the mimetypes module.
mimetypes.init()
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/jpeg", ".jpeg")
mimetypes.add_type("image/gif", ".gif")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/pdf", ".pdf")
mimetypes.add_type("font/woff", ".woff")
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("application/xml", ".xml")
mimetypes.add_type("text/plain", ".txt")


def initialize_sentry():
    """
    Initialize Sentry error tracking if SENTRY_DSN is configured.
    """
    # Guard against repeated initialization when create_app is called multiple times
    if getattr(initialize_sentry, "_initialized", False):
        return

    sentry_dsn = Environment.get("SENTRY_DSN", None)
    if sentry_dsn:
        import sentry_sdk  # type: ignore

        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=Environment.get_env(),
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            traces_sample_rate=1.0,
            # Set profiles_sample_rate to 1.0 to profile 100%
            # of sampled transactions.
            profiles_sample_rate=1.0,
        )
        initialize_sentry._initialized = True  # type: ignore[attr-defined]


log = get_logger(__name__)

# Silence SQLite and SQLAlchemy logging
_LOG_SEPARATOR_WIDTH = 72


class HealthCheckFilter(logging.Filter):
    """Filter to suppress logging for /health endpoint requests."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to suppress health check logs."""
        message = record.getMessage()
        # Check if the log message contains /health endpoint
        return "/health" not in message


def create_health_check_filter():
    """Create a filter to suppress logging for /health endpoint requests."""
    return HealthCheckFilter()


def _fmt_log_value(value: Any) -> str:
    return "<not set>" if value in (None, "") else str(value)


def _log_section(title: str) -> None:
    bar = "=" * _LOG_SEPARATOR_WIDTH
    log.info(bar)
    log.info(" %s", title)
    log.info(bar)


def _log_kv(title: str, entries: dict[str, Any]) -> None:
    if not entries:
        return
    max_key = max(len(key) for key in entries)
    log.info("[%s]", title)
    for key, value in entries.items():
        log.info("  %-*s : %s", max_key, key, _fmt_log_value(value))


def get_nodetool_package_source_folders() -> list[str]:
    """
    Thin wrapper to expose package source folders without importing the package registry at import time.

    This helper is patched in tests; the real implementation is loaded lazily to avoid heavy imports
    unless the folders are needed for reload/watch mode.
    """
    from nodetool.packages.registry import get_nodetool_package_source_folders as _impl

    return [str(path) for path in _impl()]


class ExtensionRouterRegistry:
    _instance = None
    _routers: ClassVar[list[APIRouter]] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_router(cls, router: APIRouter) -> None:
        """Register a new router from an extension."""
        if router not in cls._routers:
            cls._routers.append(router)

    @classmethod
    def get_routers(cls) -> list[APIRouter]:
        """Get all registered extension routers."""
        return cls._routers.copy()


class ServerMode(str, Enum):
    """High-level server runtime modes."""

    DESKTOP = "desktop"
    PUBLIC = "public"
    PRIVATE = "private"

    @classmethod
    def from_value(cls, value: str | ServerMode | None) -> ServerMode:
        if isinstance(value, cls):
            return value
        normalized = (value or "desktop").lower()
        try:
            return cls(normalized)
        except ValueError:
            raise ValueError(f"Invalid server mode '{value}'. Expected one of: desktop, public, private.") from None


@dataclass(slots=True)
class ServerFeatures:
    include_default_api_routers: bool = True
    include_openai_router: bool = True
    include_deploy_admin_router: bool = True
    include_deploy_collection_router: bool = True
    include_deploy_storage_router: bool = True
    enable_main_ws: bool = True
    enable_terminal_ws: bool = True
    enable_hf_download_ws: bool = True
    mount_static: bool = True
    enable_mcp: bool = False  # MCP server disabled by default, enable via --mcp


def _features_for_mode(mode: ServerMode) -> ServerFeatures:
    if mode == ServerMode.DESKTOP:
        return ServerFeatures()
    if mode == ServerMode.PUBLIC:
        return ServerFeatures(
            include_deploy_admin_router=False,
            include_deploy_collection_router=False,
            enable_terminal_ws=False,
            enable_hf_download_ws=False,
        )
    # ServerMode.PRIVATE
    return ServerFeatures(
        enable_terminal_ws=False,
        enable_hf_download_ws=False,
    )


def _load_default_routers() -> list[APIRouter]:
    """
    Lazily import and assemble the default routers to avoid heavy imports at
    module import time.
    """
    from . import (
        admin_secrets,
        asset,
        collection,
        cost,
        debug,
        fal_schema,
        file,
        font,
        job,
        memory,
        message,
        model,
        node,
        oauth,
        settings,
        skills,
        storage,
        thread,
        users,
        vibecoding,
        workflow,
        workspace,
    )

    routers: list[APIRouter] = [
        admin_secrets.router,
        asset.router,
        cost.router,
        message.router,
        thread.router,
        model.router,
        node.router,
        oauth.router,
        skills.router,
        workflow.router,
        workspace.router,
        storage.router,
        storage.temp_router,
        fal_schema.router,
        font.router,
        debug.router,
        job.router,
        settings.router,
        memory.router,
        vibecoding.router,
        collection.router,
        users.router,
    ]

    # Add file router only for non-production environments
    if not Environment.is_production():
        routers.append(file.router)

    return routers


def _load_deploy_routers(
    include_admin_router: bool = True,
    include_collection_router: bool = True,
    include_storage_router: bool = True,
) -> list[APIRouter]:
    """
    Load deployment/admin routers.

    These provide:
    - Admin operations (model downloads, cache management)
    - Collection management via /admin/collections/*
    - Storage management via /admin/storage/* and /storage/*
    - Workflow execution via /api/workflows/*
    """
    routers: list[APIRouter] = []
    if include_admin_router:
        from nodetool.deploy.admin_routes import create_admin_router

        routers.append(create_admin_router())
    if include_collection_router:
        from nodetool.deploy.collection_routes import create_collection_router

        routers.append(create_collection_router())
    if include_storage_router:
        from nodetool.deploy.storage_routes import (
            create_admin_storage_router,
            create_public_storage_router,
        )

        routers.append(create_admin_storage_router())
        routers.append(create_public_storage_router())
    return routers


async def check_ollama_availability(port: int = 11434, timeout: float = 2.0) -> bool:
    """
    Check if Ollama is responsive at the specified port.

    Args:
        port: The port to check (default: 11434)
        timeout: Request timeout in seconds (default: 2.0)

    Returns:
        True if Ollama is responsive, False otherwise
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            # Try localhost first, then 127.0.0.1
            for host in ("localhost", "127.0.0.1"):
                try:
                    response = await client.get(f"http://{host}:{port}/api/tags")
                    if response.status_code == 200:
                        return True
                except Exception as e:
                    log.debug(f"Ollama not available at {host}:{port}: {e}")
                    continue
    except Exception as e:
        log.debug(f"Failed to check Ollama availability: {e}")
    return False


def setup_ollama_url():
    """
    Set OLLAMA_API_URL environment variable if Ollama is available at default port
    and the variable is not already set.
    """
    if os.environ.get("OLLAMA_API_URL"):
        log.info(f"OLLAMA_API_URL already set to: {os.environ.get('OLLAMA_API_URL')}")
        return

    # Check if Ollama is available at default port
    try:
        # Use synchronous check during startup
        import httpx

        with httpx.Client(timeout=2.0) as client:
            # Prefer localhost (per developer request), fallback to 127.0.0.1
            candidates = [
                ("http://localhost:11434", "/api/tags"),
                ("http://127.0.0.1:11434", "/api/tags"),
            ]
            for base, path in candidates:
                try:
                    response = client.get(base + path)
                    if response.status_code == 200:
                        os.environ["OLLAMA_API_URL"] = base
                        log.info(f"Detected Ollama at {base}, enabling provider via OLLAMA_API_URL")
                        return
                except Exception:
                    continue
    except Exception as e:
        log.debug(f"Could not check Ollama availability: {e}")

    log.info("Ollama not detected at localhost:11434, using OLLAMA_API_URL from environment if provided")


def create_app(
    origins: list[str] | None = None,
    routers: list[APIRouter] | None = None,
    static_folder: str | None = None,
    apps_folder: str | None = None,
    *,
    mode: str | ServerMode | None = None,
    auth_provider: str | None = None,
    include_default_api_routers: bool | None = None,
    include_openai_router: bool | None = None,
    include_deploy_admin_router: bool | None = None,
    include_deploy_collection_router: bool | None = None,
    include_deploy_storage_router: bool | None = None,
    enable_main_ws: bool | None = None,
    enable_terminal_ws: bool | None = None,
    enable_hf_download_ws: bool | None = None,
    mount_static: bool | None = None,
    enable_mcp: bool | None = None,
):
    # Initialize Sentry only when the application is created, not on module import
    initialize_sentry()

    server_mode = ServerMode.from_value(mode)
    features = _features_for_mode(server_mode)
    feature_overrides: dict[str, bool | None] = {
        "include_default_api_routers": include_default_api_routers,
        "include_openai_router": include_openai_router,
        "include_deploy_admin_router": include_deploy_admin_router,
        "include_deploy_collection_router": include_deploy_collection_router,
        "include_deploy_storage_router": include_deploy_storage_router,
        "enable_main_ws": enable_main_ws,
        "enable_terminal_ws": enable_terminal_ws,
        "enable_hf_download_ws": enable_hf_download_ws,
        "mount_static": mount_static,
        "enable_mcp": enable_mcp,
    }
    for key, value in feature_overrides.items():
        if value is not None:
            features = replace(features, **{key: value})

    if auth_provider:
        os.environ["AUTH_PROVIDER"] = auth_provider.lower()

    origins = ["*"] if origins is None else origins
    routers = _load_default_routers() if (routers is None and features.include_default_api_routers) else (routers or [])

    # Centralized dotenv loading for consistency with deploy.fastapi_server
    from nodetool.config.environment import load_dotenv_files

    load_dotenv_files()

    auth_kind = Environment.get_auth_provider_kind()
    if server_mode == ServerMode.PUBLIC and auth_kind != "supabase":
        raise RuntimeError("Public server mode requires AUTH_PROVIDER=supabase.")
    if server_mode == ServerMode.PRIVATE and auth_kind not in ("static", "multi_user", "supabase"):
        raise RuntimeError("Private server mode requires AUTH_PROVIDER=static, multi_user, or supabase.")

    # Log loaded environment configuration (with defaults resolved by Environment)
    _log_section("NodeTool Server Startup")
    startup_vars = {
        "ENV": Environment.get("ENV"),
        "LOG_LEVEL": Environment.get("LOG_LEVEL"),
        "DEBUG": Environment.get("DEBUG"),
        "AUTH_PROVIDER": Environment.get("AUTH_PROVIDER"),
        "DB_PATH": Environment.get("DB_PATH"),
        "CHROMA_PATH": Environment.get("CHROMA_PATH"),
        "JOB_EXECUTION_STRATEGY": Environment.get("JOB_EXECUTION_STRATEGY"),
        "OLLAMA_API_URL": Environment.get("OLLAMA_API_URL"),
        "NODETOOL_API_URL": Environment.get("NODETOOL_API_URL"),
        "NODETOOL_ENABLE_TERMINAL_WS": Environment.get("NODETOOL_ENABLE_TERMINAL_WS"),
        "WORKER_ID": Environment.get("WORKER_ID"),
    }
    _log_kv("Environment", startup_vars)

    feature_flags = {
        "mode": server_mode.value,
        "include_default_api_routers": features.include_default_api_routers,
        "include_openai_router": features.include_openai_router,
        "include_deploy_admin_router": features.include_deploy_admin_router,
        "include_deploy_collection_router": features.include_deploy_collection_router,
        "include_deploy_storage_router": features.include_deploy_storage_router,
        "enable_main_ws": features.enable_main_ws,
        "enable_terminal_ws": features.enable_terminal_ws,
        "enable_hf_download_ws": features.enable_hf_download_ws,
        "enable_mcp": features.enable_mcp,
        "mount_static": features.mount_static,
    }
    _log_kv("Feature Flags", feature_flags)

    # Log key configuration details
    supabase_url = Environment.get("SUPABASE_URL")
    postgres_db = Environment.get("POSTGRES_DB")
    db_path = Environment.get("DB_PATH")
    auth_provider = Environment.get("AUTH_PROVIDER", "local")

    db_summary: dict[str, Any]
    if supabase_url:
        db_summary = {
            "backend": "Supabase",
            "supabase_url": supabase_url,
        }
    elif postgres_db:
        postgres_host = Environment.get("POSTGRES_HOST", "localhost")
        db_summary = {
            "backend": "PostgreSQL",
            "postgres_host": postgres_host,
            "postgres_db": postgres_db,
        }
    elif db_path:
        db_summary = {
            "backend": "SQLite",
            "db_path": db_path,
        }
    else:
        db_summary = {
            "backend": "<not configured>",
        }
        log.warning("No database configured (SUPABASE_URL, POSTGRES_DB, or DB_PATH)")
    _log_kv("Database", db_summary)

    _log_kv("Authentication", {"provider": auth_provider, "enforce_auth": Environment.enforce_auth()})

    # Log storage configuration
    asset_bucket = Environment.get("ASSET_BUCKET", "images")
    asset_temp_bucket = Environment.get("ASSET_TEMP_BUCKET")
    asset_domain = Environment.get("ASSET_DOMAIN")
    s3_endpoint = Environment.get("S3_ENDPOINT_URL")

    storage_summary: dict[str, Any]
    if supabase_url:
        storage_summary = {
            "backend": "Supabase",
            "asset_bucket": asset_bucket,
            "asset_temp_bucket": asset_temp_bucket,
            "asset_domain": asset_domain,
        }
    elif s3_endpoint:
        storage_summary = {
            "backend": "S3",
            "s3_endpoint_url": s3_endpoint,
            "s3_region": Environment.get("S3_REGION", "us-east-1"),
            "asset_bucket": asset_bucket,
            "asset_temp_bucket": asset_temp_bucket,
            "asset_domain": asset_domain,
        }
    else:
        storage_summary = {
            "backend": "File",
            "asset_bucket": asset_bucket,
            "asset_temp_bucket": asset_temp_bucket,
            "asset_domain": asset_domain,
        }
    _log_kv("Asset Storage", storage_summary)

    # Ollama check, env diagnostics, and security checks are deferred to
    # the async lifespan to avoid blocking app creation.

    # Use FastAPI lifespan API instead of deprecated on_event hooks
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Check if Ollama is available and set OLLAMA_API_URL if not already set.
        # Run in a thread because it makes synchronous HTTP requests with timeouts.
        await asyncio.to_thread(setup_ollama_url)

        # Log comprehensive environment diagnostics (secure masking of secrets)
        # This is particularly useful for Electron, Docker, and production deployments
        from nodetool.config.env_diagnostics import log_env_diagnostics

        log_env_diagnostics(logger=log, check_permissions=True)

        # Run startup security checks to warn about insecure configurations
        # Import is local to avoid circular imports (security module imports config which may import api)
        from nodetool.security.startup_checks import run_startup_security_checks

        run_startup_security_checks(raise_on_critical=False)

        # Validate production requirements
        if Environment.is_production():
            if not os.environ.get("SECRETS_MASTER_KEY"):
                raise RuntimeError(
                    "SECRETS_MASTER_KEY environment variable must be set in production. "
                    'Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
                )
            if not os.environ.get("ADMIN_TOKEN"):
                log.warning(
                    "ADMIN_TOKEN not set - admin endpoints (/admin/*) will not require "
                    "additional admin authentication beyond standard auth"
                )

        _log_section("NodeTool Runtime Initialization")

        # Initialize tracing (may load instrumentation libraries)
        from nodetool.observability.tracing import init_tracing

        await asyncio.to_thread(init_tracing, service_name="nodetool-api")

        # Run database migrations before starting
        from nodetool.models.migrations import run_startup_migrations

        if not RUNNING_PYTEST:
            try:
                await run_startup_migrations()
                log.info("[Startup] Database migrations completed successfully")
            except Exception as e:
                log.error(f"[Startup] Failed to run database migrations: {e}", exc_info=True)
                raise

        # Populate mock data if --mock flag is enabled
        if os.environ.get("NODETOOL_MOCK_MODE") == "1":
            log.info("[Startup] Mock mode enabled; populating database with test data")
            try:
                from nodetool.api.mock_data import populate_mock_data
                from nodetool.runtime.resources import ResourceScope

                async with ResourceScope():
                    result = await populate_mock_data(user_id="1")
                    log.info(f"[Startup] Mock data populated successfully: {result}")
            except Exception as e:
                log.error(f"[Startup] Failed to populate mock data: {e}", exc_info=True)

        # Start job execution manager cleanup task
        from nodetool.workflows.job_execution_manager import JobExecutionManager

        job_manager = JobExecutionManager.get_instance()
        await job_manager.start_cleanup_task()
        log.info("[Startup] JobExecutionManager cleanup task started")

        # Hand control back to the app
        yield

        # Shutdown: cleanup resources
        _log_section("NodeTool Server Shutdown")
        log.info("[Shutdown] Cleaning up resources")

        job_manager = JobExecutionManager.get_instance()
        await job_manager.shutdown()
        log.info("[Shutdown] JobExecutionManager shutdown complete")

        # Shutdown SQLite connection pools with WAL checkpointing
        from nodetool.runtime.db_sqlite import shutdown_all_sqlite_pools

        await shutdown_all_sqlite_pools()
        log.info("[Shutdown] SQLite connection pools shutdown complete")

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )

    from nodetool.api.middleware import ResourceScopeMiddleware
    from nodetool.runtime.resources import (
        get_static_auth_provider,
        get_user_auth_provider,
    )
    from nodetool.security.http_auth import create_http_auth_middleware

    static_provider = get_static_auth_provider()
    user_provider = get_user_auth_provider()
    enforce_auth = Environment.enforce_auth()
    auth_middleware = create_http_auth_middleware(
        static_provider=static_provider,
        user_provider=user_provider,
        enforce_auth=enforce_auth,
    )
    app.middleware("http")(auth_middleware)

    # Add admin token middleware for production admin endpoints
    if Environment.is_production():
        from nodetool.security.admin_auth import create_admin_auth_middleware

        admin_auth = create_admin_auth_middleware()
        app.middleware("http")(admin_auth)

    if not RUNNING_PYTEST:
        app.add_middleware(ResourceScopeMiddleware)  # type: ignore[arg-type]

    # Mount OpenAI-compatible endpoints
    # In production, use environment variables for configuration
    from nodetool.metadata.types import Provider

    default_provider = os.environ.get("CHAT_PROVIDER", Provider.Ollama.value)
    default_model = os.environ.get("DEFAULT_MODEL", "llama3.2:latest")
    tools_str = os.environ.get("NODETOOL_TOOLS", "")
    tools_list = [t.strip() for t in tools_str.split(",") if t.strip()] if tools_str else []

    if features.include_openai_router:
        from nodetool.api.openai import create_openai_compatible_router

        app.include_router(
            create_openai_compatible_router(
                provider=default_provider,
                default_model=default_model,
                tools=tools_list,
            )
        )

    for router in routers:
        app.include_router(router)

    # Include deploy routers for admin and production operations
    for router in _load_deploy_routers(
        include_admin_router=features.include_deploy_admin_router,
        include_collection_router=features.include_deploy_collection_router,
        include_storage_router=features.include_deploy_storage_router,
    ):
        app.include_router(router)

    for extension_router in ExtensionRouterRegistry().get_routers():
        app.include_router(extension_router)

    if not Environment.is_production():

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            log.error(f"Request validation error: {exc}")
            return JSONResponse({"detail": exc.errors()}, status_code=422)

    @app.get("/health")
    async def health_check() -> str:
        return "OK"

    @app.get("/ping")
    async def ping():
        """Health check with system information."""
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }

    @app.get("/editor/{workflow_id}")
    async def editor_redirect(workflow_id: str):
        return RedirectResponse(url="/")

    if features.enable_hf_download_ws and not Environment.is_production():
        from nodetool.integrations.huggingface.hf_websocket import (
            huggingface_download_endpoint,
        )

        app.add_websocket_route("/ws/download", huggingface_download_endpoint)

    async def _authenticate_websocket(websocket: WebSocket):
        if not enforce_auth:
            static = get_static_auth_provider()
            return None, static.user_id

        token = static_provider.extract_token_from_ws(websocket.headers, websocket.query_params)
        if not token:
            # Must accept before closing to avoid 403
            await websocket.accept()
            await websocket.close(code=1008, reason="Missing authentication")
            log.warning("WebSocket connection rejected: Missing authentication header")
            return None, None

        static_result = await static_provider.verify_token(token)
        if static_result.ok and static_result.user_id:
            return token, static_result.user_id

        if Environment.get_auth_provider_kind() == "supabase":
            if not user_provider:
                await websocket.accept()
                await websocket.close(code=1008, reason="Authentication provider not configured")
                log.warning("WebSocket connection rejected: Auth provider not configured")
                return None, None
            user_result = await user_provider.verify_token(token)
            if user_result.ok and user_result.user_id:
                return token, user_result.user_id
            await websocket.accept()
            await websocket.close(code=1008, reason="Invalid authentication")
            log.warning("WebSocket connection rejected: Invalid Supabase token")
            return None, None

        await websocket.accept()
        await websocket.close(code=1008, reason="Invalid authentication")
        log.warning("WebSocket connection rejected: Invalid token")
        return None, None

    if features.enable_main_ws:

        @app.websocket("/ws")
        async def unified_websocket_endpoint(websocket: WebSocket):
            """
            Unified WebSocket endpoint for both workflow execution and chat communications.

            This is the recommended endpoint for new integrations. It handles:
            - Workflow job execution (run_job, cancel_job, get_status, etc.)
            - Chat message processing (with AI providers)
            - Real-time bidirectional updates

            The endpoint routes messages based on their structure:
            - Messages with 'command' field: Workflow operations
            - Messages with 'role' or 'content': Chat messages
            - Control messages (stop, ping, etc.): Connection control

            See docs/websocket-api.md for detailed API documentation.
            """
            from nodetool.integrations.websocket.unified_websocket_runner import (
                UnifiedWebSocketRunner,
            )

            token, user_id = await _authenticate_websocket(websocket)
            if user_id is None:
                return
            runner = UnifiedWebSocketRunner(auth_token=token or "", user_id=user_id)
            await runner.run(websocket)

    if features.enable_terminal_ws:

        @app.websocket("/ws/terminal")
        async def terminal_websocket_endpoint(websocket: WebSocket):
            from nodetool.integrations.websocket.terminal_runner import (
                TerminalWebSocketRunner,
            )

            # Only allow terminal access when explicitly enabled and never in production
            if Environment.is_production() or not TerminalWebSocketRunner.is_enabled():
                # Must accept before closing to raise WebSocketDisconnect in tests
                await websocket.accept()
                await websocket.close(code=1008, reason="Terminal access disabled")
                return

            # Skip authentication in dev mode for convenience
            if not enforce_auth:
                token = None
                user_id = "1"  # Default dev user
            else:
                token, user_id = await _authenticate_websocket(websocket)
                if user_id is None:
                    return

            runner = TerminalWebSocketRunner(auth_token=token or "", user_id=user_id)
            await runner.run(websocket)

        # Backwards-compatible terminal websocket route (older clients/tests)
        @app.websocket("/terminal")
        async def terminal_websocket_endpoint_legacy(websocket: WebSocket):
            await terminal_websocket_endpoint(websocket)

    # =========================================================================
    # MCP Server Integration
    # =========================================================================
    # Mount the FastMCP server to expose NodeTool tools and resources to
    # MCP-compatible clients (e.g., Claude Desktop, AI assistants).
    # Enabled via --mcp flag. Endpoints available when enabled:
    #   - /mcp/sse      : SSE endpoint for streaming communication
    #   - /mcp/messages : POST endpoint for message handling
    # The MCP server is unprotected by default; authentication should be
    # configured separately if required for production use.
    if features.enable_mcp:
        import warnings

        from nodetool.api.mcp_server import mcp

        # Use sse_app() for SSE transport (provides /sse and /messages routes)
        # Note: sse_app() is deprecated in favor of http_app() but provides
        # the endpoint structure expected by existing MCP clients.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            app.mount("/mcp", mcp.sse_app())

        log.info("[Startup] MCP server mounted at /mcp (endpoints: /mcp/sse, /mcp/messages)")

    if features.mount_static and static_folder and os.path.exists(static_folder):
        log.info(f"[Startup] Mounting static folder: {static_folder}")
        app.mount("/", StaticFiles(directory=static_folder, html=True), name="static")

    return app


def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown."""
    # Don't override signal handlers - let uvicorn handle them
    # Our cleanup will happen via the FastAPI shutdown event
    pass


def run_uvicorn_server(app: Any, host: str, port: int, reload: bool) -> None:
    """
    Starts api using Uvicorn with the specified configuration.

    Args:
        app: The app to run or import string when reload=True or workers > 1.
        host: The host to run on.
        port: The port to run on.
        reload: Whether to reload the server on changes.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    editable_dirs = get_nodetool_package_source_folders()
    reload_dirs = [parent_dir, *editable_dirs] if reload else []

    use_color = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

    configure_logging(
        fmt=(
            "\x1b[90m%(asctime)s\x1b[0m | %(levelname)s | \x1b[36m%(name)s\x1b[0m | %(message)s" if use_color else None
        )
    )

    # Check for insecure authentication configuration when binding to network interfaces
    Environment.emit_auth_warnings(host, logger=log)

    # Uvicorn uses its own logging; keep level name plain for compatibility
    formatter = {
        "format": os.getenv(
            "NODETOOL_LOG_FORMAT",
            (
                "\x1b[90m%(asctime)s\x1b[0m | %(levelname)s | \x1b[36m%(name)s\x1b[0m | %(message)s"
                if use_color
                else "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            ),
        ),
        "datefmt": os.getenv("NODETOOL_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S"),
    }
    log_level = Environment.get_log_level()

    # Create health check filter instance
    health_filter = create_health_check_filter()

    uvicorn_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": formatter},
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": log_level.upper(),
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level.upper(),
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": "WARNING",  # Suppress DEBUG keepalive ping/pong messages
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": log_level.upper(),
                "propagate": False,
            },
        },
    }

    # Apply health check filter to suppress /health endpoint logs
    # Apply filter after uvicorn configures logging using a short delay
    import threading
    import time

    def apply_health_filter():
        """Apply health check filter after uvicorn configures logging."""
        time.sleep(0.2)  # Give uvicorn time to configure logging
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        # Check if filter already exists
        filter_exists = any(isinstance(f, HealthCheckFilter) for f in uvicorn_access_logger.filters)
        if not filter_exists:
            uvicorn_access_logger.addFilter(health_filter)

    filter_thread = threading.Thread(target=apply_health_filter, daemon=True)
    filter_thread.start()

    try:
        uvicorn(
            app=app,
            host=host,
            port=port,
            reload=reload,
            reload_dirs=reload_dirs,
            log_config=uvicorn_log_config,
            loop="asyncio",
            workers=1,
        )
    except KeyboardInterrupt:
        log.info("[Shutdown] Server interrupted by user (Ctrl+C)")
        # On Windows, uvicorn shutdown can hang - force exit immediately after cleanup
        if platform.system() == "Windows":
            log.info("[Shutdown] Windows detected: forcing immediate exit to prevent hanging...")
            # Use a separate thread to force exit after a short delay
            import threading
            import time

            def force_exit():
                time.sleep(1)  # Give cleanup handlers time to run
                log.info("[Shutdown] Forcing process termination...")
                os._exit(0)

            # Start the force exit timer
            exit_thread = threading.Thread(target=force_exit, daemon=True)
            exit_thread.start()

            # Let the normal shutdown proceed, but it will be terminated by the timer
        raise
    finally:
        # As a last resort on Windows, forcefully exit the process to avoid hangs
        if platform.system() == "Windows":
            os._exit(0)


if __name__ == "__main__":
    app = create_app(
        # static_folder=os.path.join(os.path.dirname(__file__), "..", "ui", "dist"),
        # apps_folder=os.path.join(os.path.dirname(__file__), "..", "ui", "apps"),
    )
    run_uvicorn_server(app, "0.0.0.0", 8000, True)
