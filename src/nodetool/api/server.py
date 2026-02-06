import asyncio
import logging
import mimetypes
import os
import platform
import sys
from contextlib import asynccontextmanager
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
        file,
        font,
        job,
        memory,
        message,
        model,
        node,
        oauth,
        skills,
        settings,
        storage,
        thread,
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
        font.router,
        debug.router,
        job.router,
        settings.router,
        memory.router,
        vibecoding.router,
    ]

    if not Environment.is_production():
        routers.append(file.router)
        routers.append(collection.router)

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
):
    # Initialize Sentry only when the application is created, not on module import
    initialize_sentry()

    origins = ["*"] if origins is None else origins
    routers = _load_default_routers() if routers is None else routers

    # Centralized dotenv loading for consistency with deploy.fastapi_server
    from nodetool.config.environment import load_dotenv_files

    load_dotenv_files()

    from nodetool.observability.tracing import init_tracing

    init_tracing(service_name="nodetool-api")

    # Log loaded environment configuration
    env_name = os.environ.get("ENV", "development")
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    debug = os.environ.get("DEBUG")

    log.info(
        "Environment configuration loaded: ENV=%s | LOG_LEVEL=%s | DEBUG=%s",
        env_name,
        log_level,
        debug,
    )

    # Log key configuration details
    supabase_url = os.environ.get("SUPABASE_URL")
    postgres_db = os.environ.get("POSTGRES_DB")
    db_path = os.environ.get("DB_PATH")
    auth_provider = os.environ.get("AUTH_PROVIDER", "local")

    if supabase_url:
        log.info(f"Database: Supabase ({supabase_url})")
    elif postgres_db:
        postgres_host = os.environ.get("POSTGRES_HOST", "localhost")
        log.info(f"Database: PostgreSQL ({postgres_host}/{postgres_db})")
    elif db_path:
        log.info(f"Database: SQLite ({db_path})")
    else:
        log.warning("No database configured (SUPABASE_URL, POSTGRES_DB, or DB_PATH)")

    log.info(f"Authentication provider: {auth_provider}")

    # Log storage configuration
    asset_bucket = os.environ.get("ASSET_BUCKET", "images")
    asset_temp_bucket = os.environ.get("ASSET_TEMP_BUCKET")
    asset_domain = os.environ.get("ASSET_DOMAIN")
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")

    if supabase_url:
        log.info("Asset storage: Supabase")
        log.debug(f"  - Asset bucket: {asset_bucket}")
        if asset_temp_bucket:
            log.debug(f"  - Temp bucket: {asset_temp_bucket}")
        if asset_domain:
            log.debug(f"  - Domain: {asset_domain}")
    elif s3_endpoint:
        log.info(f"Asset storage: S3 (endpoint={s3_endpoint})")
        log.debug(f"  - Asset bucket: {asset_bucket}")
        if asset_temp_bucket:
            log.debug(f"  - Temp bucket: {asset_temp_bucket}")
        log.debug(f"  - Region: {os.environ.get('S3_REGION', 'us-east-1')}")
        if asset_domain:
            log.debug(f"  - Domain: {asset_domain}")
    else:
        log.info("Asset storage: File-based")
        log.debug(f"  - Asset bucket: {asset_bucket}")

    # Check if Ollama is available and set OLLAMA_API_URL if not already set
    setup_ollama_url()

    # Log comprehensive environment diagnostics (secure masking of secrets)
    # This is particularly useful for Electron, Docker, and production deployments
    from nodetool.config.env_diagnostics import log_env_diagnostics

    log_env_diagnostics(logger=log, check_permissions=True)

    # Run startup security checks to warn about insecure configurations
    # Import is local to avoid circular imports (security module imports config which may import api)
    from nodetool.security.startup_checks import run_startup_security_checks

    run_startup_security_checks(raise_on_critical=False)

    # Use FastAPI lifespan API instead of deprecated on_event hooks
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Run database migrations before starting
        from nodetool.models.migrations import run_startup_migrations

        if not RUNNING_PYTEST:
            try:
                await run_startup_migrations()
                log.info("Database migrations completed successfully")
            except Exception as e:
                log.error(f"Failed to run database migrations: {e}", exc_info=True)
                raise

        # Populate mock data if --mock flag is enabled
        if os.environ.get("NODETOOL_MOCK_MODE") == "1":
            log.info("Mock mode enabled - populating database with test data")
            try:
                from nodetool.api.mock_data import populate_mock_data
                from nodetool.runtime.resources import ResourceScope

                async with ResourceScope():
                    result = await populate_mock_data(user_id="1")
                    log.info(f"Mock data populated successfully: {result}")
            except Exception as e:
                log.error(f"Failed to populate mock data: {e}", exc_info=True)

        # Start job execution manager cleanup task
        from nodetool.workflows.job_execution_manager import JobExecutionManager

        job_manager = JobExecutionManager.get_instance()
        await job_manager.start_cleanup_task()
        log.info("JobExecutionManager cleanup task started")

        # Hand control back to the app
        yield

        # Shutdown: cleanup resources
        log.info("Server shutdown initiated - cleaning up resources")

        # Import here to avoid circular imports
        from nodetool.integrations.websocket.websocket_updates import websocket_updates

        await websocket_updates.shutdown()
        log.info("WebSocket updates shutdown complete")

        job_manager = JobExecutionManager.get_instance()
        await job_manager.shutdown()
        log.info("JobExecutionManager shutdown complete")

        # Shutdown SQLite connection pools with WAL checkpointing
        from nodetool.runtime.db_sqlite import shutdown_all_sqlite_pools

        await shutdown_all_sqlite_pools()
        log.info("SQLite connection pools shutdown complete")

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
    from nodetool.api.openai import create_openai_compatible_router
    from nodetool.integrations.huggingface.hf_websocket import (
        huggingface_download_endpoint,
    )
    from nodetool.integrations.websocket.terminal_runner import (
        TerminalWebSocketRunner,
    )
    from nodetool.integrations.websocket.unified_websocket_runner import (
        UnifiedWebSocketRunner,
    )
    from nodetool.integrations.websocket.websocket_updates import websocket_updates
    from nodetool.metadata.types import Provider
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

    if not RUNNING_PYTEST:
        app.add_middleware(ResourceScopeMiddleware)  # type: ignore[arg-type]

    # Mount OpenAI-compatible endpoints with default provider set to "ollama"
    if not Environment.is_production():
        app.include_router(
            create_openai_compatible_router(
                provider=Provider.Ollama.value,
            )
        )

    for router in routers:
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

    @app.get("/editor/{workflow_id}")
    async def editor_redirect(workflow_id: str):
        return RedirectResponse(url="/")

    if not Environment.is_production():
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
        token, user_id = await _authenticate_websocket(websocket)
        if user_id is None:
            return
        runner = UnifiedWebSocketRunner(auth_token=token or "", user_id=user_id)
        await runner.run(websocket)

    @app.websocket("/ws/terminal")
    async def terminal_websocket_endpoint(websocket: WebSocket):
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

    # WebSocket endpoint for periodic system updates (e.g., system stats)
    @app.websocket("/ws/updates")
    async def updates_websocket_endpoint(websocket: WebSocket):
        await websocket_updates.handle_client(websocket)

    if static_folder and os.path.exists(static_folder):
        log.info(f"Mounting static folder: {static_folder}")
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
        log.info("Server interrupted by user (Ctrl+C)")
        # On Windows, uvicorn shutdown can hang - force exit immediately after cleanup
        if platform.system() == "Windows":
            log.info("Windows detected: forcing immediate exit to prevent hanging...")
            # Use a separate thread to force exit after a short delay
            import threading
            import time

            def force_exit():
                time.sleep(1)  # Give cleanup handlers time to run
                log.info("Forcing process termination...")
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
