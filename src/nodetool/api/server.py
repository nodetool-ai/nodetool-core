import os
import asyncio
import platform
import sys
from typing import Any, List
import dotenv
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from nodetool.config.environment import Environment

from nodetool.integrations.huggingface.huggingface_cache import (
    huggingface_download_endpoint,
)
from nodetool.integrations.websocket.websocket_runner import WebSocketRunner
from nodetool.chat.chat_websocket_runner import ChatWebSocketRunner

from fastapi import APIRouter, FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as uvicorn
import sys
from nodetool.config.logging_config import configure_logging, get_logger

from nodetool.metadata.types import Provider
from nodetool.packages.registry import get_nodetool_package_source_folders

from . import (
    asset,
    font,
    collection,
    file,
    debug,
    message,
    node,
    storage,
    workflow,
    model,
    settings,
    thread,
    job,
)
import mimetypes

from nodetool.integrations.websocket.websocket_updates import websocket_updates
from nodetool.api.openai import create_openai_compatible_router
from nodetool.api.mcp_server import create_mcp_app

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


Environment.initialize_sentry()

log = get_logger(__name__)

# Silence SQLite and SQLAlchemy logging
import logging
logging.getLogger('nodetool.models.sqlite_adapter').setLevel(logging.WARNING)
logging.getLogger('nodetool.chat.chat_websocket_runner').setLevel(logging.WARNING)


class ExtensionRouterRegistry:
    _instance = None
    _routers: List[APIRouter] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExtensionRouterRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_router(cls, router: APIRouter) -> None:
        """Register a new router from an extension."""
        if router not in cls._routers:
            cls._routers.append(router)

    @classmethod
    def get_routers(cls) -> List[APIRouter]:
        """Get all registered extension routers."""
        return cls._routers.copy()


DEFAULT_ROUTERS = [
    asset.router,
    message.router,
    thread.router,
    model.router,
    node.router,
    workflow.router,
    storage.router,
    storage.temp_router,
    font.router,
    debug.router,
    job.router,
]


if not Environment.is_production():
    DEFAULT_ROUTERS.append(file.router)
    DEFAULT_ROUTERS.append(settings.router)
    DEFAULT_ROUTERS.append(collection.router)


def create_app(
    origins: list[str] = ["*"],
    routers: list[APIRouter] = DEFAULT_ROUTERS,
    static_folder: str | None = None,
    apps_folder: str | None = None,
):
    # Centralized dotenv loading for consistency with deploy.fastapi_server
    from nodetool.config.environment import load_dotenv_files

    load_dotenv_files()
    log.info(
        "dotenv: ENV=%s | LOG_LEVEL=%s | DEBUG=%s",
        os.environ.get("ENV"),
        os.environ.get("LOG_LEVEL"),
        os.environ.get("DEBUG"),
    )

    # Use FastAPI lifespan API instead of deprecated on_event hooks
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: pre-initialize storages to avoid first-request blocking
        try:
            # Offload potential filesystem setup to threads
            await asyncio.to_thread(Environment.get_asset_storage)
            await asyncio.to_thread(Environment.get_temp_storage)
        except Exception as e:
            log.warning(f"Storage pre-initialization failed: {e}")

        # Start job execution manager cleanup task
        try:
            from nodetool.workflows.job_execution_manager import (
                JobExecutionManager,
            )

            job_manager = JobExecutionManager.get_instance()
            await job_manager.start_cleanup_task()
            log.info("JobExecutionManager cleanup task started")
        except Exception as e:
            log.error(f"Error starting JobExecutionManager: {e}")

        # Hand control back to the app
        yield

        # Shutdown: cleanup resources
        log.info("Server shutdown initiated - cleaning up resources")

        try:
            # Import here to avoid circular imports
            from nodetool.integrations.websocket.websocket_updates import (
                websocket_updates,
            )

            await websocket_updates.shutdown()
            log.info("WebSocket updates shutdown complete")
        except Exception as e:
            log.error(f"Error during websocket updates shutdown: {e}")

        # Shutdown job execution manager
        try:
            from nodetool.workflows.job_execution_manager import (
                JobExecutionManager,
            )

            job_manager = JobExecutionManager.get_instance()
            await job_manager.shutdown()
            log.info("JobExecutionManager shutdown complete")
        except Exception as e:
            log.error(f"Error during JobExecutionManager shutdown: {e}")

        # Give a moment for cleanup to complete
        await asyncio.sleep(0.1)
        log.info("Server shutdown cleanup complete")

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )

    # Mount OpenAI-compatible endpoints with default provider set to "ollama"
    app.include_router(
        create_openai_compatible_router(
            provider=Provider.Ollama.value,
        )
    )

    # Mount FastMCP server
    try:
        mcp_app = create_mcp_app()
        # Mount MCP at /mcp prefix
        app.mount("/mcp", mcp_app.get_asgi_app())
        log.info("FastMCP server mounted at /mcp")
    except Exception as e:
        log.warning(f"Could not mount FastMCP server: {e}")

    for router in routers:
        app.include_router(router)

    for extension_router in ExtensionRouterRegistry().get_routers():
        app.include_router(extension_router)

    if not Environment.is_production():

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            request: Request, exc: RequestValidationError
        ):
            print(f"Request validation error: {exc}")
            return JSONResponse({"detail": exc.errors()}, status_code=422)

    if apps_folder:
        print(f"Mounting apps folder: {apps_folder}")
        app.mount("/apps", StaticFiles(directory=apps_folder, html=True), name="apps")

    # Pre-initialization and shutdown cleanup handled via lifespan above

    @app.get("/health")
    async def health_check() -> str:
        return "OK"

    @app.get("/editor/{workflow_id}")
    async def editor_redirect(workflow_id: str):
        return RedirectResponse(url="/")

    if not Environment.is_production():
        app.add_websocket_route("/hf/download", huggingface_download_endpoint)

    @app.websocket("/predict")
    async def websocket_endpoint(websocket: WebSocket):
        await WebSocketRunner().run(websocket)

    @app.websocket("/chat")
    async def chat_websocket_endpoint(websocket: WebSocket):
        # Extract authentication information
        auth_header = websocket.headers.get("authorization")
        auth_token = None

        # Extract bearer token if present
        if auth_header and auth_header.startswith("Bearer "):
            auth_token = auth_header.replace("Bearer ", "")

        # Check for API key in query params as fallback
        if not auth_token:
            auth_token = websocket.query_params.get("api_key")

        # Create runner with authentication token
        chat_runner = ChatWebSocketRunner(auth_token=auth_token)
        await chat_runner.run(websocket)

    # WebSocket endpoint for periodic system updates (e.g., system stats)
    @app.websocket("/updates")
    async def updates_websocket_endpoint(websocket: WebSocket):
        await websocket_updates.handle_client(websocket)

    if static_folder and os.path.exists(static_folder):
        print(f"Mounting static folder: {static_folder}")
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
    if reload:
        reload_dirs = [parent_dir] + [str(dir) for dir in editable_dirs]
    else:
        reload_dirs = []

    use_color = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

    configure_logging(
        fmt=(
            "\x1b[90m%(asctime)s\x1b[0m | %(levelname)s | \x1b[36m%(name)s\x1b[0m | %(message)s"
            if use_color
            else None
        )
    )

    # Uvicorn uses its own logging; keep level name plain for compatibility
    formatter = {
        "format": os.getenv(
            "NODETOOL_LOG_FORMAT",
            (
                (
                    "\x1b[90m%(asctime)s\x1b[0m | %(levelname)s | \x1b[36m%(name)s\x1b[0m | %(message)s"
                    if use_color
                    else "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                )
            ),
        ),
        "datefmt": os.getenv("NODETOOL_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S"),
    }

    uvicorn_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": formatter},
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": os.getenv("NODETOOL_LOG_LEVEL", "INFO").upper(),
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": os.getenv("NODETOOL_LOG_LEVEL", "INFO").upper(),
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": os.getenv("NODETOOL_LOG_LEVEL", "INFO").upper(),
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": os.getenv("NODETOOL_LOG_LEVEL", "INFO").upper(),
                "propagate": False,
            },
        },
    }

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
        print("\nServer interrupted by user (Ctrl+C)")
        # On Windows, uvicorn shutdown can hang - force exit immediately after cleanup
        if platform.system() == "Windows":
            print("Windows detected: forcing immediate exit to prevent hanging...")
            # Use a separate thread to force exit after a short delay
            import threading
            import time

            def force_exit():
                time.sleep(2)  # Give cleanup handlers time to run
                print("Forcing process termination...")
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
