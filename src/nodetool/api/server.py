import os
import asyncio
import platform
import multiprocessing
from typing import Any, List
import dotenv
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from nodetool.api import collection, file, package, prediction, font
from nodetool.common.environment import Environment

from nodetool.common.huggingface_cache import huggingface_download_endpoint
from nodetool.common.websocket_runner import WebSocketRunner
from nodetool.chat.chat_websocket_runner import ChatWebSocketRunner

from fastapi import APIRouter, FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as uvicorn

from nodetool.metadata.types import Provider
from nodetool.packages.registry import get_nodetool_package_source_folders

from . import asset, job, message, node, storage, workflow, model, settings, thread
import mimetypes

from nodetool.common.websocket_updates import websocket_updates
from nodetool.api.openai import create_openai_compatible_router

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
    job.router,
    message.router,
    thread.router,
    model.router,
    node.router,
    prediction.router,
    workflow.router,
    storage.router,
    storage.temp_router,
    font.router,
]


if not Environment.is_production():
    DEFAULT_ROUTERS.append(file.router)
    DEFAULT_ROUTERS.append(settings.router)
    DEFAULT_ROUTERS.append(collection.router)
    DEFAULT_ROUTERS.append(package.router)


def create_app(
    origins: list[str] = ["*"],
    routers: list[APIRouter] = DEFAULT_ROUTERS,
    static_folder: str | None = None,
    apps_folder: str | None = None,
):
    env_file = dotenv.find_dotenv(usecwd=True)

    if env_file != "":
        print(f"Loading environment from {env_file}")
        dotenv.load_dotenv(env_file)

    app = FastAPI()

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

    # Pre-initialize storages to avoid first-request blocking due to lazy init
    @app.on_event("startup")
    async def _initialize_storages() -> None:
        try:
            # Offload potential filesystem setup to threads
            await asyncio.to_thread(Environment.get_asset_storage)
            await asyncio.to_thread(Environment.get_temp_storage)
        except Exception as e:
            Environment.get_logger().warning(f"Storage pre-initialization failed: {e}")

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

    if platform.system() == "Windows":
        loop = "asyncio"
    else:
        try:
            import uvloop  # type: ignore  # noqa: F401

            loop = "uvloop"
        except Exception:
            loop = "asyncio"

    workers = 1

    # Uvicorn requires an import string when using reload=True or workers > 1
    if (reload or workers > 1) and not isinstance(app, str):
        app = "nodetool.api.app:app"

    # Workaround: On Windows, uvicorn's reload mode may prevent clean Ctrl-C shutdown.
    # Disable reload on Windows to avoid reloader child-process hang.
    if platform.system() == "Windows" and reload:
        print("Windows detected: disabling uvicorn reload for clean Ctrl-C shutdown.")
        reload = False

    try:
        uvicorn(
            app=app,
            host=host,
            port=port,
            reload=reload,
            reload_dirs=reload_dirs,
            loop=loop,
            workers=workers,
        )
    except KeyboardInterrupt:
        # Ensure immediate termination on Windows where child processes may linger
        if platform.system() == "Windows":
            os._exit(0)
        raise


if __name__ == "__main__":
    app = create_app(
        # static_folder=os.path.join(os.path.dirname(__file__), "..", "ui", "dist"),
        # apps_folder=os.path.join(os.path.dirname(__file__), "..", "ui", "apps"),
    )
    run_uvicorn_server(app, "0.0.0.0", 8000, True)
