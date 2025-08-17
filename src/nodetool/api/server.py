#!/usr/bin/env python

from __future__ import annotations

from typing import Any, List
import os
import asyncio
import mimetypes
import platform

from fastapi import APIRouter, FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from uvicorn import run as uvicorn

from nodetool.api import asset, job, message, node, storage, workflow, model, settings, thread
from nodetool.api import file as file_api
from nodetool.api import package, prediction, font, collection, openai as openai_api
from nodetool.common.environment import Environment
from nodetool.common.websocket_updates import WebSocketUpdates

# Initialize logger and settings
log = Environment.get_logger()
Environment.initialize_sentry()

# Windows selector policy for asyncio compatibility
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure common MIME types are registered (helps on some systems)
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


class ExtensionRouterRegistry:
    _instance = None
    _routers: List[APIRouter] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExtensionRouterRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_router(cls, router: APIRouter) -> None:
        if router not in cls._routers:
            cls._routers.append(router)

    @classmethod
    def get_routers(cls) -> List[APIRouter]:
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

# In non-production, also mount admin/dev routers
if not Environment.is_production():
    DEFAULT_ROUTERS.append(file_api.router)
    DEFAULT_ROUTERS.append(settings.router)
    DEFAULT_ROUTERS.append(collection.router)
    DEFAULT_ROUTERS.append(package.router)


def create_app(
    origins: list[str] = ["*"],
    routers: list[APIRouter] = DEFAULT_ROUTERS,
    static_folder: str | None = None,
    apps_folder: str | None = None,
) -> FastAPI:
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
        openai_api.create_openai_compatible_router(provider="ollama")
    )

    for router in routers:
        app.include_router(router)

    for extension_router in ExtensionRouterRegistry().get_routers():
        app.include_router(extension_router)

    @app.get("/health")
    async def health_check() -> str:
        return "OK"

    @app.get("/editor/{workflow_id}")
    async def editor_redirect(workflow_id: str):
        return RedirectResponse(url="/")

    # Optional: websocket updates for clients
    websocket_updates = WebSocketUpdates()

    @app.websocket("/updates")
    async def updates_websocket_endpoint(websocket: WebSocket):
        await websocket_updates.handle_client(websocket)

    if apps_folder:
        print(f"Mounting apps folder: {apps_folder}")
        app.mount("/apps", StaticFiles(directory=apps_folder, html=True), name="apps")

    if static_folder and os.path.exists(static_folder):
        print(f"Mounting static folder: {static_folder}")
        app.mount("/", StaticFiles(directory=static_folder, html=True), name="static")

    return app


def run_uvicorn_server(app: Any, host: str, port: int, reload: bool) -> None:
    """
    Starts api using Uvicorn with the specified configuration.

    Args:
        app: The app to run.
        host: The host to run on.
        port: The port to run on.
        reload: Whether to reload the server on changes.
    """
    reload_dirs: list[str] = []
    # Add project root for reload if desired
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if reload and os.path.isdir(project_root):
        reload_dirs = [project_root]

    uvicorn(app=app, host=host, port=port, reload=reload, reload_dirs=reload_dirs)


# Legacy exports expected by other modules
def mount_static_folder(app: FastAPI, static_folder: str | None):
    if static_folder and os.path.exists(static_folder):
        print(f"Mounting static folder: {static_folder}")
        app.mount("/", StaticFiles(directory=static_folder, html=True), name="static")

__all__ = ["create_app", "run_uvicorn_server", "ExtensionRouterRegistry", "mount_static_folder"]
