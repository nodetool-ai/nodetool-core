"""
Production server entry point for NodeTool.

This module provides a production-ready entry point that runs the main server
with all routers including deploy routers for admin operations.

Usage:
    # As a module
    python -m nodetool.api.run_server

    # With options
    python -m nodetool.api.run_server --port 7777 --host 0.0.0.0
"""

import os
import sys

from rich.console import Console

from nodetool.api.server import create_app, run_uvicorn_server
from nodetool.config.environment import Environment, load_dotenv_files
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Provider

console = Console()
log = get_logger(__name__)


def run_server(
    host: str = "0.0.0.0",
    port: int = 7777,
    reload: bool = False,
    mode: str | None = None,
    auth_provider: str | None = None,
    include_default_api_routers: bool | None = None,
    include_openai_router: bool | None = None,
    include_deploy_admin_router: bool | None = None,
    include_deploy_collection_router: bool | None = None,
    include_deploy_storage_router: bool | None = None,
    include_deploy_workflow_router: bool | None = None,
    enable_main_ws: bool | None = None,
    enable_updates_ws: bool | None = None,
    enable_terminal_ws: bool | None = None,
    enable_hf_download_ws: bool | None = None,
    mount_static: bool | None = None,
) -> None:
    """Run the NodeTool server.

    This function starts the server using uvicorn. It's the main entry point
    for running a production NodeTool server.

    Args:
        host: Host address to serve on (default: 0.0.0.0 for all interfaces)
        port: Port to serve on (default: 7777)
        reload: Whether to reload on code changes (default: False)

    Example:
        >>> run_server(host="0.0.0.0", port=7777)
    """
    # Use centralized dotenv loading
    load_dotenv_files()

    # Log startup configuration
    env = Environment.get_env()
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    debug = os.environ.get("DEBUG")

    log.info(
        "Starting server: ENV=%s | LOG_LEVEL=%s | DEBUG=%s",
        env,
        log_level,
        debug,
    )

    # Get configuration from environment
    default_provider = os.environ.get("CHAT_PROVIDER", Provider.Ollama.value)
    default_model = os.environ.get("DEFAULT_MODEL", "llama3.2:latest")

    console.print(f"🚀 Starting NodeTool server on {host}:{port}")
    console.print("")
    console.print("=" * 70)
    console.print(f"Environment: {env}")
    console.print(f"Default Provider: {default_provider}")
    console.print(f"Default Model: {default_model}")

    if Environment.is_production():
        admin_token = os.environ.get("ADMIN_TOKEN")
        if admin_token:
            masked = f"{admin_token[:8]}...{admin_token[-4:]}" if len(admin_token) > 12 else "***"
            console.print(f"Admin Token: {masked}")
        else:
            console.print("[yellow]⚠️  Admin Token: Not configured[/yellow]")

        if not os.environ.get("SECRETS_MASTER_KEY"):
            console.print("[red]❌ SECRETS_MASTER_KEY not set - required for production[/red]")
            sys.exit(1)

    console.print("=" * 70)
    console.print("")
    console.print("ENDPOINTS:")
    console.print("  Health: GET /health, /ping")
    console.print("  API: GET /api/*")
    console.print("  Admin: /admin/* (requires X-Admin-Token in production)")
    console.print("  OpenAI-Compatible: /v1/chat/completions, /v1/models")
    console.print("  Workflows: /workflows/*")
    console.print("=" * 70)
    console.print("")

    effective_mode = mode or os.environ.get("NODETOOL_SERVER_MODE")
    # Create and run the app
    app = create_app(
        mode=effective_mode,
        auth_provider=auth_provider,
        include_default_api_routers=include_default_api_routers,
        include_openai_router=include_openai_router,
        include_deploy_admin_router=include_deploy_admin_router,
        include_deploy_collection_router=include_deploy_collection_router,
        include_deploy_storage_router=include_deploy_storage_router,
        include_deploy_workflow_router=include_deploy_workflow_router,
        enable_main_ws=enable_main_ws,
        enable_updates_ws=enable_updates_ws,
        enable_terminal_ws=enable_terminal_ws,
        enable_hf_download_ws=enable_hf_download_ws,
        mount_static=mount_static,
    )
    run_uvicorn_server(app, host, port, reload)


def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the NodeTool server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7777, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
