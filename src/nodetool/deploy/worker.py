"""
NodeTool Worker - Deployable FastAPI Server

This is the main deployable worker for NodeTool that can be deployed anywhere.
It provides a complete FastAPI server with:

1. OpenAI-compatible chat completions with SSE streaming (/v1/chat/completions)
2. Model listing (/v1/models)
3. Workflow execution (/workflows/*)
4. Admin operations (/admin/*)
5. Collection/RAG operations (/collections/*)

This worker can be deployed to:
- Local machines
- Docker containers
- RunPod serverless
- Google Cloud Run
- Any platform supporting Python/FastAPI

Usage:
    # As a module
    python -m nodetool.deploy.worker

    # Via CLI
    nodetool worker --port 8080

    # In code
    from nodetool.deploy.worker import create_worker_app, run_worker
    app = create_worker_app()
    run_worker(host="0.0.0.0", port=8000)
"""

import os
import datetime
import multiprocessing
import platform
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from rich.console import Console

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.api.openai import create_openai_compatible_router
from nodetool.types.workflow import Workflow
from nodetool.deploy.workflow_routes import (
    create_workflow_router,
)
from nodetool.deploy.admin_routes import create_admin_router
from nodetool.deploy.collection_routes import create_collection_router


console = Console()
log = get_logger(__name__)


def create_worker_app(
    remote_auth: bool = False,
    provider: str = "ollama",
    default_model: str = "gpt-oss:20b",
    tools: List[str] = [],
    workflows: List[Workflow] = [],
) -> FastAPI:
    """Create a FastAPI worker application for NodeTool operations.

    This is the main entry point for creating a deployable NodeTool worker.
    The worker provides OpenAI-compatible endpoints, workflow execution,
    admin operations, and collection management.

    Args:
        remote_auth: Whether to use remote authentication (Supabase)
        provider: Default AI provider to use (ollama, openai, anthropic, etc.)
        default_model: Default model to use when not specified by client
        tools: List of tool names to enable (e.g., ['google_search', 'browser'])
        workflows: List of workflows to make available

    Returns:
        FastAPI application instance configured as a NodeTool worker

    Example:
        >>> app = create_worker_app(
        ...     remote_auth=False,
        ...     provider="ollama",
        ...     default_model="llama3.2:latest",
        ...     tools=["google_search"]
        ... )
    """
    # Set authentication mode
    Environment.set_remote_auth(remote_auth)

    app = FastAPI(
        title="NodeTool Worker",
        version="1.0.0",
        description="Deployable NodeTool worker with OpenAI-compatible API, workflow execution, and admin operations",
    )

    @app.on_event("startup")
    async def startup_event():
        """Initialize worker on startup."""
        console.print("NodeTool worker started successfully")
        # Include routers after initialization
        try:
            app.include_router(
                create_openai_compatible_router(
                    provider=provider,
                    default_model=default_model,
                    tools=tools,
                )
            )
            # Include lightweight workflow, admin, and collection routers
            app.include_router(create_workflow_router())
            app.include_router(create_admin_router())
            app.include_router(create_collection_router())
        except Exception as e:  # noqa: BLE001
            log.error(f"Failed to include routers: {e}")

    @app.on_event("shutdown")
    async def shutdown_event():
        console.print("NodeTool worker shutting down...")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

    @app.get("/ping")
    async def ping():
        """Health check with system information."""
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.datetime.now().isoformat(),
            }
        except Exception as e:
            console.print(f"Health check error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_worker(
    host: str = "0.0.0.0",
    port: int = 8000,
    remote_auth: bool = False,
    provider: str = "ollama",
    default_model: str = "gpt-oss:20b",
    tools: List[str] = [],
    workflows: List[Workflow] = [],
):
    """Run the NodeTool worker.

    This function starts the worker server using uvicorn. It's the main
    entry point for running a deployable NodeTool worker.

    Args:
        host: Host address to serve on (default: 0.0.0.0 for all interfaces)
        port: Port to serve on (default: 8000)
        remote_auth: Whether to use remote authentication (Supabase)
        provider: Default AI provider to use
        default_model: Default model to use when not specified by client
        tools: List of tool names to enable
        workflows: List of workflows to make available

    Example:
        >>> run_worker(
        ...     host="0.0.0.0",
        ...     port=8000,
        ...     provider="ollama",
        ...     default_model="llama3.2:latest"
        ... )
    """
    # Use centralized dotenv loading for consistency
    from nodetool.config.environment import load_dotenv_files

    load_dotenv_files()
    # Simple diagnostic about changes and key flags
    log.info(
        "dotenv: ENV=%s | LOG_LEVEL=%s | DEBUG=%s",
        os.environ.get("ENV"),
        os.environ.get("LOG_LEVEL"),
        os.environ.get("DEBUG"),
    )

    app = create_worker_app(remote_auth, provider, default_model, tools, workflows)

    console.print(f"🚀 Starting NodeTool worker on {host}:{port}")
    console.print(
        f"Chat completions endpoint: http://{host}:{port}/v1/chat/completions"
    )
    console.print(f"Models endpoint: http://{host}:{port}/v1/models")
    console.print(f"Workflows endpoint: http://{host}:{port}/workflows")
    console.print("Admin endpoints:")
    console.print(f"  - Health check: http://{host}:{port}/admin/health")
    console.print(
        f"  - HuggingFace download: http://{host}:{port}/admin/models/huggingface/download"
    )
    console.print(
        f"  - Ollama download: http://{host}:{port}/admin/models/ollama/download"
    )
    console.print(f"  - Cache scan: http://{host}:{port}/admin/cache/scan")
    console.print(f"  - Cache size: http://{host}:{port}/admin/cache/size")
    console.print(
        f"  - Delete HF model: http://{host}:{port}/admin/models/huggingface/{{repo_id}}"
    )
    console.print(f"  - Workflow status: http://{host}:{port}/admin/workflows/status")
    console.print(
        "Authentication mode:",
        "Remote (Supabase)" if remote_auth else "Local (user_id=1)",
    )
    console.print("Default model:", f"{default_model} (provider: {provider})")
    console.print("Tools:", tools)
    console.print("Workflows:", [w.name for w in workflows])
    console.print("\\nSend requests with Authorization: Bearer YOUR_TOKEN header")

    # Run the server
    try:
        loop = "asyncio" if platform.system() == "Windows" else "uvloop"
        workers = max(1, multiprocessing.cpu_count())
        uvicorn.run(
            app, host=host, port=port, log_level="info", loop=loop, workers=workers
        )
    except KeyboardInterrupt:
        console.print("\\n👋 NodeTool worker stopped by user")
    except Exception as e:
        console.print(f"❌ Worker error: {e}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    # Get configuration from environment
    provider = os.getenv("CHAT_PROVIDER", "ollama")
    default_model = os.getenv("DEFAULT_MODEL", "gpt-oss:20b")
    remote_auth = os.getenv("REMOTE_AUTH", "false").lower() == "true"
    tools_str = os.getenv("NODETOOL_TOOLS", "")
    tools = (
        [tool.strip() for tool in tools_str.split(",") if tool.strip()]
        if tools_str
        else []
    )
    port = int(os.getenv("PORT", 8000))

    run_worker(
        remote_auth=remote_auth,
        provider=provider,
        default_model=default_model,
        tools=tools,
        port=port,
    )
