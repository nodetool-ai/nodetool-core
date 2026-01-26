"""
NodeTool Worker - Deployable FastAPI Server

This is the main deployable worker for NodeTool that can be deployed anywhere.
It provides a complete FastAPI server with:

1. OpenAI-compatible chat completions with SSE streaming (/v1/chat/completions)
2. Model listing (/v1/models)
3. Workflow execution (/workflows/*)
4. Admin operations (/admin/*)
   - Model downloads (HuggingFace, Ollama)
   - Cache management
   - Database operations
   - Collection/RAG management (/admin/collections/*)
   - File storage management (/admin/storage/*)
5. Public storage (read-only) (/storage/*)
6. Legacy collection routes (/collections/*)

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

import datetime
import os
import platform
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from rich.console import Console

from nodetool.api.openai import create_openai_compatible_router
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.deploy.admin_routes import create_admin_router
from nodetool.deploy.auth import (
    DEPLOYMENT_CONFIG_FILE,
    get_token_source,
    get_worker_auth_token,
)
from nodetool.deploy.collection_routes import create_collection_router
from nodetool.deploy.storage_routes import (
    create_admin_storage_router,
    create_public_storage_router,
)
from nodetool.deploy.workflow_routes import (
    create_workflow_router,
)
from nodetool.runtime.resources import get_static_auth_provider, get_user_auth_provider
from nodetool.security.http_auth import create_http_auth_middleware
from nodetool.types.workflow import Workflow

console = Console()
log = get_logger(__name__)


def create_worker_app(
    provider: str = "ollama",
    default_model: str = "gpt-oss:20b",
    tools: list[str] | None = None,
    workflows: list[Workflow] | None = None,
) -> FastAPI:
    """Create a FastAPI worker application for NodeTool operations.

    This is the main entry point for creating a deployable NodeTool worker.
    The worker provides OpenAI-compatible endpoints, workflow execution,
    admin operations, and collection management.

    Args:
        provider: Default AI provider to use (ollama, openai, anthropic, etc.)
        default_model: Default model to use when not specified by client
        tools: List of tool names to enable (e.g., ['google_search', 'browser'])
        workflows: List of workflows to make available

    Returns:
        FastAPI application instance configured as a NodeTool worker

    Example:
        >>> app = create_worker_app(
        ...     provider="ollama",
        ...     default_model="llama3.2:latest",
        ...     tools=["google_search"]
        ... )
    """
    tools = tools or []
    workflows = workflows or []

    from nodetool.config.environment import load_dotenv_files
    from nodetool.observability.tracing import init_tracing

    load_dotenv_files()
    init_tracing(service_name="nodetool-worker")

    if Environment.is_production() and not os.environ.get("SECRETS_MASTER_KEY"):
        raise RuntimeError("SECRETS_MASTER_KEY environment variable must be set for deployed workers.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        console.print("NodeTool worker started successfully")
        try:
            app.include_router(
                create_openai_compatible_router(
                    provider=provider,
                    default_model=default_model,
                    tools=tools,
                )
            )
            app.include_router(create_workflow_router())
            app.include_router(create_admin_router())
            app.include_router(create_collection_router())
            app.include_router(create_admin_storage_router())
            app.include_router(create_public_storage_router())
        except Exception as e:
            log.error(f"Failed to include routers: {e}")
        yield
        console.print("NodeTool worker shutting down...")

    app = FastAPI(
        title="NodeTool Worker",
        version="1.0.0",
        description="Deployable NodeTool worker with OpenAI-compatible API, workflow execution, and admin operations",
        lifespan=lifespan,
    )

    # Add authentication middleware
    static_provider = get_static_auth_provider()
    user_provider = get_user_auth_provider()
    enforce_auth = Environment.enforce_auth()
    auth_middleware = create_http_auth_middleware(
        static_provider=static_provider,
        user_provider=user_provider,
        enforce_auth=enforce_auth,
    )
    app.middleware("http")(auth_middleware)

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
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def run_worker(
    host: str = "0.0.0.0",
    port: int = 8000,
    provider: str = "ollama",
    default_model: str = "gpt-oss:20b",
    tools: list[str] | None = None,
    workflows: list[Workflow] | None = None,
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

    tools = tools or []
    workflows = workflows or []

    app = create_worker_app(provider, default_model, tools, workflows)

    # Get authentication info
    auth_token = get_worker_auth_token()
    token_source = get_token_source()
    masked_token = f"{auth_token[:8]}...{auth_token[-4:]}" if auth_token and len(auth_token) > 12 else "***"

    console.print(f"🚀 Starting NodeTool worker on {host}:{port}")
    console.print("")
    console.print("=" * 70)
    console.print("AUTHENTICATION")
    console.print("=" * 70)
    console.print("🔒 Status: ENABLED (all endpoints require authentication)")
    console.print(f"🔑 Token: {masked_token}")

    if token_source == "environment":
        console.print("📍 Source: Environment variable (WORKER_AUTH_TOKEN)")
    elif token_source == "config":
        console.print(f"📍 Source: Config file ({DEPLOYMENT_CONFIG_FILE})")
    else:  # generated
        console.print(f"📍 Source: Auto-generated and saved to {DEPLOYMENT_CONFIG_FILE}")
        console.print(f"💾 Full token saved to: {DEPLOYMENT_CONFIG_FILE}")

    console.print("")
    console.print("🔓 Public endpoints (no auth): /health, /ping")
    console.print("🔐 Protected endpoints: All others (use header below)")
    console.print("")
    console.print(f"   Authorization: Bearer {auth_token}")
    console.print("=" * 70)
    console.print("")

    auth_header = f' -H "Authorization: Bearer {auth_token}"'

    console.print("ENDPOINTS:")
    console.print("")
    console.print("OpenAI-Compatible:")
    console.print(f"  Chat: curl{auth_header} -X POST http://{host}:{port}/v1/chat/completions \\")
    console.print('    -H "Content-Type: application/json" \\')
    console.print(f'    -d \'{{"model": "{default_model}", "messages": [{{"role": "user", "content": "Hello"}}]}}\'')
    console.print(f"  Models: curl{auth_header} http://{host}:{port}/v1/models")
    console.print("")
    console.print("Workflows:")
    console.print(f"  curl{auth_header} http://{host}:{port}/workflows")
    console.print("")
    console.print("Admin:")
    console.print(f"  Health (public): curl http://{host}:{port}/health")
    console.print(f"  HF download: curl{auth_header} -X POST http://{host}:{port}/admin/models/huggingface/download")
    console.print(f"  Cache scan: curl{auth_header} http://{host}:{port}/admin/cache/scan")
    console.print("")
    console.print("Collections:")
    console.print(f"  List: curl{auth_header} http://{host}:{port}/admin/collections")
    console.print(f"  Create: curl{auth_header} -X POST http://{host}:{port}/admin/collections \\")
    console.print('    -H "Content-Type: application/json" \\')
    console.print('    -d \'{"name": "my_docs", "embedding_model": "all-minilm:latest"}\'')
    console.print("")
    console.print("Storage:")
    console.print(f"  Upload: curl{auth_header} -X PUT http://{host}:{port}/admin/storage/assets/file.png \\")
    console.print("    --data-binary @file.png")
    console.print(f"  Download: curl{auth_header} http://{host}:{port}/storage/assets/file.png")
    console.print("")
    console.print("Assets:")
    console.print(f"  List: curl{auth_header} 'http://{host}:{port}/admin/assets?user_id=1'")
    console.print(f"  Create: curl{auth_header} -X POST 'http://{host}:{port}/admin/assets' \\")
    console.print("    -d 'user_id=1&name=MyFolder&content_type=folder'")
    console.print(f"  Get: curl{auth_header} 'http://{host}:{port}/admin/assets/{{asset_id}}?user_id=1'")
    console.print(f"  Delete: curl{auth_header} -X DELETE 'http://{host}:{port}/admin/assets/{{asset_id}}?user_id=1'")
    console.print("")
    console.print("Auth provider:", Environment.get_auth_provider_kind())
    console.print("Default model:", f"{default_model} (provider: {provider})")
    console.print("Tools:", tools)
    console.print("Workflows:", [w.name for w in workflows])

    # Check for insecure authentication configuration when binding to network interfaces
    for warning in Environment.check_insecure_auth_binding(host):
        console.print(f"[bold red]{warning}[/bold red]")

    # Run the server
    try:
        loop = "asyncio" if platform.system() == "Windows" else "uvloop"
        # Always use single worker (multiple workers not supported with app object)
        uvicorn.run(app, host=host, port=port, log_level="info", loop=loop)
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
    tools_str = os.getenv("NODETOOL_TOOLS", "")
    tools = [tool.strip() for tool in tools_str.split(",") if tool.strip()] if tools_str else []
    port = int(os.getenv("PORT", 8000))

    run_worker(
        provider=provider,
        default_model=default_model,
        tools=tools,
        port=port,
    )
