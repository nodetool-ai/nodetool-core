import os
import sys
import shutil
import atexit
import warnings
import asyncio
import click
from typing import Optional, Dict, Any, List

from nodetool.api.workflow import from_model
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.config.settings import load_settings
from nodetool.deploy.docker import (
    generate_image_tag,
)
from nodetool.deploy.runpod_api import GPUType
from nodetool.dsl.codegen import create_dsl_modules
from nodetool.dsl.export import graph_to_dsl_py
from nodetool.deploy.progress import ProgressManager

# Add Rich for better tables and terminal output
from nodetool.types.job import JobUpdate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Create console instance
console = Console()

# Global progress manager instance
progress_manager = ProgressManager(console=console)


def cleanup_progress():
    """Cleanup function to ensure progress bars are stopped on exit."""
    progress_manager.stop()


# Register cleanup function
atexit.register(cleanup_progress)

warnings.filterwarnings("ignore")
log = get_logger(__name__)

# Define supported GPU types for RunPod REST API
# SUPPORTED_GPU_TYPES = [
#     "NVIDIA GeForce RTX 4090",
#     "NVIDIA GeForce RTX 4080",
#     "NVIDIA GeForce RTX 4070 Ti",
#     "NVIDIA GeForce RTX 4070",
#     "NVIDIA GeForce RTX 4060 Ti",
#     "NVIDIA GeForce RTX 4060",
#     "NVIDIA GeForce RTX 3090",
#     "NVIDIA GeForce RTX 3080",
#     "NVIDIA GeForce RTX 3070",
#     "NVIDIA GeForce RTX 3060",
#     "NVIDIA RTX A6000",
#     "NVIDIA RTX A5000",
#     "NVIDIA RTX A4000",
#     "NVIDIA L40S",
#     "NVIDIA L40",
#     "NVIDIA L4",
#     "NVIDIA A100 80GB PCIe",
#     "NVIDIA A100 40GB PCIe",
#     "NVIDIA H100 PCIe",
#     "NVIDIA H100 SXM5",
# ]

SUPPORTED_GPU_TYPES = GPUType.list_values()


@click.group()
def cli():
    """Nodetool CLI - A tool for managing and running Nodetool workflows and packages."""
    pass


@cli.command("mcp")
def mcp():
    """Start the NodeTool Model Context Protocol (MCP) server.

    Enables IDE integrations (e.g., Claude Code, other MCP-compatible IDEs) to
    access NodeTool workflows and capabilities."""
    from nodetool.api.mcp_server import mcp

    mcp.run()


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Host address to bind to (use 0.0.0.0 for all interfaces).")
@click.option("--port", default=8000, help="Port to listen on.", type=int)
@click.option(
    "--static-folder",
    default=None,
    help="Path to folder containing static web assets (e.g., compiled React UI).",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
)
@click.option("--apps-folder", default=None, help="Path to folder containing app bundles.")
@click.option("--force-fp16", is_flag=True, help="Force FP16 precision for ComfyUI integrations (GPU optimization).")
@click.option("--reload", is_flag=True, help="Enable auto-reload on file changes (development only).")
@click.option("--production", is_flag=True, help="Enable production mode with stricter validation and optimizations.")
@click.option(
    "--auth-provider",
    type=click.Choice(["none", "local", "static", "supabase"], case_sensitive=False),
    default=None,
    help="Select authentication provider (overrides ENV AUTH_PROVIDER)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (DEBUG level) for detailed output.",
)
def serve(
    host: str,
    port: int,
    static_folder: str | None = None,
    reload: bool = False,
    force_fp16: bool = False,
    auth_provider: str | None = None,
    apps_folder: str | None = None,
    production: bool = False,
    verbose: bool = False,
):
    """Run the FastAPI backend server for the NodeTool platform.

    Serves the REST API, WebSocket endpoints, and optionally static assets or app bundles.
    """
    from nodetool.api.server import create_app, run_uvicorn_server

    # Configure logging level based on verbose flag
    if verbose:
        from nodetool.config.logging_config import configure_logging

        configure_logging(level="DEBUG")
        os.environ["LOG_LEVEL"] = "DEBUG"
        console.print("[cyan]🐛 Verbose logging enabled (DEBUG level)[/]")

    try:
        import comfy.cli_args  # type: ignore

        comfy.cli_args.args.force_fp16 = force_fp16
    except ImportError:
        pass

    if auth_provider:
        os.environ["AUTH_PROVIDER"] = auth_provider.lower()

    if not reload:
        app = create_app(static_folder=static_folder, apps_folder=apps_folder)
    else:
        if static_folder:
            raise Exception("static folder and reload are exclusive options")
        if apps_folder:
            raise Exception("apps folder and reload are exclusive options")
        app = "nodetool.api.app:app"

    run_uvicorn_server(app=app, host=host, port=port, reload=reload)


@cli.command()
@click.argument("workflow", required=False, type=str)
@click.option(
    "--jsonl",
    is_flag=True,
    help="Output raw JSONL format instead of pretty-printed messages (for subprocess/automation use)",
)
@click.option(
    "--stdin",
    is_flag=True,
    help="Read full RunJobRequest JSON from stdin (for subprocess/automation use)",
)
@click.option(
    "--user-id",
    default="1",
    help="User ID for workflow execution (default: 1)",
)
@click.option(
    "--auth-token",
    default="local_token",
    help="Authentication token for workflow execution (default: local_token)",
)
def run(
    workflow: str | None,
    jsonl: bool,
    stdin: bool,
    user_id: str,
    auth_token: str,
):
    """Run a workflow by ID, file path, or RunJobRequest JSON.

    Interactive mode (default):
      Runs a workflow and displays pretty-printed status updates.
      Specify a workflow ID or path to a workflow JSON file.

    JSONL mode (--jsonl):
      Outputs raw JSONL (JSON Lines) format for subprocess/automation use.
      Each line is a valid JSON object representing workflow progress.

    Stdin mode (--stdin):
      Reads a complete RunJobRequest JSON from stdin.
      Useful for programmatic workflow execution.

    Examples:
      # Interactive: Run workflow by ID
      nodetool run workflow_abc123

      # Interactive: Run workflow from file
      nodetool run workflow.json

      # JSONL: Stream workflow progress as JSONL
      nodetool run workflow_abc123 --jsonl

      # Stdin: Read RunJobRequest from stdin (JSONL output)
      echo '{"workflow_id":"abc","user_id":"1","auth_token":"token","params":{}}' | nodetool run --stdin --jsonl

      # Stdin: Read RunJobRequest from file
      cat request.json | nodetool run --stdin --jsonl
    """
    import asyncio
    import json
    import os
    import sys
    import base64
    import traceback
    from typing import Any

    from nodetool.workflows.processing_context import ProcessingContext
    from nodetool.workflows.run_job_request import RunJobRequest
    from nodetool.workflows.run_workflow import run_workflow
    from nodetool.types.graph import Graph

    def _default(obj: Any) -> Any:
        """JSON serializer for objects not serializable by default json code."""
        try:
            if hasattr(obj, "model_dump") and callable(obj.model_dump):
                return obj.model_dump()
        except Exception:
            pass

        if isinstance(obj, (bytes, bytearray)):
            return {
                "__type__": "bytes",
                "base64": base64.b64encode(bytes(obj)).decode("utf-8"),
            }

        return str(obj)

    def _parse_workflow_arg(value: str) -> RunJobRequest:
        """Parse workflow argument as ID, file path, or RunJobRequest JSON."""
        # Check if it's a file
        if os.path.isfile(value):
            with open(value, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if it's a full RunJobRequest or just a workflow definition
            if "graph" in data and "user_id" not in data:
                # It's a workflow definition file
                graph = Graph(**data["graph"])
                return RunJobRequest(
                    user_id=user_id,
                    auth_token=auth_token,
                    graph=graph,
                )
            else:
                # It's a RunJobRequest JSON file
                if isinstance(data.get("graph"), dict):
                    data["graph"] = Graph(**data["graph"])
                return RunJobRequest(**data)

        # Try to parse as inline JSON
        try:
            data = json.loads(value)
            if isinstance(data.get("graph"), dict):
                data["graph"] = Graph(**data["graph"])
            return RunJobRequest(**data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Treat as workflow ID
        return RunJobRequest(
            workflow_id=value,
            user_id=user_id,
            auth_token=auth_token,
        )

    def _read_stdin_request() -> RunJobRequest:
        """Read RunJobRequest JSON from stdin."""
        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            print("Error: No request JSON provided via stdin", file=sys.stderr)
            sys.exit(1)
        try:
            req_dict = json.loads(stdin_data)
            if isinstance(req_dict.get("graph"), dict):
                req_dict["graph"] = Graph(**req_dict["graph"])
            return RunJobRequest(**req_dict)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON from stdin: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: Invalid RunJobRequest: {e}", file=sys.stderr)
            sys.exit(1)

    # Determine the request source
    if stdin:
        request = _read_stdin_request()
    elif workflow:
        try:
            request = _parse_workflow_arg(workflow)
        except Exception as e:
            if jsonl:
                err = {"type": "error", "error": str(e)}
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()
                sys.exit(1)
            else:
                console.print(
                    Panel.fit(f"Failed to prepare workflow: {e}", style="bold red")
                )
                traceback.print_exc()
                sys.exit(1)
    else:
        if jsonl:
            print("Error: Workflow argument required (or use --stdin)", file=sys.stderr)
        else:
            console.print("[red]Error: Workflow argument required (or use --stdin)[/]")
        sys.exit(1)

    # Execute the workflow
    if jsonl:
        # JSONL output mode (for subprocess/automation)
        async def run_jsonl() -> int:
            context = ProcessingContext(
                user_id=request.user_id,
                auth_token=request.auth_token,
                workflow_id=request.workflow_id,
                job_id=None,
            )

            try:
                async for msg in run_workflow(
                    request,
                    context=context,
                    use_thread=False,
                    send_job_updates=True,
                    initialize_graph=True,
                    validate_graph=True,
                ):
                    line = json.dumps(
                        msg if isinstance(msg, dict) else _default(msg),
                        default=_default,
                    )
                    sys.stdout.write(line + "\n")
                    sys.stdout.flush()
                return 0
            except Exception as e:
                err = {"type": "error", "error": str(e)}
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()
                return 1

        exit_code = asyncio.run(run_jsonl())
        sys.exit(exit_code)
    else:
        # Interactive pretty-printed mode
        async def run_interactive():
            workflow_desc = workflow or "stdin"
            console.print(
                Panel.fit(f"Running workflow {workflow_desc}...", style="blue")
            )
            try:
                async for message in run_workflow(request):
                    # Pretty-print each message coming from the runner
                    if isinstance(message, JobUpdate) and message.status == "error":
                        console.print(
                            Panel.fit(f"Error: {message.error}", style="bold red")
                        )
                        sys.exit(1)
                    else:
                        msg_type = Text(message.type, style="bold cyan")
                        console.print(f"{msg_type}: {message.model_dump_json()}")
                console.print(
                    Panel.fit("Workflow finished successfully", style="green")
                )
            except Exception as e:
                console.print(
                    Panel.fit(f"Error running workflow: {e}", style="bold red")
                )
                traceback.print_exc()
                sys.exit(1)

        asyncio.run(run_interactive())


@cli.command()
def chat():
    """Start a nodetool chat."""
    import asyncio
    from nodetool.chat.chat_cli import chat_cli

    asyncio.run(chat_cli())


@cli.command("worker")
@click.option("--host", default="0.0.0.0", help="Host address to bind to (listen on all interfaces for deployments).")
@click.option("--port", default=8000, help="Port to listen on.", type=int)
@click.option(
    "--auth-provider",
    type=click.Choice(["none", "local", "static", "supabase"], case_sensitive=False),
    default=None,
    help="Select authentication provider (overrides ENV AUTH_PROVIDER)",
)
@click.option(
    "--default-model",
    default="gpt-oss:20b",
    help="Fallback model when client doesn't specify one.",
)
@click.option(
    "--provider",
    default="ollama",
    help="AI provider for the default model (e.g., openai, anthropic, ollama).",
)
@click.option(
    "--tools",
    default="",
    help="Comma-separated list of tools to enable (e.g., google_search,browser).",
)
@click.option(
    "--workflow",
    "workflows",
    multiple=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="One or more workflow JSON files to register with the worker.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (DEBUG level) for detailed output.",
)
def worker(
    host: str,
    port: int,
    remote_auth: bool,
    provider: str,
    default_model: str,
    tools: str,
    workflows: list[str],
    verbose: bool = False,
):
    """Start a deployable worker process with OpenAI-compatible endpoints.

    Used for running NodeTool as a backend service with chat/completion API support.
    admin operations, and collection management. It can be deployed anywhere.

    Examples:
      # Start worker on default port 8000
      nodetool worker

      # Start worker on custom port
      nodetool worker --port 8080

      # Start with specific provider and model
      nodetool worker --provider openai --default-model gpt-4

      # Start with tools enabled
      nodetool worker --tools "google_search,browser"

      # Start with verbose logging
      nodetool worker --verbose
    """
    from nodetool.deploy.worker import run_worker

    # Configure logging level based on verbose flag
    if verbose:
        from nodetool.config.logging_config import configure_logging

        configure_logging(level="DEBUG")
        console.print("[cyan]🐛 Verbose logging enabled (DEBUG level)[/]")

    import json
    from nodetool.types.workflow import Workflow
    import dotenv

    dotenv.load_dotenv()

    def load_workflow(path: str) -> Workflow:
        with open(path, "r") as f:
            workflow = json.load(f)
        return Workflow.model_validate(workflow)

    loaded_workflows = [load_workflow(f) for f in workflows]

    # Parse comma-separated tools string into list
    tools_list = (
        [tool.strip() for tool in tools.split(",") if tool.strip()] if tools else []
    )

    run_worker(
        host, port, remote_auth, provider, default_model, tools_list, loaded_workflows
    )


@cli.command("dsl-export")
@click.argument("workflow_id", required=True, type=str)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Path to write the generated DSL Python file. Prints to stdout if omitted.",
    type=click.Path(resolve_path=True, dir_okay=False, file_okay=True),
)
@click.option(
    "--user-id",
    default="1",
    show_default=True,
    help="User ID for database lookup of saved workflows.",
)
def dsl_export(workflow_id: str, output: str | None, user_id: str):
    """Export a workflow to Python DSL code using its WORKFLOW_ID.

    Looks up the workflow by ID in the database; if not found, falls back to
    installed example workflows. Emits Python code that mirrors the graph using
    DSL node wrappers and connections.
    """
    import asyncio
    from typing import Any
    from nodetool.models.workflow import Workflow as WorkflowModel
    from nodetool.packages.registry import Registry

    async def _load_graph() -> Any:
        # Try database first
        wf = await WorkflowModel.find(user_id, workflow_id)
        if wf:
            return wf.get_api_graph()

        # Fallback to examples
        registry = Registry.get_instance()
        examples = registry.list_examples()
        match = next((ex for ex in examples if ex.id == workflow_id), None)
        if not match:
            raise ValueError(f"Workflow '{workflow_id}' not found in database or examples.")

        # Load full example workflow
        ex_full = registry.load_example(match.package_name or "", match.name)
        if not ex_full or not ex_full.graph:
            raise ValueError(f"Failed to load example workflow '{workflow_id}'.")
        return ex_full.graph

    try:
        graph = asyncio.run(_load_graph())
        code = graph_to_dsl_py(graph)
    except Exception as e:
        click.echo(f"Error exporting workflow '{workflow_id}': {e}", err=True)
        raise SystemExit(1)

    if output:
        try:
            with open(output, "w", encoding="utf-8") as f:
                f.write(code)
            click.echo(f"✅ Wrote DSL to {output}")
        except Exception as e:
            click.echo(f"Error writing file '{output}': {e}", err=True)
            raise SystemExit(1)
    else:
        # Print to stdout
        click.echo(code)


@cli.command("chat-server")
@click.option("--host", default="127.0.0.1", help="Host address to bind to.")
@click.option("--port", default=8080, help="Port to listen on.", type=int)
@click.option(
    "--remote-auth", is_flag=True, help="Enable remote authentication (Supabase-backed auth)."
)
@click.option(
    "--default-model",
    default="gpt-oss:20b",
    help="Default AI model to use when client doesn't specify one.",
)
@click.option(
    "--provider",
    default="ollama",
    help="AI provider for the default model.",
)
@click.option(
    "--tools",
    default="",
    help="Comma-separated list of tools (e.g., google_search,browser,google_news).",
)
@click.option(
    "--workflow",
    "workflows",
    multiple=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="One or more workflow JSON files to register.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (DEBUG level) for detailed output.",
)
def chat_server(
    host: str,
    port: int,
    auth_provider: str | None,
    provider: str,
    default_model: str,
    tools: str,
    workflows: list[str],
    verbose: bool = False,
):
    """Launch a WebSocket and Server-Sent Events (SSE) compatible chat server.

    Provides an OpenAI-compatible chat completion interface with optional tool support and custom workflows.

    Examples:
      # Start chat server on default port 8080
      nodetool chat-server

      # Start chat server on port 3000
      nodetool chat-server --port 3000

      # Start with tools
      nodetool chat-server --tools "google_search,google_news,google_images"

      # Start with verbose logging
      nodetool chat-server --verbose
    """
    from nodetool.chat.server import run_chat_server

    # Configure logging level based on verbose flag
    if verbose:
        from nodetool.config.logging_config import configure_logging

        configure_logging(level="DEBUG")
        console.print("[cyan]🐛 Verbose logging enabled (DEBUG level)[/]")
    import json
    from nodetool.types.workflow import Workflow
    import dotenv

    dotenv.load_dotenv()

    def load_workflow(path: str) -> Workflow:
        with open(path, "r") as f:
            workflow = json.load(f)
        return Workflow.model_validate(workflow)

    loaded_workflows = [load_workflow(f) for f in workflows]

    # Parse comma-separated tools string into list
    tools_list = (
        [tool.strip() for tool in tools.split(",") if tool.strip()] if tools else []
    )

    if auth_provider:
        os.environ["AUTH_PROVIDER"] = auth_provider.lower()

    run_chat_server(host, port, provider, default_model, tools_list, loaded_workflows)


@cli.command("chat-client")
@click.option(
    "--server-url",
    help="Override default OpenAI URL to point to a local chat server or custom endpoint.",
)
@click.option(
    "--runpod-endpoint",
    help="RunPod serverless endpoint ID (e.g., abc123xyz) — convenience shortcut.",
)
@click.option("--auth-token", help="HTTP authentication token for server (falls back to RUNPOD_API_KEY env var).")
@click.option("--message", help="Send a single message in non-interactive mode (no conversation loop).")
@click.option(
    "--model",
    default="gpt-4o-mini",
    help="Model to use (e.g., gpt-4o, gpt-oss:20b).",
)
@click.option(
    "--provider",
    help="AI provider when connecting to local server (e.g., openai, anthropic, ollama).",
)
def chat_client(
    server_url: Optional[str],
    auth_token: Optional[str],
    message: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    runpod_endpoint: Optional[str],
):
    """Interactive or non-interactive client for connecting to chat services.

    Supports OpenAI API, local NodeTool chat server, or RunPod serverless endpoints.
    Supports streaming responses and multi-turn conversations.

    Examples:
      # Interactive chat with OpenAI API (default)
      nodetool chat-client --auth-token sk-your-openai-key

      # Use different OpenAI model
      nodetool chat-client --auth-token sk-your-openai-key --model gpt-4

      # Connect to local NodeTool server
      nodetool chat-client --server-url http://localhost:8080

      # Connect to local server with specific model and provider
      nodetool chat-client --server-url http://localhost:8080 --model claude-3-opus-20240229 --provider anthropic

      # Send single message to OpenAI
      nodetool chat-client --message "Hello, AI!" --auth-token sk-your-openai-key

      # Connect to RunPod endpoint
      nodetool chat-client --runpod-endpoint my-runpod-endpoint-id
    """
    import asyncio
    import dotenv
    from nodetool.chat.chat_client import run_chat_client

    dotenv.load_dotenv()

    if not auth_token:
        if server_url and "api.runpod.ai" in server_url:
            auth_token = Environment.get("RUNPOD_API_KEY")
        else:
            auth_token = Environment.get("OPENAI_API_KEY")

    # If no server URL provided, use OpenAI API directly
    if not server_url:
        if runpod_endpoint:
            server_url = f"https://{runpod_endpoint}.api.runpod.ai"
        else:
            server_url = "https://api.openai.com"
            # Use provided model or default to gpt-5-mini for OpenAI
            if not model:
                model = "gpt-5-mini"
    else:
        # For local server, use provided model or default to gpt-oss:20b
        if not model:
            model = "gpt-oss:20b"

    asyncio.run(run_chat_client(server_url, auth_token, message, model))


@cli.group()
def secrets():
    """Manage encrypted secrets stored in the database with per-user encryption."""
    pass


@secrets.command("list")
@click.option("--user-id", "-u", default="1", help="User ID to list secrets for.")
@click.option("--limit", default=100, show_default=True, type=int, help="Maximum number of secrets to return.")
def secrets_list(user_id: str, limit: int) -> None:
    """List stored secret metadata without revealing values."""
    import asyncio
    from nodetool.models.secret import Secret

    async def _list() -> list[Secret]:
        items, _ = await Secret.list_for_user(user_id=user_id, limit=limit)
        return items

    secrets_for_user = asyncio.run(_list())

    if not secrets_for_user:
        console.print(f"[yellow]No secrets stored for user {user_id}.[/]")
        return

    table = Table(title=f"Secrets for user {user_id}")
    table.add_column("Key", style="cyan")
    table.add_column("Updated At", style="green")

    for secret in secrets_for_user:
        updated = secret.updated_at.isoformat() if secret.updated_at else "N/A"
        table.add_row(secret.key, updated)

    console.print(table)


@secrets.command("store")
@click.argument("key")
@click.option("--user-id", "-u", default="1", help="User ID that owns the secret.")
@click.option("--description", "-d", default=None, help="Optional description for the secret.")
@click.option(
    "--no-confirm",
    is_flag=True,
    help="Store the secret without requiring a second confirmation of the value.",
)
def secrets_store(
    key: str,
    user_id: str,
    description: Optional[str],
    no_confirm: bool,
) -> None:
    """Store or update a secret value by securely prompting for input."""
    import asyncio
    from nodetool.models.secret import Secret

    prompt_kwargs = {"hide_input": True}
    if not no_confirm:
        prompt_kwargs["confirmation_prompt"] = True

    secret_value = click.prompt(f"Enter value for secret '{key}'", **prompt_kwargs)

    if secret_value is None or secret_value == "":
        console.print("[red]Secret value cannot be empty.[/]")
        return

    async def _store() -> None:
        await Secret.upsert(
            user_id=user_id,
            key=key,
            value=secret_value,
            description=description,
        )

    asyncio.run(_store())

    # Clear the reference to the value as soon as it is stored
    secret_value = None

    console.print(f"[green]Secret '{key}' stored for user {user_id}.[/]")


@cli.command("codegen")
def codegen_cmd():
    """Regenerate DSL (Domain-Specific Language) modules from node definitions.

    Scans node packages and generates Python code for type-safe workflow creation.
    Completely wipes and recreates src/nodetool/dsl/<namespace>/ directories."""
    # Add the src directory to the Python path to allow relative imports
    src_dir = os.path.abspath("src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    base_nodes_path = os.path.join("src", "nodetool", "nodes")
    base_dsl_path = os.path.join("src", "nodetool", "dsl")

    if not os.path.isdir(base_nodes_path):
        click.echo(f"Error: Nodes directory not found at {base_nodes_path}", err=True)
        return

    namespaces = [
        d
        for d in os.listdir(base_nodes_path)
        if os.path.isdir(os.path.join(base_nodes_path, d))
    ]

    if not namespaces:
        click.echo(
            f"No subdirectories found in {base_nodes_path} to treat as namespaces.",
            err=True,
        )
        return

    for namespace in namespaces:
        source_path = os.path.join(base_nodes_path, namespace)
        output_path = os.path.join(base_dsl_path, namespace)

        # Ensure the output directory for the namespace exists
        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path, exist_ok=True)

        click.echo(
            f"Generating DSL modules from {source_path} to {output_path} for namespace '{namespace}'..."
        )
        create_dsl_modules(source_path, output_path)
        click.echo(f"✅ DSL module generation complete for namespace '{namespace}'!")

    click.echo("✅ All DSL module generation complete!")


@cli.group()
def settings():
    """Commands for viewing and editing configuration settings and secrets files."""
    pass


@settings.command("show")
def show_settings():
    """Show current settings or secrets."""
    from nodetool.config.settings import load_settings

    # Load settings and secrets
    settings_obj = load_settings()

    # Choose which model to display
    data = secrets_obj if secrets else settings_obj

    # Create a rich table
    table = Table(title="Secrets" if secrets else "Settings")

    # Add columns
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="yellow")

    from nodetool.config.configuration import get_settings_registry

    for setting in get_settings_registry():
        # Get field description from the model
        description = setting.description
        table.add_row(setting.env_var, data.get(setting.env_var, ""), description)

    # Display the table
    console.print(table)


@settings.command("edit")
@click.option("--secrets", is_flag=True, help="Edit secrets instead of settings.")
def edit_settings(secrets: bool = False):
    """Edit settings or secrets."""
    from nodetool.config.settings import (
        load_settings,
        get_system_file_path,
        SETTINGS_FILE,
        SECRETS_FILE,
    )
    import subprocess
    import yaml
    import os

    # Load current settings and secrets
    settings_obj, secrets_obj = load_settings()

    # If no specific key/value, open the file in an editor
    file_path = get_system_file_path(SECRETS_FILE if secrets else SETTINGS_FILE)

    if not os.path.exists(file_path):
        # Create the file with empty content if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            if secrets:
                yaml.dump(secrets_obj, f)
            else:
                yaml.dump(settings_obj, f)

    # Open the file in the default editor
    click.echo(f"Opening {file_path} in your default editor...")

    # Determine the editor to use
    editor = os.environ.get("EDITOR", "vi")

    try:
        subprocess.run([editor, file_path], check=True)
        click.echo(f"Settings saved to {file_path}")

    except subprocess.CalledProcessError:
        click.echo("Error: Failed to edit the file", err=True)


# Package Commands Group
@click.group()
def package():
    """Commands for managing NodeTool packages and generating documentation."""
    pass


@package.command("list")
@click.option(
    "--available", "-a", is_flag=True, help="List available packages from the registry"
)
def list_packages(available):
    """List installed or available packages."""
    from nodetool.packages.registry import Registry

    registry = Registry()

    if available:
        packages = registry.list_available_packages()
        if not packages:
            console.print(
                "[bold red]No packages available in the registry or unable to fetch package list.[/]"
            )
            return

        table = Table(title="Available Packages")
        table.add_column("Name", style="cyan")
        table.add_column("Repository ID", style="green")

        for pkg in packages:
            table.add_row(pkg.name, pkg.repo_id)

        console.print(table)
    else:
        packages = registry.list_installed_packages()
        if not packages:
            console.print("[bold yellow]No packages installed.[/]")
            return

        table = Table(title="Installed Packages")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Description", style="yellow")
        table.add_column("Nodes", style="magenta")

        for pkg in packages:
            table.add_row(
                pkg.name, pkg.version, pkg.description, str(len(pkg.nodes or []))
            )

        console.print(table)


@package.command()
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output during scanning"
)
def scan(verbose):
    """Scan current directory for nodes and create package metadata."""
    import sys
    import traceback
    from nodetool.packages.registry import (
        scan_for_package_nodes,
        save_package_metadata,
        update_pyproject_include,
    )

    try:
        print("Scanning for package nodes")
        # Scan for nodes and create package model
        package = scan_for_package_nodes(verbose=verbose)

        # Save package metadata
        save_package_metadata(package, verbose=verbose)
        # Update pyproject.toml with asset files
        update_pyproject_include(package, verbose=verbose)

        node_count = len(package.nodes or [])
        example_count = len(package.examples or [])
        asset_count = len(package.assets or [])

        click.echo(
            f"✅ Successfully created package metadata for {package.name} with:\n"
            f"  - {node_count} nodes\n"
            f"  - {example_count} examples\n"
            f"  - {asset_count} assets"
        )

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if verbose:
            traceback.print_exc()
        sys.exit(1)


@package.command()
def init():
    """Initialize a new Nodetool project."""
    import os

    if os.path.exists("pyproject.toml"):
        if not click.confirm(
            "pyproject.toml already exists. Do you want to overwrite it?"
        ):
            return

    # Gather project information
    name = click.prompt("Project name", type=str)
    version = "0.1.0"
    description = click.prompt("Description", type=str, default="")
    author = click.prompt("Author (name <email>)", type=str)
    python_version = "3.11"

    # Create pyproject.toml content
    author_name = author.split(" <")[0] if " <" in author else author
    author_email = (
        author.split(" <")[1].rstrip(">") if " <" in author else "author@example.com"
    )
    python_req = python_version.lstrip("^")

    pyproject_content = f"""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "{version}"
description = "{description}"
readme = "README.md"
authors = [
    {{name = "{author_name}", email = "{author_email}"}}
]
requires-python = ">={python_req}"

dependencies = [
    "nodetool-core @ git+https://github.com/nodetool-ai/nodetool-core.git@main",
]

[tool.hatch.build.targets.wheel]
packages = ["src/nodetool"]
"""

    # Write to pyproject.toml
    with open("pyproject.toml", "w") as f:
        f.write(pyproject_content)

    # Create basic directory structure
    os.makedirs("src/nodetool/package_metadata", exist_ok=True)

    click.echo("✅ Successfully initialized Nodetool project")
    click.echo("Created:")
    click.echo("  - pyproject.toml")
    click.echo("  - src/nodetool/package_metadata/")


@package.command()
@click.option(
    "--output-dir",
    "-o",
    default="docs",
    help="Directory where documentation will be generated",
)
@click.option(
    "--compact",
    "-c",
    is_flag=True,
    help="Generate compact documentation for LLM usage",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output during scanning"
)
def docs(output_dir: str, compact: bool, verbose: bool):
    """Generate documentation for the package nodes."""
    import os
    import sys
    import tomli
    import traceback
    from nodetool.packages.gen_docs import generate_documentation
    from nodetool.metadata.node_metadata import get_node_classes_from_module

    try:
        # Add src directory to Python path temporarily
        src_path = os.path.abspath("src")
        if not os.path.exists(src_path):
            click.echo("Error: No src directory found", err=True)
            sys.exit(1)

        sys.path.append(src_path)

        nodes_path = os.path.join(src_path, "nodetool", "nodes")
        if not os.path.exists(nodes_path):
            click.echo(
                "Error: No nodes directory found at src/nodetool/nodes", err=True
            )
            sys.exit(1)

        # Get package name from pyproject.toml
        if not os.path.exists("pyproject.toml"):
            click.echo("Error: No pyproject.toml found in current directory", err=True)
            sys.exit(1)

        with open("pyproject.toml", "rb") as f:
            pyproject_data = tomli.loads(f.read().decode())

        project_data = pyproject_data.get("project", {})
        if not project_data:
            click.echo("Error: No [project] metadata found in pyproject.toml", err=True)
            sys.exit(1)

        package_name = project_data.get("name")
        if not package_name:
            click.echo("Error: No package name found in pyproject.toml", err=True)
            sys.exit(1)

        # Note: repository URL from PEP 621 URLs is not required for docs generation

        # Discover node classes by scanning the directory
        node_classes = []
        with click.progressbar(
            length=100,
            label="Scanning for nodes",
            show_eta=False,
            show_percent=True,
        ) as bar:
            bar.update(10)

            # Scan for node classes
            for root, _, files in os.walk(nodes_path):
                for file in files:
                    if file.endswith(".py"):
                        module_path = os.path.join(root, file)
                        rel_path = os.path.relpath(module_path, src_path)
                        module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")

                        click.echo(f"Scanning module: {module_name}")

                        try:
                            classes = get_node_classes_from_module(module_name, verbose)
                            if classes:
                                node_classes.extend(classes)
                        except Exception as e:
                            traceback.print_exc()
                            if verbose:
                                click.echo(
                                    f"Error processing {module_name}: {e}", err=True
                                )

            bar.update(40)

            if not node_classes:
                click.echo("Warning: No node classes found during scanning", err=True)
            else:
                click.echo(f"Found {len(node_classes)} node classes")

            # Generate the documentation
            docs = generate_documentation(node_classes, compact)

            bar.update(40)

            # Write to output file
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "index.md"), "w") as f:
                f.write(docs)
            bar.update(10)

        click.echo(f"✅ Documentation generated in {output_dir}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        traceback.print_exc()
        sys.exit(1)


# Add package group to the main CLI
cli.add_command(package)

# Add settings group to the main CLI
cli.add_command(settings)


@cli.group()
def admin():
    """Maintenance utilities for model assets and caches.

    Manage HuggingFace and Ollama model downloads, cache inspection, and cleanup."""
    pass


@admin.command("download-hf")
@click.option("--repo-id", required=True, help="HuggingFace repository ID to download")
@click.option(
    "--cache-dir", default="/app/.cache/huggingface/hub", help="Cache directory path"
)
@click.option("--file-path", help="Specific file to download (optional)")
@click.option(
    "--allow-patterns", multiple=True, help="Patterns to allow (can specify multiple)"
)
@click.option(
    "--ignore-patterns", multiple=True, help="Patterns to ignore (can specify multiple)"
)
@click.option(
    "--server-url",
    required=True,
    help="HTTP API server URL to execute on (e.g., http://localhost:8000)",
)
def download_hf(
    repo_id: str,
    cache_dir: str,
    file_path: str | None,
    allow_patterns: tuple,
    ignore_patterns: tuple,
    server_url: str,
):
    """Download HuggingFace models with progress tracking.

    Examples:
        # Download entire model repository locally
        nodetool admin download-hf --repo-id microsoft/DialoGPT-small

        # Download with streaming progress locally
        nodetool admin download-hf --repo-id microsoft/DialoGPT-small

        # Download via HTTP API server
        nodetool admin download-hf --repo-id microsoft/DialoGPT-small --server-url http://localhost:8000

        # Download specific file via HTTP API
        nodetool admin download-hf --repo-id microsoft/DialoGPT-small --file-path config.json --server-url http://localhost:8000

        # Download with pattern filtering via HTTP API
        nodetool admin download-hf --repo-id microsoft/DialoGPT-small --allow-patterns "*.json" --allow-patterns "*.txt" --ignore-patterns "*.bin" --server-url http://localhost:8000
    """
    import asyncio

    import dotenv

    dotenv.load_dotenv()

    async def run_download():
        console.print("[bold cyan]📥 Starting HuggingFace download...[/]")
        console.print(f"Repository: {repo_id}")
        console.print(f"Cache directory: {cache_dir}")
        if file_path:
            console.print(f"File: {file_path}")
        if allow_patterns:
            console.print(f"Allow patterns: {', '.join(allow_patterns)}")
        if ignore_patterns:
            console.print(f"Ignore patterns: {', '.join(ignore_patterns)}")
        console.print(f"HTTP API Server: {server_url}")
        console.print()

        try:
            # Execute via HTTP API
            from nodetool.deploy.admin_client import AdminHTTPClient

            api_key = os.getenv("RUNPOD_API_KEY")

            client = AdminHTTPClient(server_url, auth_token=api_key)
            async for progress_update in client.download_huggingface_model(
                repo_id=repo_id,
                cache_dir=cache_dir,
                file_path=file_path,
                ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
                allow_patterns=list(allow_patterns) if allow_patterns else None,
            ):
                progress_manager._display_progress_update(progress_update)

        except Exception as e:
            console.print(f"[red]❌ Failed: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_download())


@admin.command("download-ollama")
@click.option("--model-name", required=True, help="Ollama model name to download")
@click.option(
    "--server-url",
    required=True,
    help="HTTP API server URL to execute on (e.g., http://localhost:8000)",
)
def download_ollama(
    model_name: str,
    server_url: str,
):
    """Download Ollama models with progress tracking.

    Examples:
        # Download Ollama model locally
        nodetool admin download-ollama --model-name llama3.2:latest

        # Download with streaming progress locally
        nodetool admin download-ollama --model-name llama3.2:latest

        # Download via HTTP API server
        nodetool admin download-ollama --model-name llama3.2:latest --server-url http://localhost:8000
    """
    import asyncio

    async def run_download():
        console.print("[bold cyan]📥 Starting Ollama download...[/]")
        console.print(f"Model: {model_name}")
        console.print(f"HTTP API Server: {server_url}")
        console.print()

        try:
            # Execute via HTTP API
            from nodetool.deploy.admin_client import AdminHTTPClient

            api_key = os.getenv("RUNPOD_API_KEY")

            client = AdminHTTPClient(server_url, auth_token=api_key)
            async for progress_update in client.download_ollama_model(
                model_name=model_name
            ):
                progress_manager._display_progress_update(progress_update)

        except Exception as e:
            console.print(f"[red]❌ Failed: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_download())


@admin.command("scan-cache")
@click.option(
    "--server-url",
    required=True,
    help="HTTP API server URL to execute on (e.g., http://localhost:8000)",
)
def scan_cache(server_url: str):
    """Scan HuggingFace cache and display information.

    Examples:
        # Scan cache locally
        nodetool admin scan-cache

        # Scan cache via HTTP API server
        nodetool admin scan-cache --server-url http://localhost:8000
    """
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    async def run_scan():
        console.print("[bold cyan]🔍 Scanning HuggingFace cache...[/]")
        console.print(f"HTTP API Server: {server_url}")
        console.print()

        try:
            # Execute via HTTP API
            from nodetool.deploy.admin_client import AdminHTTPClient

            api_key = os.getenv("RUNPOD_API_KEY")

            client = AdminHTTPClient(server_url, auth_token=api_key)
            result = await client.scan_cache()
            _handle_scan_cache_output(result)

        except Exception as e:
            console.print(f"[red]❌ Failed: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    def _handle_scan_cache_output(progress_update):
        """Handle scan cache specific output."""
        status = progress_update.get("status", "unknown")

        if status == "completed":
            cache_info = progress_update.get("cache_info", {})
            console.print("[green]✅ Cache scan completed[/]")

            # Display cache information
            size_on_disk = cache_info.get("size_on_disk", 0)
            size_gb = size_on_disk / (1024**3) if size_on_disk else 0

            console.print(f"[cyan]📊 Total cache size: {size_gb:.2f} GB[/]")

            repos = cache_info.get("repos", [])
            if repos:
                console.print(f"[cyan]📋 Found {len(repos)} cached repositories:[/]")

                table = Table()
                table.add_column("Repository", style="cyan")
                table.add_column("Size (GB)", style="green")
                table.add_column("Files", style="yellow")

                for repo in repos:
                    repo_size_gb = repo.get("size_on_disk", 0) / (1024**3)
                    table.add_row(
                        repo.get("repo_id", "Unknown"),
                        f"{repo_size_gb:.2f}",
                        str(repo.get("nb_files", 0)),
                    )

                console.print(table)
            else:
                console.print("[yellow]No cached repositories found[/]")

        elif status == "error":
            error = progress_update.get("error", "Unknown error")
            console.print(f"[red]❌ Error: {error}[/]")
            sys.exit(1)

    asyncio.run(run_scan())


@admin.command("delete-hf")
@click.option(
    "--repo-id", required=True, help="HuggingFace repository ID to delete from cache"
)
@click.option(
    "--server-url",
    required=True,
    help="HTTP API server URL to execute on (e.g., http://localhost:8000)",
)
def delete_hf(repo_id: str, server_url: str):
    """Delete HuggingFace model from cache.

    Examples:
        # Delete model locally
        nodetool admin delete-hf --repo-id microsoft/DialoGPT-small

        # Delete model via HTTP API server
        nodetool admin delete-hf --repo-id microsoft/DialoGPT-small --server-url http://localhost:8000
    """
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    async def run_delete():
        console.print("[bold yellow]🗑️ Deleting HuggingFace model from cache...[/]")
        console.print(f"Repository: {repo_id}")
        console.print(f"HTTP API Server: {server_url}")
        console.print()

        if not click.confirm(
            f"Are you sure you want to delete {repo_id} from the cache?"
        ):
            console.print("[yellow]❌ Operation cancelled[/]")
            return

        try:
            # Execute via HTTP API
            from nodetool.deploy.admin_client import AdminHTTPClient

            api_key = os.getenv("RUNPOD_API_KEY")

            client = AdminHTTPClient(server_url, auth_token=api_key)
            result = await client.delete_huggingface_model(repo_id=repo_id)
            progress_manager._display_progress_update(result)
        except Exception as e:
            console.print(f"[red]❌ Failed: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_delete())


@admin.command("cache-size")
@click.option(
    "--cache-dir", default="/app/.cache/huggingface/hub", help="Cache directory path"
)
@click.option(
    "--server-url",
    required=True,
    help="HTTP API server URL to execute on (e.g., http://localhost:8000)",
)
def cache_size(cache_dir: str, server_url: str, api_key: str | None):
    """Calculate total cache size.

    Examples:
        # Calculate cache size locally
        nodetool admin cache-size

        # Calculate cache size with custom directory locally
        nodetool admin cache-size --cache-dir /custom/cache/path

        # Calculate cache size via HTTP API server
        nodetool admin cache-size --server-url http://localhost:8000
    """
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    async def run_calculate():
        console.print("[bold cyan]📏 Calculating cache size...[/]")
        console.print(f"Cache directory: {cache_dir}")
        console.print(f"HTTP API Server: {server_url}")
        console.print()

        try:
            # Execute via HTTP API
            from nodetool.deploy.admin_client import AdminHTTPClient

            api_key = os.getenv("RUNPOD_API_KEY")

            client = AdminHTTPClient(server_url, auth_token=api_key)
            result = await client.get_cache_size(cache_dir=cache_dir)
            _handle_cache_size_output(result)

        except Exception as e:
            console.print(f"[red]❌ Failed: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    def _handle_cache_size_output(progress_update):
        """Handle cache size specific output."""
        if "success" in progress_update and progress_update["success"]:
            total_size = progress_update.get("total_size_bytes", 0)
            size_gb = progress_update.get("size_gb", 0)

            console.print("[green]✅ Cache size calculation completed[/]")
            console.print(
                f"[cyan]📊 Total size: {size_gb} GB ({total_size:,} bytes)[/]"
            )
        elif "status" in progress_update and progress_update["status"] == "error":
            error = progress_update.get("error", "Unknown error")
            console.print(f"[red]❌ Error: {error}[/]")
            sys.exit(1)

    asyncio.run(run_calculate())


# Add admin group to the main CLI
cli.add_command(admin)


def _handle_list_options(
    list_gpu_types: bool,
    list_cpu_flavors: bool,
    list_data_centers: bool,
    list_all_options: bool,
) -> None:
    """Handle list options and exit if any are specified."""
    from nodetool.deploy.runpod_api import (
        ComputeType,
        CPUFlavor,
        DataCenter,
        CUDAVersion,
    )

    if list_gpu_types:
        console.print("[bold cyan]Available GPU Types:[/]")
        for gpu_type in SUPPORTED_GPU_TYPES:
            console.print(f"  {gpu_type}")
        sys.exit(0)

    if list_cpu_flavors:
        console.print("[bold cyan]Available CPU Flavors:[/]")
        for cpu_flavor in CPUFlavor:
            console.print(f"  {cpu_flavor.value}")
        sys.exit(0)

    if list_data_centers:
        console.print("[bold cyan]Available Data Centers:[/]")
        for data_center in DataCenter:
            console.print(f"  {data_center.value}")
        sys.exit(0)

    if list_all_options:
        console.print("[bold cyan]Available Options:[/]")
        console.print("\n[bold]Compute Types:[/]")
        for compute_type in ComputeType:
            console.print(f"  {compute_type.value}")
        console.print("\n[bold]GPU Types:[/]")
        for gpu_type in SUPPORTED_GPU_TYPES:
            console.print(f"  {gpu_type}")
        console.print("\n[bold]CPU Flavors:[/]")
        for cpu_flavor in CPUFlavor:
            console.print(f"  {cpu_flavor.value}")
        console.print("\n[bold]Data Centers:[/]")
        for data_center in DataCenter:
            console.print(f"  {data_center.value}")
        console.print("\n[bold]CUDA Versions:[/]")
        for cuda_version in CUDAVersion:
            console.print(f"  {cuda_version.value}")
        sys.exit(0)


@cli.command("list-gcp-options")
def list_gcp_options():
    """List available Google Cloud Run configuration options for deployments.

    Shows available regions, CPU options, memory options, and Docker registry options."""
    from nodetool.deploy.google_cloud_run_api import (
        CloudRunRegion,
        CloudRunCPU,
        CloudRunMemory,
    )

    console.print("[bold cyan]Google Cloud Run Options:[/]")

    console.print("\n[bold]Regions:[/]")
    for region in CloudRunRegion:
        console.print(f"  {region.value}")

    console.print("\n[bold]CPU Options:[/]")
    for cpu in CloudRunCPU:
        console.print(f"  {cpu.value}")

    console.print("\n[bold]Memory Options:[/]")
    for memory in CloudRunMemory:
        console.print(f"  {memory.value}")

    console.print("\n[bold]Registry Options:[/]")
    console.print("  gcr.io")
    console.print("  us-docker.pkg.dev")
    console.print("  europe-docker.pkg.dev")
    console.print("  asia-docker.pkg.dev")


def _handle_docker_config_check(
    check_docker_config: bool, docker_registry: str, docker_username: str | None
) -> None:
    """Handle Docker configuration check and exit if specified."""
    if not check_docker_config:
        return

    from nodetool.deploy.docker import (
        check_docker_auth,
        get_docker_username_from_config,
        format_image_name,
    )

    console.print("🔍 Checking Docker configuration...")

    # Check Docker authentication
    is_authenticated = check_docker_auth(docker_registry)
    console.print(f"Registry: {docker_registry}")
    console.print(f"Authenticated: {'✅ Yes' if is_authenticated else '❌ No'}")

    # Check Docker username from config
    config_username = get_docker_username_from_config(docker_registry)
    if config_username:
        console.print(f"Username from Docker config: {config_username}")
    else:
        console.print("Username from Docker config: ❌ Not found")

    # Check environment and arguments
    env_username = os.getenv("DOCKER_USERNAME")
    if env_username:
        console.print(f"Username from DOCKER_USERNAME env: {env_username}")
    else:
        console.print("Username from DOCKER_USERNAME env: ❌ Not set")

    if docker_username:
        console.print(f"Username from --docker-username arg: {docker_username}")
    else:
        console.print("Username from --docker-username arg: ❌ Not provided")

    # Show final resolved username
    final_username = docker_username or env_username or config_username

    if final_username:
        console.print(f"\n🎉 Final resolved username: {final_username}")

        # Show what the full image name would be
        example_image = format_image_name(
            "my-workflow", final_username, docker_registry
        )
        example_tag = generate_image_tag()
        console.print(f"Example image name: {example_image}:{example_tag}")
    else:
        console.print("\n❌ No Docker username found!")
        console.print("To fix this, run: docker login")

    sys.exit(0)


def env_for_deploy(
    chat_provider: str,
    default_model: str,
):
    """Get environment variables for deploy."""
    # Parse comma-separated tools string into list
    env = {
        "CHAT_PROVIDER": chat_provider,
        "DEFAULT_MODEL": default_model,
    }

    # Merge settings and secrets from settings.yaml and secrets.yaml into env
    # without overriding explicitly provided values

    _settings, _secrets = load_settings()
    for _k, _v in (_settings or {}).items():
        if _v is not None and str(_v) != "" and _k not in env:
            env[_k] = str(_v)
    for _k, _v in (_secrets or {}).items():
        if _v is not None and str(_v) != "" and _k not in env:
            env[_k] = str(_v)

    master_key = os.environ.get("SECRETS_MASTER_KEY")
    if master_key:
        env.setdefault("SECRETS_MASTER_KEY", master_key)

    return env


def _populate_master_key_env(deployment: Any, master_key: str) -> None:
    from nodetool.config.deployment import (
        SelfHostedDeployment,
        RunPodDeployment,
        GCPDeployment,
    )

    def _inject(env: Optional[Dict[str, str]]) -> Dict[str, str]:
        env = dict(env) if env else {}
        env["SECRETS_MASTER_KEY"] = master_key
        return env

    if isinstance(deployment, SelfHostedDeployment):
        deployment.container.environment = _inject(deployment.container.environment)
        if deployment.proxy and deployment.proxy.services:
            for service in deployment.proxy.services:
                service.environment = _inject(service.environment)
    elif isinstance(deployment, (RunPodDeployment, GCPDeployment)):
        deployment.environment = _inject(getattr(deployment, "environment", None))


async def _export_encrypted_secrets_payload(limit: int = 1000) -> List[Dict[str, Any]]:
    from nodetool.models.secret import Secret

    secrets = await Secret.list_all(limit=limit)
    payload: List[Dict[str, Any]] = []
    for secret in secrets:
        payload.append(
            {
                "user_id": secret.user_id,
                "key": secret.key,
                "encrypted_value": secret.encrypted_value,
                "description": secret.description,
                "created_at": secret.created_at.isoformat()
                if secret.created_at
                else None,
                "updated_at": secret.updated_at.isoformat()
                if secret.updated_at
                else None,
            }
        )
    return payload


async def _import_secrets_to_worker(
    server_url: str, auth_token: str, payload: List[Dict[str, Any]]
) -> Dict[str, Any]:
    from nodetool.deploy.admin_client import AdminHTTPClient

    client = AdminHTTPClient(server_url, auth_token)
    return await client.import_secrets(payload)


def _sync_secrets_to_deployment(name: str, deployment: Any) -> None:
    server_url = getattr(deployment, "get_server_url", lambda: None)()
    if not server_url:
        console.print(
            f"[yellow]Skipping secret sync for '{name}': server URL unavailable.[/]"
        )
        return

    auth_token = getattr(deployment, "worker_auth_token", None)
    if not auth_token:
        env: Optional[Dict[str, str]] = None
        if hasattr(deployment, "environment") and getattr(deployment, "environment"):
            env = deployment.environment
        elif hasattr(deployment, "container") and getattr(
            deployment.container, "environment", None
        ):
            env = deployment.container.environment
        if env:
            auth_token = env.get("WORKER_AUTH_TOKEN")

    if not auth_token:
        console.print(
            f"[yellow]Skipping secret sync for '{name}': worker auth token unavailable.[/]"
        )
        return

    secrets_payload = asyncio.run(_export_encrypted_secrets_payload())
    if not secrets_payload:
        console.print(f"[green]No local secrets to sync for '{name}'.[/]")
        return

    try:
        asyncio.run(_import_secrets_to_worker(server_url, auth_token, secrets_payload))
        console.print(f"[green]Synced {len(secrets_payload)} secret(s) to '{name}'.[/]")
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]Warning: failed to sync secrets for '{name}': {exc}[/]"
        )


@cli.group()
def deploy():
    """Controls deployments described in deployment.yaml.

    Manage cloud and self-hosted deployments (RunPod, Google Cloud Run, self-hosted Docker, etc.)."""
    pass


@deploy.command("init")
def deploy_init():
    """Initialize a new deployment.yaml configuration file."""
    from nodetool.config.deployment import (
        init_deployment_config,
        get_deployment_config_path,
    )

    try:
        config_path = get_deployment_config_path()

        if config_path.exists():
            if not click.confirm(
                f"Deployment configuration already exists at {config_path}. Overwrite?"
            ):
                console.print("[yellow]Operation cancelled[/]")
                return

        console.print("[bold cyan]🚀 Initializing deployment configuration...[/]")
        console.print()

        init_deployment_config()

        console.print(f"[green]✅ Created deployment.yaml at {config_path}[/]")
        console.print()
        console.print("[cyan]Next steps:[/]")
        console.print("  1. Edit deployment.yaml to add your deployments")
        console.print("  2. Run 'nodetool deploy list' to see configured deployments")
        console.print("  3. Run 'nodetool deploy plan <name>' to preview changes")
        console.print("  4. Run 'nodetool deploy apply <name>' to deploy")
        console.print()
        console.print("[cyan]Example deployment types:[/]")
        console.print("  - self-hosted: Deploy to your own server via SSH")
        console.print("  - runpod: Deploy to RunPod serverless")
        console.print("  - gcp: Deploy to Google Cloud Run")

    except FileExistsError as e:
        console.print(f"[yellow]{e}[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@deploy.command("show")
@click.argument("name")
def deploy_show(name: str):
    """Display detailed information about a specific deployment."""
    from nodetool.deploy.manager import DeploymentManager
    from nodetool.config.deployment import (
        SelfHostedDeployment,
        RunPodDeployment,
        GCPDeployment,
    )

    try:
        manager = DeploymentManager()

        # Get deployment configuration
        deployment = manager.get_deployment(name)

        # Get current state
        state = manager.state_manager.read_state(name)

        # Create rich panel with deployment details
        from rich.panel import Panel

        # Build content for the panel
        content = []

        # Header
        content.append(f"[bold cyan]Deployment: {name}[/]")
        content.append(f"[cyan]Type: {deployment.type}[/]")
        content.append("")

        # Type-specific configuration
        if isinstance(deployment, SelfHostedDeployment):
            content.append("[bold]Self-Hosted Configuration:[/]")
            content.append(f"  Host: {deployment.host}")
            content.append(f"  SSH User: {deployment.ssh.user}")
            content.append(f"  Image: {deployment.image.name}:{deployment.image.tag}")
            content.append("")

            # Container details
            content.append("[bold]Container:[/]")
            content.append(f"  • {deployment.container.name}")
            content.append(f"    Port: {deployment.container.port}")
            if deployment.container.workflows:
                content.append(
                    f"    Workflows: {', '.join(deployment.container.workflows)}"
                )
            if deployment.container.gpu:
                content.append(f"    GPU: {deployment.container.gpu}")
            content.append("")

            # Paths
            content.append("[bold]Paths:[/]")
            content.append(f"  Workspace: {deployment.paths.workspace}")
            content.append(f"  HF Cache: {deployment.paths.hf_cache}")

        elif isinstance(deployment, RunPodDeployment):
            content.append("[bold]RunPod Configuration:[/]")
            content.append(f"  Image: {deployment.image.name}:{deployment.image.tag}")
            content.append(f"  Template ID: {deployment.template_id or 'Not set'}")
            content.append(f"  Endpoint ID: {deployment.endpoint_id or 'Not set'}")
            content.append("")

            if state and state.get("pod_id"):
                content.append("[bold]RunPod State:[/]")
                content.append(f"  Pod ID: {state['pod_id']}")

        elif isinstance(deployment, GCPDeployment):
            content.append("[bold]Google Cloud Run Configuration:[/]")
            content.append(f"  Project: {deployment.project_id}")
            content.append(f"  Region: {deployment.region}")
            content.append(f"  Service: {deployment.service_name}")
            content.append(f"  Image: {deployment.image.name}:{deployment.image.tag}")
            content.append(f"  CPU: {deployment.cpu}")
            content.append(f"  Memory: {deployment.memory}")
            content.append("")

        # Current state
        content.append("[bold]Status:[/]")
        if state:
            status = state.get("status", "unknown")
            status_color = {
                "running": "green",
                "active": "green",
                "stopped": "red",
                "error": "red",
                "unknown": "yellow",
            }.get(status, "white")
            content.append(f"  Status: [{status_color}]{status}[/]")

            if state.get("last_deployed"):
                content.append(f"  Last Deployed: {state['last_deployed']}")

            if state.get("compose_hash"):
                content.append(f"  Compose Hash: {state['compose_hash'][:12]}...")
        else:
            content.append("  Status: [yellow]Not deployed[/]")

        content.append("")

        # URLs and endpoints
        if isinstance(deployment, SelfHostedDeployment):
            content.append("[bold]Endpoints:[/]")
            url = f"http://{deployment.host}:{deployment.container.port}"
            content.append(f"  {deployment.container.name}: {url}")

        elif isinstance(deployment, GCPDeployment):
            if state and state.get("service_url"):
                content.append("[bold]Endpoint:[/]")
                content.append(f"  {state['service_url']}")

        elif isinstance(deployment, RunPodDeployment):
            if state and state.get("endpoint_url"):
                content.append("[bold]Endpoint:[/]")
                content.append(f"  {state['endpoint_url']}")

        # Display the panel
        panel_content = "\n".join(content)
        panel = Panel(
            panel_content,
            title="[bold]Deployment Details[/]",
            border_style="cyan",
            expand=False,
        )

        console.print(panel)

    except KeyError:
        console.print(f"[red]Deployment '{name}' not found[/]")
        console.print()
        console.print("[cyan]Available deployments:[/]")
        console.print("  Run: nodetool deploy list")
        sys.exit(1)
    except FileNotFoundError:
        console.print("[yellow]No deployment.yaml found[/]")
        console.print()
        console.print("[cyan]Create one with:[/]")
        console.print("  nodetool deploy init")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@deploy.command("add")
@click.argument("name")
@click.option(
    "--type",
    "deployment_type",
    type=click.Choice(["self-hosted", "runpod", "gcp"]),
    prompt="Deployment type",
    help="Type of deployment",
)
def deploy_add(name: str, deployment_type: str):
    """Add a new deployment to deployment.yaml interactively."""
    from nodetool.config.deployment import (
        DeploymentConfig,
        SelfHostedDeployment,
        RunPodDeployment,
        GCPDeployment,
        SSHConfig,
        ImageConfig,
        ContainerConfig,
        load_deployment_config,
        save_deployment_config,
        get_deployment_config_path,
    )

    try:
        config_path = get_deployment_config_path()

        # Load existing config
        try:
            config = load_deployment_config()
        except FileNotFoundError:
            console.print("[yellow]No deployment.yaml found. Creating new file...[/]")
            config = DeploymentConfig(deployments={})

        # Check if deployment name already exists
        if name in config.deployments:
            console.print(f"[red]Deployment '{name}' already exists[/]")
            console.print("[cyan]Use 'nodetool deploy edit {name}' to modify it[/]")
            sys.exit(1)

        console.print(f"[bold cyan]Adding new {deployment_type} deployment: {name}[/]")
        console.print()

        # Gather deployment-specific configuration
        if deployment_type == "self-hosted":
            console.print("[cyan]Self-Hosted Configuration:[/]")
            host = click.prompt("Host address", type=str)
            ssh_user = click.prompt("SSH username", type=str)
            ssh_key_path = click.prompt(
                "SSH key path", type=str, default="~/.ssh/id_rsa"
            )

            image_name = click.prompt(
                "Docker image name", type=str, default="nodetool/nodetool"
            )
            image_tag = click.prompt("Docker image tag", type=str, default="latest")

            # Container configuration
            console.print()
            console.print("[cyan]Add containers (press Ctrl+C when done):[/]")
            containers = []

            while True:
                try:
                    console.print()
                    container_name = click.prompt(
                        f"  Container #{len(containers) + 1} name", type=str
                    )
                    container_port = click.prompt("  Port", type=int)

                    # Optional GPU
                    use_gpu = click.confirm("  Assign GPU?", default=False)
                    gpu = None
                    if use_gpu:
                        gpu = click.prompt(
                            "  GPU device(s) (e.g., '0' or '0,1')", type=str
                        )

                    # Optional workflows
                    has_workflows = click.confirm(
                        "  Assign specific workflows?", default=False
                    )
                    workflows = None
                    if has_workflows:
                        workflows_str = click.prompt(
                            "  Workflow IDs (comma-separated)", type=str
                        )
                        workflows = [w.strip() for w in workflows_str.split(",")]

                    containers.append(
                        ContainerConfig(
                            name=container_name,
                            port=container_port,
                            gpu=gpu,
                            workflows=workflows,
                        )
                    )

                    if not click.confirm("  Add another container?", default=True):
                        break

                except click.exceptions.Abort:
                    break

            if not containers:
                console.print(
                    "[yellow]No containers configured. Adding default container.[/]"
                )
                containers.append(ContainerConfig(name="default", port=8000))

            deployment = SelfHostedDeployment(
                host=host,
                ssh=SSHConfig(user=ssh_user, key_path=ssh_key_path),
                image=ImageConfig(name=image_name, tag=image_tag),
                containers=containers,
            )

        elif deployment_type == "runpod":
            console.print("[cyan]RunPod Configuration:[/]")
            image_name = click.prompt("Docker image name", type=str)
            image_tag = click.prompt("Docker image tag", type=str, default="latest")
            template_id = click.prompt("Template ID (optional)", type=str, default="")
            endpoint_id = click.prompt("Endpoint ID (optional)", type=str, default="")

            deployment = RunPodDeployment(
                image=ImageConfig(name=image_name, tag=image_tag),
                template_id=template_id or None,
                endpoint_id=endpoint_id or None,
            )

        elif deployment_type == "gcp":
            console.print("[cyan]Google Cloud Run Configuration:[/]")
            project_id = click.prompt("GCP Project ID", type=str)
            region = click.prompt("Region", type=str, default="us-central1")
            service_name = click.prompt("Service name", type=str, default=name)
            image_name = click.prompt("Docker image name", type=str)
            image_tag = click.prompt("Docker image tag", type=str, default="latest")

            # Optional resource configuration
            console.print()
            configure_resources = click.confirm("Configure CPU/Memory?", default=False)
            cpu = "4"
            memory = "16Gi"
            if configure_resources:
                cpu = click.prompt("CPU cores", type=str, default="4")
                memory = click.prompt("Memory", type=str, default="16Gi")

            deployment = GCPDeployment(
                project_id=project_id,
                region=region,
                service_name=service_name,
                image=ImageConfig(name=image_name, tag=image_tag),
                cpu=cpu,
                memory=memory,
            )

        # Add deployment to config
        config.deployments[name] = deployment

        # Save config
        save_deployment_config(config)

        console.print()
        console.print(f"[green]✅ Deployment '{name}' added to {config_path}[/]")
        console.print()
        console.print("[cyan]Next steps:[/]")
        console.print(f"  1. Review configuration: nodetool deploy show {name}")
        console.print(f"  2. Preview changes: nodetool deploy plan {name}")
        console.print(f"  3. Deploy: nodetool deploy apply {name}")

    except click.exceptions.Abort:
        console.print()
        console.print("[yellow]Operation cancelled[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@deploy.command("edit")
@click.argument("name", required=False)
def deploy_edit(name: Optional[str]):
    """Edit deployment configuration file.

    If a deployment name is provided, opens the file and shows the deployment location.
    Otherwise, opens the entire deployment.yaml file.
    """
    from nodetool.config.deployment import (
        get_deployment_config_path,
        load_deployment_config,
    )
    import subprocess

    try:
        config_path = get_deployment_config_path()

        if not config_path.exists():
            console.print("[yellow]No deployment.yaml found[/]")
            console.print()
            console.print("[cyan]Create one with:[/]")
            console.print("  nodetool deploy init")
            sys.exit(1)

        # Verify deployment name exists if provided
        if name:
            try:
                dep_config = load_deployment_config()
                if name not in dep_config.deployments:
                    console.print(f"[red]Deployment '{name}' not found[/]")
                    console.print()
                    console.print("[cyan]Available deployments:[/]")
                    for dep_name in dep_config.deployments.keys():
                        console.print(f"  • {dep_name}")
                    sys.exit(1)

                console.print(
                    f"[cyan]Opening deployment.yaml (deployment: {name})...[/]"
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Could not verify deployment: {e}[/]")
        else:
            console.print("[cyan]Opening deployment.yaml...[/]")

        console.print(f"[cyan]File: {config_path}[/]")
        console.print()

        # Determine editor to use
        editor = os.environ.get("EDITOR", "vi")

        try:
            subprocess.run([editor, str(config_path)], check=True)
            console.print()
            console.print("[green]✅ File saved[/]")
            console.print()
            console.print("[cyan]Next steps:[/]")
            if name:
                console.print(f"  1. Review changes: nodetool deploy show {name}")
                console.print(f"  2. Preview deployment: nodetool deploy plan {name}")
                console.print(f"  3. Apply changes: nodetool deploy apply {name}")
            else:
                console.print("  1. Review deployments: nodetool deploy list")
                console.print(
                    "  2. Show specific deployment: nodetool deploy show <name>"
                )

        except subprocess.CalledProcessError:
            console.print("[red]Error: Failed to edit the file[/]")
            sys.exit(1)

    except FileNotFoundError:
        console.print("[yellow]No deployment.yaml found[/]")
        console.print()
        console.print("[cyan]Create one with:[/]")
        console.print("  nodetool deploy init")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@deploy.command("list")
def deploy_list():
    """List all configured deployments and their status."""
    from nodetool.deploy.manager import DeploymentManager

    try:
        manager = DeploymentManager()
        deployments = manager.list_deployments()

        if not deployments:
            console.print("[yellow]No deployments configured[/]")
            console.print()
            console.print("[cyan]Create a deployment.yaml file to get started:[/]")
            console.print("  nodetool deploy init")
            return

        table = Table(title="Deployments")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Host/Location", style="magenta")
        table.add_column("Last Deployed", style="blue")

        for deployment in deployments:
            status_color = {
                "running": "green",
                "active": "green",
                "stopped": "red",
                "error": "red",
                "unknown": "yellow",
            }.get(deployment["status"], "white")

            status = f"[{status_color}]{deployment['status']}[/]"

            # Get host/location based on type
            location = deployment.get("host", "")
            if not location and "project" in deployment:
                location = f"{deployment['project']}/{deployment.get('region', '')}"

            last_deployed = deployment.get("last_deployed", "Never")
            if last_deployed and last_deployed != "Never":
                # Format timestamp
                from datetime import datetime

                try:
                    dt = datetime.fromisoformat(last_deployed.replace("Z", "+00:00"))
                    last_deployed = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass

            table.add_row(
                deployment["name"], deployment["type"], status, location, last_deployed
            )

        console.print(table)

    except FileNotFoundError:
        console.print("[yellow]No deployment.yaml found[/]")
        console.print()
        console.print("[cyan]Create one with:[/]")
        console.print("  nodetool deploy init")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@deploy.command("plan")
@click.argument("name")
def deploy_plan(name: str):
    """Show what changes will be made without executing deployment."""
    from nodetool.deploy.manager import DeploymentManager

    try:
        manager = DeploymentManager()
        plan = manager.plan(name)

        console.print(f"[bold cyan]Deployment Plan: {plan['deployment_name']}[/]")
        console.print(f"[cyan]Type: {plan.get('type', 'self-hosted')}[/]")
        console.print(f"[cyan]Host: {plan.get('host', 'N/A')}[/]")
        console.print()

        if plan.get("changes"):
            console.print("[bold yellow]Changes:[/]")
            for change in plan["changes"]:
                console.print(f"  • {change}")
            console.print()

        if plan.get("will_create"):
            console.print("[bold green]Will Create:[/]")
            for item in plan["will_create"]:
                console.print(f"  + {item}")
            console.print()

        if plan.get("will_update"):
            console.print("[bold yellow]Will Update:[/]")
            for item in plan["will_update"]:
                console.print(f"  ~ {item}")
            console.print()

        if plan.get("will_destroy"):
            console.print("[bold red]Will Destroy:[/]")
            for item in plan["will_destroy"]:
                console.print(f"  - {item}")
            console.print()

        if (
            not plan.get("changes")
            and not plan.get("will_create")
            and not plan.get("will_update")
        ):
            console.print("[green]✅ No changes - deployment is up to date[/]")

    except KeyError:
        console.print(f"[red]Deployment '{name}' not found[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@deploy.command("apply")
@click.argument("name")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without executing"
)
def deploy_apply(name: str, dry_run: bool):
    """Apply deployment configuration to target platform."""
    import asyncio
    from nodetool.deploy.manager import DeploymentManager
    from nodetool.security.master_key import MasterKeyManager

    try:
        manager = DeploymentManager()

        deployment = manager.get_deployment(name)

        # Get master key from keychain (same as local system)
        if dry_run:
            master_key = "dry-run-placeholder"
        else:
            try:
                master_key = asyncio.run(MasterKeyManager.get_master_key())
            except Exception as e:
                console.print(
                    f"[red]Failed to retrieve master key from keychain: {e}[/]"
                )
                sys.exit(1)

        _populate_master_key_env(deployment, master_key)

        if dry_run:
            console.print("[yellow]Dry run mode - no changes will be made[/]")
            console.print()

        console.print(f"[bold cyan]Applying deployment: {name}[/]")

        results = manager.apply(name, dry_run=dry_run)

        console.print()
        console.print("[bold]Deployment Steps:[/]")
        for step in results.get("steps", []):
            console.print(f"  {step}")

        if results.get("errors"):
            console.print()
            console.print("[bold red]Errors:[/]")
            for error in results["errors"]:
                console.print(f"  ❌ {error}")
            sys.exit(1)

        if results["status"] == "success":
            console.print()
            console.print("[bold green]✅ Deployment successful![/]")

            if not dry_run:
                _sync_secrets_to_deployment(name, deployment)

    except KeyError:
        console.print(f"[red]Deployment '{name}' not found[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@deploy.command("status")
@click.argument("name")
def deploy_status(name: str):
    """Get current status of a deployment."""
    from nodetool.deploy.manager import DeploymentManager

    try:
        manager = DeploymentManager()
        status = manager.status(name)

        console.print(f"[bold cyan]Deployment Status: {status['deployment_name']}[/]")
        console.print(f"[cyan]Host: {status['host']}[/]")
        console.print(f"[cyan]Status: {status.get('status', 'unknown')}[/]")
        console.print()

        if status.get("last_deployed"):
            console.print(f"Last deployed: {status['last_deployed']}")
            console.print()

        if status.get("containers"):
            console.print("[bold]Containers:[/]")
            for container in status["containers"]:
                console.print(f"  • {container['name']}: {container['status']}")
                if "url" in container:
                    console.print(f"    URL: {container['url']}")
            console.print()

        if status.get("live_status"):
            console.print("[bold]Live Status:[/]")
            console.print(status["live_status"])

        if status.get("live_status_error"):
            console.print(
                f"[yellow]Could not get live status: {status['live_status_error']}[/]"
            )

    except KeyError:
        console.print(f"[red]Deployment '{name}' not found[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@deploy.command("logs")
@click.argument("name")
@click.option("--service", help="Specific service/container name")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option(
    "--tail", default=100, type=int, help="Number of lines from end (default: 100)"
)
def deploy_logs(name: str, service: Optional[str], follow: bool, tail: int):
    """View logs from deployed containers."""
    from nodetool.deploy.manager import DeploymentManager

    try:
        manager = DeploymentManager()

        console.print(f"[cyan]Fetching logs for: {name}[/]")
        if service:
            console.print(f"[cyan]Service: {service}[/]")
        console.print()

        logs = manager.logs(name, service=service, follow=follow, tail=tail)

        # Print logs directly
        print(logs)

    except KeyError:
        console.print(f"[red]Deployment '{name}' not found[/]")
        sys.exit(1)
    except NotImplementedError as e:
        console.print(f"[yellow]{e}[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@deploy.command("destroy")
@click.argument("name")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def deploy_destroy(name: str, force: bool):
    """Destroy deployment (stop and remove all resources)."""
    from nodetool.deploy.manager import DeploymentManager

    try:
        manager = DeploymentManager()

        if not force:
            if not click.confirm(
                f"Are you sure you want to destroy deployment '{name}'?"
            ):
                console.print("[yellow]Operation cancelled[/]")
                return

        console.print(f"[bold yellow]Destroying deployment: {name}[/]")

        results = manager.destroy(name, force=force)

        console.print()
        console.print("[bold]Destruction Steps:[/]")
        for step in results.get("steps", []):
            console.print(f"  {step}")

        if results.get("errors"):
            console.print()
            console.print("[bold red]Errors:[/]")
            for error in results["errors"]:
                console.print(f"  ❌ {error}")
            sys.exit(1)

        if results["status"] == "success":
            console.print()
            console.print("[bold green]✅ Deployment destroyed[/]")

    except KeyError:
        console.print(f"[red]Deployment '{name}' not found[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@deploy.group("workflows")
def deploy_workflows():
    """Manage workflows on deployed instances."""
    pass


@deploy_workflows.command("sync")
@click.argument("deployment_name")
@click.argument("workflow_id")
def deploy_workflows_sync(deployment_name: str, workflow_id: str):
    """Sync a local workflow to a deployed instance.

    Automatically downloads referenced models (HuggingFace, Ollama) and syncs assets."""
    import asyncio
    from io import BytesIO
    from nodetool.deploy.manager import DeploymentManager
    from nodetool.deploy.admin_client import AdminHTTPClient
    from nodetool.models.workflow import Workflow
    from nodetool.models.asset import Asset as AssetModel
    from nodetool.api.workflow import from_model
    from nodetool.config.environment import Environment
    from nodetool.deploy.sync import extract_models

    async def extract_and_download_models(
        workflow_data: dict, client: AdminHTTPClient
    ) -> int:
        """Extract model references from workflow and download them on remote."""
        models = extract_models(workflow_data)

        if not models:
            return 0

        console.print(f"[cyan]Found {len(models)} model(s) to download[/]")

        downloaded_count = 0
        for model in models:
            try:
                model_type = model.get("type", "")

                # Handle HuggingFace models
                if model_type.startswith("hf."):
                    repo_id = model.get("repo_id")
                    console.print(f"  [cyan]Downloading HF model: {repo_id}[/]")

                    # Start download (streaming progress)
                    last_status = None
                    async for progress in client.download_huggingface_model(
                        repo_id=repo_id,
                        file_path=model.get("path"),
                        ignore_patterns=model.get("ignore_patterns"),
                        allow_patterns=model.get("allow_patterns"),
                    ):
                        last_status = progress.get("status")
                        if last_status == "downloading":
                            file_name = progress.get("file", "")
                            percent = progress.get("percent", 0)
                            console.print(
                                f"    [yellow]{file_name}: {percent:.1f}%[/]",
                                end="\r",
                            )
                        elif last_status == "complete":
                            console.print(f"    [green]✓ Downloaded {repo_id}[/]")

                    # Stream completed - mark as downloaded
                    if last_status != "complete":
                        console.print(f"    [green]✓ Downloaded {repo_id}[/]")
                    downloaded_count += 1

                # Handle Ollama models
                elif (
                    model_type == "language_model" and model.get("provider") == "ollama"
                ):
                    model_id = model.get("id")
                    console.print(f"  [cyan]Downloading Ollama model: {model_id}[/]")

                    last_status = None
                    async for progress in client.download_ollama_model(
                        model_name=model_id
                    ):
                        last_status = progress.get("status")
                        if last_status and last_status != "success":
                            console.print(f"    [yellow]{last_status}[/]", end="\r")
                        elif last_status == "success":
                            console.print(f"    [green]✓ Downloaded {model_id}[/]")

                    # Stream completed - mark as downloaded
                    if last_status != "success":
                        console.print(f"    [green]✓ Downloaded {model_id}[/]")
                    downloaded_count += 1

            except Exception as e:
                console.print(
                    f"    [red]✗ Failed to download model: {e}[/]"
                )

        return downloaded_count

    async def extract_and_sync_assets(
        workflow_data: dict, client: AdminHTTPClient
    ) -> int:
        """Extract asset references from workflow and sync them to remote."""
        asset_ids = set()

        # Extract asset IDs from constant nodes
        for node in workflow_data.get("graph", {}).get("nodes", []):
            node_type = node.get("type", "")
            if node_type.startswith("nodetool.constant."):
                value = node.get("data", {}).get("value", {})
                if isinstance(value, dict):
                    # Check for asset_id field
                    asset_id = value.get("asset_id")
                    if asset_id:
                        asset_ids.add(asset_id)

        if not asset_ids:
            return 0

        console.print(f"[cyan]Found {len(asset_ids)} asset(s) to sync[/]")

        # Get local storage
        storage = Environment.get_asset_storage()
        synced_count = 0

        for asset_id in asset_ids:
            try:
                # Get local asset metadata
                asset = await AssetModel.get(asset_id)
                if not asset:
                    console.print(f"  [yellow]⚠️  Asset {asset_id} not found locally, skipping[/]")
                    continue

                console.print(f"  [cyan]Syncing asset: {asset.name}[/]")

                # Check if asset already exists on remote
                try:
                    remote_asset = await client.get_asset(asset_id)
                    console.print(f"    [yellow]Asset already exists on remote, skipping[/]")
                    synced_count += 1
                    continue
                except Exception:
                    # Asset doesn't exist, continue with sync
                    pass

                # Create asset metadata on remote (preserve asset ID)
                await client.create_asset(
                    id=asset.id,
                    user_id=asset.user_id,
                    name=asset.name,
                    content_type=asset.content_type,
                    parent_id=asset.parent_id,
                    workflow_id=asset.workflow_id,
                    metadata=asset.metadata,
                )

                # Upload asset file if it's not a folder
                if asset.content_type != "folder" and asset.file_name:
                    # Download from local storage
                    stream = BytesIO()
                    await storage.download(asset.file_name, stream)
                    file_data = stream.getvalue()

                    # Upload to remote storage
                    await client.upload_asset_file(asset.file_name, file_data)

                    # Upload thumbnail if exists
                    if asset.has_thumbnail and asset.thumb_file_name:
                        thumb_stream = BytesIO()
                        await storage.download(asset.thumb_file_name, thumb_stream)
                        thumb_data = thumb_stream.getvalue()
                        await client.upload_asset_file(asset.thumb_file_name, thumb_data)

                console.print(f"    [green]✓ Synced {asset.name}[/]")
                synced_count += 1

            except Exception as e:
                console.print(f"    [red]✗ Failed to sync asset {asset_id}: {e}[/]")

        return synced_count

    async def run_sync():
        try:
            manager = DeploymentManager()
            deployment = manager.get_deployment(deployment_name)

            # Get server URL from deployment
            server_url = deployment.get_server_url()
            if not server_url:
                console.print(
                    f"[red]Cannot determine server URL for deployment '{deployment_name}'[/]"
                )
                console.print(
                    "[yellow]The deployment may not be active yet. Try deploying first with:[/]"
                )
                console.print(f"  nodetool deploy apply {deployment_name}")
                sys.exit(1)

            console.print(f"[bold cyan]🔄 Syncing workflow to {deployment_name}...[/]")
            console.print(f"[cyan]Server: {server_url}[/]")
            console.print(f"[cyan]Workflow ID: {workflow_id}[/]")
            console.print()

            # Get local workflow
            workflow = await Workflow.get(workflow_id)
            if workflow is None:
                console.print(f"[red]❌ Workflow not found locally: {workflow_id}[/]")
                sys.exit(1)

            # Get auth token from deployment (for self-hosted deployments)
            from nodetool.config.deployment import SelfHostedDeployment

            auth_token = None
            if isinstance(deployment, SelfHostedDeployment):
                auth_token = deployment.worker_auth_token

            # Create client
            client = AdminHTTPClient(server_url, auth_token=auth_token)

            # Sync assets first
            workflow_data = from_model(workflow).model_dump()
            synced_assets = await extract_and_sync_assets(workflow_data, client)
            if synced_assets > 0:
                console.print(f"[green]✅ Synced {synced_assets} asset(s)[/]")
                console.print()

            # Download models required by the workflow
            synced_models = await extract_and_download_models(workflow_data, client)
            if synced_models > 0:
                console.print(f"[green]✅ Downloaded {synced_models} model(s)[/]")
                console.print()

            # Sync workflow
            result = await client.update_workflow(workflow_id, workflow_data)

            if result.get("status") == "ok":
                console.print("[green]✅ Workflow synced successfully[/]")
            else:
                console.print(f"[yellow]⚠️ Remote response: {result}[/]")

            # Close database connections
            from nodetool.models.base_model import close_all_database_adapters
            try:
                await close_all_database_adapters()
            except Exception:
                pass  # Ignore errors during cleanup

            # Give asyncio a chance to clean up any remaining tasks
            await asyncio.sleep(0.1)

            # Cancel any remaining tasks to allow clean shutdown
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            return 0

        except KeyError:
            console.print(f"[red]Deployment '{deployment_name}' not found[/]")
            return 1
        except Exception as e:
            console.print(f"[red]❌ Failed to sync workflow: {e}[/]")
            import traceback

            traceback.print_exc()
            return 1

    exit_code = asyncio.run(run_sync())
    sys.exit(exit_code)


@deploy_workflows.command("list")
@click.argument("deployment_name")
def deploy_workflows_list(deployment_name: str):
    """List workflows on a deployed instance."""
    import asyncio
    from nodetool.deploy.manager import DeploymentManager
    from nodetool.deploy.admin_client import AdminHTTPClient

    async def run_list():
        try:
            manager = DeploymentManager()
            deployment = manager.get_deployment(deployment_name)

            # Get server URL from deployment
            server_url = deployment.get_server_url()
            if not server_url:
                console.print(
                    f"[red]Cannot determine server URL for deployment '{deployment_name}'[/]"
                )
                console.print(
                    "[yellow]The deployment may not be active yet. Try deploying first with:[/]"
                )
                console.print(f"  nodetool deploy apply {deployment_name}")
                sys.exit(1)

            console.print(
                f"[bold cyan]📋 Fetching workflows from {deployment_name}...[/]"
            )
            console.print(f"[cyan]Server: {server_url}[/]")
            console.print()

            # Get auth token from deployment (for self-hosted deployments)
            from nodetool.config.deployment import SelfHostedDeployment

            auth_token = None
            if isinstance(deployment, SelfHostedDeployment):
                auth_token = deployment.worker_auth_token

            # Get workflows from remote
            client = AdminHTTPClient(server_url, auth_token=auth_token)
            result = await client.list_workflows()

            workflows = result.get("workflows", [])

            if not workflows:
                console.print("[yellow]No workflows found on remote instance[/]")
                return

            table = Table(title=f"Workflows on '{deployment_name}'")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Description", style="yellow")

            for workflow in workflows:
                table.add_row(
                    workflow.get("id", ""),
                    workflow.get("name", ""),
                    workflow.get("description", "")[:50]
                    if workflow.get("description")
                    else "",
                )

            console.print(table)
            console.print()
            console.print(f"[cyan]Total: {len(workflows)} workflow(s)[/]")

        except KeyError:
            console.print(f"[red]Deployment '{deployment_name}' not found[/]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Failed to list workflows: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_list())


@deploy_workflows.command("delete")
@click.argument("deployment_name")
@click.argument("workflow_id")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def deploy_workflows_delete(deployment_name: str, workflow_id: str, force: bool):
    """Delete a workflow from a deployed instance."""
    import asyncio
    from nodetool.deploy.manager import DeploymentManager
    from nodetool.deploy.admin_client import AdminHTTPClient

    async def run_delete():
        try:
            manager = DeploymentManager()
            deployment = manager.get_deployment(deployment_name)

            # Get server URL from deployment
            server_url = deployment.get_server_url()
            if not server_url:
                console.print(
                    f"[red]Cannot determine server URL for deployment '{deployment_name}'[/]"
                )
                console.print(
                    "[yellow]The deployment may not be active yet. Try deploying first with:[/]"
                )
                console.print(f"  nodetool deploy apply {deployment_name}")
                sys.exit(1)

            if not force:
                if not click.confirm(
                    f"Are you sure you want to delete workflow '{workflow_id}' from '{deployment_name}'?"
                ):
                    console.print("[yellow]Operation cancelled[/]")
                    return

            console.print(
                f"[bold yellow]🗑️ Deleting workflow from {deployment_name}...[/]"
            )
            console.print(f"[cyan]Server: {server_url}[/]")
            console.print(f"[cyan]Workflow ID: {workflow_id}[/]")
            console.print()

            # Get auth token from deployment (for self-hosted deployments)
            from nodetool.config.deployment import SelfHostedDeployment

            auth_token = None
            if isinstance(deployment, SelfHostedDeployment):
                auth_token = deployment.worker_auth_token

            # Delete from remote
            client = AdminHTTPClient(server_url, auth_token=auth_token)
            result = await client.delete_workflow(workflow_id)

            if result.get("status") == "ok":
                console.print("[green]✅ Workflow deleted successfully[/]")
            else:
                console.print(f"[yellow]⚠️ Remote response: {result}[/]")

        except KeyError:
            console.print(f"[red]Deployment '{deployment_name}' not found[/]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Failed to delete workflow: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_delete())


@deploy_workflows.command("run")
@click.argument("deployment_name")
@click.argument("workflow_id")
@click.option(
    "--params",
    "-p",
    multiple=True,
    help='Workflow parameters in key=value format (e.g., -p prompt="Hello")',
)
def deploy_workflows_run(deployment_name: str, workflow_id: str, params: tuple):
    """Run a workflow on a deployed instance."""
    import asyncio
    from nodetool.deploy.manager import DeploymentManager
    from nodetool.deploy.admin_client import AdminHTTPClient

    async def run_workflow():
        try:
            manager = DeploymentManager()
            deployment = manager.get_deployment(deployment_name)

            # Get server URL from deployment
            server_url = deployment.get_server_url()
            if not server_url:
                console.print(
                    f"[red]Cannot determine server URL for deployment '{deployment_name}'[/]"
                )
                console.print(
                    "[yellow]The deployment may not be active yet. Try deploying first with:[/]"
                )
                console.print(f"  nodetool deploy apply {deployment_name}")
                sys.exit(1)

            # Parse parameters
            workflow_params = {}
            for param in params:
                if "=" not in param:
                    console.print(f"[red]Invalid parameter format: {param}[/]")
                    console.print(
                        "[yellow]Use key=value format, e.g., -p prompt='Hello'[/]"
                    )
                    sys.exit(1)
                key, value = param.split("=", 1)
                workflow_params[key] = value

            console.print(
                f"[bold cyan]▶️  Running workflow on {deployment_name}...[/]"
            )
            console.print(f"[cyan]Server: {server_url}[/]")
            console.print(f"[cyan]Workflow ID: {workflow_id}[/]")
            if workflow_params:
                console.print(f"[cyan]Parameters: {workflow_params}[/]")
            console.print()

            # Get auth token from deployment (for self-hosted deployments)
            from nodetool.config.deployment import SelfHostedDeployment

            auth_token = None
            if isinstance(deployment, SelfHostedDeployment):
                auth_token = deployment.worker_auth_token

            # Run workflow on remote
            client = AdminHTTPClient(server_url, auth_token=auth_token)
            result = await client.run_workflow(workflow_id, workflow_params)

            console.print("[green]✅ Workflow executed successfully[/]")
            console.print("\n[bold]Results:[/]")

            # Display results
            import json
            console.print(json.dumps(result.get("results", {}), indent=2))

        except KeyError:
            console.print(f"[red]Deployment '{deployment_name}' not found[/]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Failed to run workflow: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_workflow())


@deploy.group("database")
def deploy_database():
    """Manage database on deployed instances."""
    pass


@deploy_database.command("get")
@click.argument("deployment_name")
@click.argument("table")
@click.argument("key")
def deploy_database_get(deployment_name: str, table: str, key: str):
    """Get an item from database table by key."""
    import asyncio
    from nodetool.deploy.manager import DeploymentManager
    from nodetool.deploy.admin_client import AdminHTTPClient

    async def run_get():
        try:
            manager = DeploymentManager()
            deployment = manager.get_deployment(deployment_name)

            # Get server URL from deployment
            server_url = deployment.get_server_url()
            if not server_url:
                console.print(
                    f"[red]Cannot determine server URL for deployment '{deployment_name}'[/]"
                )
                sys.exit(1)

            # Get auth token from deployment (for self-hosted deployments)
            from nodetool.config.deployment import SelfHostedDeployment

            auth_token = None
            if isinstance(deployment, SelfHostedDeployment):
                auth_token = deployment.worker_auth_token

            # Get item from database
            client = AdminHTTPClient(server_url, auth_token=auth_token)
            item = await client.db_get(table, key)

            console.print(f"[bold cyan]Item from {table}/{key}:[/]")
            import json
            console.print(json.dumps(item, indent=2))

        except KeyError:
            console.print(f"[red]Deployment '{deployment_name}' not found[/]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Failed to get item: {e}[/]")
            sys.exit(1)

    asyncio.run(run_get())


@deploy_database.command("save")
@click.argument("deployment_name")
@click.argument("table")
@click.argument("json_data")
def deploy_database_save(deployment_name: str, table: str, json_data: str):
    """Save an item to database table."""
    import asyncio
    import json
    from nodetool.deploy.manager import DeploymentManager
    from nodetool.deploy.admin_client import AdminHTTPClient

    async def run_save():
        try:
            manager = DeploymentManager()
            deployment = manager.get_deployment(deployment_name)

            # Get server URL from deployment
            server_url = deployment.get_server_url()
            if not server_url:
                console.print(
                    f"[red]Cannot determine server URL for deployment '{deployment_name}'[/]"
                )
                sys.exit(1)

            # Parse JSON data
            try:
                item = json.loads(json_data)
            except json.JSONDecodeError as e:
                console.print(f"[red]Invalid JSON: {e}[/]")
                sys.exit(1)

            # Get auth token from deployment (for self-hosted deployments)
            from nodetool.config.deployment import SelfHostedDeployment

            auth_token = None
            if isinstance(deployment, SelfHostedDeployment):
                auth_token = deployment.worker_auth_token

            # Save item to database
            client = AdminHTTPClient(server_url, auth_token=auth_token)
            result = await client.db_save(table, item)

            console.print(f"[green]✅ Item saved to {table}[/]")
            console.print(json.dumps(result, indent=2))

        except KeyError:
            console.print(f"[red]Deployment '{deployment_name}' not found[/]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Failed to save item: {e}[/]")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_save())


@deploy_database.command("delete")
@click.argument("deployment_name")
@click.argument("table")
@click.argument("key")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def deploy_database_delete(deployment_name: str, table: str, key: str, force: bool):
    """Delete an item from database table by key."""
    import asyncio
    from nodetool.deploy.manager import DeploymentManager
    from nodetool.deploy.admin_client import AdminHTTPClient

    async def run_delete():
        try:
            manager = DeploymentManager()
            deployment = manager.get_deployment(deployment_name)

            # Get server URL from deployment
            server_url = deployment.get_server_url()
            if not server_url:
                console.print(
                    f"[red]Cannot determine server URL for deployment '{deployment_name}'[/]"
                )
                sys.exit(1)

            if not force:
                if not click.confirm(
                    f"Are you sure you want to delete {table}/{key} from '{deployment_name}'?"
                ):
                    console.print("[yellow]Operation cancelled[/]")
                    return

            # Get auth token from deployment (for self-hosted deployments)
            from nodetool.config.deployment import SelfHostedDeployment

            auth_token = None
            if isinstance(deployment, SelfHostedDeployment):
                auth_token = deployment.worker_auth_token

            # Delete item from database
            client = AdminHTTPClient(server_url, auth_token=auth_token)
            await client.db_delete(table, key)

            console.print(f"[green]✅ Item {table}/{key} deleted successfully[/]")

        except KeyError:
            console.print(f"[red]Deployment '{deployment_name}' not found[/]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Failed to delete item: {e}[/]")
            sys.exit(1)

    asyncio.run(run_delete())


@deploy.group("collections")
def deploy_collections():
    """Manage vector database collections on deployed instances."""
    pass


@deploy_collections.command("sync")
@click.argument("deployment_name")
@click.argument("collection_name")
def deploy_collections_sync(deployment_name: str, collection_name: str):
    """Sync a local ChromaDB collection to a deployed instance.

    Creates collection on remote if needed and syncs all documents, embeddings, and metadata."""
    import asyncio

    from nodetool.deploy.manager import DeploymentManager
    from nodetool.deploy.admin_client import AdminHTTPClient
    from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
        get_async_chroma_client,
    )

    async def run_sync():
        try:
            manager = DeploymentManager()
            deployment = manager.get_deployment(deployment_name)

            if deployment is None:
                console.print(f"[red]Deployment '{deployment_name}' not found[/]")
                sys.exit(1)

            # Get server URL
            server_url = deployment.get_server_url()

            console.print(f"🔄 Syncing collection to {deployment_name}...")
            console.print(f"Server: {server_url}")
            console.print(f"Collection: {collection_name}")
            console.print()

            # Get auth token from deployment (for self-hosted deployments)
            from nodetool.config.deployment import SelfHostedDeployment

            auth_token = None
            if isinstance(deployment, SelfHostedDeployment):
                auth_token = deployment.worker_auth_token

            client = AdminHTTPClient(server_url, auth_token=auth_token)

            # Get local collection
            chroma_client = await get_async_chroma_client()
            collection = await chroma_client.get_collection(name=collection_name)

            # Get collection metadata to extract embedding model
            collection_metadata = collection.metadata
            embedding_model = collection_metadata.get(
                "embedding_model", "all-minilm:latest"
            )

            # Create collection on remote if it doesn't exist
            try:
                console.print(
                    f"Creating collection '{collection_name}' with embedding model '{embedding_model}'..."
                )
                await client.create_collection(
                    name=collection_name, embedding_model=embedding_model
                )
                console.print("[green]✓ Collection created[/]")
            except Exception as e:
                # Collection might already exist, that's ok
                if "already exists" in str(e).lower():
                    console.print("[yellow]⚠️ Collection already exists[/]")
                else:
                    console.print(f"[yellow]⚠️ {e}[/]")

            existing_count = await collection.count()
            console.print(f"Found {existing_count} items in local collection")
            console.print()

            # Sync in batches
            batch_size = 10
            synced_count = 0

            for i in range(0, existing_count, batch_size):
                batch = await collection.get(
                    include=["metadatas", "documents", "embeddings"],
                    limit=batch_size,
                    offset=i,
                )

                if batch["ids"]:
                    # Convert numpy arrays to lists for JSON serialization
                    embeddings = (
                        [emb.tolist() for emb in batch["embeddings"]]
                        if batch["embeddings"] is not None
                        else []
                    )

                    # Convert None metadatas to dicts with placeholder
                    # ChromaDB requires non-empty metadata dicts
                    metadatas = (
                        [
                            meta if meta is not None and meta else {"_empty": "true"}
                            for meta in batch["metadatas"]
                        ]
                        if batch["metadatas"] is not None
                        else []
                    )

                    await client.add_to_collection(
                        collection_name=collection_name,
                        documents=batch["documents"],
                        ids=batch["ids"],
                        metadatas=metadatas,
                        embeddings=embeddings,
                    )
                    synced_count += len(batch["ids"])
                    console.print(
                        f"  Synced batch {i // batch_size + 1} ({synced_count}/{existing_count} items)"
                    )

            console.print(
                f"[green]✅ Synced {synced_count} items to collection '{collection_name}'[/]"
            )

        except KeyError:
            console.print(f"[red]Deployment '{deployment_name}' not found[/]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Failed to sync collection: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_sync())


# Add deploy group to main CLI
cli.add_command(deploy)


@cli.group()
def sync():
    """Synchronize database entries with a remote NodeTool server.

    Push local workflows and data to remote deployments."""
    pass


@sync.command("workflow")
@click.option("--id", "workflow_id", required=True, help="Workflow ID to sync.")
@click.option(
    "--server-url",
    required=True,
    help="Remote server base URL (e.g., http://localhost:8000).",
)
def sync_workflow(workflow_id: str, server_url: str):
    """Push a local workflow to a remote NodeTool server."""
    import asyncio

    from nodetool.deploy.admin_client import AdminHTTPClient
    from nodetool.models.workflow import Workflow
    import dotenv

    dotenv.load_dotenv()

    async def run_sync():
        try:
            console.print("[bold cyan]🔄 Syncing workflow to remote...[/]")
            # Get local workflow as a dict directly from the adapter
            workflow = await Workflow.get(workflow_id)
            if workflow is None:
                console.print(f"[red]❌ Workflow not found: {workflow_id}[/]")
                raise SystemExit(1)
            # Use optional API key for auth if present
            api_key = os.getenv("RUNPOD_API_KEY")
            client = AdminHTTPClient(server_url, auth_token=api_key)
            res = await client.update_workflow(
                workflow_id, from_model(workflow).model_dump()
            )

            status = res.get("status", "ok")
            if status == "ok":
                console.print("[green]✅ Workflow synced successfully[/]")
            else:
                console.print(f"[yellow]⚠️ Remote response: {res}[/]")
        except Exception as e:
            console.print(f"[red]❌ Failed to sync workflow: {e}[/]")
            raise SystemExit(1)

    asyncio.run(run_sync())


# Add sync group to the main CLI
cli.add_command(sync)


# ---- Proxy Commands ----


@cli.command("proxy")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Path to proxy configuration YAML file",
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind to",
)
@click.option(
    "--port",
    default=443,
    type=int,
    help="Port to bind to",
)
@click.option(
    "--no-tls",
    is_flag=True,
    help="Disable TLS (serve HTTP only)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (DEBUG level)",
)
def proxy(
    config: str,
    host: str,
    port: int,
    no_tls: bool,
    verbose: bool,
):
    """Start the async Docker reverse proxy server.

    The proxy routes HTTP requests to Docker containers, starting them on-demand
    and stopping them after an idle timeout. It supports Let's Encrypt ACME
    for TLS certificate management.

    Examples:
      # Start proxy with HTTPS on port 443
      nodetool proxy --config /etc/proxy/config.yaml

      # Start proxy on HTTP port 8080 (no TLS)
      nodetool proxy --config /etc/proxy/config.yaml --port 8080 --no-tls

      # Start with verbose logging
      nodetool proxy --config /etc/proxy/config.yaml --verbose
    """
    import asyncio

    from nodetool.proxy.config import load_config_with_env
    from nodetool.proxy.server import run_proxy_app

    try:
        # Configure logging if verbose
        if verbose:
            from nodetool.config.logging_config import configure_logging

            configure_logging(level="DEBUG")
            console.print("[cyan]🐛 Verbose logging enabled (DEBUG level)[/]")

        # Load configuration
        console.print(f"[cyan]Loading proxy configuration from {config}[/]")
        proxy_config = load_config_with_env(config)

        console.print(
            f"[green]✅ Configuration loaded[/]\n"
            f"Domain: {proxy_config.global_.domain}\n"
            f"Services: {len(proxy_config.services)}\n"
            f"Idle timeout: {proxy_config.global_.idle_timeout}s"
        )

        # Display services
        if verbose:
            table = Table(title="Configured Services")
            table.add_column("Name", style="cyan")
            table.add_column("Path", style="magenta")
            table.add_column("Image", style="green")
            table.add_column("Internal Port", style="yellow")
            table.add_column("Host Port", style="blue")

            for service in proxy_config.services:
                table.add_row(
                    service.name,
                    service.path,
                    service.image,
                    str(service.internal_port),
                    str(service.host_port or "auto"),
                )
            console.print(table)

        # Run proxy
        use_tls = not no_tls
        if use_tls:
            console.print(
                f"[cyan]Starting proxy with TLS on {host}:{port}[/]"
            )
        else:
            console.print(f"[cyan]Starting proxy on {host}:{port} (HTTP only)[/]")

        asyncio.run(run_proxy_app(proxy_config, host=host, port=port, use_tls=use_tls))

    except FileNotFoundError as e:
        console.print(f"[red]❌ {e}[/]")
        raise SystemExit(1)
    except ValueError as e:
        console.print(f"[red]❌ Configuration error: {e}[/]")
        raise SystemExit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Proxy interrupted by user[/]")
        raise SystemExit(0)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/]")
        raise SystemExit(1)

@cli.command("proxy-daemon")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Path to proxy configuration YAML file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (DEBUG level)",
)
def proxy_daemon(config: str, verbose: bool):
    """Run the FastAPI proxy with ACME HTTP + HTTPS listeners concurrently."""
    import asyncio

    from nodetool.proxy.config import load_config_with_env
    from nodetool.proxy.server import run_proxy_daemon

    if verbose:
        from nodetool.config.logging_config import configure_logging

        configure_logging(level="DEBUG")
        console.print("[cyan]🐛 Verbose logging enabled (DEBUG level)[/]")

    console.print(f"[cyan]Loading proxy configuration from {config}[/]")
    proxy_config = load_config_with_env(config)

    console.print(
        f"[green]✅ Configuration loaded[/]\n"
        f"Domain: {proxy_config.global_.domain}\n"
        f"Services: {len(proxy_config.services)}\n"
        f"HTTP port: {proxy_config.global_.listen_http}\n"
        f"HTTPS port: {proxy_config.global_.listen_https}\n"
        f"Connect mode: {proxy_config.global_.connect_mode}"
    )

    asyncio.run(run_proxy_daemon(proxy_config))


@cli.command("proxy-status")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Path to proxy configuration YAML file",
)
@click.option(
    "--server-url",
    default="http://localhost/status",
    help="URL of proxy status endpoint",
)
@click.option(
    "--bearer-token",
    help="Bearer token for authentication (defaults to config value)",
)
def proxy_status(config: str, server_url: str, bearer_token: str):
    """Check the status of proxy services.

    Connects to the running proxy server and displays the status of all
    managed containers (running, stopped, not created, etc.).

    Examples:
      # Check status using local config
      nodetool proxy-status --config /etc/proxy/config.yaml

      # Check remote proxy status
      nodetool proxy-status --config /etc/proxy/config.yaml \\
        --server-url https://proxy.example.com/status \\
        --bearer-token MY_TOKEN
    """
    import asyncio

    from nodetool.proxy.config import load_config_with_env

    async def check_status():
        import httpx

        try:
            # Load config for bearer token if not provided
            if not bearer_token:
                proxy_config = load_config_with_env(config)
                token = proxy_config.global_.bearer_token
            else:
                token = bearer_token

            # Fetch status from proxy
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    server_url,
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()

            # Parse and display status
            status_data = response.json()

            table = Table(title="Proxy Service Status")
            table.add_column("Service", style="cyan")
            table.add_column("Path", style="magenta")
            table.add_column("Status", style="yellow")
            table.add_column("Host Port", style="blue")
            table.add_column("Last Access", style="green")

            for service in status_data:
                status = service["status"]
                status_color = (
                    "green" if status == "running" else "yellow" if status == "stopped" else "red"
                )
                status_text = f"[{status_color}]{status}[/{status_color}]"

                last_access = service.get("last_access_epoch")
                if last_access:
                    import datetime

                    dt = datetime.datetime.fromtimestamp(last_access)
                    last_access_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    last_access_str = "Never"

                table.add_row(
                    service["name"],
                    service["path"],
                    status_text,
                    str(service.get("host_port") or "-"),
                    last_access_str,
                )

            console.print(table)

        except httpx.ConnectError:
            console.print(
                f"[red]❌ Failed to connect to proxy at {server_url}[/]"
            )
            raise SystemExit(1)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                console.print("[red]❌ Authentication failed (invalid bearer token)[/]")
            else:
                console.print(f"[red]❌ Proxy error: {e.response.status_code}[/]")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/]")
            raise SystemExit(1)

    asyncio.run(check_status())


@cli.command("proxy-validate-config")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Path to proxy configuration YAML file",
)
def proxy_validate_config(config: str):
    """Validate proxy configuration file.

    Loads and validates the proxy YAML configuration, checking for
    errors in service definitions and global settings.

    Examples:
      # Validate configuration
      nodetool proxy-validate-config --config /etc/proxy/config.yaml
    """
    from nodetool.proxy.config import load_config_with_env

    try:
        console.print(f"[cyan]Validating configuration: {config}[/]")
        proxy_config = load_config_with_env(config)

        console.print(
            "[green]✅ Configuration is valid[/]\n"
            f"Domain: {proxy_config.global_.domain}\n"
            f"Services: {len(proxy_config.services)}\n"
            f"Idle timeout: {proxy_config.global_.idle_timeout}s"
        )

        # Display services
        table = Table(title="Services")
        table.add_column("Name", style="cyan")
        table.add_column("Path", style="magenta")
        table.add_column("Image", style="green")
        table.add_column("Internal Port", style="yellow")
        table.add_column("Host Port", style="blue")

        for service in proxy_config.services:
            table.add_row(
                service.name,
                service.path,
                service.image,
                str(service.internal_port),
                str(service.host_port or "auto"),
            )
        console.print(table)

    except FileNotFoundError as e:
        console.print(f"[red]❌ {e}[/]")
        raise SystemExit(1)
    except ValueError as e:
        console.print(f"[red]❌ Configuration error: {e}[/]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/]")
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
