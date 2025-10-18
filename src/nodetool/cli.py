import os
import sys
import shutil
import atexit
import warnings
import click
from typing import Optional

from nodetool.api.workflow import from_model
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.config.settings import load_settings
from nodetool.deploy.docker import (
    generate_image_tag,
)
from nodetool.deploy.runpod_api import GPUType
from nodetool.dsl.codegen import create_dsl_modules
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
    """Start a nodetool MCP server."""
    from nodetool.api.mcp_server import mcp

    mcp.run()


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Host address to serve on.")
@click.option("--port", default=8000, help="Port to serve on.", type=int)
@click.option(
    "--static-folder",
    default=None,
    help="Path to the static folder to serve.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
)
@click.option("--apps-folder", default=None, help="Path to the apps folder.")
@click.option("--force-fp16", is_flag=True, help="Force FP16.")
@click.option("--reload", is_flag=True, help="Reload the server on changes.")
@click.option("--production", is_flag=True, help="Run in production mode.")
@click.option(
    "--remote-auth",
    is_flag=True,
    help="Use single local user with id 1 for authentication. Will be ingnored on production.",
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
    remote_auth: bool = False,
    apps_folder: str | None = None,
    production: bool = False,
    verbose: bool = False,
):
    """Serve the Nodetool API server."""
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

    Environment.set_remote_auth(remote_auth)

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

    from nodetool.config.logging_config import configure_logging

    configure_logging(level="DEBUG")

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
@click.option("--host", default="0.0.0.0", help="Host address to serve on.")
@click.option("--port", default=8000, help="Port to serve on.", type=int)
@click.option(
    "--remote-auth", is_flag=True, help="Use remote authentication (Supabase)."
)
@click.option(
    "--default-model",
    default="gpt-oss:20b",
    help="Default AI model to use when not specified by client.",
)
@click.option(
    "--provider",
    default="ollama",
    help="AI provider to use.",
)
@click.option(
    "--tools",
    default="",
    help="Comma-separated list of tools to use (e.g., 'google_search,browser').",
)
@click.option(
    "--workflow",
    "workflows",
    multiple=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="One or more workflow files to use.",
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
    """Start a NodeTool worker (deployable server).

    The worker provides OpenAI-compatible endpoints, workflow execution,
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


@cli.command("chat-server")
@click.option("--host", default="127.0.0.1", help="Host address to serve on.")
@click.option("--port", default=8080, help="Port to serve on.", type=int)
@click.option(
    "--remote-auth", is_flag=True, help="Use remote authentication (Supabase)."
)
@click.option(
    "--default-model",
    default="gpt-oss:20b",
    help="Default AI model to use when not specified by client.",
)
@click.option(
    "--provider",
    default="ollama",
    help="AI provider to use.",
)
@click.option(
    "--tools",
    default="",
    help="Comma-separated list of tools to use (e.g., 'google_search,google_news,google_images').",
)
@click.option(
    "--workflow",
    "workflows",
    multiple=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="One or more workflow files to use.",
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
    remote_auth: bool,
    provider: str,
    default_model: str,
    tools: str,
    workflows: list[str],
    verbose: bool = False,
):
    """Start a chat server SSE protocol.

    Examples:
      # Start WebSocket server on default port 8080
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

    run_chat_server(
        host, port, remote_auth, provider, default_model, tools_list, loaded_workflows
    )


@cli.command("chat-client")
@click.option(
    "--server-url",
    help="URL of the chat server to connect to. If not provided, uses OpenAI API directly.",
)
@click.option(
    "--runpod-endpoint",
    help="RunPod endpoint to use. Convenience option to not specify --server-url.",
)
@click.option("--auth-token", help="Authentication token")
@click.option("--message", help="Send a single message (non-interactive mode).")
@click.option(
    "--model",
    default="gpt-4o-mini",
    help="AI model to use (default: gpt-4o-mini for OpenAI, gpt-oss:20b for local server).",
)
@click.option(
    "--provider",
    help="AI provider to use when connecting to local server (e.g., 'openai', 'anthropic', 'ollama').",
)
def chat_client(
    server_url: Optional[str],
    auth_token: Optional[str],
    message: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    runpod_endpoint: Optional[str],
):
    """Connect to OpenAI API or a NodeTool chat server using OpenAI Chat Completions API.

    By default (no --server-url), connects directly to OpenAI API.
    With --server-url, connects to a NodeTool chat server with OpenAI compatibility.

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

    asyncio.run(run_chat_client(server_url, auth_token, message, model, provider))


@cli.command("codegen")
def codegen_cmd():
    """Generate DSL modules from node definitions."""
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
    """Commands for managing NodeTool settings and secrets."""
    pass


@settings.command("show")
@click.option("--secrets", is_flag=True, help="Show secrets instead of settings.")
@click.option("--mask", is_flag=True, help="Mask secret values with ****.")
def show_settings(secrets: bool, mask: bool):
    """Show current settings or secrets."""
    from nodetool.config.settings import load_settings

    # Load settings and secrets
    settings_obj, secrets_obj = load_settings()

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
        masked_value = "****" if setting.is_secret else data.get(setting.env_var, "")
        table.add_row(setting.env_var, masked_value, description)

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
    """Commands for managing Nodetool packages."""
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
    """Commands for admin operations (model downloads, cache management, health checks)."""
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
    """List available Google Cloud Run options."""
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

    return env


@cli.group()
def deploy():
    """Manage deployments via deployment.yaml configuration."""
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
    from nodetool.deploy.manager import DeploymentManager

    try:
        manager = DeploymentManager()

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
    """Sync a workflow to a deployed instance."""
    import asyncio
    from nodetool.deploy.manager import DeploymentManager
    from nodetool.deploy.admin_client import AdminHTTPClient
    from nodetool.models.workflow import Workflow
    from nodetool.api.workflow import from_model

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

            # Sync to remote
            client = AdminHTTPClient(server_url, auth_token=auth_token)
            result = await client.update_workflow(
                workflow_id, from_model(workflow).model_dump()
            )

            if result.get("status") == "ok":
                console.print("[green]✅ Workflow synced successfully[/]")
            else:
                console.print(f"[yellow]⚠️ Remote response: {result}[/]")

        except KeyError:
            console.print(f"[red]Deployment '{deployment_name}' not found[/]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Failed to sync workflow: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_sync())


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

            # Get workflows from remote
            client = AdminHTTPClient(server_url)
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

            # Delete from remote
            client = AdminHTTPClient(server_url)
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


# Add deploy group to main CLI
cli.add_command(deploy)


@cli.group()
def sync():
    """Commands to sync local database items with a remote NodeTool server."""
    pass


@sync.command("workflow")
@click.option("--id", "workflow_id", required=True, help="Workflow ID to sync")
@click.option(
    "--server-url",
    required=True,
    help="Remote NodeTool server base URL (e.g., http://localhost:8000)",
)
def sync_workflow(workflow_id: str, server_url: str):
    """Sync a local workflow to a remote database via admin routes."""
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


if __name__ == "__main__":
    cli()
