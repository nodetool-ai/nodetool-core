import os
import sys
import shutil
import click
from nodetool.common.configuration import get_settings_registry
from nodetool.common.environment import Environment
from nodetool.deploy.runpod_api import GPUType
from nodetool.dsl.codegen import create_dsl_modules

# silence warnings on the command line
import warnings

# Add Rich for better tables and terminal output
from nodetool.types.job import JobUpdate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import Optional

import dotenv

dotenv.load_dotenv()

# Create console instance
console = Console()

warnings.filterwarnings("ignore")
log = Environment.get_logger()

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


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Host address to serve on.")
@click.option("--port", default=8000, help="Port to serve on.", type=int)
@click.option("--worker-url", default=None, help="URL of the worker to connect to.")
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
def serve(
    host: str,
    port: int,
    static_folder: str | None = None,
    reload: bool = False,
    force_fp16: bool = False,
    remote_auth: bool = False,
    worker_url: str | None = None,
    apps_folder: str | None = None,
    production: bool = False,
):
    """Serve the Nodetool API server."""
    from nodetool.api.server import create_app, run_uvicorn_server

    try:
        import comfy.cli_args  # type: ignore

        comfy.cli_args.args.force_fp16 = force_fp16
    except ImportError:
        pass

    Environment.set_remote_auth(remote_auth)

    if worker_url:
        Environment.set_worker_url(worker_url)

    if Environment.is_production():
        Environment.set_nodetool_api_url("https://api.nodetool.ai")
    else:
        Environment.set_nodetool_api_url(f"http://127.0.0.1:{port}")

    if not reload:
        app = create_app(static_folder=static_folder, apps_folder=apps_folder)
    else:
        if static_folder:
            raise Exception("static folder and reload are exclusive options")
        if apps_folder:
            raise Exception("apps folder and reload are exclusive options")
        app = "nodetool.api.app:app"

    run_uvicorn_server(app=app, host=host, port=port, reload=reload)


@cli.command("worker")
@click.option("--host", default="127.0.0.1", help="Host address to serve on.")
@click.option("--port", default=8001, help="Port to serve on.", type=int)
@click.option("--force-fp16", is_flag=True, help="Force FP16.")
@click.option("--reload", is_flag=True, help="Reload the server on changes.")
def worker(
    host: str,
    port: int,
    reload: bool = False,
    force_fp16: bool = False,
):
    """Start a Nodetool worker instance."""
    from nodetool.api.server import run_uvicorn_server

    try:
        import comfy.cli_args  # type: ignore
        import comfy.model_management  # type: ignore
        import comfy.utils  # type: ignore
        from nodes import init_extra_nodes  # type: ignore

        comfy.cli_args.args.force_fp16 = force_fp16
    except ImportError:
        pass

    app = "nodetool.api.worker:app"
    init_extra_nodes()
    run_uvicorn_server(app=app, host=host, port=port, reload=reload)


@cli.command()
@click.argument("workflow", type=str)
def run(workflow: str):
    """Run a workflow by ID or from a local JSON definition file."""
    import asyncio
    import json
    import os
    import sys
    import traceback

    from nodetool.workflows.run_job_request import RunJobRequest
    from nodetool.workflows.run_workflow import run_workflow
    from nodetool.types.graph import Graph

    # Determine whether the provided argument is a file path or an ID
    is_file = os.path.isfile(workflow)

    try:
        if is_file:
            with open(workflow, "r", encoding="utf-8") as f:
                workflow_json = json.load(f)

            assert "graph" in workflow_json, "Graph not found in workflow JSON"
            graph = Graph(**workflow_json["graph"])

            request = RunJobRequest(
                user_id="1",
                auth_token="local_token",
                graph=graph,
            )
        else:
            # Treat the argument as a workflow ID
            request = RunJobRequest(
                workflow_id=workflow,
                user_id="1",
                auth_token="local_token",
            )
    except Exception as e:
        console.print(Panel.fit(f"Failed to prepare workflow: {e}", style="bold red"))
        traceback.print_exc()
        sys.exit(1)

    async def run_workflow_async():
        console.print(Panel.fit(f"Running workflow {workflow}...", style="blue"))
        try:
            async for message in run_workflow(request):
                print(message)
                # Pretty-print each message coming from the runner
                if isinstance(message, JobUpdate) and message.status == "error":
                    console.print(
                        Panel.fit(f"Error: {message.error}", style="bold red")
                    )
                    sys.exit(1)
                else:
                    msg_type = Text(message.type, style="bold cyan")
                    console.print(f"{msg_type}: {message.model_dump_json()}")
            console.print(Panel.fit("Workflow finished successfully", style="green"))
        except Exception as e:
            console.print(Panel.fit(f"Error running workflow: {e}", style="bold red"))
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_workflow_async())


@cli.command()
def chat():
    """Start a nodetool chat."""
    import asyncio
    from nodetool.chat.chat_cli import chat_cli

    asyncio.run(chat_cli())


@cli.command("chat-server")
@click.option("--host", default="127.0.0.1", help="Host address to serve on.")
@click.option("--port", default=8080, help="Port to serve on.", type=int)
@click.option(
    "--remote-auth", is_flag=True, help="Use remote authentication (Supabase)."
)
@click.option(
    "--default-model",
    default="gemma3n:latest",
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
def chat_server(
    host: str,
    port: int,
    remote_auth: bool,
    provider: str,
    default_model: str,
    tools: str,
    workflows: list[str],
):
    """Start a chat server SSE protocol.

    Examples:
      # Start WebSocket server on default port 8080
      nodetool chat-server

      # Start chat server on port 3000
      nodetool chat-server --port 3000

      # Start with tools
      nodetool chat-server --tools "google_search,google_news,google_images"
    """
    from nodetool.chat.server import run_chat_server
    import json
    from nodetool.types.workflow import Workflow

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
@click.option("--auth-token", help="Authentication token")
@click.option("--message", help="Send a single message (non-interactive mode).")
@click.option(
    "--model",
    default="gpt-4o-mini",
    help="AI model to use (default: gpt-4o-mini for OpenAI, gemma3n:latest for local server).",
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
    """
    import asyncio
    from nodetool.chat.chat_client import run_chat_client

    if not auth_token:
        auth_token = Environment.get("OPENAI_API_KEY")

    # If no server URL provided, use OpenAI API directly
    if not server_url:
        server_url = "https://api.openai.com"
        # Use provided model or default to gpt-4o-mini for OpenAI
        if not model:
            model = "gpt-4o-mini"
    else:
        # For local server, use provided model or default to gemma3n:latest
        if not model:
            model = "gemma3n:latest"

    asyncio.run(run_chat_client(server_url, auth_token, message, model, provider))


@cli.command("explorer")
@click.option("--dir", "-d", default=".", help="Directory to start exploring from.")
def explorer(dir: str):
    """Explore files in an interactive text UI."""
    from nodetool.file_explorer import FileExplorer
    import curses

    explorer = FileExplorer(dir)
    curses.wrapper(explorer.run)


# Add this after the other @cli commands but before the package group


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
    from nodetool.common.settings import load_settings

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

    settings_registry = get_settings_registry()
    for setting in settings_registry:
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
    from nodetool.common.settings import (
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
    settings_registry = get_settings_registry()

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
        with click.progressbar(
            length=100,
            label="Scanning for nodes",
            show_eta=False,
            show_percent=True,
        ) as bar:
            bar.update(10)
            # Scan for nodes and create package model
            package = scan_for_package_nodes(verbose=verbose)
            bar.update(80)

            # Save package metadata
            save_package_metadata(package, verbose=verbose)
            # Update pyproject.toml with asset files
            update_pyproject_include(package, verbose=verbose)
            bar.update(10)

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
    python_version = "^3.10"

    # Create pyproject.toml content
    pyproject_content = f"""[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "{name}"
version = "{version}"
description = "{description}"
readme = "README.md"
authors = ["{author}"]
packages = [{{ include = "nodetool", from = "src" }}]
package-mode = true
include = []

[tool.poetry.dependencies]
python = "{python_version}"
nodetool-core = {{ git = "https://github.com/nodetool-ai/nodetool-core.git", rev = "main" }}
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
            project_data = pyproject_data.get("tool", {}).get("poetry", {})

        if not project_data:
            click.echo("Error: No project metadata found in pyproject.toml", err=True)
            sys.exit(1)

        package_name = project_data.get("name")
        if not package_name:
            click.echo("Error: No package name found in pyproject.toml", err=True)
            sys.exit(1)

        repository = project_data.get("repository")
        if not repository:
            click.echo("Error: No repository found in pyproject.toml", err=True)
            sys.exit(1)

        repository.split("/")[-2]

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
                allow_patterns=list(allow_patterns) if allow_patterns else None
            ):
                _display_progress_update(progress_update)

        except Exception as e:
            console.print(f"[red]❌ Failed: {e}[/]")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_download())


def _display_progress_update(progress_update):
    """Shared function to display progress updates in consistent format."""
    status = progress_update.get("status", "unknown")
    message = progress_update.get("message", "")

    if status == "starting":
        console.print(f"[blue]🚀 {message}[/]")
    elif status == "progress":
        if "current_file" in progress_update:
            current_file = progress_update["current_file"]
            if "file_progress" in progress_update:
                file_num = progress_update["file_progress"]
                total_files = progress_update["total_files"]
                console.print(
                    f"[yellow]📁 [{file_num}/{total_files}] {current_file}[/]"
                )
            else:
                console.print(f"[yellow]📁 {current_file}[/]")
        else:
            console.print(f"[yellow]⚙️ {message}[/]")

        # Show progress info if available
        if "downloaded_size" in progress_update and "total_size" in progress_update:
            downloaded = progress_update["downloaded_size"]
            total = progress_update["total_size"]
            if total > 0:
                pct = (downloaded / total) * 100
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                console.print(
                    f"[cyan]📊 Progress: {downloaded_mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)[/]"
                )

    elif status == "completed":
        console.print(f"[green]✅ {message}[/]")
        if "downloaded_files" in progress_update:
            console.print(
                f"[green]📋 Downloaded {progress_update['downloaded_files']} files[/]"
            )
    elif status.startswith("pulling"):
        # Handle Ollama pulling status with digest info
        digest = progress_update.get("digest", "")
        total = progress_update.get("total")
        completed = progress_update.get("completed")
        
        # Extract the layer ID from status (e.g., "pulling aeda25e63ebd")
        layer_id = status.replace("pulling ", "") if " " in status else "unknown"
        
        if digest and "sha256:" in digest:
            # Show shortened digest for readability
            short_digest = digest.split(":")[-1][:12] if ":" in digest else digest[:12]
            console.print(f"[yellow]🐋 Pulling layer {layer_id} (sha256:{short_digest})[/]")
        else:
            console.print(f"[yellow]🐋 Pulling layer {layer_id}[/]")
        
        # Show size information if available
        if total:
            total_mb = total / (1024 * 1024)
            if completed:
                completed_mb = completed / (1024 * 1024)
                pct = (completed / total) * 100 if total > 0 else 0
                console.print(
                    f"[cyan]📊 Progress: {completed_mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)[/]"
                )
            else:
                console.print(f"[cyan]📦 Size: {total_mb:.1f} MB[/]")
    elif status == "error":
        error = progress_update.get("error", "Unknown error")
        console.print(f"[red]❌ Error: {error}[/]")
        sys.exit(1)
    elif status == "healthy":
        console.print("[green]✅ System is healthy[/]")

        # Display system information for health checks
        console.print(
            f"[cyan]🖥️ Platform: {progress_update.get('platform', 'Unknown')}[/]"
        )
        console.print(
            f"[cyan]🐍 Python: {progress_update.get('python_version', 'Unknown')}[/]"
        )
        console.print(
            f"[cyan]🏠 Hostname: {progress_update.get('hostname', 'Unknown')}[/]"
        )

        # Memory info
        memory = progress_update.get("memory", {})
        if isinstance(memory, dict):
            console.print(
                f"[cyan]💾 Memory: {memory.get('available_gb', 0):.1f}GB available / {memory.get('total_gb', 0):.1f}GB total ({memory.get('used_percent', 0)}% used)[/]"
            )

        # Disk info
        disk = progress_update.get("disk", {})
        if isinstance(disk, dict):
            console.print(
                f"[cyan]💿 Disk: {disk.get('free_gb', 0):.1f}GB free / {disk.get('total_gb', 0):.1f}GB total ({disk.get('used_percent', 0)}% used)[/]"
            )

        # GPU info
        gpus = progress_update.get("gpus", [])
        if isinstance(gpus, list) and gpus:
            console.print("[cyan]🎮 GPUs:[/]")
            for i, gpu in enumerate(gpus):
                name = gpu.get("name", "Unknown")
                used_mb = gpu.get("memory_used_mb", 0)
                total_mb = gpu.get("memory_total_mb", 0)
                used_pct = (used_mb / total_mb * 100) if total_mb > 0 else 0
                console.print(
                    f"[cyan]  GPU {i}: {name} - {used_mb}MB/{total_mb}MB ({used_pct:.1f}% used)[/]"
                )
        elif gpus == "unavailable":
            console.print("[yellow]🎮 GPUs: Not available[/]")



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
                _display_progress_update(progress_update)

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
            _display_progress_update(result)
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
def cache_size(
    cache_dir: str, server_url: str, api_key: str | None
):
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
        generate_image_tag,
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


@cli.command("deploy")
@click.option(
    "--workflow-id",
    "workflow_ids",
    multiple=True,
    help="Workflow ID to deploy (can specify multiple).",
)
@click.option(
    "--docker-username",
    help="Docker Hub username or organization (auto-detected from docker login if not provided)",
)
@click.option(
    "--docker-registry",
    default="docker.io",
    help="Docker registry URL (default: docker.io for Docker Hub)",
)
@click.option("--tag", help="Tag of the Docker image (default: auto-generated hash)")
@click.option(
    "--platform",
    default="linux/amd64",
    help="Docker build platform (default: linux/amd64 for RunPod compatibility)",
)
@click.option(
    "--template-name", help="Name of the RunPod template (defaults to image name)"
)
# Skip options
@click.option("--skip-build", is_flag=True, help="Skip Docker build")
@click.option("--skip-push", is_flag=True, help="Skip pushing to registry")
@click.option("--skip-template", is_flag=True, help="Skip creating RunPod template")
@click.option("--skip-endpoint", is_flag=True, help="Skip creating RunPod endpoint")
# Cache options
@click.option("--no-cache", is_flag=True, help="Disable Docker Hub cache optimization")
@click.option(
    "--no-auto-push", is_flag=True, help="Disable automatic push during optimized build"
)
@click.option(
    "--check-docker-config", is_flag=True, help="Check Docker configuration and exit"
)
# Endpoint compute configuration
@click.option(
    "--compute-type",
    type=click.Choice(["CPU", "GPU"]),
    default="GPU",
    help="Compute type for the endpoint",
)
@click.option(
    "--gpu-types",
    multiple=True,
    type=click.Choice(SUPPORTED_GPU_TYPES),
    help="GPU types to use (can specify multiple). Use actual GPU model names.",
)
@click.option("--gpu-count", type=int, help="Number of GPUs per worker")
@click.option(
    "--cpu-flavors",
    multiple=True,
    type=click.Choice(["cpu3c", "cpu3g", "cpu5c", "cpu5g"]),
    help="CPU flavors to use for CPU compute (can specify multiple)",
)
@click.option("--vcpu-count", type=int, help="Number of vCPUs for CPU compute")
@click.option(
    "--data-centers",
    multiple=True,
    help="Preferred data center locations (can specify multiple)",
)
# Endpoint scaling configuration
@click.option(
    "--workers-min", type=int, default=0, help="Minimum number of workers (default: 0)"
)
@click.option(
    "--workers-max", type=int, default=1, help="Maximum number of workers (default: 3)"
)
@click.option(
    "--idle-timeout",
    type=int,
    default=60,
    help="Seconds before scaling down idle workers (default: 5)",
)
# Endpoint advanced configuration
@click.option(
    "--execution-timeout", type=int, help="Maximum execution time in milliseconds"
)
@click.option(
    "--flashboot", is_flag=True, help="Enable flashboot for faster worker startup"
)
@click.option(
    "--network-volume-id",
    help="Network volume ID to attach to workers (models will be stored at /runpod-volume)",
)
@click.option(
    "--allowed-cuda-versions",
    multiple=True,
    help="Allowed CUDA versions (can specify multiple)",
)
@click.option(
    "--tools",
    default="",
    help="Comma-separated list of tools to use for chat handler (e.g., 'google_search,google_news,google_images').",
)
@click.option(
    "--name",
    help="Name for the endpoint (required for all deployments)",
)
# List options
@click.option(
    "--list-gpu-types", is_flag=True, help="List all available GPU types and exit"
)
@click.option(
    "--list-cpu-flavors", is_flag=True, help="List all available CPU flavors and exit"
)
@click.option(
    "--list-data-centers", is_flag=True, help="List all available data centers and exit"
)
@click.option(
    "--list-all-options", is_flag=True, help="List all available options and exit"
)
@click.option(
    "--local-docker",
    is_flag=True,
    help="Run local docker container instead of deploying to RunPod",
)
def deploy(
    workflow_ids: tuple[str, ...],
    docker_username: str | None,
    docker_registry: str,
    tag: str | None,
    platform: str,
    template_name: str | None,
    skip_build: bool,
    skip_push: bool,
    skip_template: bool,
    skip_endpoint: bool,
    no_cache: bool,
    no_auto_push: bool,
    check_docker_config: bool,
    compute_type: str,
    gpu_types: tuple,
    gpu_count: int | None,
    cpu_flavors: tuple,
    vcpu_count: int | None,
    data_centers: tuple,
    workers_min: int,
    workers_max: int,
    idle_timeout: int,
    execution_timeout: int | None,
    flashboot: bool,
    network_volume_id: str | None,
    allowed_cuda_versions: tuple,
    tools: str,
    name: str | None,
    list_gpu_types: bool,
    list_cpu_flavors: bool,
    list_data_centers: bool,
    list_all_options: bool,
    local_docker: bool,
):
    """Deploy workflow or chat handler to RunPod serverless infrastructure.

    Examples:
      # Basic workflow deployment
      nodetool deploy --workflow-id abc123 --name my-workflow

      # Deploy multiple workflows
      nodetool deploy --workflow-id abc123 --workflow-id def456 --workflow-id ghi789 --name multi-workflow

      # With specific GPU and regions
      nodetool deploy --workflow-id abc123 --name gpu-workflow --gpu-types "NVIDIA GeForce RTX 4090" --gpu-types "NVIDIA L40S" --data-centers US-CA-2 --data-centers US-GA-1

      # CPU-only endpoint
      nodetool deploy --workflow-id abc123 --name cpu-workflow --compute-type CPU --cpu-flavors cpu3c --cpu-flavors cpu5c

      # Check Docker configuration
      nodetool deploy --check-docker-config

      # List available options
      nodetool deploy --list-gpu-types
      nodetool deploy --list-all-options
    """
    import dotenv

    dotenv.load_dotenv()

    # Handle list options (these don't require workflow-id)
    _handle_list_options(
        list_gpu_types, list_cpu_flavors, list_data_centers, list_all_options
    )

    # Handle Docker config check (doesn't require workflow-id)
    _handle_docker_config_check(check_docker_config, docker_registry, docker_username)

    # Call the main deployment function
    from nodetool.deploy.deploy_to_runpod import deploy_to_runpod

    # Parse comma-separated tools string into list
    tools_list = (
        [tool.strip() for tool in tools.split(",") if tool.strip()] if tools else None
    )

    deploy_to_runpod(
        workflow_ids=list(workflow_ids) if workflow_ids else None,
        docker_username=docker_username,
        docker_registry=docker_registry,
        image_name=name,
        tag=tag,
        platform=platform,
        template_name=template_name,
        skip_build=skip_build,
        skip_push=skip_push,
        skip_template=skip_template,
        skip_endpoint=skip_endpoint,
        no_cache=no_cache,
        no_auto_push=no_auto_push,
        compute_type=compute_type,
        gpu_types=gpu_types,
        gpu_count=gpu_count,
        cpu_flavors=cpu_flavors,
        vcpu_count=vcpu_count,
        data_centers=data_centers,
        workers_min=workers_min,
        workers_max=workers_max,
        idle_timeout=idle_timeout,
        execution_timeout=execution_timeout,
        flashboot=flashboot,
        network_volume_id=network_volume_id,
        allowed_cuda_versions=allowed_cuda_versions,
        name=name,
        local_docker=local_docker,
        tools=tools_list,
    )


@cli.command("test-runpod")
@click.option("--endpoint-id", required=True, help="RunPod endpoint ID")
@click.option("--api-key", help="RunPod API key (can also use RUNPOD_API_KEY env var)")
@click.option(
    "--params", type=click.Path(exists=True), help="JSON file with workflow parameters"
)
@click.option("--params-json", help="Inline JSON string with workflow parameters")
@click.option(
    "--timeout", type=int, default=600, help="Timeout in seconds (default: 600)"
)
def test_runpod(
    endpoint_id: str,
    api_key: str | None,
    params: str | None,
    params_json: str | None,
    timeout: int,
):
    """Test deployed NodeTool workflow on RunPod serverless infrastructure.

    Examples:
      # Basic test with no parameters
      nodetool test-runpod --endpoint-id abc123def456

      # Test with JSON file parameters
      nodetool test-runpod --endpoint-id abc123def456 --params test_params.json

      # Test with inline JSON parameters
      nodetool test-runpod --endpoint-id abc123def456 --params-json '{"text": "Hello World"}'
    """
    import json
    import time
    import traceback

    try:
        import runpod
    except ImportError:
        console.print("[bold red]❌ Error: runpod library not found[/]")
        console.print("Install it with: pip install runpod")
        sys.exit(1)

    # Get API key from argument or environment
    api_key = api_key or os.getenv("RUNPOD_API_KEY")
    if not api_key:
        console.print("[bold red]❌ Error: RunPod API key is required[/]")
        console.print(
            "Provide it via --api-key argument or RUNPOD_API_KEY environment variable"
        )
        sys.exit(1)

    # Configure runpod library
    runpod.api_key = api_key
    endpoint = runpod.Endpoint(endpoint_id)

    # Get workflow parameters
    workflow_params = {}
    if params:
        try:
            with open(params, "r") as f:
                workflow_params = json.load(f)
        except Exception as e:
            console.print(
                f"[bold red]❌ Failed to load parameters from {params}: {e}[/]"
            )
            sys.exit(1)
    elif params_json:
        try:
            workflow_params = json.loads(params_json)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]❌ Invalid JSON in --params-json: {e}[/]")
            sys.exit(1)
    else:
        console.print(
            "[bold yellow]⚠️ No parameters provided, using empty parameters[/]"
        )

    console.print("[bold cyan]🧪 Testing RunPod workflow...[/]")
    console.print(f"Endpoint ID: {endpoint_id}")
    console.print(f"Parameters: {json.dumps(workflow_params, indent=2)}")
    console.print(f"Timeout: {timeout} seconds")

    try:
        console.print("[bold blue]🚀 Starting workflow execution...[/]")

        job = endpoint.run(workflow_params)

        console.print(f"Job status: {job.status()}")
        start_time = time.time()

        while job.status() in ("RUNNING", "IN_PROGRESS", "IN_QUEUE"):
            time.sleep(1)
            elapsed = int(time.time() - start_time)

            if elapsed >= timeout:
                console.print(f"[bold red]⏰ Job timed out after {timeout} seconds[/]")
                sys.exit(1)

            console.print(f"Job status: {job.status()} (elapsed: {elapsed}s)")

        result = job.output()

        console.print("[bold green]✅ Job completed successfully![/]")
        elapsed = int(time.time() - start_time)
        console.print(f"Execution completed in {elapsed} seconds")

        # Display results
        console.print("\n[bold cyan]📊 Job Results:[/]")
        console.print(json.dumps(result, indent=2))

        console.print("\n[bold green]✅ Test completed successfully![/]")

    except TimeoutError:
        console.print("\n[bold red]⏰ Job timed out[/]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]🛑 Test interrupted by user[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Test failed: {e}[/]")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
