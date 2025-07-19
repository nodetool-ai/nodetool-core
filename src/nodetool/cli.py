import os
import sys
import shutil
import click
from nodetool.common.configuration import get_settings_registry
from nodetool.common.environment import Environment
from nodetool.dsl.codegen import create_dsl_modules

# silence warnings on the command line
import warnings

# Add Rich for better tables and terminal output
from nodetool.types.job import JobUpdate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import List, Optional


# Create console instance
console = Console()

warnings.filterwarnings("ignore")
log = Environment.get_logger()


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
    from nodetool.workflows.read_graph import read_graph
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
                    console.print(Panel.fit(f"Error: {message.error}", style="bold red"))
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
    from nodetool.common.settings import SETTINGS_FILE, SECRETS_FILE
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


@cli.command("deploy")
@click.option("--workflow-id", help="Workflow ID to deploy (required unless using list options)")
@click.option("--docker-username", help="Docker Hub username or organization (auto-detected from docker login if not provided)")
@click.option("--docker-registry", default="docker.io", help="Docker registry URL (default: docker.io for Docker Hub)")
@click.option("--image-name", help="Base name of the Docker image (defaults to sanitized workflow name)")
@click.option("--tag", help="Tag of the Docker image (default: auto-generated hash)")
@click.option("--platform", default="linux/amd64", help="Docker build platform (default: linux/amd64 for RunPod compatibility)")
@click.option("--template-name", help="Name of the RunPod template (defaults to image name)")
# Skip options
@click.option("--skip-build", is_flag=True, help="Skip Docker build")
@click.option("--skip-push", is_flag=True, help="Skip pushing to registry")
@click.option("--skip-template", is_flag=True, help="Skip creating RunPod template")
@click.option("--skip-endpoint", is_flag=True, help="Skip creating RunPod endpoint")
@click.option("--check-docker-config", is_flag=True, help="Check Docker configuration and exit")
# Endpoint compute configuration
@click.option("--compute-type", type=click.Choice(["CPU", "GPU"]), default="GPU", help="Compute type for the endpoint")
@click.option("--gpu-types", multiple=True, type=click.Choice(["ADA_24", "ADA_32_PRO", "ADA_48_PRO", "ADA_80_PRO", "AMPERE_16", "AMPERE_24", "AMPERE_48", "AMPERE_80", "HOPPER_141"]), help="GPU types to use (can specify multiple)")
@click.option("--gpu-count", type=int, help="Number of GPUs per worker")
@click.option("--cpu-flavors", multiple=True, type=click.Choice(["cpu3c", "cpu3g", "cpu5c", "cpu5g"]), help="CPU flavors to use for CPU compute (can specify multiple)")
@click.option("--vcpu-count", type=int, help="Number of vCPUs for CPU compute")
@click.option("--data-centers", multiple=True, help="Preferred data center locations (can specify multiple)")
# Endpoint scaling configuration
@click.option("--workers-min", type=int, default=0, help="Minimum number of workers (default: 0)")
@click.option("--workers-max", type=int, default=3, help="Maximum number of workers (default: 3)")
@click.option("--idle-timeout", type=int, default=5, help="Seconds before scaling down idle workers (default: 5)")
@click.option("--scaler-type", type=click.Choice(["QUEUE_DELAY", "REQUEST_COUNT"]), default="QUEUE_DELAY", help="Type of auto-scaler (default: QUEUE_DELAY)")
@click.option("--scaler-value", type=int, default=4, help="Threshold value for the scaler (default: 4)")
# Endpoint advanced configuration
@click.option("--execution-timeout", type=int, help="Maximum execution time in milliseconds")
@click.option("--flashboot", is_flag=True, help="Enable flashboot for faster worker startup")
@click.option("--network-volume-id", help="Network volume ID to attach to workers")
@click.option("--allowed-cuda-versions", multiple=True, help="Allowed CUDA versions (can specify multiple)")
# List options
@click.option("--list-gpu-types", is_flag=True, help="List all available GPU types and exit")
@click.option("--list-cpu-flavors", is_flag=True, help="List all available CPU flavors and exit")
@click.option("--list-data-centers", is_flag=True, help="List all available data centers and exit")
@click.option("--list-all-options", is_flag=True, help="List all available options and exit")
def deploy(
    workflow_id: str | None,
    docker_username: str | None,
    docker_registry: str,
    image_name: str | None,
    tag: str | None,
    platform: str,
    template_name: str | None,
    skip_build: bool,
    skip_push: bool,
    skip_template: bool,
    skip_endpoint: bool,
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
    scaler_type: str,
    scaler_value: int,
    execution_timeout: int | None,
    flashboot: bool,
    network_volume_id: str | None,
    allowed_cuda_versions: tuple,
    list_gpu_types: bool,
    list_cpu_flavors: bool,
    list_data_centers: bool,
    list_all_options: bool,
):
    """Deploy workflow to RunPod serverless infrastructure.
    
    Examples:
      # Basic deployment
      nodetool deploy --workflow-id abc123
      
      # With specific GPU and regions
      nodetool deploy --workflow-id abc123 --gpu-types AMPERE_24 --gpu-types ADA_48_PRO --data-centers US-CA-2 --data-centers US-GA-1
      
      # CPU-only endpoint
      nodetool deploy --workflow-id abc123 --compute-type CPU --cpu-flavors cpu3c --cpu-flavors cpu5c
      
      # Check Docker configuration
      nodetool deploy --check-docker-config
      
      # List available options
      nodetool deploy --list-gpu-types
      nodetool deploy --list-all-options
    """
    import sys
    import os
    import traceback
    from nodetool.deploy.deploy_to_runpod import (
        ComputeType, GPUType, CPUFlavor, DataCenter, ScalerType, CUDAVersion,
        check_docker_auth, get_docker_username_from_config,
        format_image_name, sanitize_name, generate_image_tag, fetch_workflow_from_db,
        build_docker_image, push_to_registry, create_or_update_runpod_template,
        create_runpod_endpoint
    )
    
    # Handle list options (these don't require workflow-id)
    if list_gpu_types:
        console.print("[bold cyan]Available GPU Types:[/]")
        for gpu_type in GPUType:
            console.print(f"  {gpu_type.value}")
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
        for gpu_type in GPUType:
            console.print(f"  {gpu_type.value}")
        console.print("\n[bold]CPU Flavors:[/]")
        for cpu_flavor in CPUFlavor:
            console.print(f"  {cpu_flavor.value}")
        console.print("\n[bold]Data Centers:[/]")
        for data_center in DataCenter:
            console.print(f"  {data_center.value}")
        console.print("\n[bold]Scaler Types:[/]")
        for scaler_type in ScalerType:
            console.print(f"  {scaler_type.value}")
        console.print("\n[bold]CUDA Versions:[/]")
        for cuda_version in CUDAVersion:
            console.print(f"  {cuda_version.value}")
        sys.exit(0)
    
    # Handle Docker config check (doesn't require workflow-id)
    if check_docker_config:
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
        final_username = (
            docker_username or 
            env_username or 
            config_username
        )
        
        if final_username:
            console.print(f"\n🎉 Final resolved username: {final_username}")
            
            # Show what the full image name would be
            example_image = format_image_name("my-workflow", final_username, docker_registry)
            example_tag = generate_image_tag()
            console.print(f"Example image name: {example_image}:{example_tag}")
        else:
            console.print("\n❌ No Docker username found!")
            console.print("To fix this, run: docker login")
            
        sys.exit(0)
    
    # Validate that workflow-id is provided for deployment operations
    if not workflow_id:
        console.print("❌ Error: --workflow-id is required for deployment operations")
        console.print("Use --help to see available options or use one of the list commands:")
        console.print("  --list-gpu-types, --list-cpu-flavors, --list-data-centers, --list-all-options")
        sys.exit(1)
    
    # Get Docker username from multiple sources
    docker_username = (
        docker_username or 
        os.getenv("DOCKER_USERNAME") or 
        get_docker_username_from_config(docker_registry)
    )
    
    if not docker_username and not (skip_build and skip_push):
        console.print("Error: Docker username is required for building and pushing images.")
        console.print("Provide it via one of these methods:")
        console.print("1. Command line: --docker-username myusername")
        console.print("2. Environment variable: export DOCKER_USERNAME=myusername")
        console.print("3. Docker login: docker login (will be read from ~/.docker/config.json)")
        sys.exit(1)
    
    if docker_username:
        console.print(f"Using Docker username: {docker_username}")
    
    # Generate unique tag if not provided
    if tag:
        image_tag = tag
        console.print(f"Using provided tag: {image_tag}")
    else:
        image_tag = generate_image_tag()
        console.print(f"Generated unique tag: {image_tag}")
    
    # Check if Docker is running
    if not skip_build:
        try:
            from nodetool.deploy.deploy_to_runpod import run_command
            run_command("docker --version", capture_output=True)
        except:
            console.print("Error: Docker is not installed or not running")
            sys.exit(1)
    
    # Fetch workflow from database
    workflow_path, workflow_name = fetch_workflow_from_db(workflow_id)
    
    # Set defaults based on workflow name
    base_image_name = image_name or sanitize_name(workflow_name)
    console.print(f"Using base image name: {base_image_name}")
    
    # Format full image name with registry and username
    if docker_username:
        full_image_name = format_image_name(base_image_name, docker_username, docker_registry)
        console.print(f"Full image name: {full_image_name}")
    else:
        full_image_name = base_image_name
    
    template_name = template_name or base_image_name
    console.print(f"Using template name: {template_name}")
    
    template_id = None
    endpoint_id = None
    
    try:
        # Build Docker image with embedded workflow
        if not skip_build:
            build_docker_image(workflow_path, full_image_name, image_tag, platform)
        
        if not skip_push:
            push_to_registry(full_image_name, image_tag, docker_registry)
        
        if not skip_template:
            template_id = create_or_update_runpod_template(template_name, full_image_name, image_tag)
        
        if not skip_endpoint and template_id:
            # Convert GPU types from string values
            gpu_type_ids = list(gpu_types) if gpu_types else None
            cpu_flavor_ids = list(cpu_flavors) if cpu_flavors else None
            data_center_ids = list(data_centers) if data_centers else None
            allowed_cuda_versions_list = list(allowed_cuda_versions) if allowed_cuda_versions else None
            
            endpoint_id = create_runpod_endpoint(
                template_id=template_id,
                name=workflow_id,
                compute_type=compute_type,
                gpu_type_ids=gpu_type_ids,
                gpu_count=gpu_count,
                cpu_flavor_ids=cpu_flavor_ids,
                vcpu_count=vcpu_count,
                data_center_ids=data_center_ids,
                workers_min=workers_min,
                workers_max=workers_max,
                idle_timeout=idle_timeout,
                scaler_type=scaler_type,
                scaler_value=scaler_value,
                execution_timeout_ms=execution_timeout,
                flashboot=flashboot,
                network_volume_id=network_volume_id,
                allowed_cuda_versions=allowed_cuda_versions_list,
            )
        
        console.print(f"\n🎉 Deployment completed successfully!")
        console.print(f"Workflow ID: {workflow_id}")
        console.print(f"Image: {full_image_name}:{image_tag}")
        console.print(f"Platform: {platform}")
        if template_id:
            console.print(f"Template ID: {template_id}")
        if endpoint_id:
            console.print(f"Endpoint ID: {endpoint_id}")
        
    except Exception as e:
        console.print(f"[bold red]Deployment failed: {e}[/]")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up workflow file
        if 'workflow_path' in locals() and os.path.exists(workflow_path):
            os.unlink(workflow_path)


@cli.command("test-runpod")
@click.option("--endpoint-id", required=True, help="RunPod endpoint ID")
@click.option("--api-key", help="RunPod API key (can also use RUNPOD_API_KEY env var)")
@click.option("--params", type=click.Path(exists=True), help="JSON file with workflow parameters")
@click.option("--params-json", help="Inline JSON string with workflow parameters")
@click.option("--output", help="Output file for results (default: auto-generated)")
@click.option("--timeout", type=int, default=60, help="Timeout in seconds (default: 60)")
def test_runpod(
    endpoint_id: str,
    api_key: str | None,
    params: str | None,
    params_json: str | None,
    output: str | None,
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
      
      # Test with custom timeout and output file
      nodetool test-runpod --endpoint-id abc123def456 --timeout 120 --output results.json
    """
    import json
    import time
    import traceback
    from datetime import datetime
    from typing import Dict, Any
    
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
        console.print("Provide it via --api-key argument or RUNPOD_API_KEY environment variable")
        sys.exit(1)
    
    # Configure runpod library
    runpod.api_key = api_key
    endpoint = runpod.Endpoint(endpoint_id)
    
    # Get workflow parameters
    workflow_params = {}
    if params:
        try:
            with open(params, 'r') as f:
                workflow_params = json.load(f)
        except Exception as e:
            console.print(f"[bold red]❌ Failed to load parameters from {params}: {e}[/]")
            sys.exit(1)
    elif params_json:
        try:
            workflow_params = json.loads(params_json)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]❌ Invalid JSON in --params-json: {e}[/]")
            sys.exit(1)
    else:
        console.print("[bold yellow]⚠️ No parameters provided, using empty parameters[/]")
    
    console.print(f"[bold cyan]🧪 Testing RunPod workflow...[/]")
    console.print(f"Endpoint ID: {endpoint_id}")
    console.print(f"Parameters: {json.dumps(workflow_params, indent=2)}")
    console.print(f"Timeout: {timeout} seconds")
    
    try:
        console.print(f"[bold blue]🚀 Starting workflow execution...[/]")
        
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
        
        console.print(f"[bold green]✅ Job completed successfully![/]")
        elapsed = int(time.time() - start_time)
        console.print(f"Execution completed in {elapsed} seconds")
        
        # Display results
        console.print(f"\n[bold cyan]📊 Job Results:[/]")
        console.print(json.dumps(result, indent=2))
        
        # Save results
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"runpod_result_{timestamp}.json"
        
        try:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            
            console.print(f"[bold green]💾 Results saved to: {output}[/]")
            
        except Exception as e:
            console.print(f"[bold red]❌ Failed to save results: {e}[/]")
        
        console.print(f"\n[bold green]✅ Test completed successfully![/]")
        
    except TimeoutError:
        console.print(f"\n[bold red]⏰ Job timed out[/]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print(f"\n[bold yellow]🛑 Test interrupted by user[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Test failed: {e}[/]")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
