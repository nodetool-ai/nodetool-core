import os
import click
from nodetool.common.environment import Environment

# silence warnings on the command line
import warnings

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

    os.environ["ENV"] = "production" if production else "development"

    try:
        import comfy.cli_args  # type: ignore

        comfy.cli_args.args.force_fp16 = force_fp16
    except ImportError:
        pass

    Environment.set_remote_auth(remote_auth)

    if worker_url:
        Environment.set_worker_url(worker_url)

    if Environment.is_production():
        Environment.set_nodetool_api_url(f"https://api.nodetool.ai")
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
@click.argument("workflow_id", type=str)
def run(workflow_id: str):
    """Run a workflow from a file."""
    import asyncio
    import traceback
    from nodetool.workflows.run_job_request import RunJobRequest
    from nodetool.workflows.run_workflow import run_workflow

    request = RunJobRequest(
        workflow_id=workflow_id, user_id="1", auth_token="local_token"
    )

    async def run_workflow_async():
        print("Running workflow...")
        try:
            async for message in run_workflow(request):
                # Print message type and content
                if hasattr(message, "type"):
                    print(f"{message.type}: {message.model_dump_json()}")
                else:
                    print(message)
            print("Workflow finished")
        except Exception as e:
            print(f"Error running workflow: {e}")
            traceback.print_exc()
            exit(1)

    asyncio.run(run_workflow_async())


@cli.command()
def chat():
    """Start a nodetool chat."""
    import asyncio
    from nodetool.chat.chat import chat_cli

    asyncio.run(chat_cli())


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
    from tabulate import tabulate
    from nodetool.packages.registry import Registry

    registry = Registry()

    if available:
        packages = registry.list_available_packages()
        if not packages:
            click.echo(
                "No packages available in the registry or unable to fetch package list."
            )
            return

        headers = ["Name", "Repository ID"]
        table_data = [[pkg.name, pkg.repo_id] for pkg in packages]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        packages = registry.list_installed_packages()
        if not packages:
            click.echo("No packages installed.")
            return

        headers = ["Name", "Version", "Description", "Nodes"]
        table_data = [
            [pkg.name, pkg.version, pkg.description, len(pkg.nodes or [])]
            for pkg in packages
        ]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


@package.command()
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output during scanning"
)
def scan(verbose):
    """Scan current directory for nodes and create package metadata."""
    import os
    import sys
    import tomli
    import json
    import traceback
    from nodetool.metadata.node_metadata import (
        EnumEncoder,
        PackageModel,
        get_node_classes_from_module,
    )

    sys.path.append(os.path.abspath("src"))

    try:
        # Check for pyproject.toml in current directory
        if not os.path.exists("pyproject.toml"):
            click.echo("Error: No pyproject.toml found in current directory", err=True)
            sys.exit(1)

        # Read pyproject.toml
        with open("pyproject.toml", "rb") as f:
            pyproject_data = tomli.loads(f.read().decode())

        # Extract metadata
        project_data = pyproject_data.get("project", {})
        if not project_data:
            project_data = pyproject_data.get("tool", {}).get("poetry", {})

        if not project_data:
            click.echo("Error: No project metadata found in pyproject.toml", err=True)
            sys.exit(1)

        repo_id = project_data.get("repository", "").split("/")[-2:]
        repo_id = "/".join(repo_id)

        # Create package model
        package = PackageModel(
            name=project_data.get("name", ""),
            description=project_data.get("description", ""),
            version=project_data.get("version", "0.1.0"),
            authors=project_data.get("authors", []),
            repo_id=repo_id,
        )

        # Add src directory to Python path temporarily
        src_path = os.path.abspath("src/nodetool/nodes")
        if os.path.exists(src_path):
            with click.progressbar(
                length=100,
                label="Scanning for nodes",
                show_eta=False,
                show_percent=True,
            ) as bar:
                bar.update(10)

                # Discover nodes
                for root, _, files in os.walk(src_path):
                    for file in files:
                        if file.endswith(".py"):
                            module_path = os.path.join(root, file)
                            rel_path = os.path.relpath(module_path, src_path)
                            module_name = os.path.splitext(rel_path)[0].replace(
                                os.sep, "."
                            )

                            if verbose:
                                click.echo(f"Scanning module: {module_name}")

                            try:
                                full_module_name = f"nodetool.nodes.{module_name}"
                                node_classes = get_node_classes_from_module(
                                    full_module_name, verbose
                                )
                                if node_classes:
                                    assert package.nodes is not None
                                    package.nodes.extend(
                                        node_class.metadata()
                                        for node_class in node_classes
                                    )
                            except Exception as e:
                                if verbose:
                                    click.echo(
                                        f"Error processing {module_name}: {e}", err=True
                                    )

                bar.update(90)

            # Write the single nodes.json file in the root directory
            os.makedirs("src/nodetool/package_metadata", exist_ok=True)
            with open(f"src/nodetool/package_metadata/{package.name}.json", "w") as f:
                json.dump(
                    package.model_dump(exclude_defaults=True),
                    f,
                    indent=2,
                    cls=EnumEncoder,
                )

        click.echo(
            f"✅ Successfully created package metadata for {package.name} with {len(package.nodes or [])} total nodes"
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
include = ["src/nodetool/package-metadata/{name}.json"]

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
def docs(output_dir: str, compact: bool):
    """Generate documentation for the package nodes."""
    import os
    import sys
    import tomli
    import traceback
    from nodetool.packages.gen_docs import generate_documentation

    try:
        # Add src directory to Python path temporarily
        src_path = os.path.abspath("src")
        if not os.path.exists(src_path):
            click.echo("Error: No src directory found", err=True)
            sys.exit(1)

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

        owner = repository.split("/")[-2]

        # Generate documentation
        with click.progressbar(
            length=100,
            label="Generating documentation",
            show_eta=False,
            show_percent=True,
        ) as bar:
            bar.update(10)
            # Generate the documentation
            docs = generate_documentation(package_name, compact)

            # Write to output file
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "index.md"), "w") as f:
                f.write(docs)
            bar.update(90)

        click.echo(f"✅ Documentation generated in {output_dir}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        traceback.print_exc()
        sys.exit(1)


# Add package group to the main CLI
cli.add_command(package)


if __name__ == "__main__":
    cli()
