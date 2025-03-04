# CLI Implementation
import sys
import traceback
import click
from tabulate import tabulate
import os
import tomli
import json

from nodetool.metadata.node_metadata import (
    EnumEncoder,
    PackageModel,
    get_node_classes_from_module,
)
from nodetool.packages.registry import (
    Registry,
    get_package_metadata_from_pip,
    validate_repo_id,
)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    Nodetool Package Manager CLI.

    This tool helps you manage packages for the Nodetool ecosystem.
    """
    pass


@cli.command("list")
@click.option(
    "--available", "-a", is_flag=True, help="List available packages from the registry"
)
def list_packages(available):
    """List installed or available packages."""
    registry = Registry()

    if available:
        packages = registry.list_available_packages()
        if not packages:
            click.echo(
                "No packages available in the registry or unable to fetch package list."
            )
            return

        print(packages)
        headers = ["Name", "Repository ID"]
        table_data = [[pkg.name, pkg.repo_id] for pkg in packages]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        packages = registry.list_installed_packages()
        if not packages:
            click.echo("No packages installed.")
            return

        headers = ["Name", "Repository ID"]
        table_data = [[pkg.name, pkg.repo_id] for pkg in packages]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


@cli.command("install")
@click.argument("repo_id")
@click.option(
    "--local",
    "-l",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Install from a local directory instead of the registry",
)
@click.option(
    "--namespace",
    "-n",
    type=str,
    help="Namespace to install from the package",
)
def install_package_cmd(repo_id, local, namespace):
    """Install a package by repository ID (owner/project) or from a local directory."""
    registry = Registry()

    try:
        # If installing from local path, skip repo_id validation
        if not local:
            is_valid, error_msg = validate_repo_id(repo_id)
            if not is_valid:
                click.echo(f"Error: {error_msg}", err=True)
                sys.exit(1)

        with click.progressbar(
            length=100,
            label=f"Installing {repo_id}",
            show_eta=False,
            show_percent=True,
        ) as bar:
            bar.update(10)  # Start progress
            try:
                registry.install_package(
                    repo_id,
                    local_path=local,
                )
                success = True
            except Exception as e:
                success = False
                traceback.print_exc()
                raise e
            bar.update(90)  # Complete progress

        if success:
            click.echo(f"✅ Successfully installed {repo_id}")
        else:
            click.echo(f"❌ Failed to install {repo_id}", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command("uninstall")
@click.argument("repo_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def uninstall_package_cmd(repo_id, yes):
    """Uninstall a package by repository ID (owner/project)."""
    registry = Registry()

    try:
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            click.echo(f"Error: {error_msg}", err=True)
            sys.exit(1)

        # Confirm uninstallation
        if not yes and not click.confirm(
            f"Are you sure you want to uninstall {repo_id}?"
        ):
            click.echo("Uninstallation cancelled.")
            return

        success = registry.uninstall_package(repo_id)
        if success:
            click.echo(f"✅ Successfully uninstalled {repo_id}")
        else:
            click.echo(f"❌ Failed to uninstall {repo_id}", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command("update")
@click.argument("repo_id")
def update_package_cmd(repo_id):
    """Update a package to the latest version."""
    registry = Registry()

    try:
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            click.echo(f"Error: {error_msg}", err=True)
            sys.exit(1)

        with click.progressbar(
            length=100,
            label=f"Updating {repo_id}",
            show_eta=False,
            show_percent=True,
        ) as bar:
            bar.update(10)  # Start progress
            success = registry.update_package(repo_id)
            bar.update(90)  # Complete progress

        if success:
            click.echo(f"✅ Successfully updated {repo_id}")
        else:
            click.echo(f"❌ Failed to update {repo_id}", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command("info")
@click.argument("repo_id")
def package_info(repo_id):
    """Show detailed information about an installed package."""
    registry = Registry()

    try:
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            click.echo(f"Error: {error_msg}", err=True)
            sys.exit(1)

        package = get_package_metadata_from_pip(repo_id)
        if not package:
            click.echo(f"Package {repo_id} is not installed.", err=True)
            sys.exit(1)

        # Display package information
        click.echo(f"📦 Package: {package.name}")
        click.echo(f"📝 Description: {package.description}")
        click.echo(f"🔖 Version: {package.version}")
        click.echo(f"🔗 Repository: https://github.com/{package.repo_id}")

        if package.authors:
            click.echo(f"👤 Authors: {', '.join(package.authors)}")

        if package.nodes:
            click.echo(f"🧩 Nodes ({len(package.nodes)}):")
            for node in package.nodes:
                click.echo(f"  - {node.title}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command("scan")
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output during scanning"
)
def scan_package(verbose):
    """Scan current directory for nodes and create nodes.json metadata file."""
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

        # Create package model
        package = PackageModel(
            name=project_data.get("name", ""),
            description=project_data.get("description", ""),
            version=project_data.get("version", "0.1.0"),
            authors=project_data.get("authors", []),
        )

        # Add src directory to Python path temporarily
        src_path = os.path.abspath("src")
        if os.path.exists(src_path):
            sys.path.insert(0, src_path)

            # Find all Python modules under src
            with click.progressbar(
                length=100,
                label="Scanning for nodes",
                show_eta=False,
                show_percent=True,
            ) as bar:
                bar.update(10)

                # Track nodes by module
                module_nodes = {}

                # Discover nodes
                for root, _, files in os.walk(src_path):
                    for file in files:
                        if file.endswith(".py"):
                            module_path = os.path.join(root, file)
                            rel_path = os.path.relpath(module_path, src_path)
                            module_name = os.path.splitext(rel_path)[0].replace(
                                os.sep, "."
                            )
                            module_dir = os.path.dirname(module_path)

                            if verbose:
                                click.echo(f"Scanning module: {module_name}")

                            try:
                                node_classes = get_node_classes_from_module(
                                    module_name, verbose
                                )
                                if node_classes:
                                    # Create nodes.json in the module directory
                                    module_package = PackageModel(
                                        name=project_data.get("name", ""),
                                        description=project_data.get("description", ""),
                                        version=project_data.get("version", "0.1.0"),
                                        authors=project_data.get("authors", []),
                                        nodes=[
                                            node_class.metadata()
                                            for node_class in node_classes
                                        ],
                                    )

                                    nodes_json_path = os.path.join(
                                        module_dir, "nodes.json"
                                    )
                                    with open(nodes_json_path, "w") as f:
                                        json.dump(
                                            module_package.model_dump(
                                                exclude_defaults=True
                                            ),
                                            f,
                                            indent=2,
                                            cls=EnumEncoder,
                                        )

                                    if verbose:
                                        click.echo(
                                            f"Created nodes.json in {module_dir}"
                                        )

                                    assert package.nodes is not None
                                    # Add to package total
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

            # Remove src from path
            sys.path.remove(src_path)

        click.echo(
            f"✅ Successfully created nodes.json files for {len(package.nodes or [])} total nodes"
        )

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
