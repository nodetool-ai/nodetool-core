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

        headers = ["Name", "Version", "Description", "Nodes"]
        table_data = [
            [pkg.name, pkg.version, pkg.description, len(pkg.nodes or [])]
            for pkg in packages
        ]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


@cli.command("scan")
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output during scanning"
)
def scan_package(verbose):
    """Scan current directory for nodes and create a single nodes.json metadata file."""
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
        src_path = os.path.abspath("src/nodetool/nodes")
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
                                node_classes = get_node_classes_from_module(
                                    module_name, verbose
                                )
                                if node_classes:
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


if __name__ == "__main__":
    cli()
