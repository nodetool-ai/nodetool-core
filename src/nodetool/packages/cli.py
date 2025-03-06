# CLI Implementation
import sys
import traceback
import click
from tabulate import tabulate
import os
import tomli
import json
import importlib.util
import importlib.machinery

from nodetool.metadata.node_metadata import (
    EnumEncoder,
    PackageModel,
    get_node_classes_from_module,
)
from nodetool.packages.registry import (
    Registry,
)
from nodetool.packages.gen_docs import generate_documentation


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


@cli.command("init")
def init_project():
    """Initialize a new Nodetool project with pyproject.toml."""
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


@cli.command("docs")
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
def generate_docs(output_dir: str, compact: bool):
    """Generate documentation for the package nodes and setup GitHub Pages."""
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

        # Create index.html and CSS files
        os.makedirs(output_dir, exist_ok=True)

        # Create styles.css
        css_content = """
        :root {
            --primary-color: #2563eb;
            --background-color: #ffffff;
            --text-color: #1f2937;
            --code-bg: #f3f4f6;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--background-color);
        }

        h1, h2, h3 {
            color: var(--primary-color);
            margin-top: 2rem;
        }

        .node-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        code {
            background: var(--code-bg);
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.9em;
        }

        pre {
            background: var(--code-bg);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
        }

        nav {
            padding: 1rem 0;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 2rem;
        }

        .search-box {
            width: 100%;
            padding: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        }
        """

        with open(os.path.join(output_dir, "styles.css"), "w") as f:
            f.write(css_content)

        # Create base index.html
        index_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{package_name} Documentation</title>
            <meta name="description" content="Documentation for {package_name} nodes">
            <link rel="stylesheet" href="styles.css">
            <link rel="icon" href="data:,">
        </head>
        <body>
            <nav>
                <h1>{package_name}</h1>
                <input type="text" class="search-box" placeholder="Search nodes..." id="searchBox">
            </nav>
            <main id="content">
                <!-- Node documentation will be inserted here -->
            </main>
            <script>
                // Simple search functionality
                document.getElementById('searchBox').addEventListener('input', function(e) {{
                    const searchTerm = e.target.value.toLowerCase();
                    document.querySelectorAll('.node-card').forEach(card => {{
                        const text = card.textContent.toLowerCase();
                        card.style.display = text.includes(searchTerm) ? 'block' : 'none';
                    }});
                }});
            </script>
        </body>
        </html>
        """

        with open(os.path.join(output_dir, "index.html"), "w") as f:
            f.write(index_content)

        # Continue with existing documentation generation
        with click.progressbar(
            length=100,
            label="Generating documentation",
            show_eta=False,
            show_percent=True,
        ) as bar:
            bar.update(10)

            # Process each Python module under src/nodetool/nodes
            for root, _, files in os.walk(nodes_path):
                for file in files:
                    if file.endswith(".py"):
                        module_path = os.path.join(root, file)
                        rel_path = os.path.relpath(module_path, nodes_path)
                        module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                        full_module_name = f"nodetool.nodes.{module_name}"

                        try:
                            generate_documentation(
                                full_module_name, output_dir, compact=compact
                            )
                        except Exception as e:
                            click.echo(f"Error processing {module_name}: {e}", err=True)

            bar.update(90)

        click.echo(f"✅ Successfully generated documentation in {output_dir}")

        # Create GitHub Actions workflow for Pages
        workflow_dir = ".github/workflows"
        os.makedirs(workflow_dir, exist_ok=True)

        workflow_content = """name: Deploy Documentation
on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Pages
        uses: actions/configure-pages@v4
        
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './docs'
          
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
"""

        workflow_path = os.path.join(workflow_dir, "deploy-docs.yml")
        with open(workflow_path, "w") as f:
            f.write(workflow_content)

        click.echo("\nSetup GitHub Pages deployment:")
        click.echo("  - Created .github/workflows/deploy-docs.yml")
        click.echo("\nTo enable GitHub Pages:")
        click.echo("1. Go to your repository settings")
        click.echo("2. Navigate to Pages section")
        click.echo("3. Select 'GitHub Actions' as the source")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
