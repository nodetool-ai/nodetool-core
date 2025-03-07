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

        repository = project_data.get("repository")
        if not repository:
            click.echo("Error: No repository found in pyproject.toml", err=True)
            sys.exit(1)

        owner = repository.split("/")[-2]

        # Create index.html and CSS files
        os.makedirs(output_dir, exist_ok=True)

        # Create initial index.md with front matter
        index_content = f"""---
layout: default
title: {package_name} Documentation
nav_order: 1
permalink: /
---

# {package_name} Documentation

## Available Nodes

"""
        # Track all module documentation files
        module_links = []

        # Continue with documentation generation
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

                        # Generate documentation file with front matter for navigation
                        doc_content = f"""---
layout: default
title: {module_name}
parent: Nodes
has_children: false
nav_order: 2
---

"""
                        doc_content += generate_documentation(
                            full_module_name,
                            compact=compact,
                        )

                        # Create a separate file for each module
                        module_filename = f"{module_name.replace('.', '_')}.md"
                        with open(os.path.join(output_dir, module_filename), "w") as f:
                            f.write(doc_content)

                        # Add to module links for the index
                        module_links.append(f"- [{module_name}]({module_filename})")

            bar.update(90)

        # Create a nodes.md file as a parent for all node pages
        nodes_content = """---
layout: default
title: Nodes
nav_order: 2
has_children: true
permalink: /nodes
---

# Nodes

This section contains documentation for all available nodes.
"""
        with open(os.path.join(output_dir, "nodes.md"), "w") as f:
            f.write(nodes_content)

        # Add module links to index.md
        index_content += "\n".join(sorted(module_links))
        index_content += """

## Navigation

Use the sidebar navigation to explore detailed documentation for each node.
"""
        with open(os.path.join(output_dir, "index.md"), "w") as f:
            f.write(index_content)

        # Create GitHub Actions workflow for Pages
        workflow_dir = ".github/workflows"
        os.makedirs(workflow_dir, exist_ok=True)

        workflow_content = """name: Deploy Jekyll with GitHub Pages dependencies preinstalled

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Pages
        uses: actions/configure-pages@v5
      - name: Build with Jekyll
        uses: actions/jekyll-build-pages@v1
        with:
          source: ./docs
          destination: ./_site
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
"""

        workflow_path = os.path.join(workflow_dir, "deploy-docs.yml")
        with open(workflow_path, "w") as f:
            f.write(workflow_content)

        # Create Jekyll configuration
        jekyll_config = f"""title: {package_name} Documentation
description: Documentation for {package_name} nodes
remote_theme: just-the-docs/just-the-docs
baseurl: "{repository}"
url: "{repository}"

# Build settings
markdown: kramdown
theme: just-the-docs
plugins:
  - jekyll-feed
  - jekyll-remote-theme

# Navigation Structure
nav_sort: case_sensitive
heading_anchors: true

# Enable search
search_enabled: true
search:
  heading_level: 3
  previews: 3
  preview_words_before: 5
  preview_words_after: 10
  tokenizer_separator: /[\s/]+/
  rel_url: true
  button: false

# Aux links for the upper right navigation
aux_links:
  "View on GitHub":
    - "{repository}"

aux_links_new_tab: true

# Color scheme
color_scheme: dark

# Exclude files from processing
exclude:
  - Gemfile
  - Gemfile.lock
  - node_modules
  - vendor
"""

        with open(os.path.join(output_dir, "_config.yml"), "w") as f:
            f.write(jekyll_config)

        # Create Gemfile
        gemfile_content = """source "https://rubygems.org"

gem "jekyll"
gem "github-pages", group: :jekyll_plugins
gem "just-the-docs", "0.7.0"
gem "jekyll-remote-theme"
"""

        with open(os.path.join(output_dir, "Gemfile"), "w") as f:
            f.write(gemfile_content)

        click.echo("\nSetup GitHub Pages deployment:")
        click.echo("  - Created .github/workflows/deploy-docs.yml")
        click.echo("  - Created docs/_config.yml")
        click.echo("  - Created docs/Gemfile")
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
