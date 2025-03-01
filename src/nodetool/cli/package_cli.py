"""
CLI commands for managing nodetool packages.

This module provides CLI commands for:
- Listing available packages
- Installing packages
- Uninstalling packages
- Updating packages
- Searching for packages
"""

import argparse
import sys
import logging
from typing import List, Optional

from nodetool.common.package_registry import Registry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def list_packages(args: argparse.Namespace) -> None:
    """List available or installed packages."""
    registry = Registry()

    if args.installed:
        packages = registry.list_installed_packages()
        if not packages:
            print("No packages installed.")
            return

        print("Installed packages:")
        for package in packages:
            print(f"  {package.name} (v{package.version}) - {package.description}")
    else:
        packages = registry.list_available_packages()
        if not packages:
            print("No packages available in the registry.")
            return

        print("Available packages:")
        for package in packages:
            print(
                f"  {package['name']} (v{package['version']}) - {package['description']}"
            )


def install_package(args: argparse.Namespace) -> None:
    """Install a package."""
    registry = Registry()

    if args.package_name:
        success = registry.install_package(args.package_name)
        if not success:
            sys.exit(1)
    else:
        print("Please specify a package name to install.")
        sys.exit(1)


def uninstall_package(args: argparse.Namespace) -> None:
    """Uninstall a package."""
    registry = Registry()

    if args.package_name:
        success = registry.uninstall_package(args.package_name)
        if not success:
            sys.exit(1)
    else:
        print("Please specify a package name to uninstall.")
        sys.exit(1)


def update_package(args: argparse.Namespace) -> None:
    """Update a package."""
    registry = Registry()

    if args.package_name:
        success = registry.update_package(args.package_name)
        if not success:
            sys.exit(1)
    elif args.all:
        # Update all installed packages
        packages = registry.list_installed_packages()
        if not packages:
            print("No packages installed.")
            return

        for package in packages:
            print(f"Updating {package.name}...")
            registry.update_package(package.name)
    else:
        print(
            "Please specify a package name to update or use --all to update all packages."
        )
        sys.exit(1)


def search_packages(args: argparse.Namespace) -> None:
    """Search for packages."""
    registry = Registry()

    if not args.query:
        print("Please specify a search query.")
        sys.exit(1)

    # Get all available packages
    packages = registry.list_available_packages()
    if not packages:
        print("No packages available in the registry.")
        return

    # Filter packages by query
    query = args.query.lower()
    matching_packages = [
        package
        for package in packages
        if query in package["name"].lower()
        or query in package["description"].lower()
        or any(query in tag.lower() for tag in package.get("tags", []))
    ]

    if not matching_packages:
        print(f"No packages found matching '{args.query}'.")
        return

    print(f"Packages matching '{args.query}':")
    for package in matching_packages:
        print(f"  {package['name']} (v{package['version']}) - {package['description']}")
        if package.get("tags"):
            print(f"    Tags: {', '.join(package['tags'])}")


def generate_package(args: argparse.Namespace) -> None:
    """Generate package metadata from a GitHub repository or local folder."""
    registry = Registry()

    if args.folder_path:
        package = registry.generate_package_from_folder(args.folder_path)
        if not package:
            sys.exit(1)
    elif args.github_repo:
        package = registry.generate_package_from_github(args.github_repo)
        if not package:
            sys.exit(1)
    else:
        print("Please specify either a GitHub repository URL or a local folder path.")
        sys.exit(1)


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the package management subcommand parser."""
    package_parser = subparsers.add_parser(
        "package",
        help="Manage nodetool packages",
    )

    package_subparsers = package_parser.add_subparsers(
        dest="package_command",
        help="Package management commands",
    )

    # List command
    list_parser = package_subparsers.add_parser(
        "list",
        help="List available or installed packages",
    )
    list_parser.add_argument(
        "--installed",
        action="store_true",
        help="List only installed packages",
    )
    list_parser.set_defaults(func=list_packages)

    # Install command
    install_parser = package_subparsers.add_parser(
        "install",
        help="Install a package",
    )
    install_parser.add_argument(
        "package_name",
        help="Name of the package to install",
    )
    install_parser.set_defaults(func=install_package)

    # Uninstall command
    uninstall_parser = package_subparsers.add_parser(
        "uninstall",
        help="Uninstall a package",
    )
    uninstall_parser.add_argument(
        "package_name",
        help="Name of the package to uninstall",
    )
    uninstall_parser.set_defaults(func=uninstall_package)

    # Update command
    update_parser = package_subparsers.add_parser(
        "update",
        help="Update a package",
    )
    update_parser.add_argument(
        "package_name",
        nargs="?",
        help="Name of the package to update",
    )
    update_parser.add_argument(
        "--all",
        action="store_true",
        help="Update all installed packages",
    )
    update_parser.set_defaults(func=update_package)

    # Search command
    search_parser = package_subparsers.add_parser(
        "search",
        help="Search for packages",
    )
    search_parser.add_argument(
        "query",
        help="Search query",
    )
    search_parser.set_defaults(func=search_packages)

    # Generate command
    generate_parser = package_subparsers.add_parser(
        "generate",
        help="Generate package metadata from a GitHub repository or local folder",
    )
    generate_group = generate_parser.add_mutually_exclusive_group(required=True)
    generate_group.add_argument(
        "--github-repo",
        help="GitHub repository URL",
    )
    generate_group.add_argument(
        "--folder-path",
        help="Path to local folder containing the package",
    )
    generate_parser.set_defaults(func=generate_package)

    # Set default function for package parser
    package_parser.set_defaults(func=lambda _: package_parser.print_help())
