#!/usr/bin/env python3
"""
Script for generating and submitting package metadata to the nodetool package registry.

This script helps package authors to:
1. Generate package metadata YAML file from a GitHub repository or local folder
2. Validate the package metadata
3. Submit the package metadata to the registry (via PR)

Usage:
    python generate_package.py --github-repo https://github.com/johndoe/my-package
    python generate_package.py --folder-path /path/to/my-package

Requirements:
    - PyGithub (pip install PyGithub)
    - GitPython (pip install GitPython)
    - tomli (pip install tomli)
"""

import os
import sys
import argparse
import logging
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

# Add parent directory to path to import nodetool modules
sys.path.append(str(Path(__file__).parent.parent))

from nodetool.common.package_registry import (
    extract_package_metadata_from_repo,
    extract_package_metadata_from_folder,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_github_repo(repo_url: str) -> bool:
    """Validate that the GitHub repository exists and is accessible."""
    # Extract owner and repo name from URL
    if "github.com" not in repo_url:
        logger.error(f"Invalid GitHub repository URL: {repo_url}")
        return False

    # Simple validation for now
    return True


def validate_folder_path(folder_path: str) -> bool:
    """Validate that the folder path exists and contains a pyproject.toml file."""
    if not os.path.isdir(folder_path):
        logger.error(f"Invalid folder path: {folder_path}")
        return False

    if not os.path.exists(os.path.join(folder_path, "pyproject.toml")):
        logger.error(f"No pyproject.toml found in folder: {folder_path}")
        return False

    return True


def clone_registry_repo(registry_repo: str, target_dir: str) -> bool:
    """Clone the package registry repository."""
    try:
        subprocess.check_call(
            ["git", "clone", registry_repo, target_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Successfully cloned registry repository to {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error cloning registry repository: {e}")
        return False


def create_branch(repo_dir: str, branch_name: str) -> bool:
    """Create a new branch in the repository."""
    try:
        subprocess.check_call(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Created branch: {branch_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating branch: {e}")
        return False


def commit_and_push(repo_dir: str, package_name: str, branch_name: str) -> bool:
    """Commit and push changes to the repository."""
    try:
        # Add the package file
        subprocess.check_call(
            ["git", "add", f"packages/{package_name}.yaml"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Commit the changes
        subprocess.check_call(
            ["git", "commit", "-m", f"Add package metadata for {package_name}"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Push the changes
        subprocess.check_call(
            ["git", "push", "origin", branch_name],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info(f"Successfully pushed changes to branch: {branch_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error committing and pushing changes: {e}")
        return False


def create_pull_request(
    repo_dir: str, package_name: str, branch_name: str, github_token: str
) -> bool:
    """Create a pull request to the registry repository."""
    # Try to import PyGithub
    try:
        from github import Github
    except ImportError:
        logger.error("PyGithub is not installed. Install it with: pip install PyGithub")
        return False

    try:
        # Create GitHub client
        g = Github(github_token)

        # Get the repository
        repo = g.get_repo("nodetool/package-registry")

        # Create pull request
        pr = repo.create_pull(
            title=f"Add package metadata for {package_name}",
            body=f"This PR adds metadata for the {package_name} package.",
            head=branch_name,
            base="main",
        )

        logger.info(f"Successfully created pull request: {pr.html_url}")
        return True
    except Exception as e:
        logger.error(f"Error creating pull request: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit package metadata to the registry"
    )

    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--github-repo", help="GitHub repository URL")
    source_group.add_argument(
        "--folder-path", help="Path to local folder containing the package"
    )

    # Output options
    parser.add_argument("--output", help="Path to output package metadata YAML file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Registry submission options
    parser.add_argument(
        "--submit", action="store_true", help="Submit package metadata to the registry"
    )
    parser.add_argument(
        "--registry-repo",
        default="https://github.com/nodetool/package-registry.git",
        help="Registry repository URL",
    )
    parser.add_argument("--github-token", help="GitHub token for creating pull request")

    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    # Generate package metadata
    package_metadata = None

    if args.github_repo:
        # Validate GitHub repository
        if not validate_github_repo(args.github_repo):
            sys.exit(1)

        # Generate package metadata from GitHub repository
        package_metadata = extract_package_metadata_from_repo(args.github_repo)
    elif args.folder_path:
        # Validate folder path
        if not validate_folder_path(args.folder_path):
            sys.exit(1)

        # Generate package metadata from local folder
        package_metadata = extract_package_metadata_from_folder(args.folder_path)

    if not package_metadata:
        logger.error("Failed to generate package metadata")
        sys.exit(1)

    # Determine output path
    output_path = args.output or f"{package_metadata.name}.yaml"

    # Create the directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    # Write package metadata to YAML file
    try:
        import yaml

        with open(output_path, "w") as f:
            yaml.dump(package_metadata.model_dump(), f, sort_keys=False)
        logger.info(f"Successfully wrote package metadata to {output_path}")
    except Exception as e:
        logger.error(f"Error writing package metadata to {output_path}: {e}")
        sys.exit(1)

    # Submit package metadata to the registry
    if args.submit:
        if not args.github_token:
            logger.error(
                "GitHub token is required for submitting package metadata to the registry"
            )
            sys.exit(1)

        # Create temporary directory for cloning registry repository
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone registry repository
            if not clone_registry_repo(args.registry_repo, temp_dir):
                sys.exit(1)

            # Create packages directory if it doesn't exist
            packages_dir = os.path.join(temp_dir, "packages")
            os.makedirs(packages_dir, exist_ok=True)

            # Copy package metadata to packages directory
            shutil.copy(
                output_path, os.path.join(packages_dir, f"{package_metadata.name}.yaml")
            )

            # Create branch
            branch_name = f"add-package-{package_metadata.name}"
            if not create_branch(temp_dir, branch_name):
                sys.exit(1)

            # Commit and push changes
            if not commit_and_push(temp_dir, package_metadata.name, branch_name):
                sys.exit(1)

            # Create pull request
            if not create_pull_request(
                temp_dir, package_metadata.name, branch_name, args.github_token
            ):
                sys.exit(1)

            logger.info(
                f"Successfully submitted package metadata for {package_metadata.name} to the registry"
            )


if __name__ == "__main__":
    main()
