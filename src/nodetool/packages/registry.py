"""
Package registry module for nodetool application.

This module provides functionality for managing node packages in the nodetool ecosystem.
It handles:

- Loading/saving package metadata from YAML files
- Installing/uninstalling packages
- Listing available packages
- Updating packages

Key components:
- PackageModel: Package metadata model that defines the structure of package information
- NodeMetadata: Metadata model for individual nodes within a package
- Registry: Package registry manager that handles all package operations
- File locations:
  - Package metadata: ~/.config/nodetool/packages/
  - Package registry: GitHub repository with YAML files

Packages are identified by their repository ID (repo_id) which follows the format <owner>/<project>.
For example: "nodetool/package-registry".

The module uses a combination of local storage and remote registry to manage packages.
Packages are installed using either uv or pip package managers, with preference for uv.

Usage:
    registry = Registry()
    packages = registry.list_installed_packages()
    registry.install_package("nodetool/my-package")
"""

from enum import Enum
import json
import os
import yaml
import subprocess
import requests
import tomli
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from urllib.parse import urlparse

from nodetool.common.settings import get_system_file_path
from nodetool.metadata.node_metadata import PackageModel


# Constants
PACKAGES_DIR = "packages"
REGISTRY_URL = (
    "https://raw.githubusercontent.com/nodetool-ai/nodetool-registry/main/index.json"
)
DEFAULT_REGISTRY_REPO = "https://github.com/nodetool/package-registry.git"


def json_serializer(obj: Any) -> dict:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, Enum):
        return obj.value
    raise TypeError("Type not serializable")


def validate_repo_id(repo_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a repo_id follows the <owner>/<project> format.

    Args:
        repo_id: The repository ID to validate

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing:
            - Boolean indicating if the repo_id is valid
            - Error message if invalid, None otherwise
    """
    if not repo_id:
        return False, "Repository ID cannot be empty"

    # Check for the owner/project format using regex
    pattern = r"^[a-zA-Z0-9][-a-zA-Z0-9_]*\/[a-zA-Z0-9][-a-zA-Z0-9_]*$"
    if not re.match(pattern, repo_id):
        return (
            False,
            f"Invalid repository ID format: {repo_id}. Must be in the format <owner>/<project>",
        )

    return True, None


def get_packages_dir() -> Path:
    """
    Get the path to the packages directory.

    This function retrieves the system-specific path where package metadata is stored.
    It ensures the directory exists by creating it if necessary.

    Returns:
        Path: A Path object pointing to the packages directory.
    """
    packages_dir = get_system_file_path(PACKAGES_DIR)
    os.makedirs(packages_dir, exist_ok=True)
    return packages_dir


def get_package_metadata_from_github(github_repo: str) -> Optional[PackageModel]:
    """
    Get package metadata from a GitHub repository.

    This function fetches the pyproject.toml file from a GitHub repository and
    extracts package metadata from it. It supports both standard project metadata
    and Poetry-style metadata.

    Args:
        github_repo: GitHub repository URL (e.g. "https://github.com/nodetool-ai/nodetool-comfy")

    Returns:
        PackageModel: A populated PackageModel instance if successful
        None: If metadata could not be retrieved or parsed

    Raises:
        requests.HTTPError: If the GitHub API request fails
    """
    # Extract repo_id from github_repo URL
    parsed_url = urlparse(github_repo)
    path_parts = parsed_url.path.strip("/").split("/")
    if len(path_parts) < 2:
        print(f"Invalid GitHub repository URL: {github_repo}")
        return None

    repo_id = f"{path_parts[0]}/{path_parts[1]}"

    # Get the pyproject.toml file from the GitHub repository
    raw_url = f"https://raw.githubusercontent.com/{repo_id}/main/pyproject.toml"

    response = requests.get(raw_url)
    response.raise_for_status()
    pyproject_data = tomli.loads(response.text)

    # Extract metadata
    project_data = pyproject_data.get("project", {})
    if not project_data:
        project_data = pyproject_data.get("tool", {}).get("poetry", {})

    if not project_data:
        print(f"No project metadata found in pyproject.toml for {github_repo}")
        return None

    return PackageModel(
        name=project_data.get("name", github_repo),
        description=project_data.get("description", ""),
        version=project_data.get("version", "0.1.0"),
        authors=project_data.get("authors", []),
        repo_id=repo_id,
    )


def get_package_manager_command() -> List[str]:
    """
    Return the appropriate package manager command (uv or pip).

    This function checks if the uv package manager is available on the system.
    If uv is available, it returns a command list for using uv's pip interface.
    Otherwise, it falls back to the standard pip command.

    Returns:
        List[str]: A list containing the package manager command components
                  (either ["uv", "pip"] or ["pip"])
    """
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=False)
        return ["uv", "pip"]
    except FileNotFoundError:
        return ["pip"]


def extract_node_metadata_from_package(
    package: PackageModel, verbose: bool = False
) -> PackageModel:
    """
    Extract metadata from all node classes in a package.

    This function:
    1. Finds all node classes in the namespaces provided by the package
    2. Extracts metadata from each node class
    3. Adds the metadata to the package model

    Args:
        package: The package model to extract node metadata from
        verbose: Whether to print verbose output

    Returns:
        The updated package model with node metadata
    """
    from nodetool.metadata.node_metadata import get_node_classes_from_namespace

    # Initialize empty nodes list (ensure it's never None)
    package.nodes = []

    # Process each namespace provided by the package
    for namespace in package.namespaces:
        try:
            if verbose:
                print(f"Searching for nodes in namespace: {namespace}")
            node_classes = get_node_classes_from_namespace(namespace, verbose=verbose)

            # Extract metadata from each node class
            for node_class in node_classes:
                try:
                    node_metadata = node_class.metadata()
                    package.nodes.append(node_metadata)
                    if verbose:
                        print(f"Added metadata for node: {node_metadata.title}")
                except Exception as e:
                    print(
                        f"Error extracting metadata from node class {node_class.__name__}: {e}"
                    )
        except Exception as e:
            print(f"Error processing namespace {namespace}: {e}")

    return package


class PackageInfo(BaseModel):
    """
    Package information model for nodetool.
    This is the model for the package index in the registry.
    """

    name: str
    description: str
    repo_id: str = Field(description="Repository ID in the format <owner>/<project>")
    namespaces: List[str] = Field(
        default_factory=list, description="Namespaces provided by this package"
    )


class Registry:
    """
    Package registry manager for nodetool.

    This class provides methods for managing packages in the nodetool ecosystem,
    including listing, installing, uninstalling, and updating packages.
    It interacts with both local package metadata and the remote package registry.

    Packages are identified by their repository ID (repo_id) which follows the format <owner>/<project>.
    For example: "nodetool/package-registry".

    The repo_id is used to:
    - Uniquely identify packages
    - Generate local metadata filenames (replacing '/' with '--')
    - Construct GitHub URLs when needed

    Usage:
        registry = Registry()
        packages = registry.list_installed_packages()
        registry.install_package("nodetool/my-package")
    """

    def __init__(self, registry_url: str = REGISTRY_URL):
        """
        Initialize the registry manager.

        Args:
            registry_url: URL of the remote package registry.
                         Defaults to the official nodetool package registry.
        """
        self.registry_url = registry_url
        self.packages_dir = get_packages_dir()
        self.pkg_mgr = get_package_manager_command()

    def _get_package_filename(self, repo_id: str) -> str:
        """
        Get the filename for a package's metadata file.

        Args:
            repo_id: Repository ID in the format <owner>/<project>

        Returns:
            str: The filename for the package's metadata file
        """
        owner, project = repo_id.split("/")
        return f"{project}.json"

    def list_installed_packages(self) -> List[PackageModel]:
        """
        List all installed packages.

        This method scans the local packages directory for YAML metadata files
        and loads them into PackageModel instances.

        Returns:
            List[PackageModel]: A list of PackageModel instances representing installed packages

        Note:
            Errors loading individual package metadata files are logged but don't stop the process.
        """
        packages = []

        for yaml_file in self.packages_dir.glob("*.json"):
            try:
                with open(yaml_file, "r") as f:
                    package_data = json.load(f)
                    packages.append(PackageModel(**package_data))
            except Exception as e:
                print(f"Error loading package metadata from {yaml_file}: {e}")

        return packages

    def list_available_packages(self) -> List[PackageInfo]:
        """
        List all available packages from the registry.

        This method fetches the package index from the remote registry
        and returns the list of available packages.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing package information
                                 from the registry

        Note:
            Returns an empty list if the registry cannot be reached or the index cannot be parsed.
        """
        response = requests.get(self.registry_url)
        response.raise_for_status()
        packages_data = response.json()["packages"]

        return [PackageInfo(**package) for package in packages_data]

    def get_package_metadata(
        self,
        repo_id: str,
    ) -> Optional[PackageModel]:
        """
        Get metadata for a specific package from the local package registry.

        Args:
            repo_id: Repository ID in the format <owner>/<project>

        Returns:
            PackageModel: Package metadata if found
            None: If package metadata could not be found or loaded
        """
        # Validate repo_id format
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            raise ValueError(error_msg)

        package_file = self.packages_dir / self._get_package_filename(repo_id)

        if package_file.exists():
            with open(package_file, "r") as f:
                package_data = json.load(f)
                return PackageModel(**package_data)
        else:
            return None

    def get_package_info(self, repo_id: str) -> Optional[PackageInfo]:
        """
        Get package info for a specific package from the local package registry.
        """
        all_packages = self.list_available_packages()
        for available_package in all_packages:
            if available_package.repo_id == repo_id:
                return available_package
        return None

    def install_package(self, repo_id: str) -> None:
        """
        Install a package by repository ID.

        This method:
        1. Validates the repo_id format
        2. Installs the package using the appropriate package manager
        3. Finds all node classes in the package and adds their metadata to the package
        4. Saves the metadata locally

        Args:
            repo_id: Repository ID in the format <owner>/<project>

        Returns:
            bool: True if installation was successful, False otherwise

        Note:
            If installation fails, any created metadata file is removed.
        """
        # Validate repo_id format
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            raise ValueError(error_msg)

        # find package info from registry
        package_info = self.get_package_info(repo_id)
        if not package_info:
            raise ValueError(f"Package {repo_id} not found in registry")

        # Install package via package manager
        try:
            # Use github_repo from package if available, otherwise construct from repo_id
            install_url = f"git+https://github.com/{repo_id}"
            subprocess.check_call([*self.pkg_mgr, "install", install_url])
            print(f"Successfully installed package {repo_id}")

            # Try to get metadata from the installed package
            package = get_package_metadata_from_pip(repo_id)
            if not package:
                raise ValueError(f"Failed to get pip metadata for {repo_id}")

            package_info = self.get_package_info(repo_id)
            if not package_info:
                raise ValueError(f"Failed to get package info for {repo_id}")

            package.namespaces = package_info.namespaces

            # Extract node metadata from the package
            package = extract_node_metadata_from_package(package, verbose=True)

            # Save updated package metadata with nodes
            package_file = self.packages_dir / self._get_package_filename(repo_id)
            with open(package_file, "w") as f:
                json.dump(package.model_dump(), f, indent=4, default=json_serializer)

            # We ensure package.nodes is never None in extract_node_metadata_from_package,
            # but the type checker doesn't know that, so we check again here
            node_count = len(package.nodes) if package.nodes is not None else 0
            print(f"Saved metadata for {node_count} nodes from package {repo_id}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing package {repo_id}: {e}")
            # Clean up metadata file if installation failed
            package_file = self.packages_dir / self._get_package_filename(repo_id)
            if package_file.exists():
                package_file.unlink()
            raise e

    def uninstall_package(self, repo_id: str) -> bool:
        """
        Uninstall a package by repository ID.

        This method:
        1. Validates the repo_id format
        2. Removes the local package metadata file
        3. Uninstalls the package using the appropriate package manager

        Args:
            repo_id: Repository ID in the format <owner>/<project>

        Returns:
            bool: True if uninstallation was successful, False otherwise

        Note:
            Returns False if the package is not installed or if uninstallation fails.
        """
        # Validate repo_id format
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            raise ValueError(error_msg)

        package = self.get_package_metadata(repo_id)

        if not package:
            print(f"Package {repo_id} not installed")
            return False

        # Remove package metadata
        package_file = self.packages_dir / self._get_package_filename(repo_id)
        if package_file.exists():
            package_file.unlink()

        # Uninstall package via package manager
        try:
            subprocess.check_call([*self.pkg_mgr, "uninstall", "-y", package.name])
            print(f"Successfully uninstalled package {repo_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error uninstalling package {repo_id}: {e}")
            return False

    def update_package(self, repo_id: str) -> bool:
        """
        Update a package to the latest version.

        This method:
        1. Validates the repo_id format
        2. Checks if the package is installed
        3. Updates the package using the appropriate package manager
        4. Updates the local package metadata
        5. Updates the node metadata for all nodes in the package

        Args:
            repo_id: Repository ID in the format <owner>/<project>

        Returns:
            bool: True if update was successful, False otherwise

        Note:
            Returns False if the package is not installed or if the update fails.
        """
        # Validate repo_id format
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            raise ValueError(error_msg)

        package = self.get_package_metadata(repo_id)

        if not package:
            print(f"Package {repo_id} not installed")
            return False

        # Update package via package manager
        try:
            # Use github_repo from package if available, otherwise construct from repo_id
            install_url = f"https://github.com/{repo_id}"
            subprocess.check_call([*self.pkg_mgr, "install", "--upgrade", install_url])

            # Try to get updated metadata from GitHub
            updated_package = get_package_metadata_from_github(install_url)
            if not updated_package:
                updated_package = package  # Fall back to existing metadata

            # Extract node metadata from the package
            updated_package = extract_node_metadata_from_package(
                updated_package, verbose=True
            )

            # Save updated package metadata
            package_file = self.packages_dir / self._get_package_filename(repo_id)
            with open(package_file, "w") as f:
                yaml.dump(updated_package.model_dump(), f)

            # We ensure updated_package.nodes is never None in extract_node_metadata_from_package,
            # but the type checker doesn't know that, so we check again here
            node_count = (
                len(updated_package.nodes) if updated_package.nodes is not None else 0
            )
            print(f"Successfully updated package {repo_id} with {node_count} nodes")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error updating package {repo_id}: {e}")
            return False


def get_package_metadata_from_pip(repo_id: str) -> Optional[PackageModel]:
    """
    Get package metadata from an installed pip package.

    This function uses pip's metadata API to extract information about an installed package
    and converts it to a PackageModel instance.

    Args:
        repo_id: Repository ID in the format <owner>/<project>

    Returns:
        PackageModel: A populated PackageModel instance if successful
        None: If metadata could not be retrieved or parsed
    """
    # Use pip's metadata API to get package information
    import importlib.metadata as metadata

    is_valid, error_msg = validate_repo_id(repo_id)
    if not is_valid:
        raise ValueError(error_msg)

    owner, project = repo_id.split("/")

    # Get package distribution
    dist = metadata.distribution(project)

    # Extract metadata
    metadata_dict = {k: v for k, v in dist.metadata.items()}  # type: ignore

    # Parse author information
    authors = []
    if "Author" in metadata_dict:
        authors.append(metadata_dict["Author"])
    elif "Author-email" in metadata_dict:
        authors.append(metadata_dict["Author-email"])

    return PackageModel(
        name=project,
        description=metadata_dict.get("Summary", ""),
        version=metadata_dict.get("Version", "0.1.0"),
        authors=authors,
        namespaces=[],
        repo_id=repo_id,
    )
