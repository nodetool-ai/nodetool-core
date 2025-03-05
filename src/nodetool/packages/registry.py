"""
Package registry module for nodetool application.

This module provides functionality for managing node packages in the nodetool ecosystem.
It handles:
- Installing packages from GitHub repositories or local paths
- Uninstalling packages
- Listing installed and available packages
- Updating packages to their latest versions

Key components:
- PackageInfo: Model for package information in the registry index
- PackageModel: Package metadata model imported from nodetool.metadata.node_metadata
- Registry: Package registry manager that handles all package operations
- Helper functions:
  - validate_repo_id: Validates repository IDs in owner/project format
  - get_package_metadata_from_github: Extracts metadata from GitHub repository
  - get_package_metadata_from_pip: Extracts metadata from installed pip packages
  - discover_node_packages: Scans for installed packages in nodetool.nodes namespace

Package management:
- Uses either uv or pip as the package manager (preferring uv if available)
- Packages are identified by their repository ID (repo_id) in the format <owner>/<project>
- Package metadata is retrieved from pyproject.toml files and nodes.json
- Available packages are listed from a central registry at REGISTRY_URL

Usage:
    registry = Registry()
    packages = registry.list_installed_packages()
    registry.install_package("owner/project")
    registry.uninstall_package("owner/project")
    registry.update_package("owner/project")
"""

from enum import Enum
import json
import os
import subprocess
import requests
import tomli
import re
import pkgutil
import importlib
import importlib.metadata
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
    Simplified package registry manager that works with Python's package system.

    Packages are discovered through the nodetool.nodes namespace and metadata
    is stored in nodes.json files within each package.
    """

    def __init__(self):
        self.pkg_mgr = get_package_manager_command()

    def pip_install(
        self, install_path: str, editable: bool = False, upgrade: bool = False
    ) -> None:
        """
        Call the pip install command.
        """
        if upgrade:
            subprocess.check_call([*self.pkg_mgr, "install", "--upgrade", install_path])
        elif editable:
            subprocess.check_call([*self.pkg_mgr, "install", "-e", install_path])
        else:
            subprocess.check_call([*self.pkg_mgr, "install", install_path])

    def pip_uninstall(self, package_name: str) -> None:
        """
        Call the pip uninstall command.
        """
        subprocess.check_call([*self.pkg_mgr, "uninstall", package_name, "--yes"])

    def list_installed_packages(self) -> List[PackageModel]:
        """List all installed node packages."""
        return discover_node_packages()

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
        response = requests.get(REGISTRY_URL)
        response.raise_for_status()
        packages_data = response.json()["packages"]

        return [PackageInfo(**package) for package in packages_data]

    def install_package(self, repo_id: str, local_path: Optional[str] = None) -> None:
        """Install a package by repository ID or from local path."""
        if local_path:
            self.pip_install(local_path, editable=True)
        else:
            install_path = f"git+https://github.com/{repo_id}"
            self.pip_install(install_path)

    def uninstall_package(self, repo_id: str) -> bool:
        """Uninstall a package by repository ID."""
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            raise ValueError(error_msg)

        _, project = repo_id.split("/")
        try:
            self.pip_uninstall(project)
            print(f"Successfully uninstalled package {repo_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error uninstalling package {repo_id}: {e}")
            return False

    def update_package(self, repo_id: str) -> bool:
        """Update a package to the latest version."""
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            raise ValueError(error_msg)

        try:
            install_url = f"git+https://github.com/{repo_id}"
            self.pip_install(install_url, upgrade=True)
            print(f"Successfully updated package {repo_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error updating package {repo_id}: {e}")
            return False


def discover_node_packages() -> list[PackageModel]:
    """
    Discover all installed node packages by finding packages that start with 'nodetool-'.

    This function:
    1. Gets all installed packages using importlib.metadata
    2. Filters for packages starting with 'nodetool-'
    3. Creates PackageModel instances for each discovered package
    4. Handles both regular installations and editable installs

    Returns:
        List[PackageModel]: List of discovered packages with their node metadata
    """
    packages = []
    import sys

    # First try to get the package's development location (for editable installs)
    for path in sys.path:
        if "nodetool-" in path:
            package_path = Path(path) / "nodetool" / "package_metadata"
            metadata_files = list(package_path.glob("*.json"))
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        packages.append(PackageModel(**metadata))
                except Exception as e:
                    print(f"Error processing {metadata_file}: {e}")

    # Get all installed distributions
    for dist in importlib.metadata.distributions():
        package_name = dist.metadata["Name"]
        if not package_name.startswith("nodetool-"):
            continue

        # If no dev location found, try site-packages
        base_path = str(dist.locate_file("nodetool/package"))
        metadata_files = list(Path(base_path).glob("*.json"))

        for metadata_file in metadata_files:
            with open(metadata_file) as f:
                try:
                    metadata = json.load(f)
                    packages.append(PackageModel(**metadata))
                except Exception as e:
                    print(f"Error processing {metadata_file}: {e}")

    return packages


# def discover_node_packages() -> list[PackageModel]:
#     """
#     Discover all installed node packages by finding packages that start with 'nodetool-'.

#     This function:
#     1. Gets all installed packages using importlib.metadata
#     2. Filters for packages starting with 'nodetool-'
#     3. Creates PackageModel instances for each discovered package
#     4. Handles both regular installations and editable installs

#     Returns:
#         List[PackageModel]: List of discovered packages with their node metadata
#     """
#     packages = []
#     import sys

#     # First try to get the package's development location (for editable installs)
#     for path in sys.path:
#         if "nodetool-" in path:
#             nodes_path = Path(path) / "nodetool" / "nodes"
#             metadata_files = list(nodes_path.glob("**/nodes.json"))
#             for metadata_file in metadata_files:
#                 try:
#                     with open(metadata_file) as f:
#                         metadata = json.load(f)
#                         packages.append(PackageModel(**metadata))
#                 except Exception as e:
#                     print(f"Error processing {metadata_file}: {e}")

#     # Get all installed distributions
#     for dist in importlib.metadata.distributions():
#         package_name = dist.metadata["Name"]
#         if not package_name.startswith("nodetool-"):
#             continue

#         # If no dev location found, try site-packages
#         base_path = str(dist.locate_file("nodetool/nodes"))
#         metadata_files = list(Path(base_path).glob("**/nodes.json"))

#         for metadata_file in metadata_files:
#             with open(metadata_file) as f:
#                 try:
#                     metadata = json.load(f)
#                     packages.append(PackageModel(**metadata))
#                 except Exception as e:
#                     print(f"Error processing {metadata_file}: {e}")

#     return packages
