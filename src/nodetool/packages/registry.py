"""
Package registry module for nodetool application.

This module provides functionality for managing node packages in the nodetool ecosystem.
It handles:
- Installing packages from GitHub repositories or local paths
- Uninstalling packages
- Listing installed and available packages
- Updating packages to their latest versions
- Managing example workflows from installed packages
- Managing package-provided assets from installed packages

Key components:
- PackageInfo: Model for package information in the registry index
- PackageModel: Package metadata model imported from nodetool.metadata.node_metadata
- Registry: Unified package registry manager that handles all package operations,
  including examples and assets
- AssetInfo: Model for asset information from package-provided assets
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

Example and asset management:
- Registry provides functionality to discover, load, and manage example workflows and assets
- Examples and assets are stored in 'examples' and 'assets' directories within each package, respectively
- Supports caching of examples and assets for better performance
- Allows searching examples by ID or name, and assets by file name
- Provides ability to save new examples (in development mode only)

Usage:
    registry = Registry()
    packages = registry.list_installed_packages()
    registry.install_package("owner/project")
    registry.uninstall_package("owner/project")
    registry.update_package("owner/project")

    examples = registry.list_examples()
    example = registry.find_example_by_name("example_name")
"""

from enum import Enum
import json
import os
import subprocess
import requests
import tomli
import re
import importlib
import importlib.metadata
from nodetool.config.logging_config import get_logger
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from urllib.parse import urlparse
import httpx
import asyncio
import click
import tomlkit

from nodetool.config.environment import Environment
from nodetool.config.settings import get_system_file_path
from nodetool.metadata.node_metadata import NodeMetadata, PackageModel, ExampleMetadata
from nodetool.packages.types import AssetInfo, PackageInfo
from nodetool.types.workflow import Workflow
from nodetool.types.graph import Graph as APIGraph
from nodetool.workflows.base_node import get_node_class, split_camel_case


# Constants
PACKAGES_DIR = "packages"
# New wheel-based package index (PEP 503 compliant)
PACKAGE_INDEX_URL = "https://nodetool-ai.github.io/nodetool-registry/simple/"
# Legacy JSON registry for backward compatibility
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
            - Booleae indicating if the repo_id is valid
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
    extracts package metadata using PEP 621 `[project]` metadata.

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

    response = requests.get(
        raw_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=30,
    )
    response.raise_for_status()
    pyproject_data = tomli.loads(response.text)

    # Extract metadata (PEP 621 only)
    project_data = pyproject_data.get("project", {})
    if not project_data:
        raise ValueError(
            f"No PEP 621 [project] metadata found in pyproject.toml for {github_repo}"
        )

    # Authors can be list of tables
    raw_authors = project_data.get("authors", [])
    authors: list[str] = []
    if (
        isinstance(raw_authors, list)
        and raw_authors
        and isinstance(raw_authors[0], dict)
    ):
        for a in raw_authors:
            name = a.get("name")
            email = a.get("email")
            if name and email:
                authors.append(f"{name} <{email}>")
            elif name:
                authors.append(str(name))
            elif email:
                authors.append(str(email))
    elif isinstance(raw_authors, list):
        authors = [str(a) for a in raw_authors]

    return PackageModel(
        name=project_data.get("name", github_repo),
        description=project_data.get("description", ""),
        version=project_data.get("version", "0.1.0"),
        authors=authors,
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


class Registry:
    """
    Unified package registry manager that works with Python's package system.

    This class combines functionality for managing packages, examples, and assets.
    Packages are discovered through the nodetool.nodes namespace and metadata
    is stored in nodes.json files within each package.
    """

    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Registry()
        return cls._instance

    def __init__(self):
        self.pkg_mgr = get_package_manager_command()
        self._node_cache = None  # Cache for node metadata
        self._packages_cache = None  # Cache for installed packages
        self._examples_cache = (
            {}
        )  # Cache for loaded examples by package_name:example_name
        self._example_search_cache: Optional[Dict[str, Any]] = None
        self._index_available = None  # Cache for package index availability
        self.logger = get_logger(__name__)

    def pip_install(
        self,
        install_path: str,
        editable: bool = False,
        upgrade: bool = False,
        use_index: bool = True,
    ) -> None:
        """
        Call the pip install command with optional wheel index support.

        Args:
            install_path: Package name or path to install
            editable: Install in editable mode
            upgrade: Upgrade to latest version
            use_index: Use the NodeTool package index for wheel-based installation
        """
        cmd = [*self.pkg_mgr, "install"]

        # Add package index if using wheel-based installation
        # Skip index for: paths, git URLs, file URLs, editable installs
        if (
            use_index
            and not editable
            and not install_path.startswith(
                ("/", ".", "git+", "file://", "http://", "https://")
            )
        ):
            cmd.extend(["--index-url", PACKAGE_INDEX_URL])

        if upgrade:
            cmd.append("--upgrade")
        if editable:
            cmd.append("-e")

        cmd.append(install_path)
        subprocess.check_call(cmd)

    def pip_uninstall(self, package_name: str) -> None:
        """
        Call the pip uninstall command.
        """
        # Always use standard pip for uninstall operations, not uv
        subprocess.check_call(["pip", "uninstall", package_name, "--yes"])

    def list_installed_packages(self) -> List[PackageModel]:
        """List all installed node packages."""
        if self._packages_cache is None:
            self._packages_cache = discover_node_packages()
        return self._packages_cache

    def find_package_by_name(self, name: str) -> Optional[PackageModel]:
        """Find a package by name."""
        return next(
            (
                package
                for package in self.list_installed_packages()
                if package.name == name
            ),
            None,
        )

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
        response = requests.get(
            REGISTRY_URL,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=30,
        )
        response.raise_for_status()
        packages_data = response.json()["packages"]

        return [PackageInfo(**package) for package in packages_data]

    def check_package_index_available(self) -> bool:
        """Check if the wheel-based package index is available.

        Returns:
            bool: True if the package index is reachable, False otherwise
        """
        if self._index_available is not None:
            return self._index_available

        try:
            response = requests.get(
                PACKAGE_INDEX_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                timeout=10,
            )
            self._index_available = response.status_code == 200
            if self._index_available:
                self.logger.info(f"Package index available at {PACKAGE_INDEX_URL}")
            else:
                self.logger.warning(
                    f"Package index returned status {response.status_code}"
                )
        except Exception as e:
            self._index_available = False
            self.logger.warning(f"Package index not available: {e}")

        return self._index_available

    def get_install_command_for_package(self, repo_id: str) -> str:
        """Get the recommended install command for a package.

        Args:
            repo_id: Repository ID in format owner/project

        Returns:
            str: The pip install command string
        """
        package_name = repo_id.split("/")[1]

        if self.check_package_index_available():
            return f"pip install --index-url {PACKAGE_INDEX_URL} {package_name}"
        else:
            return f"pip install git+https://github.com/{repo_id}"

    def get_package_installation_info(self, repo_id: str) -> Dict[str, Any]:
        """Get comprehensive installation information for a package.

        Args:
            repo_id: Repository ID in format owner/project

        Returns:
            Dict containing installation methods and recommendations
        """
        package_name = repo_id.split("/")[1]
        index_available = self.check_package_index_available()

        return {
            "package_name": package_name,
            "repo_id": repo_id,
            "wheel_available": index_available,
            "recommended_command": self.get_install_command_for_package(repo_id),
            "wheel_command": f"pip install --index-url {PACKAGE_INDEX_URL} {package_name}",
            "git_command": f"pip install git+https://github.com/{repo_id}",
            "package_index_url": PACKAGE_INDEX_URL,
        }

    def install_package(
        self, repo_id: str, local_path: Optional[str] = None, use_git: bool = False
    ) -> None:
        """Install a package by repository ID or from local path.

        Args:
            repo_id: Repository ID in format owner/project (e.g., 'nodetool-ai/nodetool-base')
            local_path: Local path for editable install
            use_git: Force git-based installation instead of wheel-based
        """
        if local_path:
            self.pip_install(local_path, editable=True, use_index=False)
        elif use_git:
            # Fallback to git-based installation
            install_path = f"git+https://github.com/{repo_id}"
            self.pip_install(install_path, use_index=False)
        else:
            # Try wheel-based installation first, fallback to git if it fails
            package_name = repo_id.split("/")[1]

            try:
                if self.check_package_index_available():
                    self.logger.info(f"Installing {package_name} from wheel index")
                    self.pip_install(package_name, use_index=True)
                else:
                    raise Exception("Package index not available")
            except (subprocess.CalledProcessError, Exception) as e:
                self.logger.warning(
                    f"Wheel installation failed: {e}. Falling back to git installation."
                )
                install_path = f"git+https://github.com/{repo_id}"
                self.pip_install(install_path, use_index=False)

        # Clear the cache since we've installed a new package
        self.clear_packages_cache()

    def uninstall_package(self, repo_id: str) -> bool:
        """Uninstall a package by repository ID."""
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            raise ValueError(error_msg)

        _, project = repo_id.split("/")
        try:
            self.pip_uninstall(project)
            print(f"Successfully uninstalled package {repo_id}")
            # Clear the cache since we've uninstalled a package
            self.clear_packages_cache()
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error uninstalling package {repo_id}: {e}")
            return False

    def update_package(self, repo_id: str, use_git: bool = False) -> bool:
        """Update a package to the latest version.

        Args:
            repo_id: Repository ID in format owner/project
            use_git: Force git-based update instead of wheel-based
        """
        is_valid, error_msg = validate_repo_id(repo_id)
        if not is_valid:
            raise ValueError(error_msg)

        try:
            if use_git:
                # Fallback to git-based update
                install_url = f"git+https://github.com/{repo_id}"
                self.pip_install(install_url, upgrade=True, use_index=False)
            else:
                # Try wheel-based update first, fallback to git if it fails
                package_name = repo_id.split("/")[1]

                try:
                    if self.check_package_index_available():
                        self.logger.info(f"Updating {package_name} from wheel index")
                        self.pip_install(package_name, upgrade=True, use_index=True)
                    else:
                        raise Exception("Package index not available")
                except (subprocess.CalledProcessError, Exception) as e:
                    self.logger.warning(
                        f"Wheel update failed: {e}. Falling back to git update."
                    )
                    install_url = f"git+https://github.com/{repo_id}"
                    self.pip_install(install_url, upgrade=True, use_index=False)

            print(f"Successfully updated package {repo_id}")
            # Clear the cache since we've updated a package
            self.clear_packages_cache()
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error updating package {repo_id}: {e}")
            return False

    async def search_nodes(self, query: str = "") -> List[Dict[str, Any]]:
        """
        Search for nodes across all available packages asynchronously.

        This method fetches node metadata from all available packages in parallel and
        filters them based on the provided query string. The results are
        cached for subsequent searches.

        Args:
            query: Optional search string to filter nodes by name or description
                  If empty, returns all available nodes

        Returns:
            List[Dict[str, Any]]: A list of node metadata dictionaries matching the query
        """
        # Use cached nodes if available
        if self._node_cache is None:
            self._node_cache = await self._fetch_all_nodes_async()

        # If no query, return all nodes
        if not query:
            return self._node_cache

        # Filter nodes based on query (case-insensitive)
        query = query.lower()
        return [
            node
            for node in self._node_cache
            if query in node.get("name", "").lower()
            or query in node.get("description", "").lower()
        ]

    async def _fetch_all_nodes_async(self) -> List[Dict[str, Any]]:
        """
        Fetch node metadata from all available packages asynchronously.

        This method:
        1. Gets all available packages from the registry
        2. For each package, fetches the package metadata JSON file in parallel
        3. Extracts node information from each metadata file

        Returns:
            List[Dict[str, Any]]: A list of node metadata dictionaries
        """
        all_nodes = []
        available_packages = await asyncio.to_thread(self.list_available_packages)

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create tasks for all package metadata fetches
            tasks = []
            for package in available_packages:
                # Construct the URL to the package metadata file
                package_name = package.repo_id.split("/")[1]
                metadata_url = f"https://raw.githubusercontent.com/{package.repo_id}/main/src/nodetool/package_metadata/{package_name}.json"
                tasks.append(
                    self._fetch_package_nodes(client, metadata_url, package.repo_id)
                )

            # Run all tasks in parallel and gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for package_nodes in results:
                if isinstance(package_nodes, Exception):
                    # Skip failed requests
                    continue
                all_nodes.extend(package_nodes)  # type: ignore

        return all_nodes

    async def _fetch_package_nodes(
        self, client: httpx.AsyncClient, url: str, repo_id: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch node metadata for a single package.

        Args:
            client: httpx AsyncClient to use for the request
            url: URL to the package metadata file
            repo_id: Repository ID of the package

        Returns:
            List[Dict[str, Any]]: List of node metadata dictionaries for the package

        Raises:
            Exception: If the request fails or the response cannot be parsed
        """
        try:
            response = await client.get(url)
            response.raise_for_status()

            # Parse the node metadata
            package_metadata = response.json()
            package_nodes = package_metadata.get("nodes", [])

            # Add package information to each node
            for node in package_nodes:
                node["package"] = repo_id

            return package_nodes

        except (httpx.RequestError, json.JSONDecodeError) as e:
            print(f"Error fetching nodes from {repo_id}: {e}")
            raise

    async def get_package_for_node_type(self, node_type: str) -> Optional[str]:
        """
        Get the package that provides a specific node type.

        Args:
            node_type: The type identifier of the node

        Returns:
            Optional[str]: The repository ID of the package providing the node,
                          or None if the node type is not found
        """
        # Ensure the node cache is populated
        if self._node_cache is None:
            self._node_cache = await self._fetch_all_nodes_async()

        # Search for the node type in the cache
        for node in self._node_cache:
            if node.get("node_type") == node_type:
                return node.get("package")

        return None

    def clear_node_cache(self) -> None:
        """
        Clear the node metadata cache.

        This forces the next search to fetch fresh node metadata.
        """
        self._node_cache = None

    def get_all_installed_nodes(self) -> list[NodeMetadata]:
        """
        Get all nodes from installed packages only.

        This is a synchronous method that only looks at locally installed packages
        without making any network requests.

        Returns:
            List[Dict[str, Any]]: A list of node metadata dictionaries from installed packages
        """
        all_nodes = []
        installed_packages = self.list_installed_packages()

        for package in installed_packages:
            if package.nodes:
                all_nodes.extend(package.nodes)

        return all_nodes

    def find_node_by_type(self, node_type: str) -> Optional[Dict[str, Any]]:
        """
        Find a node by its type identifier from installed packages.

        This is a synchronous method that searches through installed packages only.

        Args:
            node_type: The type identifier of the node (e.g., "namespace.ClassName")

        Returns:
            Optional[Dict[str, Any]]: The node metadata if found, None otherwise
        """
        installed_packages = self.list_installed_packages()

        for package in installed_packages:
            if package.nodes:
                for node in package.nodes:
                    node_dict = (
                        node.model_dump() if hasattr(node, "model_dump") else dict(node)
                    )
                    if node_dict.get("node_type") == node_type:
                        node_dict["package"] = package.repo_id
                        node_dict["installed"] = True
                        return node_dict

        return None

    def clear_packages_cache(self) -> None:
        """
        Clear the installed packages cache.

        This forces the next call to list_installed_packages to re-discover packages.
        """
        self._packages_cache = None

    def clear_examples_cache(self) -> None:
        """
        Clear the loaded examples cache.

        This forces the next call to load_example to re-load from disk.
        """
        self._examples_cache = {}
        self._example_search_cache = None

    def clear_index_cache(self) -> None:
        """Clear the package index availability cache."""
        self._index_available = None

    def _populate_example_search_cache(self) -> None:
        """
        Load necessary example data into an in-memory cache for searching.
        """
        if self._example_search_cache is not None:
            return

        self.logger.info("Populating example workflow search cache...")
        self._example_search_cache = {}
        packages = self.list_installed_packages()

        for package in packages:
            if not package.examples:
                continue

            for example_meta in package.examples:
                if not package.source_folder:
                    continue

                example_path = (
                    Path(package.source_folder)
                    / "nodetool"
                    / "examples"
                    / package.name
                    / f"{example_meta.name}.json"
                )

                if not example_path.exists():
                    continue

                with open(example_path, "r", encoding="utf-8") as f:
                    try:
                        workflow_data = json.load(f)

                        # Extract and cache node types for search
                        graph_data = workflow_data.get("graph", {})
                        nodes = graph_data.get("nodes", [])
                        node_types = []
                        node_titles = []

                        for node in nodes:
                            node_type = node.get("type", "")
                            node_types.append(node_type.lower())

                            # Try to get the node class to extract its title
                            try:
                                from nodetool.workflows.base_node import get_node_class

                                node_class = get_node_class(node_type)
                                if node_class:
                                    node_titles.append(node_class.get_title().lower())
                            except Exception:
                                # If we can't get the node class, skip title extraction
                                pass

                        # Store only essential data for search
                        cached_item = {
                            "id": workflow_data.get("id"),  # For deduplication
                            "_node_types": node_types,  # For searching
                            "_node_titles": node_titles,  # For searching by title
                        }

                        cache_key = f"{package.name}:{example_meta.name}"
                        self._example_search_cache[cache_key] = cached_item
                    except json.JSONDecodeError:
                        self.logger.warning(
                            f"Skipping corrupted example JSON: {example_path}"
                        )
                        continue
        self.logger.info(
            f"Cached {len(self._example_search_cache)} example workflows for search."
        )

    def _load_example_from_file(self, file_path: str, package_name: str) -> Workflow:
        """
        Load a single example workflow from a JSON file.

        Args:
            file_path: The full path to the example workflow JSON file
            package_name: Name of the package providing this example

        Returns:
            ExampleWorkflow: The loaded example workflow with metadata
        """
        try:
            with open(file_path, "r") as f:
                props = json.load(f)
                props["package_name"] = package_name
                if not Environment.is_production():
                    props["path"] = file_path
                workflow = Workflow(**props)
                return workflow
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Error decoding JSON for example workflow {file_path}: {e}"
            )
            # Return an empty Workflow with the name indicating it is broken
            now_str = datetime.now().isoformat()
            return Workflow(
                id="",
                name=f"[ERROR] {os.path.basename(file_path)}",
                tags=[],
                graph=APIGraph(nodes=[], edges=[]),
                access="",
                created_at=now_str,
                updated_at=now_str,
                description=f"Error loading this workflow: {str(e)}",
                package_name=package_name,
                path=file_path,
            )

    def _load_examples_from_directory(
        self, directory: str, package_name: str
    ) -> List[ExampleMetadata]:
        """
        Load all example workflows from a directory.

        Args:
            directory: The directory containing example workflow JSON files
            package_name: Name of the package providing these examples

        Returns:
            List[Workflow]: A list of all loaded example workflows from the directory
        """
        if not os.path.exists(directory):
            self.logger.warning(f"Examples directory does not exist: {directory}")
            return []

        # Define the package-specific examples directory
        package_dir = os.path.join(directory, package_name)
        os.makedirs(package_dir, exist_ok=True)

        if not os.path.exists(package_dir):
            self.logger.warning(
                f"Package examples directory does not exist: {package_dir}"
            )
            return []

        examples = []
        for name in os.listdir(package_dir):
            # Skip files starting with underscore (_) and non-JSON files
            if name.startswith("_") or not name.endswith(".json"):
                continue

            file_path = os.path.join(package_dir, name)
            workflow = self._load_example_from_file(file_path, package_name)
            examples.append(
                ExampleMetadata(
                    id=workflow.id,
                    name=workflow.name,
                    description=workflow.description,
                    tags=workflow.tags or [],
                )
            )

        return examples

    def clear_cache(self) -> None:
        """Clear all caches (packages, nodes, examples, and index) to force fresh data on next calls."""
        self.clear_packages_cache()
        self.clear_node_cache()
        self.clear_examples_cache()
        self.clear_index_cache()

    def list_examples(self) -> List[Workflow]:
        """
        List all example workflows from installed packages.

        This method retrieves example metadata from installed packages and
        converts them to Workflow objects without scanning the filesystem.

        Returns:
            List[Workflow]: A list of all example workflows
        """
        examples = []
        packages = self.list_installed_packages()

        for package in packages:
            if package.examples:
                for example_meta in package.examples:
                    # Create Workflow from ExampleMetadata
                    now_str = datetime.now().isoformat()
                    workflow = Workflow(
                        id=example_meta.id,
                        name=example_meta.name,
                        description=example_meta.description,
                        tags=example_meta.tags or [],
                        graph=APIGraph(
                            nodes=[], edges=[]
                        ),  # Empty graph as we don't load the full workflow
                        access="public",
                        created_at=now_str,
                        updated_at=now_str,
                        package_name=package.name,
                        path=None,  # Path not available from metadata
                    )
                    examples.append(workflow)
        return examples

    def find_example_by_id(self, id: str) -> Optional[Workflow]:
        """
        Find an example workflow by its ID.

        Args:
            id: The ID of the workflow to find

        Returns:
            Optional[Workflow]: The found workflow or None if not found
        """
        examples = self.list_examples()
        example = next((ex for ex in examples if ex.id == id), None)
        return example if example else None

    def find_example_by_name(self, name: str) -> Optional[Workflow]:
        """
        Find an example workflow by its name.

        Args:
            name: The name of the workflow to find

        Returns:
            Optional[Workflow]: The found workflow or None if not found
        """
        examples = self.list_examples()
        example = next((ex for ex in examples if ex.name == name), None)
        return example if example else None

    def save_example(self, workflow: Workflow) -> Workflow:
        """
        Save a workflow as an example in the specified package.

        Args:
            workflow: The workflow object to save

        Returns:
            Workflow: The saved workflow

        Raises:
            ValueError: If the package_name is invalid or the package is not found

        Note:
            This function removes the user_id field before saving and
            invalidates the cached examples.
        """
        if Environment.is_production():
            raise ValueError("Saving examples is only allowed in dev mode")

        assert workflow.package_name is not None

        package = self.find_package_by_name(workflow.package_name)
        if not package:
            raise ValueError(f"Package {workflow.package_name} not found")

        src_folders = get_nodetool_package_source_folders()

        package_folder = next(
            (
                src_folder
                for src_folder in src_folders
                if package.name in str(src_folder)
            ),
            None,
        )

        if not package_folder:
            raise ValueError(
                f"Package {workflow.package_name} not found in source editable folders"
            )

        path = (
            package_folder
            / "nodetool"
            / "examples"
            / package.name
            / f"{workflow.name}.json"
        )
        os.makedirs(path.parent, exist_ok=True)

        # Find the package folder
        with open(path, "w") as f:
            json.dump(workflow.model_dump(), f, indent=2)

        # Invalidate the cached examples
        self.clear_examples_cache()

        return workflow

    def _load_assets_from_directory(
        self, directory: str, package_name: str
    ) -> List[AssetInfo]:
        """
        Load all asset files from a directory.

        Args:
            directory: The directory containing asset files
            package_name: Name of the package providing these assets

        Returns:
            List[AssetInfo]: A list of asset information objects
        """
        if not os.path.exists(directory):
            return []

        # Define the package-specific assets directory
        package_dir = os.path.join(directory, package_name)
        os.makedirs(package_dir, exist_ok=True)

        if not os.path.exists(package_dir):
            return []

        assets: List[AssetInfo] = []
        for name in os.listdir(package_dir):
            if name.startswith("_"):
                continue
            assets.append(AssetInfo(package_name=package_name, name=name, path=""))

        return assets

    def list_assets(self) -> List[AssetInfo]:
        """
        List all asset files from installed packages.

        This method retrieves asset metadata from installed packages
        without scanning the filesystem.

        Returns:
            List[AssetInfo]: A list of all asset files
        """
        assets = []
        packages = self.list_installed_packages()

        for package in packages:
            if package.assets:
                assets.extend(package.assets)

        return assets

    def find_asset_by_name(
        self, name: str, package_name: Optional[str] = None
    ) -> Optional[AssetInfo]:
        """
        Find an asset file by its file name, optionally filtering by package name.

        Args:
            name: The file name of the asset to find
            package_name: Optional package name to filter by

        Returns:
            Optional[AssetInfo]: The found asset or None if not found
        """
        if package_name:
            return next(
                (
                    asset
                    for asset in self.list_assets()
                    if asset.name == name and asset.package_name == package_name
                ),
                None,
            )
        else:
            return next(
                (asset for asset in self.list_assets() if asset.name == name),
                None,
            )

    def load_example(self, package_name: str, example_name: str) -> Optional[Workflow]:
        """
        Load a single example workflow from disk given package name and example name.

        This method uses the package's source folder to construct the path to the example file.
        Results are cached for performance.

        Args:
            package_name: The name of the package containing the example
            example_name: The name of the example workflow to load

        Returns:
            Optional[Workflow]: The loaded workflow with full data, or None if not found

        Raises:
            ValueError: If the package is not found
        """
        # Check cache first
        cache_key = f"{package_name}:{example_name}"
        if cache_key in self._examples_cache:
            return self._examples_cache[cache_key]

        package = self.find_package_by_name(package_name)
        if not package:
            raise ValueError(f"Package {package_name} not found")

        if not package.source_folder:
            raise ValueError(f"Package {package_name} does not have a source folder")

        # Construct the path to the example file
        # Examples are stored in: source_folder/nodetool/examples/package_name/example_name.json
        example_path = (
            Path(package.source_folder)
            / "nodetool"
            / "examples"
            / package_name
            / f"{example_name}.json"
        )

        if not example_path.exists():
            self._examples_cache[cache_key] = None  # Cache the None result too
            return None

        workflow = self._load_example_from_file(str(example_path), package_name)
        self._examples_cache[cache_key] = workflow
        return workflow

    def search_example_workflows(self, query: str = "") -> List[Workflow]:
        """
        Search for example workflows that contain nodes matching the query.

        This method searches through node types to find workflows that use specific nodes.
        The search is optimized using a lightweight in-memory cache that stores only
        node types and workflow IDs.

        Args:
            query: The search string to find in node types

        Returns:
            List[Workflow]: A list of workflows that contain nodes with types matching the query
        """
        if not query:
            return self.list_examples()

        self._populate_example_search_cache()

        matching_workflows = []
        query = query.lower()

        self.logger.info(f"Searching for query: '{query}'")

        # To avoid adding the same workflow multiple times
        matched_workflow_ids = set()

        if self._example_search_cache is None:
            self.logger.warning("Search cache is not populated.")
            return []

        for cache_key, workflow_data in self._example_search_cache.items():
            workflow_id = workflow_data.get("id", "")
            if workflow_id and workflow_id in matched_workflow_ids:
                continue

            node_types = workflow_data.get("_node_types", [])
            node_titles = workflow_data.get("_node_titles", [])
            found_match = any(query in node_type for node_type in node_types) or any(
                query in title for title in node_titles
            )

            if found_match:
                self.logger.info(f"Found match in workflow '{cache_key}'")
                package_name, example_name = cache_key.split(":", 1)
                workflow = self.load_example(package_name, example_name)
                if workflow:
                    matching_workflows.append(workflow)
                    if workflow.id:
                        matched_workflow_ids.add(workflow.id)

        self.logger.info(f"Found {len(matching_workflows)} matching workflows.")
        return matching_workflows


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
    seen_names: set[str] = set()
    for path in sys.path:
        if "nodetool-" in path:
            base_path = Path(path)
            # Support both project-root on sys.path with src layout and direct src on sys.path
            candidate_dirs = [
                base_path / "nodetool" / "package_metadata",
                base_path / "src" / "nodetool" / "package_metadata",
            ]
            # If sys.path already points to a src folder, prefer that
            if base_path.name == "src":
                candidate_dirs.insert(0, base_path / "nodetool" / "package_metadata")

            for package_path in candidate_dirs:
                if not package_path.exists():
                    continue
                metadata_files = list(package_path.glob("*.json"))
                for metadata_file in metadata_files:
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            name = metadata.get("name")
                            if isinstance(name, str):
                                if name in seen_names:
                                    continue
                                seen_names.add(name)
                            # Prefer src folder if present for a cleaner source path
                            source_folder = (
                                str(base_path / "src")
                                if (base_path / "src").exists()
                                else str(base_path)
                            )
                            metadata["source_folder"] = source_folder
                            packages.append(PackageModel(**metadata))
                    except Exception as e:
                        print(f"Error processing {metadata_file}: {e}")
                # Avoid scanning both root and src when one already provided metadata
                if metadata_files:
                    break

    # Get all installed distributions
    visited_paths = set()
    for dist in importlib.metadata.distributions():
        package_name = dist.metadata["Name"]
        if not package_name.startswith("nodetool-"):
            continue
        # Skip if already discovered from editable location
        if package_name in seen_names:
            continue

        # If no dev location found, try site-packages
        base_path = str(dist.locate_file("nodetool/package_metadata"))
        if base_path in visited_paths:
            continue
        visited_paths.add(base_path)
        metadata_files = list(Path(base_path).glob("*.json"))

        for metadata_file in metadata_files:
            with open(metadata_file) as f:
                try:
                    metadata = json.load(f)
                    metadata["source_folder"] = str(Path(base_path).parent.parent)
                    pkg = PackageModel(**metadata)
                    packages.append(pkg)
                    if isinstance(pkg.name, str):
                        seen_names.add(pkg.name)
                except Exception as e:
                    print(f"Error processing {metadata_file}: {e}")

    return packages


def get_nodetool_package_source_folders() -> List[Path]:
    """
    Get a list of all editable source folders from nodetool packages.

    Returns:
        List[Path]: A list of Path objects to source folders of all editable nodetool packages
    """
    source_folders = []
    import sys

    # Check for editable installs in sys.path
    for path in sys.path:
        if "nodetool-" in path:
            source_path = Path(path)
            # Prefer the src directory if it exists (hatch/uv editable layout)
            if (source_path / "src").exists():
                source_path = source_path / "src"
            if source_path.exists():
                source_folders.append(source_path)

    return source_folders


def scan_for_package_nodes(verbose: bool = False) -> PackageModel:
    """Scan current directory for nodes and create package metadata."""
    import os
    import sys
    import tomli
    import json
    from nodetool.metadata.node_metadata import (
        EnumEncoder,
        PackageModel,
        get_node_classes_from_module,
    )

    sys.path.append(os.path.abspath("src"))

    # Check for pyproject.toml in current directory
    if not os.path.exists("pyproject.toml"):
        print("Error: No pyproject.toml found in current directory")
        sys.exit(1)

    # Read pyproject.toml
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomli.loads(f.read().decode())

    # Extract metadata (PEP 621 only)
    project_data = pyproject_data.get("project", {})
    if not project_data:
        print("Error: No [project] metadata found in pyproject.toml")
        sys.exit(1)

    # Name and version
    package_name = project_data.get("name", "")
    version = project_data.get("version", "0.1.0")

    # Description
    description = project_data.get("description", "")

    # Authors: PEP 621 may use list of tables
    raw_authors = project_data.get("authors", [])
    authors: list[str] = []
    try:
        if (
            isinstance(raw_authors, list)
            and raw_authors
            and isinstance(raw_authors[0], dict)
        ):
            for a in raw_authors:
                name = a.get("name")
                email = a.get("email")
                if name and email:
                    authors.append(f"{name} <{email}>")
                elif name:
                    authors.append(str(name))
                elif email:
                    authors.append(str(email))
        elif isinstance(raw_authors, list):
            authors = [str(a) for a in raw_authors]
    except Exception:
        # Safe fallback
        authors = []

    # Repository URL -> repo_id extraction
    repo_url: str | None = None
    # PEP 621: [project].urls.Repository or .Source
    urls = project_data.get("urls") if isinstance(project_data, dict) else None
    if isinstance(urls, dict):
        repo_url = urls.get("Repository") or urls.get("Source") or urls.get("Homepage")
    else:
        repo_url = None

    def _to_repo_id(url: str | None) -> str | None:
        if not url or not isinstance(url, str):
            return None
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            path = parsed.path.strip("/")
            if not path:
                return None
            # Remove trailing .git if present
            if path.endswith(".git"):
                path = path[:-4]
            # Expect format: owner/repo[/...]
            owner_repo = "/".join(path.split("/")[:2])
            return owner_repo or None
        except Exception:
            return None

    repo_id = _to_repo_id(repo_url) or ""

    # Discover examples and assets using unified Registry
    registry = Registry()

    # Create package model
    package = PackageModel(
        name=package_name,
        description=description,
        version=version,
        authors=authors,
        repo_id=repo_id,
        nodes=[],
        examples=registry._load_examples_from_directory(
            "src/nodetool/examples", package_name
        ),
        assets=registry._load_assets_from_directory(
            "src/nodetool/assets", package_name
        ),
    )

    # Add src directory to Python path temporarily
    src_path = os.path.abspath("src/nodetool/nodes")
    if os.path.exists(src_path):
        # Discover nodes
        module_names = []
        for root, _, files in os.walk(src_path):
            for file in files:
                if file.endswith(".py"):
                    module_path = os.path.join(root, file)
                    rel_path = os.path.relpath(module_path, src_path)
                    module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                    module_names.append(module_name)

        with click.progressbar(
            length=len(module_names),
            label="Scanning for nodes",
            show_eta=True,
            show_percent=True,
        ) as bar:
            for module_name in module_names:
                bar.update(1)
                if verbose:
                    click.echo(f"Scanning module: {module_name}")

                full_module_name = f"nodetool.nodes.{module_name}"
                node_classes = get_node_classes_from_module(full_module_name, verbose)
                if node_classes:
                    assert package.nodes is not None
                    package.nodes.extend(
                        node_class.get_metadata()
                        for node_class in node_classes
                        if node_class.is_visible()
                    )

        # Write the single nodes.json file in the root directory
        os.makedirs("src/nodetool/package_metadata", exist_ok=True)
        # Construct the metadata file path using package.name
        metadata_file_path = f"src/nodetool/package_metadata/{package.name}.json"
        with open(metadata_file_path, "w") as f:
            json.dump(
                package.model_dump(exclude_defaults=True),
                f,
                indent=2,
                cls=EnumEncoder,
            )

    print(
        f"✅ Successfully created package metadata for {package.name} with {len(package.nodes or [])} total nodes, {len(package.examples or [])} examples, and {len(package.assets or [])} assets"
    )

    return package  # Return the package model


def save_package_metadata(package: PackageModel, verbose: bool = False):
    """
    Save the package metadata to a JSON file.

    This function saves all package components (nodes, examples, and assets)
    to the metadata file at: src/nodetool/package_metadata/{package.name}.json

    Args:
        package: The package model to save
        verbose: Whether to print verbose output during saving

    Returns:
        str: The path to the saved metadata file
    """
    import os
    import json
    from nodetool.metadata.node_metadata import EnumEncoder

    # Create metadata directory if it doesn't exist
    os.makedirs("src/nodetool/package_metadata", exist_ok=True)

    # Generate the package metadata
    metadata = package.model_dump(exclude_defaults=True)

    # Save to package_metadata directory
    metadata_path = f"src/nodetool/package_metadata/{package.name}.json"
    if verbose:
        print(f"Saving package metadata to {metadata_path}")

    with open(metadata_path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            cls=EnumEncoder,
        )

    # Log the component counts
    node_count = len(package.nodes or [])
    example_count = len(package.examples or [])
    asset_count = len(package.assets or [])

    print(
        f"Saved metadata with {node_count} nodes, {example_count} examples, and {asset_count} assets"
    )

    return metadata_path


def update_pyproject_include(package: PackageModel, verbose: bool = False) -> None:
    """Ensure package assets are included for both Poetry and PEP 621 + Hatch formats.

    This updates either:
    - `[tool.hatch.build.targets.wheel]` for PEP 621 + Hatch projects
    - `[tool.poetry.include]` for legacy Poetry projects

    Provides backward compatibility for both package management systems.
    """
    pyproject_path = "pyproject.toml"
    if not os.path.exists(pyproject_path):
        if verbose:
            print("pyproject.toml not found, skipping update")
        return

    # Read the pyproject.toml file
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse with tomlkit to preserve formatting and comments
    data = tomlkit.parse(content)

    # Handle both Poetry and uv/hatchling formats
    if "project" in data:
        # New PEP 621 + Hatch format - use [tool.hatch.build.targets.wheel]
        if "tool" not in data:
            data["tool"] = tomlkit.table()  # type: ignore

        tool_section = data["tool"]  # type: ignore
        if "hatch" not in tool_section:  # type: ignore
            tool_section["hatch"] = tomlkit.table()  # type: ignore

        hatch_section = tool_section["hatch"]  # type: ignore
        if "build" not in hatch_section:  # type: ignore
            hatch_section["build"] = tomlkit.table()  # type: ignore

        build_section = hatch_section["build"]  # type: ignore
        if "targets" not in build_section:  # type: ignore
            build_section["targets"] = tomlkit.table()  # type: ignore

        targets_section = build_section["targets"]  # type: ignore
        if "wheel" not in targets_section:  # type: ignore
            targets_section["wheel"] = tomlkit.table()  # type: ignore

        wheel_section = targets_section["wheel"]  # type: ignore

        # Get existing artifacts list or create new one
        if "artifacts" not in wheel_section:  # type: ignore
            wheel_section["artifacts"] = tomlkit.array()  # type: ignore
            patterns = wheel_section["artifacts"]  # type: ignore
        else:
            patterns = wheel_section["artifacts"]  # type: ignore
    else:
        # Legacy Poetry format - use [tool.poetry.include]
        if "tool" not in data:
            data["tool"] = tomlkit.table()  # type: ignore

        tool_section = data["tool"]  # type: ignore
        if "poetry" not in tool_section:  # type: ignore
            tool_section["poetry"] = tomlkit.table()  # type: ignore

        poetry = tool_section["poetry"]  # type: ignore

        # Get existing include list or create new one
        if "include" not in poetry:  # type: ignore
            poetry["include"] = tomlkit.array()  # type: ignore
            patterns = poetry["include"]  # type: ignore
        else:
            include_item = poetry["include"]  # type: ignore
            # Convert to list if it's a single string
            if isinstance(include_item, str):
                poetry["include"] = tomlkit.array()  # type: ignore
                poetry["include"].append(include_item)  # type: ignore
                patterns = poetry["include"]  # type: ignore
            else:
                patterns = include_item

    # Convert absolute source paths to patterns relative to package root
    metadata_rel = f"package_metadata/{package.name}.json"
    asset_rels = [
        f"assets/{package.name}/{asset.name}" for asset in package.assets or []
    ]

    for rel in [metadata_rel, *asset_rels]:
        if rel not in patterns:  # type: ignore
            patterns.append(rel)  # type: ignore
            if verbose:
                print(f"Added package-data pattern: {rel}")

    # Write back the file
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(data))

    if verbose:
        build_tool = (
            "hatch.build.targets.wheel.artifacts"
            if "project" in data
            else "poetry.include"
        )
        print(f"Updated {pyproject_path} {build_tool} with asset files")


def load_node_packages():
    from nodetool.metadata.node_metadata import get_node_classes_from_namespace
    import importlib

    registry = Registry()
    packages = registry.list_installed_packages()

    total_loaded = 0
    for package in packages:
        if package.nodes:
            # Collect unique namespaces from this package
            namespaces = set()
            for node_metadata in package.nodes:
                node_type = node_metadata.node_type
                namespace_parts = node_type.split(".")[:-1]
                if len(namespace_parts) >= 2:
                    namespace = ".".join(namespace_parts)
                    namespaces.add(namespace)

            # Load each unique namespace from this package
            for namespace in namespaces:
                try:
                    # Try to import the module directly
                    if namespace.startswith("nodetool.nodes."):
                        module_path = namespace
                    else:
                        module_path = f"nodetool.nodes.{namespace}"

                    importlib.import_module(module_path)
                    total_loaded += 1
                except ImportError:
                    # Try alternative approach
                    try:
                        if namespace.startswith("nodetool."):
                            namespace_suffix = namespace[9:]
                            get_node_classes_from_namespace(
                                f"nodetool.nodes.{namespace_suffix}"
                            )
                            total_loaded += 1
                        else:
                            get_node_classes_from_namespace(
                                f"nodetool.nodes.{namespace}"
                            )
                            total_loaded += 1
                    except Exception:
                        pass


async def main():
    """
    Main function to run smoke tests for the registry module.
    """
    print("--- Running Smoke Tests for nodetool.packages.registry ---")

    print("\n--- Testing get_packages_dir ---")
    packages_dir = get_packages_dir()
    print(f"Packages directory: {packages_dir}")

    print("\n--- Testing get_package_manager_command ---")
    pkg_mgr_cmd = get_package_manager_command()
    print(f"Package manager command: {pkg_mgr_cmd}")

    print("\n--- Testing discover_node_packages ---")
    installed_discovered_packages = discover_node_packages()
    print(
        f"Discovered {len(installed_discovered_packages)} installed node packages (via discover_node_packages)."
    )
    for pkg in installed_discovered_packages:
        print(f"  - {pkg.name} ({pkg.repo_id if hasattr(pkg, 'repo_id') else 'N/A'})")

    print("\n--- Testing get_nodetool_package_source_folders ---")
    source_folders = get_nodetool_package_source_folders()
    print(f"Found {len(source_folders)} nodetool package source folders.")
    for folder in source_folders:
        print(f"  - {folder}")

    # Initialize Registry
    registry = Registry()

    print("\n--- Testing registry.list_installed_packages ---")
    installed_packages = registry.list_installed_packages()
    print(f"Found {len(installed_packages)} installed packages (via registry).")
    for pkg in installed_packages:
        print(f"  - {pkg.name} ({pkg.repo_id if hasattr(pkg, 'repo_id') else 'N/A'})")

    # Test list_available_packages
    print("\n--- Testing registry.list_available_packages ---")
    available_packages = registry.list_available_packages()
    print(f"Found {len(available_packages)} available packages from registry.")
    if available_packages:
        print(f"  First few: {[pkg.name for pkg in available_packages[:3]]}")

    # Test search_nodes
    print("\n--- Testing registry.search_nodes ---")
    print("Searching for all nodes (empty query)...")
    all_nodes = await registry.search_nodes("huggingface")
    print(f"Found {len(all_nodes)} nodes in total.")
    if all_nodes:
        print(f"  Sample node name: {all_nodes[0].get('name') if all_nodes else 'N/A'}")

    # Test get_package_for_node_type
    print("\n--- Testing registry.get_package_for_node_type ---")
    # This test depends on search_nodes populating the cache.
    sample_node_type = "huggingface.text_to_image.StableDiffusion"
    await registry.search_nodes()  # This might repopulate or confirm emptiness due to prior errors

    package_repo_id = await registry.get_package_for_node_type(sample_node_type)
    if package_repo_id:
        print(f"Package for node type '{sample_node_type}': {package_repo_id}")
    else:
        print(
            f"No package found for node type '{sample_node_type}' (this is expected if cache is empty, node type doesn't exist, or due to prior network issues)."
        )

    # Test unified example and asset functionality
    print("\n--- Testing unified Registry (formerly ExampleRegistry) ---")

    # Test list_examples
    print("\n--- Testing Registry.list_examples ---")
    examples = registry.list_examples()
    print(f"Listed {len(examples)} example workflows.")
    if examples:
        for example in examples:  # Show first few examples
            print(f"  - {example.name} from {example.package_name} in {example.path}")

    # Test find_example_by_name
    if examples:
        example_name = examples[0].name
        print(f"\n--- Testing Registry.find_example_by_name with '{example_name}' ---")
        found_example = registry.find_example_by_name(example_name)
        if found_example:
            print(f"Found example: {found_example.name}")
        else:
            print(f"Example '{example_name}' not found.")

    # Test cache clearing
    print("\n--- Testing Registry.clear_cache ---")
    registry.clear_cache()
    print("Example cache cleared successfully.")

    # Test list_assets
    print("\n--- Testing Registry.list_assets ---")
    assets = registry.list_assets()
    print(f"Listed {len(assets)} asset files.")
    if assets:
        for asset in assets[:3]:  # Show first few assets
            print(f"  - {asset.name} from {asset.package_name} in {asset.path}")

    # Test find_asset_by_name
    if assets:
        asset_name = assets[0].name
        print(f"\n--- Testing Registry.find_asset_by_name with '{asset_name}' ---")
        found_asset = registry.find_asset_by_name(asset_name)
        if found_asset:
            print(f"Found asset: {found_asset.name}")
        else:
            print(f"Asset '{asset_name}' not found.")

    print("=== Testing Example Workflow Search ===\n")

    # Test 1: Empty query (should return all examples)
    print("Test 1: Empty query (return all examples)")
    all_examples = registry.search_example_workflows("")
    print(f"Found {len(all_examples)} examples")
    print()

    # Test 2: Search for specific node type (RealESRGAN)
    print("Test 2: Search for 'Chat' in node types")
    results = registry.search_example_workflows("Chat")
    print(f"Found {len(results)} workflows with 'Chat' nodes")
    for workflow in results:
        print(f"  - {workflow.name}: {workflow.description[:60]}...")
    print()

    # Test 3: Search for node title (upscaler)
    print("Test 3: Search for 'LLM' in node titles")
    results = registry.search_example_workflows("LLM")
    print(f"Found {len(results)} workflows with 'LLM' in node titles")
    for workflow in results:
        print(f"  - {workflow.name}: {workflow.description[:60]}...")
    print()

    # Test 4: Search for word in node description
    print("Test 4: Search for 'upscaled' in node descriptions")
    results = registry.search_example_workflows("upscaled")
    print(f"Found {len(results)} workflows with 'upscaled' in node descriptions")
    for workflow in results:
        print(f"  - {workflow.name}: {workflow.description[:60]}...")
    print()

    # Test 5: Search for huggingface
    print("Test 5: Search for 'huggingface' in all node properties")
    results = registry.search_example_workflows("huggingface")
    print(f"Found {len(results)} workflows with 'huggingface' nodes")
    for workflow in results:
        print(f"  - {workflow.name}")
        # Show which nodes matched
        for node in workflow.graph.nodes:
            if "huggingface" in node.type.lower():
                print(f"    Node type: {node.type}")

    print()

    # Test 6: Search for "image" in descriptions
    print("Test 6: Search for 'image' in node fields")
    results = registry.search_example_workflows("image")
    print(f"Found {len(results)} workflows with 'image' in node properties")
    for workflow in results:
        print(f"  - {workflow.name}")
    print()

    # Test 7: Case insensitive search
    print("Test 7: Case insensitive search for 'REALESRGAN'")
    results = registry.search_example_workflows("REALESRGAN")
    print(f"Found {len(results)} workflows (case insensitive)")
    print()

    # Test 8: Non-matching query
    print("Test 8: Non-matching query")
    results = registry.search_example_workflows("xyz_nonexistent_query")
    print(f"Found {len(results)} workflows (should be 0)")
    print()

    print("=== Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
