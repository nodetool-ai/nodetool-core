from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from nodetool.packages.registry import (
    Registry,
    PackageModel,
    validate_repo_id,
)

from nodetool.api.utils import current_user
from nodetool.packages.types import PackageInfo

router = APIRouter(prefix="/api/packages", tags=["packages"])


class PackageListResponse(BaseModel):
    packages: List[PackageInfo]
    count: int


class InstalledPackageListResponse(BaseModel):
    packages: List[PackageModel]
    count: int


class NodeSearchResponse(BaseModel):
    nodes: List[dict]
    count: int


class PackageForNodeResponse(BaseModel):
    node_type: str
    package: Optional[str] = None
    found: bool


# Initialize registry
registry = Registry()


@router.get("/available", response_model=PackageListResponse)
async def list_available_packages(
    user: str = Depends(current_user),
) -> PackageListResponse:
    """List all available packages from the registry."""
    packages = registry.list_available_packages()
    return PackageListResponse(packages=packages, count=len(packages))


@router.get("/nodes/search", response_model=NodeSearchResponse)
async def search_nodes(
    query: str = "", user: str = Depends(current_user)
) -> NodeSearchResponse:
    """
    Search for nodes across all available packages.

    Args:
        query: Optional search string to filter nodes by name or description

    Returns:
        NodeSearchResponse: List of nodes matching the search query
    """
    nodes = await registry.search_nodes(query)
    return NodeSearchResponse(nodes=nodes, count=len(nodes))


@router.get("/nodes/package", response_model=PackageForNodeResponse)
async def get_package_for_node(
    node_type: str, user: str = Depends(current_user)
) -> PackageForNodeResponse:
    """
    Get the package that provides a specific node type.

    Args:
        node_type: The type identifier of the node

    Returns:
        PackageForNodeResponse: Information about the package providing the node
    """
    package = await registry.get_package_for_node_type(node_type)
    return PackageForNodeResponse(
        node_type=node_type, package=package, found=package is not None
    )


@router.get("/installed", response_model=InstalledPackageListResponse)
async def list_installed_packages(
    user: str = Depends(current_user),
) -> InstalledPackageListResponse:
    """List all installed packages."""
    packages = registry.list_installed_packages()
    return InstalledPackageListResponse(packages=packages, count=len(packages))



@router.get("/nodes/all", response_model=NodeSearchResponse)
async def get_all_nodes(user: str = Depends(current_user)) -> NodeSearchResponse:
    """
    Get all available nodes from all packages.

    Returns:
        NodeSearchResponse: List of all available nodes
    """
    # Reuse the search_nodes functionality with an empty query to get all nodes
    nodes = await registry.search_nodes("")
    return NodeSearchResponse(nodes=nodes, count=len(nodes))
