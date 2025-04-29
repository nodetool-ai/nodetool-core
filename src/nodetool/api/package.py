from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from nodetool.packages.registry import (
    Registry,
    PackageInfo,
    PackageModel,
    validate_repo_id,
)

from nodetool.api.utils import current_user

router = APIRouter(prefix="/api/packages", tags=["packages"])


# Models for API requests and responses
class PackageInstallRequest(BaseModel):
    repo_id: str = Field(description="Repository ID in the format <owner>/<project>")


class PackageUninstallRequest(BaseModel):
    repo_id: str = Field(description="Repository ID in the format <owner>/<project>")


class PackageResponse(BaseModel):
    success: bool
    message: str


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


@router.post("/install", response_model=PackageResponse)
async def install_package(
    request: PackageInstallRequest, user: str = Depends(current_user)
) -> PackageResponse:
    """Install a package from the registry."""
    is_valid, error_msg = validate_repo_id(request.repo_id)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        registry.install_package(request.repo_id)
        return PackageResponse(
            success=True, message=f"Package {request.repo_id} installed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/uninstall", response_model=PackageResponse)
async def uninstall_package(
    request: PackageUninstallRequest, user: str = Depends(current_user)
) -> PackageResponse:
    """Uninstall a package."""
    is_valid, error_msg = validate_repo_id(request.repo_id)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    success = registry.uninstall_package(request.repo_id)
    if success:
        return PackageResponse(
            success=True, message=f"Package {request.repo_id} uninstalled successfully"
        )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Package {request.repo_id} not found or could not be uninstalled",
        )


@router.post("/update", response_model=PackageResponse)
async def update_package(
    repo_id: str, user: str = Depends(current_user)
) -> PackageResponse:
    """Update an installed package."""
    is_valid, error_msg = validate_repo_id(repo_id)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    success = registry.update_package(repo_id)
    if success:
        return PackageResponse(
            success=True, message=f"Package {repo_id} updated successfully"
        )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Package {repo_id} not found or could not be updated",
        )


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
