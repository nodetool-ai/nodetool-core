"""Asset management tools.

These tools provide functionality for managing NodeTool assets including:
- Listing assets
- Getting asset details
- Searching assets
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from nodetool.models.asset import Asset as AssetModel
from nodetool.packages.registry import Registry
from nodetool.runtime.resources import ResourceScope, maybe_scope


@asynccontextmanager
async def _ensure_resource_scope():
    """Bind a ResourceScope only when one is not already active."""
    if maybe_scope() is not None:
        yield
        return

    async with ResourceScope():
        yield


class AssetTools:
    """Asset management tools."""

    @staticmethod
    async def list_assets(
        source: str = "user",
        parent_id: str | None = None,
        query: str | None = None,
        content_type: str | None = None,
        package_name: str | None = None,
        limit: int = 100,
        user_id: str = "1",
    ) -> dict[str, Any]:
        """
        List or search assets with flexible filtering options.

        Args:
            source: Asset source ("user" or "package")
            parent_id: Filter by parent folder ID (user assets only)
            query: Search query for asset names (min 2 chars)
            content_type: Filter by type ("image", "video", "audio", "text", "folder")
            package_name: Filter package assets by package name
            limit: Maximum number of assets to return (default: 100)
            user_id: User ID (default: "1")

        Returns:
            Dictionary with assets list and pagination info
        """
        async with _ensure_resource_scope():
            if source == "package":
                registry = Registry.get_instance()
                all_assets = registry.list_assets()

                if package_name:
                    all_assets = [a for a in all_assets if a.package_name == package_name]

                if query and len(query.strip()) >= 2:
                    query_lower = query.strip().lower()
                    all_assets = [a for a in all_assets if query_lower in a.name.lower()]

                all_assets = all_assets[:limit]

                results = [
                    {
                        "id": f"pkg:{asset.package_name}/{asset.name}",
                        "name": asset.name,
                        "package_name": asset.package_name,
                        "virtual_path": f"/api/assets/packages/{asset.package_name}/{asset.name}",
                        "source": "package",
                    }
                    for asset in all_assets
                ]

                return {
                    "assets": results,
                    "next": None,
                    "total": len(results),
                }

            if query:
                if len(query.strip()) < 2:
                    raise ValueError("Search query must be at least 2 characters long")

                assets, next_cursor, folder_paths = await AssetModel.search_assets_global(
                    user_id=user_id,
                    query=query.strip(),
                    content_type=content_type,
                    limit=limit,
                )

                results = []
                for i, asset in enumerate(assets):
                    asset_dict = await _asset_to_dict(asset)
                    folder_info = (
                        folder_paths[i]
                        if i < len(folder_paths)
                        else {
                            "folder_name": "Unknown",
                            "folder_path": "Unknown",
                            "folder_id": "",
                        }
                    )
                    asset_dict["folder_name"] = folder_info["folder_name"]
                    asset_dict["folder_path"] = folder_info["folder_path"]
                    asset_dict["folder_id"] = folder_info["folder_id"]
                    asset_dict["source"] = "user"
                    results.append(asset_dict)

                return {
                    "assets": results,
                    "next": next_cursor,
                    "total": len(results),
                }

            if content_type is None and parent_id is None:
                parent_id = user_id

            assets, next_cursor = await AssetModel.paginate(
                user_id=user_id,
                parent_id=parent_id,
                content_type=content_type,
                limit=limit,
            )

            results = []
            for asset in assets:
                asset_dict = await _asset_to_dict(asset)
                asset_dict["source"] = "user"
                results.append(asset_dict)

            return {
                "assets": results,
                "next": next_cursor,
                "total": len(results),
            }

    @staticmethod
    async def get_asset(
        asset_id: str,
        user_id: str = "1",
    ) -> dict[str, Any]:
        """
        Get detailed information about a specific asset.

        Args:
            asset_id: The ID of the asset
            user_id: User ID (default: "1")

        Returns:
            Asset details including URLs and metadata
        """
        async with _ensure_resource_scope():
            asset = await AssetModel.find(user_id, asset_id)
            if not asset:
                raise ValueError(f"Asset {asset_id} not found")

            return await _asset_to_dict(asset)

    @staticmethod
    def get_tool_functions() -> dict[str, Any]:
        """Get all asset tool functions."""
        return {
            "list_assets": AssetTools.list_assets,
            "get_asset": AssetTools.get_asset,
        }


async def _asset_to_dict(asset: AssetModel) -> dict[str, Any]:
    """
    Convert an Asset model to a dictionary for API responses.

    Args:
        asset: The AssetModel instance to convert

    Returns:
        Dictionary with asset information including URLs
    """
    from nodetool.runtime.resources import require_scope

    storage = require_scope().get_asset_storage()

    if asset.content_type != "folder":
        get_url = await storage.get_url(asset.file_name)
    else:
        get_url = None

    # Use on-demand thumbnail endpoint for assets that support thumbnails
    # This ensures thumbnails are generated on-the-fly if they don't exist
    if asset.has_thumbnail:
        thumb_url = f"/api/assets/{asset.id}/thumbnail?t={int(asset.updated_at.timestamp())}"
    else:
        thumb_url = None

    return {
        "id": asset.id,
        "user_id": asset.user_id,
        "workflow_id": asset.workflow_id,
        "parent_id": asset.parent_id,
        "name": asset.name,
        "content_type": asset.content_type,
        "size": asset.size,
        "metadata": asset.metadata,
        "created_at": asset.created_at.isoformat(),
        "get_url": get_url,
        "thumb_url": thumb_url,
        "duration": asset.duration,
    }
