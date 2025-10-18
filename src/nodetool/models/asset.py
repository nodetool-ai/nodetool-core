"""
Defines the Asset database model.

Represents a digital asset within the nodetool system, such as images, videos, or folders.
Includes metadata like name, content type, user association, parent folder (for hierarchy),
and optional workflow association.
"""

from typing import Dict, Optional, Literal, Sequence
from datetime import datetime
from nodetool.types.content_types import CONTENT_TYPE_TO_EXTENSION
from nodetool.config.logging_config import get_logger

from nodetool.models.base_model import (
    DBModel,
    DBField,
    create_time_ordered_uuid,
    DBIndex,
)
from nodetool.models.condition_builder import Field

log = get_logger(__name__)


@DBIndex(["user_id", "parent_id"])
class Asset(DBModel):
    """Database model representing a digital asset (file, folder, etc.)."""

    @classmethod
    def get_table_schema(cls):
        """Returns the database table schema for assets."""
        return {"table_name": "nodetool_assets"}

    type: Literal["asset"] = "asset"
    id: str = DBField()
    user_id: str = DBField(default="")
    workflow_id: str | None = DBField(default=None)
    parent_id: str = DBField(default="")
    file_id: str | None = DBField(default="")
    name: str = DBField(default="")
    content_type: str = DBField(default="")
    size: Optional[int] = DBField(default=None)  # File size in bytes (None for folders)
    metadata: dict | None = DBField(default=None)
    created_at: datetime = DBField(default_factory=datetime.now)
    duration: Optional[float] = DBField(default=None)

    @property
    def file_extension(self) -> str:
        """
        Get the file extension of the asset.

        For example, if the content type is "image/jpeg", this will return "jpeg".
        """
        return (
            CONTENT_TYPE_TO_EXTENSION[self.content_type]  # type: ignore
            if self.content_type in CONTENT_TYPE_TO_EXTENSION
            else "bin"
        )

    @property
    def has_thumbnail(self) -> bool:
        """
        Returns True if the asset type supports thumbnails.
        """
        return self.content_type.startswith("image/") or self.content_type.startswith(
            "video/"
        )

    @property
    def file_name(self) -> str:
        """
        Get the file name of the asset.
        """
        return f"{self.id}.{self.file_extension}"

    @property
    def thumb_file_name(self) -> str:
        """
        Get the file name of the thumbnail.
        """
        return f"{self.id}_thumb.jpg"

    @classmethod
    async def create(
        cls,
        user_id: str,
        name: str,
        content_type: str,
        metadata: dict | None = None,
        parent_id: str | None = None,
        workflow_id: str | None = None,
        duration: float | None = None,
        size: Optional[int] = None,
        **kwargs,
    ):
        """Creates a new asset record in the database.

        Generates a time-ordered UUID for the asset ID.
        Sets the parent_id to the user_id if not otherwise specified.

        Args:
            user_id: The ID of the owner user.
            name: The name of the asset.
            content_type: The MIME type of the asset content.
            metadata: Optional dictionary for additional metadata.
            parent_id: Optional ID of the parent asset (e.g., folder).
            workflow_id: Optional ID of an associated workflow.
            duration: Optional duration (e.g., for video/audio assets).
            size: Optional file size in bytes (default: None for folders).
            **kwargs: Additional fields to set on the model.

        Returns:
            The newly created and saved Asset instance.
        """
        return await super().create(
            id=create_time_ordered_uuid(),
            name=name,
            user_id=user_id,
            parent_id=parent_id or user_id,
            workflow_id=workflow_id,
            content_type=content_type,
            duration=duration,
            size=size,
            created_at=datetime.now(),
            metadata=metadata,
            **kwargs,
        )

    @classmethod
    async def find(cls, user_id: str, asset_id: str):
        """
        Find an asset by user_id and asset_id.
        """
        item = await cls.get(asset_id)
        if item and item.user_id == user_id:
            return item
        return None

    @classmethod
    async def paginate(
        cls,
        user_id: str,
        parent_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        content_type: Optional[str] = None,
        limit: int = 100,
        start_key: str | None = None,
        reverse: bool = False,
    ):
        """
        Paginate assets for a user using boto3.
        Applies filters for parent_id if provided.
        Returns a tuple of a list of Assets and the last evaluated key for pagination.
        Last key is "" if there are no more items to be returned.
        """

        condition = Field("user_id").equals(user_id)

        if parent_id:
            condition = condition.and_(Field("parent_id").equals(parent_id))
        if workflow_id:
            condition = condition.and_(Field("workflow_id").equals(workflow_id))
        if start_key:
            condition = condition.and_(Field("id").greater_than(start_key))
        if content_type:
            condition = condition.and_(
                Field("content_type").like((content_type or "") + "%")
            )

        return await cls.query(condition, limit, reverse)

    @classmethod
    async def get_children(cls, parent_id: str) -> Sequence["Asset"]:
        """
        Fetch all child assets for a given parent_id.
        """
        items, _ = await cls.query(Field("parent_id").equals(parent_id))
        return items

    @classmethod
    async def get_assets_recursive(cls, user_id: str, folder_id: str) -> Dict:
        """Recursively fetches all assets within a given folder for a user.

        Args:
            user_id: The ID of the user whose assets are being fetched.
            folder_id: The ID of the starting folder.

        Returns:
            A dictionary containing a list of assets, structured hierarchically
            with 'children' keys for subfolders. Returns an empty list if the
            initial folder is not found or not owned by the user.
        """

        async def recursive_fetch(current_folder_id):
            assets, _ = await cls.paginate(
                user_id=user_id, parent_id=current_folder_id, limit=10000
            )
            result = []
            for asset in assets:
                if asset.user_id != user_id:
                    continue

                asset_dict = asset.dict()
                if asset.content_type == "folder":
                    asset_dict["children"] = await recursive_fetch(asset.id)
                result.append(asset_dict)

            return result

        folder = await cls.find(user_id, folder_id)
        if not folder:
            log.warning(f"Folder {folder_id} not found for user {user_id}")
            return {"assets": []}

        folder_dict = folder.model_dump()
        folder_dict["children"] = await recursive_fetch(folder_id)

        return {"assets": [folder_dict]}

    @classmethod
    async def search_assets_global(
        cls,
        user_id: str,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 100,
        start_key: Optional[str] = None,
    ):
        """
        **Global Asset Search (Model Layer)**

        Search assets globally across all folders belonging to the current user and return path information.

        **Security:** Only returns assets owned by the specified user_id (user isolation enforced).
        **Performance:** Uses batch queries to avoid N+1 problems when fetching folder paths.
        **Search Behavior:** Uses contains matching (LIKE %query%) for user-friendly search.

        Note: Local search (within current folder) is handled in the frontend by filtering already-loaded folder assets.

        Args:
            user_id: The ID of the user whose assets are being searched
            query: Search term to match against asset names (automatically sanitized)
            content_type: Optional content type filter (e.g., "image", "text")
            limit: Maximum number of results to return (default 100)
            start_key: Pagination key for continuing search

        Returns:
            Tuple of (assets, next_cursor, folder_paths) where:
            - assets: List of Asset objects matching the search (filtered to current user only)
            - next_cursor: Pagination cursor for next page (None if no more results)
            - folder_paths: List of dicts with folder context for each asset

        Example:
            assets, cursor, paths = Asset.search_assets_global("user123", "photo", limit=50)
        """
        # Use raw trimmed query to preserve characters like '_' so substring search works as expected.
        # Parameters are safely bound by adapters, so no manual wildcard escaping here.
        sanitized_query = query.strip()

        # Build base condition for user and name search (consistent contains search for better UX)
        condition = (
            Field("user_id")
            .equals(user_id)
            .and_(Field("name").like(f"%{sanitized_query}%"))
        )

        # Add content_type filter if specified
        if content_type:
            condition = condition.and_(
                Field("content_type").like((content_type or "") + "%")
            )

        # Add pagination
        if start_key:
            condition = condition.and_(Field("id").greater_than(start_key))

        # Execute query
        assets, next_cursor = await cls.query(condition, limit)

        # Get folder path information for each asset
        folder_paths = await cls.get_asset_path_info(
            user_id, [asset.id for asset in assets]
        )

        # Convert folder_paths dict to list in same order as assets
        folder_path_list = []
        for asset in assets:
            if asset.id in folder_paths:
                folder_path_list.append(folder_paths[asset.id])
            else:
                folder_path_list.append(
                    {
                        "folder_name": "Unknown",
                        "folder_path": "Unknown",
                        "folder_id": asset.parent_id or "",
                    }
                )

        return assets, next_cursor, folder_path_list

    @classmethod
    async def get_asset_path_info(
        cls, user_id: str, asset_ids: list[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Get folder path information for given asset IDs using batch queries to avoid N+1 problem.

        Args:
            user_id: The ID of the user who owns the assets.
            asset_ids: List of asset IDs to get path information for.

        Returns:
            Dictionary mapping asset_id to folder info:
            {asset_id: {folder_name, folder_path, folder_id}}
        """
        if not asset_ids:
            return {}

        result = {}

        # Step 1: Batch fetch all requested assets
        asset_condition = Field("user_id").equals(user_id)
        # Create an OR condition for all asset IDs
        if len(asset_ids) == 1:
            asset_condition = asset_condition.and_(Field("id").equals(asset_ids[0]))
        else:
            id_conditions = [Field("id").equals(asset_id) for asset_id in asset_ids]
            # Combine with OR - this assumes the query system supports it
            # If not, we'll need to make multiple queries in batches
            combined_id_condition = id_conditions[0]
            for condition in id_conditions[1:]:
                combined_id_condition = combined_id_condition.or_(condition)
            asset_condition = asset_condition.and_(combined_id_condition)

        try:
            assets, _ = await cls.query(asset_condition, limit=len(asset_ids) * 2)
        except Exception:
            log.warning(
                f"Batch asset query failed, falling back to individual queries for user {user_id}"
            )
            return await cls._get_asset_path_info_fallback(user_id, asset_ids)

        # Create a map of assets by ID for quick lookup
        assets_by_id = {asset.id: asset for asset in assets}

        # Step 2: Collect all parent IDs we need to fetch
        all_parent_ids = set()
        for asset in assets:
            current_id = asset.parent_id
            while current_id and current_id != user_id:
                all_parent_ids.add(current_id)
                # We'll need to do this iteratively since we don't know the full hierarchy yet
                break  # Just collect immediate parents for now

        # Step 3: Batch fetch all parent folders
        parent_assets = {}
        if all_parent_ids:
            parent_condition = Field("user_id").equals(user_id)
            if len(all_parent_ids) == 1:
                parent_condition = parent_condition.and_(
                    Field("id").equals(list(all_parent_ids)[0])
                )
            else:
                parent_id_conditions = [
                    Field("id").equals(parent_id) for parent_id in all_parent_ids
                ]
                combined_parent_condition = parent_id_conditions[0]
                for condition in parent_id_conditions[1:]:
                    combined_parent_condition = combined_parent_condition.or_(condition)
                parent_condition = parent_condition.and_(combined_parent_condition)

            try:
                parent_results, _ = await cls.query(
                    parent_condition, limit=len(all_parent_ids) * 2
                )
                parent_assets = {asset.id: asset for asset in parent_results}
            except Exception:
                log.warning(f"Batch parent query failed for user {user_id}")

        # Step 4: Build path information for each requested asset
        for asset_id in asset_ids:
            if asset_id not in assets_by_id:
                continue

            asset = assets_by_id[asset_id]

            # If asset is in root folder (parent_id == user_id), handle specially
            if asset.parent_id == user_id:
                result[asset_id] = {
                    "folder_name": "Home",
                    "folder_path": "Home",
                    "folder_id": user_id,
                }
                continue

            # Build folder path by walking up the parent chain using cached parents
            folder_path_parts = []
            folder_ids = []
            current_id = asset.parent_id

            # Walk up the folder hierarchy using cached data
            while current_id and current_id != user_id:
                if current_id in parent_assets:
                    parent_folder = parent_assets[current_id]
                    folder_path_parts.append(parent_folder.name)
                    folder_ids.append(parent_folder.id)
                    current_id = parent_folder.parent_id
                else:
                    # If we don't have the parent cached, we need to fetch it
                    # This can happen for deeply nested folders
                    parent_folder = await cls.find(user_id, current_id)
                    if not parent_folder:
                        break
                    folder_path_parts.append(parent_folder.name)
                    folder_ids.append(parent_folder.id)
                    current_id = parent_folder.parent_id

            # Add Home as root
            folder_path_parts.append("Home")
            folder_ids.append(user_id)

            # Reverse to get path from root to immediate parent
            folder_path_parts.reverse()
            folder_ids.reverse()

            # Get immediate parent info
            immediate_parent_name = (
                folder_path_parts[-1] if folder_path_parts else "Home"
            )
            immediate_parent_id = folder_ids[-1] if folder_ids else user_id

            result[asset_id] = {
                "folder_name": immediate_parent_name,
                "folder_path": " / ".join(folder_path_parts),
                "folder_id": immediate_parent_id,
            }

        return result

    @classmethod
    async def _get_asset_path_info_fallback(
        cls, user_id: str, asset_ids: list[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Fallback method for get_asset_path_info when batch queries fail.
        Uses individual queries but with better error handling.
        """
        result = {}

        for asset_id in asset_ids:
            try:
                asset = await cls.find(user_id, asset_id)
                if not asset:
                    continue

                # If asset is in root folder (parent_id == user_id), handle specially
                if asset.parent_id == user_id:
                    result[asset_id] = {
                        "folder_name": "Home",
                        "folder_path": "Home",
                        "folder_id": user_id,
                    }
                    continue

                # Build folder path by walking up the parent chain
                folder_path_parts = []
                folder_ids = []
                current_id = asset.parent_id

                # Walk up the folder hierarchy
                while current_id and current_id != user_id:
                    parent_folder = await cls.find(user_id, current_id)
                    if not parent_folder:
                        break

                    folder_path_parts.append(parent_folder.name)
                    folder_ids.append(parent_folder.id)
                    current_id = parent_folder.parent_id

                # Add Home as root
                folder_path_parts.append("Home")
                folder_ids.append(user_id)

                # Reverse to get path from root to immediate parent
                folder_path_parts.reverse()
                folder_ids.reverse()

                # Get immediate parent info
                immediate_parent_name = (
                    folder_path_parts[-1] if folder_path_parts else "Home"
                )
                immediate_parent_id = folder_ids[-1] if folder_ids else user_id

                result[asset_id] = {
                    "folder_name": immediate_parent_name,
                    "folder_path": " / ".join(folder_path_parts),
                    "folder_id": immediate_parent_id,
                }
            except Exception as e:
                log.warning(f"Error getting path info for asset {asset_id}: {e}")
                # Provide a fallback result
                result[asset_id] = {
                    "folder_name": "Unknown",
                    "folder_path": "Unknown",
                    "folder_id": "",
                }

        return result
