"""
Defines the Asset database model.

Represents a digital asset within the nodetool system, such as images, videos, or folders.
Includes metadata like name, content type, user association, parent folder (for hierarchy),
and optional workflow association.
"""

from typing import Dict, Optional, Literal, Sequence
from datetime import datetime
from nodetool.common.content_types import CONTENT_TYPE_TO_EXTENSION
from nodetool.common.environment import Environment

from nodetool.models.base_model import (
    DBModel,
    DBField,
    create_time_ordered_uuid,
    DBIndex,
)
from nodetool.models.condition_builder import Field

log = Environment.get_logger()


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
    def create(
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
        return super().create(
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
    def find(cls, user_id: str, asset_id: str):
        """
        Find an asset by user_id and asset_id.
        """
        item = cls.get(asset_id)
        if item and item.user_id == user_id:
            return item
        return None

    @classmethod
    def paginate(
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

        return cls.query(condition, limit, reverse)

    @classmethod
    def get_children(cls, parent_id: str) -> Sequence["Asset"]:
        """
        Fetch all child assets for a given parent_id.
        """
        items, _ = cls.query(Field("parent_id").equals(parent_id))
        return items

    @classmethod
    def get_assets_recursive(cls, user_id: str, folder_id: str) -> Dict:
        """Recursively fetches all assets within a given folder for a user.

        Args:
            user_id: The ID of the user whose assets are being fetched.
            folder_id: The ID of the starting folder.

        Returns:
            A dictionary containing a list of assets, structured hierarchically
            with 'children' keys for subfolders. Returns an empty list if the
            initial folder is not found or not owned by the user.
        """

        def recursive_fetch(current_folder_id):
            assets, _ = cls.paginate(
                user_id=user_id, parent_id=current_folder_id, limit=10000
            )
            result = []
            for asset in assets:
                if asset.user_id != user_id:
                    continue

                asset_dict = asset.dict()
                if asset.content_type == "folder":
                    asset_dict["children"] = recursive_fetch(asset.id)
                result.append(asset_dict)

            return result

        folder = cls.find(user_id, folder_id)
        if not folder:
            log.warning(f"Folder {folder_id} not found for user {user_id}")
            return {"assets": []}

        folder_dict = folder.model_dump()
        folder_dict["children"] = recursive_fetch(folder_id)

        return {"assets": [folder_dict]}

    @classmethod
    def search_assets_global(
        cls,
        user_id: str,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 100,
        start_key: Optional[str] = None,
    ):
        """
        Search assets globally across all user folders and return path information.
        
        Note: Local search is handled in the frontend by filtering already-loaded folder assets.
        
        Args:
            user_id: The ID of the user whose assets are being searched.
            query: Search term to match against asset names.
            content_type: Optional content type filter.
            limit: Maximum number of results to return.
            start_key: Pagination key for continuing search.
            
        Returns:
            Tuple of (assets, next_cursor, folder_paths) where:
            - assets: List of Asset objects matching the search
            - next_cursor: Pagination cursor for next page (None if no more results)
            - folder_paths: List of dicts with folder context for each asset
        """
        # Build base condition for user and name search (global search only)
        condition = Field("user_id").equals(user_id).and_(
            Field("name").like(f"%{query}%")
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
        assets, next_cursor = cls.query(condition, limit)
        
        # Get folder path information for each asset
        folder_paths = cls.get_asset_path_info(user_id, [asset.id for asset in assets])
        
        # Convert folder_paths dict to list in same order as assets
        folder_path_list = []
        for asset in assets:
            if asset.id in folder_paths:
                folder_path_list.append(folder_paths[asset.id])
            else:
                folder_path_list.append({
                    'folder_name': 'Unknown',
                    'folder_path': 'Unknown',
                    'folder_id': asset.parent_id or ''
                })
        
        return assets, next_cursor, folder_path_list

    @classmethod
    def get_asset_path_info(cls, user_id: str, asset_ids: list[str]) -> Dict[str, Dict[str, str]]:
        """
        Get folder path information for given asset IDs.
        
        Args:
            user_id: The ID of the user who owns the assets.
            asset_ids: List of asset IDs to get path information for.
            
        Returns:
            Dictionary mapping asset_id to folder info:
            {asset_id: {folder_name, folder_path, folder_id}}
        """
        result = {}
        
        for asset_id in asset_ids:
            asset = cls.find(user_id, asset_id)
            if not asset:
                continue
                
            # If asset is in root folder (parent_id == user_id), handle specially
            if asset.parent_id == user_id:
                result[asset_id] = {
                    'folder_name': 'Home',
                    'folder_path': 'Home',
                    'folder_id': user_id
                }
                continue
            
            # Build folder path by walking up the parent chain
            folder_path_parts = []
            folder_ids = []
            current_id = asset.parent_id
            
            # Walk up the folder hierarchy
            while current_id and current_id != user_id:
                parent_folder = cls.find(user_id, current_id)
                if not parent_folder:
                    break
                    
                folder_path_parts.append(parent_folder.name)
                folder_ids.append(parent_folder.id)
                current_id = parent_folder.parent_id
            
            # Add Home as root
            folder_path_parts.append('Home')
            folder_ids.append(user_id)
            
            # Reverse to get path from root to immediate parent
            folder_path_parts.reverse()
            folder_ids.reverse()
            
            # Get immediate parent info
            immediate_parent_name = folder_path_parts[-1] if folder_path_parts else 'Home'
            immediate_parent_id = folder_ids[-1] if folder_ids else user_id
            
            result[asset_id] = {
                'folder_name': immediate_parent_name,
                'folder_path': ' / '.join(folder_path_parts),
                'folder_id': immediate_parent_id
            }
        
        return result
