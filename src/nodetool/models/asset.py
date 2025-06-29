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
    size: int = DBField(default=0)  # File size in bytes
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
        size: int = 0,
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
            size: File size in bytes (default: 0).
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
