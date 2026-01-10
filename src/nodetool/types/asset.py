from typing import Any, Optional

from pydantic import BaseModel
from pydantic import Field as PydanticField

from nodetool.types.content_types import CONTENT_TYPE_TO_EXTENSION


class Asset(BaseModel):
    id: str
    user_id: str
    workflow_id: str | None
    parent_id: str
    name: str
    content_type: str
    size: int | None = None  # File size in bytes (None for folders)
    metadata: dict[str, Any] | None = None
    created_at: str
    get_url: str | None
    thumb_url: str | None
    duration: float | None = None

    @property
    def file_extension(self) -> str:
        """
        Get the file extension of the asset.

        For example, if the content type is "image/jpeg", this will return "jpeg".
        """
        return CONTENT_TYPE_TO_EXTENSION.get(self.content_type, "bin")  # type: ignore

    @property
    def file_name(self) -> str:
        """
        Get the file name of the asset.
        """
        return f"{self.id}.{self.file_extension}"


class AssetUpdateRequest(BaseModel):
    name: str | None
    parent_id: str | None
    content_type: str | None
    data: str | None = None
    metadata: dict | None = None
    duration: float | None = None
    size: int | None = None


class AssetCreateRequest(BaseModel):
    workflow_id: str | None = None
    parent_id: str | None = None
    name: str
    content_type: str
    metadata: dict | None = None
    duration: float | None = None
    size: int | None = None


class AssetDownloadRequest(BaseModel):
    asset_ids: list[str]


class AssetList(BaseModel):
    next: str | None
    assets: list[Asset]


class TempAsset(BaseModel):
    get_url: str
    put_url: str


class AssetWithPath(BaseModel):
    # All existing Asset fields
    id: str
    user_id: str
    workflow_id: str | None
    parent_id: str | None
    name: str
    content_type: str
    size: int | None
    metadata: dict[str, Any] | None = None
    created_at: str
    get_url: str | None
    thumb_url: str | None
    duration: float | None

    # New fields for search context
    folder_name: str = PydanticField(..., description="Direct parent folder name")
    folder_path: str = PydanticField(..., description="Full path breadcrumb")
    folder_id: str = PydanticField(..., description="Parent folder ID for navigation")


class AssetSearchResult(BaseModel):
    assets: list[AssetWithPath]
    next_cursor: str | None = None
    total_count: int
    is_global_search: bool
