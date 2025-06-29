from typing import Any
from pydantic import BaseModel

from nodetool.common.content_types import CONTENT_TYPE_TO_EXTENSION


class Asset(BaseModel):
    id: str
    user_id: str
    workflow_id: str | None
    parent_id: str
    name: str
    content_type: str
    size: int = 0  # File size in bytes
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
        return (
            CONTENT_TYPE_TO_EXTENSION[self.content_type]  # type: ignore
            if self.content_type in CONTENT_TYPE_TO_EXTENSION
            else "bin"
        )

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
    size: int = 0


class AssetDownloadRequest(BaseModel):
    asset_ids: list[str]


class AssetList(BaseModel):
    next: str | None
    assets: list[Asset]


class TempAsset(BaseModel):
    get_url: str
    put_url: str
