#!/usr/bin/env python

import asyncio
import datetime
import os
from io import BytesIO
import re
from uuid import uuid4
import zipfile
import mimetypes
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, Response
from fastapi.responses import StreamingResponse, FileResponse
from nodetool.models.condition_builder import Field
from nodetool.types.asset import (
    Asset,
    AssetCreateRequest,
    AssetDownloadRequest,
    AssetList,
    AssetUpdateRequest,
    TempAsset,
)
from nodetool.api.utils import current_user
from nodetool.common.environment import Environment
from typing import Dict, List, Optional, Tuple, Union
from nodetool.models.asset import Asset as AssetModel
from nodetool.models.workflow import Workflow
from nodetool.packages.registry import Registry
from pydantic import BaseModel, Field as PydanticField
from nodetool.common.media_utils import (
    create_image_thumbnail,
    create_video_thumbnail,
    get_audio_duration,
    get_video_duration,
)


def from_model(asset: AssetModel):
    storage = Environment.get_asset_storage()
    if asset.content_type != "folder":
        get_url = storage.get_url(asset.file_name)
    else:
        get_url = None

    if asset.has_thumbnail:
        thumb_url = storage.get_url(asset.thumb_file_name)
    else:
        thumb_url = None

    return Asset(
        id=asset.id,
        user_id=asset.user_id,
        workflow_id=asset.workflow_id,
        parent_id=asset.parent_id,
        name=asset.name,
        content_type=asset.content_type,
        metadata=asset.metadata,
        created_at=asset.created_at.isoformat(),
        get_url=get_url,
        thumb_url=thumb_url,
        duration=asset.duration,
    )


# Define Pydantic models for package assets
class PackageAsset(BaseModel):
    id: str
    name: str
    package_name: str
    virtual_path: str = PydanticField(
        ..., description="Virtual path to access the asset"
    )


class PackageAssetList(BaseModel):
    assets: List[PackageAsset]


log = Environment.get_logger()
router = APIRouter(prefix="/api/assets", tags=["assets"])


@router.get("/")
async def index(
    parent_id: Optional[str] = None,
    content_type: Optional[str] = None,
    cursor: Optional[str] = None,
    page_size: Optional[int] = None,
    user: str = Depends(current_user),
    duration: Optional[int] = None,
) -> AssetList:
    """
    Returns all assets for a given user or workflow.
    """
    if page_size is None:
        page_size = 10000

    if content_type is None and parent_id is None:
        parent_id = user

    assets, next_cursor = AssetModel.paginate(
        user_id=user,
        parent_id=parent_id,
        content_type=content_type,
        limit=page_size,
        start_key=cursor,
    )

    assets = [from_model(asset) for asset in assets]

    return AssetList(next=next_cursor, assets=assets)


# Routes for package assets
@router.get("/packages", response_model=PackageAssetList)
async def list_package_assets():
    """
    List all assets from installed nodetool packages.
    """
    try:
        # Create an instance of Registry
        registry = Registry()

        # Get all assets from installed packages
        assets = registry.list_assets()

        # Convert to Pydantic models
        formatted_assets = [
            PackageAsset(
                id=f"pkg:{asset.package_name}/{asset.name}",
                name=asset.name,
                package_name=asset.package_name,
                virtual_path=f"/api/assets/packages/{asset.package_name}/{asset.name}",
            )
            for asset in assets
        ]

        return PackageAssetList(assets=formatted_assets)
    except Exception as e:
        log.exception(f"Error listing package assets: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing package assets: {str(e)}"
        )


@router.get("/packages/{package_name}", response_model=PackageAssetList)
async def list_package_assets_by_package(package_name: str):
    """
    List all assets from a specific nodetool package.
    """
    try:
        # Create an instance of Registry
        registry = Registry()

        # Get all assets from installed packages
        all_assets = registry.list_assets()

        # Filter assets by package name
        package_assets = [
            asset for asset in all_assets if asset.package_name == package_name
        ]

        # Convert to Pydantic models
        formatted_assets = [
            PackageAsset(
                id=f"pkg:{asset.package_name}/{asset.name}",
                name=asset.name,
                package_name=asset.package_name,
                virtual_path=f"/api/assets/packages/{asset.package_name}/{asset.name}",
            )
            for asset in package_assets
        ]

        return PackageAssetList(assets=formatted_assets)
    except Exception as e:
        log.exception(f"Error listing assets for package {package_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing assets for package {package_name}: {str(e)}",
        )


@router.get("/packages/{package_name}/{asset_name}")
async def get_package_asset(package_name: str, asset_name: str):
    """
    Serve a specific asset file from a nodetool package.
    """
    try:
        # Create an instance of Registry
        registry = Registry()

        package = registry.find_package_by_name(package_name)
        if not package:
            raise HTTPException(
                status_code=404,
                detail=f"Package '{package_name}' not found",
            )

        # Find the asset by name and package name
        asset = registry.find_asset_by_name(asset_name, package_name)

        # Verify the asset exists
        if not asset:
            raise HTTPException(
                status_code=404,
                detail=f"Asset '{asset_name}' not found in package '{package_name}'",
            )

        asset_path = os.path.join(
            str(package.source_folder), "nodetool", "assets", package_name, asset_name
        )

        # Get the physical path to the asset file
        if not os.path.exists(asset_path):
            raise HTTPException(
                status_code=404, detail=f"Asset file not found at path: {asset_path}"
            )

        # Determine the content type based on file extension
        content_type, _ = mimetypes.guess_type(asset_path)
        if not content_type:
            # Default to binary if content type can't be determined
            content_type = "application/octet-stream"

        # Return the file
        return FileResponse(
            path=asset_path, media_type=content_type, filename=asset_name
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(
            f"Error serving asset {asset_name} from package {package_name}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Error serving asset: {str(e)}")


@router.get("/{id}")
async def get(id: str, user: str = Depends(current_user)) -> Asset:
    """
    Returns the asset for the given id.
    """
    if id == user:
        return Asset(
            user_id=user,
            id=user,
            name="Home",
            content_type="folder",
            parent_id="",
            workflow_id=None,
            get_url=None,
            thumb_url=None,
            created_at="",
        )
    asset = AssetModel.find(user, id)
    if asset is None:
        log.info("Asset not found: %s", id)
        raise HTTPException(status_code=404, detail="Asset not found")
    return from_model(asset)


@router.put("/{id}")
async def update(
    id: str,
    req: AssetUpdateRequest,
    user: str = Depends(current_user),
) -> Asset:
    """
    Updates the asset for the given id.
    """
    asset = AssetModel.find(user, id)

    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    if req.content_type:
        asset.content_type = req.content_type
    if req.metadata:
        asset.metadata = req.metadata
    if req.name:
        asset.name = req.name.strip()
    if req.parent_id:
        asset.parent_id = req.parent_id
    if req.data:
        storage = Environment.get_asset_storage()
        await storage.upload(asset.file_name, BytesIO(req.data.encode("utf-8")))

    asset.save()
    return from_model(asset)


@router.delete("/{id}")
async def delete(id: str, user: str = Depends(current_user)):
    """
    Deletes the asset for the given id. If the asset is a folder, it deletes all contents recursively.
    """
    try:
        asset = AssetModel.find(user, id)
        if asset is None:
            log.info(f"Asset not found: {id}")
            raise HTTPException(status_code=404, detail="Asset not found")
        deleted_asset_ids = []
        if asset.content_type == "folder":
            deleted_asset_ids = await delete_folder(user, id)
        else:
            await delete_single_asset(asset)
            deleted_asset_ids = [id]
        return {"deleted_asset_ids": deleted_asset_ids}
    except Exception as e:
        log.exception(f"Asset deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting asset: {str(e)}")


async def delete_folder(user_id: str, folder_id: str) -> List[str]:
    deleted_asset_ids = []
    try:
        assets, next_cursor = AssetModel.paginate(
            user_id=user_id, parent_id=folder_id, limit=10000
        )
        # Delete children first
        for index, asset in enumerate(assets, 1):
            if asset.content_type == "folder":
                subfolder_deleted_ids = await delete_folder(user_id, asset.id)
                deleted_asset_ids.extend(subfolder_deleted_ids)
            else:
                await delete_single_asset(asset)
                deleted_asset_ids.append(asset.id)

        # Delete folder
        folder = AssetModel.find(user_id, folder_id)
        if folder:
            await delete_single_asset(folder)
            deleted_asset_ids.append(folder_id)
        else:
            log.warning(f"Folder not found when trying to delete: {folder_id}")
        log.info(f"Total assets deleted: {len(deleted_asset_ids)}")
        return deleted_asset_ids
    except Exception as e:
        log.exception(
            f"Error in delete_folder function for folder {folder_id}: {str(e)}"
        )
        raise


async def delete_single_asset(asset: AssetModel):
    try:
        asset.delete()
        storage = Environment.get_asset_storage()
        try:
            await storage.delete(asset.thumb_file_name)
        except Exception as e:
            log.warning(f"Error deleting thumbnail for asset {asset.id}: {e}")
        try:
            await storage.delete(asset.file_name)
        except Exception as e:
            log.warning(f"Error deleting file for asset {asset.id}: {e}")
    except Exception as e:
        log.exception(
            f"Error in delete_single_asset function for asset {asset.id}: {str(e)}"
        )
        raise


@router.post("/")
async def create(
    file: UploadFile | None = None,
    json: str | None = Form(None),
    user: str = Depends(current_user),
) -> Asset:
    """
    Create a new asset.
    """
    if json is None:
        raise HTTPException(status_code=400, detail="Missing JSON body")

    req = AssetCreateRequest.model_validate_json(json)
    asset = None
    duration = None
    file_io = None
    thumbnail = None

    if req.workflow_id:
        workflow = Workflow.get(req.workflow_id)
        if workflow and workflow.user_id != user:
            raise HTTPException(status_code=404, detail="Workflow not found")

    try:
        if file:
            file_content = await file.read()
            file_io = BytesIO(file_content)
            storage = Environment.get_asset_storage()

            if "video" in req.content_type:
                thumbnail = await create_video_thumbnail(file_io, 512, 512)
            elif "audio" in req.content_type:
                duration = get_audio_duration(file_io)
            elif "image" in req.content_type:
                thumbnail = await create_image_thumbnail(file_io, 512, 512)

        asset = AssetModel.create(
            workflow_id=req.workflow_id,
            user_id=user,
            parent_id=req.parent_id,
            name=req.name,
            content_type=req.content_type,
            metadata=req.metadata,
            duration=duration,
        )
        if file_io:
            file_io.seek(0)
            await storage.upload(asset.file_name, file_io)

            if thumbnail:
                await storage.upload(asset.thumb_file_name, thumbnail)

    except Exception as e:
        log.exception(e, stack_info=True)
        if asset:
            asset.delete()
        raise HTTPException(status_code=500, detail="Error uploading asset")

    return from_model(asset)


@router.post("/download")
async def download_assets(
    req: AssetDownloadRequest,
    user: str = Depends(current_user),
):
    """
    Create a ZIP file containing the requested assets and return it for download.
    Maintains folder structure based on asset.parent_id relationships.
    """
    if not req.asset_ids:
        raise HTTPException(status_code=400, detail="No asset IDs provided")

    zip_buffer = BytesIO()
    storage = Environment.get_asset_storage()

    asset_paths: Dict[str, str] = {}
    all_assets: Dict[str, AssetModel] = {}

    def fetch_all_assets(asset_ids: List[str]):
        for asset_id in asset_ids:
            asset = AssetModel.get(asset_id)
            if asset:
                all_assets[asset.id] = asset
                if asset.parent_id and asset.parent_id not in all_assets:
                    fetch_all_assets([asset.parent_id])
                if asset.content_type == "folder":
                    child_assets = AssetModel.get_children(asset.id)
                    child_asset_ids = [child.id for child in child_assets]
                    if child_asset_ids:
                        fetch_all_assets(child_asset_ids)

    fetch_all_assets(req.asset_ids)

    def get_asset_path(asset: AssetModel) -> str:
        if asset.id in asset_paths:
            return asset_paths[asset.id]

        if not asset.parent_id or asset.parent_id not in all_assets:
            path = asset.name
        else:
            parent_path = get_asset_path(all_assets[asset.parent_id])
            path = f"{parent_path}/{asset.name}"

        asset_paths[asset.id] = path
        return path

    async def fetch_asset_content(
        asset: AssetModel,
    ) -> Tuple[str, Union[BytesIO, None]]:
        try:
            if asset.user_id != user:
                raise HTTPException(
                    status_code=403,
                    detail=f"You don't have permission to download asset: {asset.id}",
                )

            asset_path = get_asset_path(asset)

            if asset.content_type == "folder":
                return f"{asset_path}/", None
            else:
                # Check if the file extension is already present
                if asset.name.lower().endswith(f".{asset.file_extension.lower()}"):
                    file_path = asset_path
                else:
                    file_path = f"{asset_path}.{asset.file_extension}"

                file_content = BytesIO()
                await storage.download(asset.file_name, file_content)
                file_content.seek(0)
                return file_path, file_content
        except Exception as e:
            log.warning(f"Error downloading asset {asset.id}: {str(e)}")
            return "", None

    asset_contents = await asyncio.gather(
        *[fetch_asset_content(asset) for asset in all_assets.values()]
    )

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path, content in asset_contents:
            if file_path and content is not None:
                zip_file.writestr(file_path, content.getvalue())

    zip_buffer.seek(0)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"assets_{timestamp}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/{folder_id}/recursive")
async def get_assets_recursive(folder_id: str, user: str = Depends(current_user)):
    """
    Get all assets in a folder recursively, including the folder structure.
    """
    assets = AssetModel.get_assets_recursive(user, folder_id)
    return assets


@router.get("/by-filename/{filename}")
async def get_by_filename(filename: str, user: str = Depends(current_user)) -> Asset:
    """
    Returns the asset for the given filename.
    """
    # Query for assets by the filename
    assets, _ = AssetModel.query(condition=Field("file_name").equals(filename))
    # Get the first matching asset if any exist
    asset = next(iter(assets), None)

    if asset is None:
        log.info("Asset not found with filename: %s", filename)
        raise HTTPException(status_code=404, detail="Asset not found")

    return from_model(asset)
