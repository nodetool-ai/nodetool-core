#!/usr/bin/env python

import datetime
import os
from io import BytesIO
import zipfile
import mimetypes
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from nodetool.models.condition_builder import Field
from nodetool.types.asset import (
    Asset,
    AssetCreateRequest,
    AssetDownloadRequest,
    AssetList,
    AssetUpdateRequest,
    AssetWithPath,
    AssetSearchResult,
)
from nodetool.api.utils import current_user
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from typing import Dict, List, Optional, Tuple, Union
from nodetool.models.asset import Asset as AssetModel
from nodetool.models.workflow import Workflow
from nodetool.packages.registry import Registry
from pydantic import BaseModel, Field as PydanticField
from nodetool.media.common.media_utils import (
    create_image_thumbnail,
    create_video_thumbnail,
    get_audio_duration,
)


def from_model(asset: AssetModel):
    storage = Environment.get_asset_storage()
    if asset.content_type != "folder":
        # Note: Pre-signed URLs from storage providers (S3, etc.) typically
        # include their own cache headers. If using file storage, consider
        # implementing a separate endpoint with proper cache headers.
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
        size=asset.size,
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


# Constants
MIN_SEARCH_QUERY_LENGTH = 2
DEFAULT_SEARCH_PAGE_SIZE = 200

log = get_logger(__name__)
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

    assets, next_cursor = await AssetModel.paginate(
        user_id=user,
        parent_id=parent_id,
        content_type=content_type,
        limit=page_size,
        start_key=cursor,
    )

    assets = [from_model(asset) for asset in assets]

    return AssetList(next=next_cursor, assets=assets)


@router.get("/search")
async def search_assets_global(
    query: str,
    content_type: Optional[str] = None,
    page_size: Optional[int] = 100,
    cursor: Optional[str] = None,
    user: str = Depends(current_user),
) -> AssetSearchResult:
    """
    **Global Asset Search**

    Search assets globally across all folders belonging to the current user with folder path information.

    **Features:**
    - Searches asset names using contains matching (finds matches anywhere in filename)
    - Provides folder breadcrumb information for each result
    - Supports content type filtering (e.g., "image", "text")
    - Includes pagination for large result sets
    - Returns only current user's assets (user isolation)

    **Examples:**
    - `GET /api/assets/search?query=photo` - Find all assets with "photo" in name
    - `GET /api/assets/search?query=sunset&content_type=image` - Find images with "sunset"
    - `GET /api/assets/search?query=doc&page_size=50` - Find "doc" assets, 50 per page

    Note: Local search (within current folder) is handled efficiently in the frontend
    by filtering already-loaded folder assets.

    Args:
        query: Search term (minimum 2 characters, case insensitive)
        content_type: Optional content type filter (e.g., "image", "text", "video")
        page_size: Results per page (default 200, max recommended 1000)
        cursor: Pagination cursor for next page
        user: Current user ID (automatically provided)

    Returns:
        AssetSearchResult with assets and folder path information (current user's assets only)
    """
    # Validate query length
    if len(query.strip()) < MIN_SEARCH_QUERY_LENGTH:
        raise HTTPException(
            status_code=400, detail="Search query must be at least 2 characters long"
        )

    try:
        # Search assets globally using the model's search method
        assets, next_cursor, folder_paths = await AssetModel.search_assets_global(
            user_id=user,
            query=query.strip(),
            content_type=content_type,
            limit=page_size or DEFAULT_SEARCH_PAGE_SIZE,
            start_key=cursor,
        )

        # Convert to AssetWithPath objects
        assets_with_path = []
        for i, asset in enumerate(assets):
            asset_data = from_model(asset)
            folder_info = (
                folder_paths[i]
                if i < len(folder_paths)
                else {
                    "folder_name": "Unknown",
                    "folder_path": "Unknown",
                    "folder_id": "",
                }
            )

            asset_with_path = AssetWithPath(
                **asset_data.model_dump(),
                folder_name=folder_info["folder_name"],
                folder_path=folder_info["folder_path"],
                folder_id=folder_info["folder_id"],
            )
            assets_with_path.append(asset_with_path)

        return AssetSearchResult(
            assets=assets_with_path,
            next_cursor=next_cursor,
            total_count=len(assets_with_path),
            is_global_search=True,
        )

    except Exception as e:
        log.exception(f"Error searching assets for user {user}: {str(e)}")
        raise HTTPException(status_code=500, detail="Search temporarily unavailable")


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

        # Return the file with caching headers
        # Package assets are immutable, so we can cache them for a long time
        return FileResponse(
            path=asset_path,
            media_type=content_type,
            filename=asset_name,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",  # 1 year
                "ETag": f'"{package_name}-{asset_name}"',
            },
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
            size=None,
            get_url=None,
            thumb_url=None,
            created_at="",
        )
    asset = await AssetModel.find(user, id)
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
    asset = await AssetModel.find(user, id)

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
    if req.size is not None:
        asset.size = req.size
    if req.data:
        storage = Environment.get_asset_storage()
        data_bytes = req.data.encode("utf-8")
        asset.size = len(data_bytes)  # Update size when data is updated
        await storage.upload(asset.file_name, BytesIO(data_bytes))

    await asset.save()
    return from_model(asset)


@router.delete("/{id}")
async def delete(id: str, user: str = Depends(current_user)):
    """
    Deletes the asset for the given id. If the asset is a folder, it deletes all contents recursively.
    """
    try:
        asset = await AssetModel.find(user, id)
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
        assets, next_cursor = await AssetModel.paginate(
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
        folder = await AssetModel.find(user_id, folder_id)
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
        await asset.delete()
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
    file_size = req.size  # Default size from request

    if req.workflow_id:
        workflow = await Workflow.get(req.workflow_id)
        if workflow and workflow.user_id != user:
            raise HTTPException(status_code=404, detail="Workflow not found")

    try:
        storage = None
        if file:
            file_content = await file.read()
            file_size = len(file_content)  # Calculate actual file size
            file_io = BytesIO(file_content)
            storage = Environment.get_asset_storage()

            if "video" in req.content_type:
                thumbnail = await create_video_thumbnail(file_io, 512, 512)
            elif "audio" in req.content_type:
                duration = get_audio_duration(file_io)
            elif "image" in req.content_type:
                thumbnail = await create_image_thumbnail(file_io, 512, 512)

        asset = await AssetModel.create(
            workflow_id=req.workflow_id,
            user_id=user,
            parent_id=req.parent_id,
            name=req.name,
            content_type=req.content_type,
            metadata=req.metadata,
            duration=duration,
            size=file_size,
        )
        if file_io and storage:
            file_io.seek(0)
            await storage.upload(asset.file_name, file_io)

            if thumbnail:
                await storage.upload(asset.thumb_file_name, thumbnail)

    except Exception as e:
        log.exception(e, stack_info=True)
        if asset:
            await asset.delete()
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
    log.info(f"User '{user}' initiated download for asset IDs: {req.asset_ids}")
    if not req.asset_ids:
        raise HTTPException(status_code=400, detail="No asset IDs provided")

    zip_buffer = BytesIO()
    storage = Environment.get_asset_storage()

    # This dictionary will hold all assets to be included in the zip, plus their parents for path construction.
    all_assets_with_parents: Dict[str, AssetModel] = {}
    # This set will hold just the assets that should be included in the zip file content.
    assets_to_zip: Dict[str, AssetModel] = {}

    # Step 1: Fetch the requested assets and all their descendants.
    queue = list(req.asset_ids)
    processed_ids = set()
    while queue:
        asset_id = queue.pop(0)
        if asset_id in processed_ids:
            continue
        processed_ids.add(asset_id)

        asset = await AssetModel.get(asset_id)
        if asset:
            assets_to_zip[asset.id] = asset
            all_assets_with_parents[asset.id] = asset
            if asset.content_type == "folder":
                child_assets = await AssetModel.get_children(asset.id)
                queue.extend([child.id for child in child_assets])
    log.info(f"Found {len(assets_to_zip)} assets/folders to include in the zip.")

    # Step 2: Fetch all necessary ancestors for path construction.
    parents_to_fetch = set()
    for asset in assets_to_zip.values():
        if asset.parent_id and asset.parent_id not in all_assets_with_parents:
            parents_to_fetch.add(asset.parent_id)

    while parents_to_fetch:
        parent_id = parents_to_fetch.pop()
        if parent_id in all_assets_with_parents:
            continue

        parent_asset = await AssetModel.get(parent_id)
        if parent_asset:
            all_assets_with_parents[parent_id] = parent_asset
            if (
                parent_asset.parent_id
                and parent_asset.parent_id not in all_assets_with_parents
            ):
                parents_to_fetch.add(parent_asset.parent_id)

    asset_paths: Dict[str, str] = {}

    def get_asset_path(asset: AssetModel) -> str:
        if asset.id in asset_paths:
            return asset_paths[asset.id]

        if not asset.parent_id or asset.parent_id not in all_assets_with_parents:
            path = asset.name
        else:
            parent_path = get_asset_path(all_assets_with_parents[asset.parent_id])
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

    asset_contents = []
    # Only iterate over the assets we actually want to zip, not the parents used for pathing.
    for asset in assets_to_zip.values():
        content = await fetch_asset_content(asset)
        asset_contents.append(content)

    used_paths: Dict[str, int] = {}
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path, content in asset_contents:
            if file_path and content is not None:
                unique_path = file_path
                if file_path in used_paths:
                    name, ext = os.path.splitext(file_path)
                    index = used_paths[file_path]
                    while True:
                        dedup_path = f"{name}_{index}{ext}"
                        if dedup_path not in used_paths:
                            unique_path = dedup_path
                            break
                        index += 1
                    used_paths[file_path] = index + 1
                    used_paths[unique_path] = 1
                else:
                    used_paths[file_path] = 1

                zip_file.writestr(unique_path, content.getvalue())

    zip_buffer.seek(0)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"assets_{timestamp}.zip"

    log.info(f"Sending ZIP file '{filename}' with {len(asset_contents)} items.")
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/{folder_id}/recursive")
async def get_assets_recursive(folder_id: str, user: str = Depends(current_user)):
    """
    Get all assets in a folder recursively, including the folder structure.
    """
    assets = await AssetModel.get_assets_recursive(user, folder_id)
    return assets


@router.get("/by-filename/{filename}")
async def get_by_filename(filename: str, user: str = Depends(current_user)) -> Asset:
    """
    Returns the asset for the given filename.
    """
    # Query for assets by the filename
    assets, _ = await AssetModel.query(condition=Field("file_name").equals(filename))
    # Get the first matching asset if any exist
    asset = next(iter(assets), None)

    if asset is None:
        log.info("Asset not found with filename: %s", filename)
        raise HTTPException(status_code=404, detail="Asset not found")

    return from_model(asset)
