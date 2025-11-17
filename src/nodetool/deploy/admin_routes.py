"""
Admin routes for the lightweight NodeTool FastAPI server.

This module encapsulates endpoints under /admin, including:
- HuggingFace model download (SSE)
- Ollama model download (SSE)
- Cache scan and size
- Delete HF model
- Workflow registry status
- Collection management (CRUD operations)
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.deploy.admin_operations import (
    calculate_cache_size,
    delete_hf_model,
    download_hf_model,
    download_ollama_model,
    scan_hf_cache,
)
from nodetool.indexing.ingestion import find_input_nodes
from nodetool.indexing.service import index_file_to_collection
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    get_async_chroma_client,
)
from nodetool.models.asset import Asset as AssetModel
from nodetool.models.workflow import Workflow
from nodetool.runtime.resources import require_scope
from nodetool.types.asset import Asset, AssetList

if TYPE_CHECKING:
    from nodetool.models.database_adapter import DatabaseAdapter


# Collection-related Pydantic models
class CollectionCreate(BaseModel):
    name: str
    embedding_model: str


class CollectionResponse(BaseModel):
    name: str
    count: int
    metadata: dict[str, Any] | None
    workflow_name: str | None = None


class CollectionList(BaseModel):
    collections: List[CollectionResponse]
    count: int


class CollectionModify(BaseModel):
    name: str | None = None
    metadata: dict[str, str] | None = None


class AddToCollection(BaseModel):
    documents: List[str]
    ids: List[str]
    metadatas: List[dict[str, str]]
    embeddings: List[List[float]]


class IndexResponse(BaseModel):
    path: str
    error: Optional[str] = None


async def asset_from_model(asset: AssetModel) -> Asset:
    """Convert AssetModel to Asset API response."""
    storage = require_scope().get_asset_storage()
    if asset.content_type != "folder":
        get_url = await storage.get_url(asset.file_name)
    else:
        get_url = None

    if asset.has_thumbnail:
        thumb_url = await storage.get_url(asset.thumb_file_name)
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


async def get_model_adapter(table: str) -> DatabaseAdapter:
    if table == "workflows":
        return await Workflow.adapter()
    elif table == "assets":
        return await AssetModel.adapter()
    else:
        raise ValueError(f"Unknown table: {table}")


def create_admin_router() -> APIRouter:
    router = APIRouter()

    @router.post("/admin/models/huggingface/download")
    async def download_huggingface_model_endpoint(request: Request):
        """Download HuggingFace model with optional streaming progress."""
        try:
            data = await request.json()
            repo_id = data.get("repo_id")

            if not repo_id:
                raise HTTPException(status_code=400, detail="repo_id is required")

            async def generate_sse():
                try:
                    async for chunk in download_hf_model(
                        repo_id=repo_id,
                        cache_dir=data.get("cache_dir", "/app/.cache/huggingface/hub"),
                        file_path=data.get("file_path"),
                        ignore_patterns=data.get("ignore_patterns"),
                        allow_patterns=data.get("allow_patterns"),
                        stream=data.get("stream", True),
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_data = {"status": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            print(f"HuggingFace download error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/admin/models/ollama/download")
    async def download_ollama_model_endpoint(request: Request):
        """Download Ollama model with optional streaming progress."""
        try:
            data = await request.json()
            model_name = data.get("model_name")

            if not model_name:
                raise HTTPException(status_code=400, detail="model_name is required")

            async def generate_sse():
                try:
                    async for chunk in download_ollama_model(
                        model_name=model_name, stream=data.get("stream", True)
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_data = {"status": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            print(f"Ollama download error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/admin/cache/scan")
    async def scan_cache():
        """Scan HuggingFace cache directory."""
        try:
            results = []
            async for chunk in scan_hf_cache():
                results.append(chunk)
            return (
                results[0]
                if results
                else {"status": "error", "message": "No cache data"}
            )
        except Exception as e:
            print(f"Cache scan error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/admin/cache/size")
    async def get_cache_size(cache_dir: str = "/app/.cache/huggingface/hub"):
        """Calculate total cache size."""
        try:
            results = []
            async for chunk in calculate_cache_size(cache_dir=cache_dir):
                results.append(chunk)
            return (
                results[0]
                if results
                else {"status": "error", "message": "No size data"}
            )
        except Exception as e:
            print(f"Cache size calculation error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.delete("/admin/models/huggingface/{repo_id:path}")
    async def delete_huggingface_model_endpoint(repo_id: str):
        """Delete HuggingFace model from cache."""
        try:
            results = []
            async for chunk in delete_hf_model(repo_id=repo_id):
                results.append(chunk)
            return (
                results[0]
                if results
                else {"status": "error", "message": "Delete failed"}
            )
        except Exception as e:
            print(f"HuggingFace model deletion error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Database adapter operations
    @router.post("/admin/db/{table}/save")
    async def db_save(table: str, item: Dict[str, Any]):
        """Save an item to the specified table using the database adapter."""
        try:
            adapter = await get_model_adapter(table)
            await adapter.save(item)
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/admin/db/{table}/{key}")
    async def db_get(table: str, key: str):
        """Get an item by primary key from the specified table."""
        try:
            adapter = await get_model_adapter(table)
            item = await adapter.get(key)
            if item is None:
                raise HTTPException(status_code=404, detail="Not found")
            return item
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.delete("/admin/db/{table}/{key}")
    async def db_delete(table: str, key: str):
        """Delete an item by primary key from the specified table."""
        try:
            adapter = await get_model_adapter(table)
            await adapter.delete(key)
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Collection management endpoints
    @router.post("/admin/collections", response_model=CollectionResponse)
    async def create_collection(req: CollectionCreate) -> CollectionResponse:
        """Create a new collection."""
        try:
            client = await get_async_chroma_client()
            metadata = {
                "embedding_model": req.embedding_model,
            }
            collection = await client.create_collection(
                name=req.name, metadata=metadata
            )
            return CollectionResponse(
                name=collection.name,
                metadata=collection.metadata,
                count=0,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/admin/collections", response_model=CollectionList)
    async def list_collections(
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> CollectionList:
        """List all collections."""
        try:
            client = await get_async_chroma_client()
            collections = await client.list_collections()

            async def get_workflow_name(metadata: dict[str, str]) -> str | None:
                if workflow_id := metadata.get("workflow"):
                    workflow = await Workflow.get(workflow_id)
                    if workflow:
                        return workflow.name
                return None

            return CollectionList(
                collections=[
                    CollectionResponse(
                        name=col.name,
                        metadata=col.metadata or {},
                        workflow_name=await get_workflow_name(col.metadata or {}),
                        count=col.count(),
                    )
                    for col in collections
                ],
                count=len(collections),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/admin/collections/{name}", response_model=CollectionResponse)
    async def get_collection(name: str) -> CollectionResponse:
        """Get a specific collection by name."""
        try:
            client = await get_async_chroma_client()
            collection = await client.get_collection(name=name)
            count = await collection.count()
            return CollectionResponse(
                name=collection.name,
                metadata=collection.metadata,
                count=count,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.put("/admin/collections/{name}")
    async def update_collection(name: str, req: CollectionModify):
        """Update a collection."""
        try:
            client = await get_async_chroma_client()
            collection = await client.get_collection(name=name)
            metadata = collection.metadata.copy()
            metadata.update(req.metadata or {})
            await collection.modify(name=req.name, metadata=metadata)
            return CollectionResponse(
                name=collection.name,
                metadata=collection.metadata,
                count=await collection.count(),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.delete("/admin/collections/{name}")
    async def delete_collection(name: str):
        """Delete a collection."""
        try:
            client = await get_async_chroma_client()
            await client.delete_collection(name=name)
            return {"message": f"Collection {name} deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/admin/collections/{name}/add")
    async def add_to_collection(name: str, req: AddToCollection):
        """Add a file to a collection."""
        try:
            client = await get_async_chroma_client()
            collection = await client.get_collection(name=name)
            await collection.add(documents=req.documents, ids=req.ids, metadatas=req.metadatas, embeddings=req.embeddings)
            return {"message": f"Documents added to collection {name} successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Asset management endpoints
    @router.get("/admin/assets", response_model=AssetList)
    async def list_assets(
        user: str = Depends(current_user),
        user_id: Optional[str] = "1",
        parent_id: Optional[str] = None,
        content_type: Optional[str] = None,
        cursor: Optional[str] = None,
        page_size: Optional[int] = 100,
    ) -> AssetList:
        """List assets (admin endpoint - no user restrictions)."""
        try:
            effective_user = user_id or user
            if page_size is None or page_size > 10000:
                page_size = 10000

            if content_type is None and parent_id is None:
                parent_id = effective_user

            assets, next_cursor = await AssetModel.paginate(
                user_id=effective_user,
                parent_id=parent_id,
                content_type=content_type,
                limit=page_size,
                start_key=cursor,
            )

            assets = await asyncio.gather(*[asset_from_model(asset) for asset in assets])

            return AssetList(next=next_cursor, assets=assets)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/admin/assets", response_model=Asset)
    async def create_asset(
        user: str = Depends(current_user),
        data: Dict[str, Any] = Body(...),
    ) -> Asset:
        """Create a new asset (admin endpoint - no user restrictions)."""
        try:
            # Extract id separately to pass via kwargs
            asset_id = data.get("id")
            kwargs = {}
            if asset_id:
                kwargs["id"] = asset_id

            asset = await AssetModel.create(
                user_id=data.get("user_id", user),
                name=data.get("name", ""),
                content_type=data.get("content_type", ""),
                parent_id=data.get("parent_id"),
                workflow_id=data.get("workflow_id"),
                metadata=data.get("metadata"),
                **kwargs,
            )
            return await asset_from_model(asset)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/admin/assets/{asset_id}", response_model=Asset)
    async def get_asset(
        asset_id: str,
        user: str = Depends(current_user),
        user_id: Optional[str] = "1",
    ) -> Asset:
        """Get a single asset by ID (admin endpoint - no user restrictions)."""
        try:
            uid = user_id or user

            # Handle special case for user root folder
            if asset_id == uid:
                return Asset(
                    user_id=uid,
                    id=uid,
                    name="Home",
                    content_type="folder",
                    parent_id="",
                    workflow_id=None,
                    size=None,
                    get_url=None,
                    thumb_url=None,
                    created_at="",
                )

            asset = await AssetModel.get(asset_id)
            if asset is None:
                raise HTTPException(status_code=404, detail="Asset not found")
            return await asset_from_model(asset)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.delete("/admin/assets/{asset_id}")
    async def delete_asset(asset_id: str, user: str = Depends(current_user)):
        """Delete an asset (recursive for folders) (admin endpoint - no user restrictions)."""
        try:
            asset = await AssetModel.get(asset_id)
            if asset is None:
                raise HTTPException(status_code=404, detail="Asset not found")
            if asset.user_id != user:
                raise HTTPException(status_code=403, detail="Asset access denied")

            deleted_asset_ids = []

            async def delete_folder(uid: str, folder_id: str) -> List[str]:
                ids = []
                try:
                    assets, _ = await AssetModel.paginate(
                        user_id=uid, parent_id=folder_id, limit=10000
                    )
                    # Delete children first
                    for a in assets:
                        if a.content_type == "folder":
                            subfolder_ids = await delete_folder(uid, a.id)
                            ids.extend(subfolder_ids)
                        else:
                            await delete_single_asset(a)
                            ids.append(a.id)

                    # Delete folder itself
                    folder = await AssetModel.find(uid, folder_id)
                    if folder:
                        await delete_single_asset(folder)
                        ids.append(folder_id)
                    return ids
                except Exception as e:
                    from nodetool.config.logging_config import get_logger

                    log = get_logger(__name__)
                    log.exception(
                        f"Error in delete_folder for folder {folder_id}: {str(e)}"
                    )
                    raise

            async def delete_single_asset(a: AssetModel):
                try:
                    await a.delete()
                    storage = require_scope().get_asset_storage()
                    with suppress(Exception):
                        await storage.delete(a.thumb_file_name)
                    with suppress(Exception):
                        await storage.delete(a.file_name)
                except Exception as e:
                    from nodetool.config.logging_config import get_logger

                    log = get_logger(__name__)
                    log.exception(
                        f"Error in delete_single_asset for asset {a.id}: {str(e)}"
                    )
                    raise

            if asset.content_type == "folder":
                deleted_asset_ids = await delete_folder(asset.user_id, asset_id)
            else:
                await delete_single_asset(asset)
                deleted_asset_ids = [asset_id]

            return {"deleted_asset_ids": deleted_asset_ids}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router
