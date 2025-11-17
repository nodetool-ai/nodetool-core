#!/usr/bin/env python

import asyncio
import os
import shutil
import tempfile
import traceback
from typing import List, Optional

import aiofiles
import chromadb
from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from nodetool.indexing.service import index_file_to_collection
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    get_async_chroma_client,
    get_async_collection,
)
from nodetool.models.workflow import Workflow

router = APIRouter(prefix="/api/collections", tags=["collections"])

DEFAULT_EMBEDDING_MODEL = "all-minilm:latest"


class Document(BaseModel):
    text: str
    doc_id: str
    metadata: dict[str, str] = {}


class CollectionCreate(BaseModel):
    name: str
    embedding_model: str


class CollectionResponse(BaseModel):
    name: str
    count: int
    metadata: chromadb.CollectionMetadata
    workflow_name: str | None = None


class CollectionList(BaseModel):
    collections: List[CollectionResponse]
    count: int


class CollectionModify(BaseModel):
    name: str | None = None
    metadata: dict[str, str] | None = None


@router.post("/", response_model=CollectionResponse)
async def create_collection(
    req: CollectionCreate,
) -> CollectionResponse:
    """Create a new collection"""
    client = await get_async_chroma_client()
    metadata = {
        "embedding_model": req.embedding_model,
    }
    collection = await client.create_collection(name=req.name, metadata=metadata)
    return CollectionResponse(
        name=collection.name,
        metadata=collection.metadata,
        count=0,
    )


@router.get("/", response_model=CollectionList)
async def list_collections(
    _offset: Optional[int] = None,
    _limit: Optional[int] = None,
) -> CollectionList:
    """List all collections"""
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


@router.get("/{name}", response_model=CollectionResponse)
async def get(name: str) -> CollectionResponse:
    """Get a specific collection by name"""
    client = await get_async_chroma_client()
    collection = await client.get_collection(name=name)
    count = await collection.count()
    return CollectionResponse(
        name=collection.name,
        metadata=collection.metadata,
        count=count,
    )


@router.put("/{name}")
async def update_collection(name: str, req: CollectionModify):
    """Update a collection"""
    client = await get_async_chroma_client()
    collection = await client.get_collection(name=name)
    metadata = collection.metadata.copy()
    metadata.update(req.metadata or {})

    if workflow_id := metadata.get("workflow"):
        workflow = await Workflow.get(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Validate workflow input nodes
        graph = workflow.graph
        collection_input, file_input = find_input_nodes(graph)
        if not collection_input:
            raise HTTPException(
                status_code=400, detail="Workflow must have a CollectionInput node"
            )
        if not file_input:
            raise HTTPException(
                status_code=400,
                detail="Workflow must have a FileInput or DocumentFileInput node",
            )

    await collection.modify(name=req.name, metadata=metadata)
    return CollectionResponse(
        name=collection.name,
        metadata=collection.metadata,
        count=await collection.count(),
    )


@router.delete("/{name}")
async def delete_collection(name: str):
    """Delete a collection"""
    client = await get_async_chroma_client()
    await client.delete_collection(name=name)
    return {"message": f"Collection {name} deleted successfully"}


class IndexResponse(BaseModel):
    path: str
    error: Optional[str] = None


def find_input_nodes(graph: dict) -> tuple[str | None, str | None]:
    # Re-exported for backward compatibility; actual implementation moved.
    from nodetool.indexing.ingestion import find_input_nodes as _find

    return _find(graph)


@router.post("/{name}/index", response_model=IndexResponse)
async def index(
    name: str,
    file: UploadFile = File(...),
    _authorization: Optional[str] = Header(None),
) -> IndexResponse:
    await get_async_collection(name)
    token = "local_token"

    # Save uploaded file temporarily
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename or "uploaded_file")
    try:
        # Write uploaded file to disk asynchronously in chunks
        async with aiofiles.open(tmp_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await buffer.write(chunk)

        file_path = tmp_path
        mime_type = file.content_type or "application/octet-stream"

        error = await index_file_to_collection(name, file_path, mime_type, token)
        if error:
            return IndexResponse(path=file.filename or "unknown", error=error)

        return IndexResponse(path=file.filename or "unknown", error=None)
    except Exception as e:
        from nodetool.config.logging_config import get_logger

        log = get_logger(__name__)
        log.error(f"Error indexing file {file.filename}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=str(e)
        ) from e
    finally:
        # Ensure temporary directory is cleaned up without blocking
        await asyncio.to_thread(shutil.rmtree, tmp_dir)
        await file.close()  # Close the uploaded file handle
