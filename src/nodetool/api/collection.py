#!/usr/bin/env python

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Header, UploadFile, File
from nodetool.common.environment import Environment
from nodetool.types.job import JobUpdate
from pydantic import BaseModel
from nodetool.common.chroma_client import (
    get_chroma_client,
    get_collection,
)
import chromadb
import os
import shutil
import tempfile
import traceback

from nodetool.metadata.types import Collection, FilePath
from nodetool.models.workflow import Workflow
from nodetool.indexing.service import index_file_to_collection
from nodetool.indexing.ingestion import (
    default_ingestion_workflow,
    find_input_nodes,
)
import aiofiles
import asyncio

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
    client = get_chroma_client()
    metadata = {
        "embedding_model": req.embedding_model,
    }
    collection = client.create_collection(name=req.name, metadata=metadata)
    return CollectionResponse(
        name=collection.name,
        metadata=collection.metadata,
        count=0,
    )


@router.get("/", response_model=CollectionList)
async def list_collections(
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> CollectionList:
    """List all collections"""
    client = get_chroma_client()
    collection_names = client.list_collections(offset=offset, limit=limit)

    collections = [client.get_collection(name) for name in collection_names]

    def get_workflow_name(metadata: dict[str, str]) -> str | None:
        if workflow_id := metadata.get("workflow"):
            workflow = Workflow.get(workflow_id)
            if workflow:
                return workflow.name
        return None

    return CollectionList(
        collections=[
            CollectionResponse(
                name=col.name,
                metadata=col.metadata or {},
                workflow_name=get_workflow_name(col.metadata or {}),
                count=col.count(),
            )
            for col in collections
        ],
        count=client.count_collections(),
    )


@router.get("/{name}", response_model=CollectionResponse)
async def get(name: str) -> CollectionResponse:
    """Get a specific collection by name"""
    client = get_chroma_client()
    collection = client.get_collection(name=name)
    count = collection.count()
    return CollectionResponse(
        name=collection.name,
        metadata=collection.metadata,
        count=count,
    )


@router.put("/{name}")
async def update_collection(name: str, req: CollectionModify):
    """Update a collection"""
    client = get_chroma_client()
    collection = client.get_collection(name=name)
    metadata = collection.metadata.copy()
    metadata.update(req.metadata or {})

    if workflow_id := metadata.get("workflow"):
        workflow = Workflow.get(workflow_id)
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

    collection.modify(name=req.name, metadata=metadata)
    return CollectionResponse(
        name=collection.name,
        metadata=collection.metadata,
        count=collection.count(),
    )


@router.delete("/{name}")
async def delete_collection(name: str):
    """Delete a collection"""
    client = get_chroma_client()
    client.delete_collection(name=name)
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
    authorization: Optional[str] = Header(None),
) -> IndexResponse:
    collection = get_collection(name)
    token = "local_token"

    # Save uploaded file temporarily
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename or "uploaded_file")
    try:
        # Async copy of uploaded file to disk
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
        Environment.get_logger().error(f"Error indexing file {file.filename}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure temporary directory is cleaned up
        shutil.rmtree(tmp_dir)
        await file.close()  # Close the uploaded file handle
