#!/usr/bin/env python

from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File
from langchain_text_splitters import (
    ExperimentalMarkdownSyntaxTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from pydantic import BaseModel
from nodetool.api.utils import current_user
from nodetool.common.chroma_client import (
    get_chroma_client,
    get_collection,
)
import chromadb
from markitdown import MarkItDown
import pymupdf
import pymupdf4llm
import os
import shutil
import tempfile

from nodetool.metadata.types import Collection, FilePath
from nodetool.models.workflow import Workflow
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.run_job_request import RunJobRequest

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


def chunk_documents_recursive(
    documents: List[Document],
    chunk_size: int = 4096,
    chunk_overlap: int = 2048,
) -> tuple[dict[str, str], list[dict]]:
    """Split documents into chunks using LangChain's recursive character splitting.
    This method provides more semantic splitting by attempting to break at natural
    text boundaries.

    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        Tuple of (id_to_text_mapping, metadata_list)
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Initialize the splitter with common text boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    ids_docs = {}
    metadatas = []

    for doc in documents:
        # Convert to LangChain document format and split
        splits = splitter.split_text(doc.text)

        # Create document IDs and collect metadata
        for i, text in enumerate(splits):
            doc_id = f"{doc.doc_id}:{i}"
            ids_docs[doc_id] = text
            metadatas.append(doc.metadata)

    return ids_docs, metadatas


def chunk_documents_markdown(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[dict[str, str], list[dict]]:
    """Split markdown documents based on headers and then recursively.

    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk in characters after header splitting
        chunk_overlap: Number of characters to overlap between recursive chunks

    Returns:
        Tuple of (id_to_text_mapping, metadata_list)
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Initialize markdown splitter
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )

    # Initialize recursive splitter for further chunking
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    ids_docs = {}
    metadatas = []
    chunk_index = 0

    for doc in documents:
        # Split by headers first
        md_splits = markdown_splitter.split_text(doc.text)

        # Further split the header-based chunks recursively
        final_splits = recursive_splitter.split_documents(md_splits)

        # Create document IDs and collect metadata for final chunks
        for split_doc in final_splits:
            # Use a running index to ensure unique IDs across header splits
            doc_id = f"{doc.doc_id}:{chunk_index}"
            ids_docs[doc_id] = split_doc.page_content
            # Carry over original document metadata, potentially add header metadata later
            metadatas.append(doc.metadata.copy())
            chunk_index += 1
        chunk_index = 0  # Reset chunk index for the next document

    return ids_docs, metadatas


def default_ingestion_workflow(
    collection: chromadb.Collection, file_path: str, mime_type: str
) -> None:
    """Process a file and add it to the collection using the default ingestion workflow.

    Args:
        collection: ChromaDB collection to add documents to
        file_path: Path to the file to process
        mime_type: MIME type of the file
    """
    # Convert file to documents
    if mime_type == "application/pdf":
        with open(file_path, "rb") as f:
            pdf_data = f.read()
            doc = pymupdf.open(stream=pdf_data, filetype="pdf")
            md_text = pymupdf4llm.to_markdown(doc)
            documents = [Document(text=md_text, doc_id=file_path)]
    else:
        md = MarkItDown()
        documents = [
            Document(text=md.convert(file_path).text_content, doc_id=file_path)
        ]

    # Chunk documents and upsert to collection
    ids_docs, _ = chunk_documents_markdown(
        documents,
        chunk_size=4096,
        chunk_overlap=256,
    )
    collection.upsert(
        documents=list(ids_docs.values()),
        ids=list(ids_docs.keys()),
    )


def find_input_nodes(graph: dict) -> tuple[str | None, str | None]:
    """Find the collection input and file input node names from a workflow graph.

    Args:
        graph: The workflow graph to search

    Returns:
        Tuple of (collection_input_name, file_input_name) where each may be None if not found
    """
    collection_input = None
    file_input = None

    for node in graph["nodes"]:
        if node["type"] == "nodetool.input.CollectionInput":
            collection_input = node["data"]["name"]
        elif node["type"] in (
            "nodetool.input.FileInput",
            "nodetool.input.DocumentFileInput",
        ):
            file_input = node["data"]["name"]

    return collection_input, file_input


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
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_path = tmp_path
        mime_type = file.content_type or "application/octet-stream"

        if workflow_id := collection.metadata.get("workflow"):
            processing_context = ProcessingContext(
                user_id="1",
                auth_token=token,
                workflow_id=workflow_id,
            )
            req = RunJobRequest(
                workflow_id=workflow_id,
                user_id="1",
                auth_token=token,
            )
            workflow = await processing_context.get_workflow(workflow_id)
            req.graph = workflow.graph
            req.params = {}

            collection_input, file_input = find_input_nodes(req.graph.model_dump())
            if collection_input:
                req.params[collection_input] = Collection(name=name)
            if file_input:
                # Use the temporary file path
                req.params[file_input] = FilePath(path=file_path)

            async for msg in run_workflow(req):
                if msg.get("type") == "job_update":
                    if msg.get("status") == "completed":
                        break
                    elif msg.get("status") == "failed":
                        return IndexResponse(
                            path=file.filename or "unknown", error=msg.get("error")
                        )
        else:
            # Use the temporary file path and determined mime type
            default_ingestion_workflow(collection, file_path, mime_type)

        return IndexResponse(path=file.filename or "unknown", error=None)
    except Exception as e:
        # Catch potential errors during file processing or workflow execution
        return IndexResponse(path=file.filename or "unknown", error=str(e))
    finally:
        # Ensure temporary directory is cleaned up
        shutil.rmtree(tmp_dir)
        await file.close()  # Close the uploaded file handle
