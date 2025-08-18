"""
Indexing utilities: document chunking and default ingestion workflow.

Extracted from nodetool.api.collection to allow reuse in the lightweight server.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import chromadb
from typing import Any
from markitdown import MarkItDown
import pymupdf
import pymupdf4llm


class Document:
    def __init__(self, text: str, doc_id: str, metadata: Dict[str, str] | None = None):
        self.text = text
        self.doc_id = doc_id
        self.metadata = metadata or {}


def chunk_documents_recursive(
    documents: List[Document],
    chunk_size: int = 4096,
    chunk_overlap: int = 2048,
) -> tuple[dict[str, str], list[dict]]:
    """Split documents using LangChain recursive character splitter.

    Returns tuple of (id_to_text_mapping, metadata_list)
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    ids_docs: Dict[str, str] = {}
    metadatas: List[dict] = []

    for doc in documents:
        splits = splitter.split_text(doc.text)
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
    """Split markdown documents by headers then recursively.

    Returns tuple of (id_to_text_mapping, metadata_list)
    """
    from langchain_text_splitters import (
        ExperimentalMarkdownSyntaxTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    ids_docs: Dict[str, str] = {}
    metadatas: List[dict] = []
    chunk_index = 0

    for doc in documents:
        md_splits = markdown_splitter.split_text(doc.text)
        final_splits = recursive_splitter.split_documents(md_splits)

        for split_doc in final_splits:
            doc_id = f"{doc.doc_id}:{chunk_index}"
            ids_docs[doc_id] = split_doc.page_content
            metadatas.append(doc.metadata.copy())
            chunk_index += 1
        chunk_index = 0

    return ids_docs, metadatas


def default_ingestion_workflow(
    collection: chromadb.Collection, file_path: str, mime_type: str
) -> None:
    """Process a file and add it to the collection using default ingestion workflow."""
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

    ids_docs, _ = chunk_documents_markdown(
        documents,
        chunk_size=4096,
        chunk_overlap=256,
    )
    collection.upsert(
        documents=list(ids_docs.values()),
        ids=list(ids_docs.keys()),
    )


async def default_ingestion_workflow_async(
    collection: Any, file_path: str, mime_type: str
) -> None:
    """Async version of default ingestion that works with AsyncChromaCollection."""
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

    ids_docs, _ = chunk_documents_markdown(
        documents,
        chunk_size=4096,
        chunk_overlap=256,
    )
    await collection.upsert(
        documents=list(ids_docs.values()),
        ids=list(ids_docs.keys()),
    )


def find_input_nodes(graph: dict) -> Tuple[str | None, str | None]:
    """Find the collection and file input node names from a workflow graph."""
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
