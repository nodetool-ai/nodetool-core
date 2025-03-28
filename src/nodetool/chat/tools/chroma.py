"""
Search and database tools module.

This module provides tools for semantic and keyword searching:
- ChromaTextSearchTool: Semantic search in ChromaDB
- ChromaHybridSearchTool: Combined semantic/keyword search
- SemanticDocSearchTool: Search documentation semantically
- KeywordDocSearchTool: Search documentation by keywords
"""

from typing import Any
import chromadb
from chromadb.api.types import IncludeEnum
from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool
from pydantic import Field
from nodetool.metadata.types import TextChunk
from typing import Literal


class ChromaTextSearchTool(Tool):
    name = "chroma_text_search"
    description = (
        "Search all ChromaDB collections for similar text using semantic search"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to search for",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 10,
            },
        },
        "required": ["text"],
    }

    def __init__(self, workspace_dir: str, collection: chromadb.Collection):
        super().__init__(workspace_dir)
        self.collection = collection

    async def process(self, context: ProcessingContext, params: dict) -> dict[str, str]:
        result = self.collection.query(
            query_texts=[params["text"]],
            n_results=params.get("n_results", 5),
        )

        # Sort results by ID for consistency
        if result["documents"] is None:
            return {}

        return dict(
            zip(
                result["ids"][0],
                result["documents"][0],
            )
        )


class ChromaIndexTool(Tool):
    name = "chroma_index"
    description = "Index a text chunk into a ChromaDB collection"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text content to index",
            },
            "source_id": {
                "type": "string",
                "description": "Unique identifier for the source of the text",
            },
            "metadata": {
                "type": "object",
                "description": "Metadata to associate with the text chunk",
                "default": {},
            },
        },
        "required": ["text", "source_id"],
    }

    def __init__(self, workspace_dir: str, collection: chromadb.Collection):
        super().__init__(workspace_dir)
        self.collection = collection

    def _generate_document_id(self, source_id: str) -> str:
        """Generate a unique document ID from the source ID."""
        import hashlib

        return f"{source_id}-{hashlib.md5(source_id.encode()).hexdigest()[:8]}"

    async def process(self, context: ProcessingContext, params: dict) -> dict[str, str]:
        text = params["text"]
        source_id = params["source_id"]
        metadata = params.get("metadata", {})

        if not source_id.strip():
            return {"error": "The source ID cannot be empty"}

        document_id = self._generate_document_id(source_id)

        self.collection.add(
            ids=[document_id],
            documents=[text],
            metadatas=[metadata],
        )

        return {
            "status": "success",
            "document_id": document_id,
            "message": f"Successfully indexed text chunk with ID {document_id}",
        }


class ChromaHybridSearchTool(Tool):
    name = "chroma_hybrid_search"
    description = (
        "Search all ChromaDB collections using both semantic and keyword-based search"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to search for",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to return per collection",
                "default": 5,
            },
            "k_constant": {
                "type": "number",
                "description": "Constant for reciprocal rank fusion",
                "default": 60.0,
            },
            "min_keyword_length": {
                "type": "integer",
                "description": "Minimum length for keyword tokens",
                "default": 3,
            },
        },
        "required": ["text"],
    }

    def __init__(self, workspace_dir: str, collection: chromadb.Collection):
        super().__init__(workspace_dir)
        self.collection = collection

    def _get_keyword_query(self, text: str, min_length: int) -> dict:
        import re

        pattern = r"[ ,.!?\-_=|]+"
        query_tokens = [
            token.strip()
            for token in re.split(pattern, text.lower())
            if len(token.strip()) >= min_length
        ]

        if not query_tokens:
            return {}

        if len(query_tokens) > 1:
            return {"$or": [{"$contains": token} for token in query_tokens]}
        return {"$contains": query_tokens[0]}

    async def process(self, context: ProcessingContext, params: dict) -> dict[str, str]:
        try:
            if not params["text"].strip():
                return {"error": "Search text cannot be empty"}

            n_results = params.get("n_results", 5)
            k_constant = params.get("k_constant", 60.0)
            min_keyword_length = params.get("min_keyword_length", 3)

            # Semantic search
            semantic_results = self.collection.query(
                query_texts=[params["text"]],
                n_results=n_results * 2,
                include=[IncludeEnum.documents],
            )

            # Keyword search
            keyword_query = self._get_keyword_query(params["text"], min_keyword_length)
            if keyword_query:
                keyword_results = self.collection.query(
                    query_texts=[params["text"]],
                    n_results=n_results * 2,
                    where_document=keyword_query,
                    include=[IncludeEnum.documents],
                )
            else:
                keyword_results = semantic_results

            # Combine results using reciprocal rank fusion
            combined_scores = {}

            if semantic_results["documents"]:
                # Process semantic results
                for rank, (id_, doc) in enumerate(
                    zip(
                        semantic_results["ids"][0],
                        semantic_results["documents"][0],
                    )
                ):
                    score = 1 / (rank + k_constant)
                    combined_scores[id_] = {"doc": doc, "score": score}

            if keyword_results["documents"]:
                # Process keyword results
                for rank, (id_, doc) in enumerate(
                    zip(
                        keyword_results["ids"][0],
                        keyword_results["documents"][0],
                    )
                ):
                    score = 1 / (rank + k_constant)
                    if id_ in combined_scores:
                        combined_scores[id_]["score"] += score
                    else:
                        combined_scores[id_] = {"doc": doc, "score": score}

            # Sort and take top results
            sorted_results = sorted(
                combined_scores.items(), key=lambda x: x[1]["score"], reverse=True
            )[:n_results]

            # Convert to simple id->document dictionary
            return {str(id_): item["doc"] for id_, item in sorted_results}

        except Exception as e:
            return {"error": str(e)}


class SemanticDocSearchTool(Tool):
    name = "semantic_doc_search"
    description = "Search documentation using semantic similarity"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The text to search for in the documentation",
            },
        },
        "required": ["query"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            from nodetool.chat.help import semantic_search_documentation

            results = semantic_search_documentation(params["query"])
            return {
                "results": [
                    {
                        "id": result.id,
                        "content": result.content,
                        "metadata": result.metadata,
                    }
                    for result in results
                ]
            }
        except Exception as e:
            return {"error": str(e)}


class KeywordDocSearchTool(Tool):
    name = "keyword_doc_search"
    description = "Search documentation using keyword matching"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The text to search for in the documentation",
            },
        },
        "required": ["query"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            from nodetool.chat.help import keyword_search_documentation

            results = keyword_search_documentation(params["query"])
            return {
                "results": [
                    {
                        "id": result.id,
                        "content": result.content,
                        "metadata": result.metadata,
                    }
                    for result in results
                ]
            }
        except Exception as e:
            return {"error": str(e)}


class ChromaRecursiveSplitAndIndexTool(Tool):
    name = "chroma_recursive_split_and_index"
    description = (
        "Split text into chunks recursively and index them into a ChromaDB collection"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text content to split and index",
            },
            "document_id": {
                "type": "string",
                "description": "Base identifier for the source document",
            },
            "chunk_size": {
                "type": "integer",
                "description": "Maximum size of each chunk in characters",
                "default": 1000,
            },
            "chunk_overlap": {
                "type": "integer",
                "description": "Number of characters to overlap between chunks",
                "default": 200,
            },
            "separators": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of separators for recursive splitting",
                "default": ["\n\n", "\n", "."],
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata to associate with all chunks",
                "default": {},
            },
        },
        "required": ["text", "document_id"],
    }

    def __init__(self, workspace_dir: str, collection: chromadb.Collection):
        super().__init__(workspace_dir)
        self.collection = collection

    async def _split_text_recursive(
        self, text: str, document_id: str, params: dict
    ) -> list[TextChunk]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params.get("chunk_size", 1000),
            chunk_overlap=params.get("chunk_overlap", 200),
            separators=params.get("separators", ["\n\n", "\n", "."]),
            length_function=len,
            is_separator_regex=False,
            add_start_index=True,
        )

        docs = splitter.split_documents([Document(page_content=text)])

        return [
            TextChunk(
                text=doc.page_content,
                source_id=f"{document_id}:{i}",
                start_index=doc.metadata.get("start_index", 0),
            )
            for i, doc in enumerate(docs)
        ]

    def _generate_document_id(self, source_id: str) -> str:
        """Generate a unique document ID from the source ID."""
        import hashlib

        return f"{source_id}-{hashlib.md5(source_id.encode()).hexdigest()[:8]}"

    async def process(self, context: ProcessingContext, params: dict) -> dict[str, Any]:
        text = params["text"]
        document_id = params["document_id"]
        base_metadata = params.get("metadata", {})

        if not text.strip():
            return {"error": "The text cannot be empty"}

        if not document_id.strip():
            return {"error": "The document ID cannot be empty"}

        # Split the text
        try:
            chunks = await self._split_text_recursive(text, document_id, params)
        except Exception as e:
            return {"error": f"Text splitting failed: {str(e)}"}

        # Index each chunk
        indexed_ids = []
        try:
            for i, chunk in enumerate(chunks):
                # Generate a unique ID for this chunk
                unique_id = self._generate_document_id(f"{chunk.source_id}:{i}")

                # Combine base metadata with chunk-specific metadata
                metadata = {**base_metadata}
                if hasattr(chunk, "start_index"):
                    metadata["start_index"] = chunk.start_index

                # Index the chunk
                self.collection.add(
                    ids=[unique_id],
                    documents=[chunk.text],
                    metadatas=[metadata],
                )
                indexed_ids.append(unique_id)

        except Exception as e:
            return {
                "error": f"Indexing failed: {str(e)}",
                "indexed_count": len(indexed_ids),
                "total_chunks": len(chunks),
            }

        return {
            "status": "success",
            "indexed_count": len(indexed_ids),
            "document_id": document_id,
            "message": f"Successfully indexed {len(indexed_ids)} chunks from document {document_id}",
        }


class ChromaMarkdownSplitAndIndexTool(Tool):
    name = "chroma_markdown_split_and_index"
    description = "Split markdown text into chunks based on headers and index them into a ChromaDB collection"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The markdown text content to split and index",
            },
            "document_id": {
                "type": "string",
                "description": "Base identifier for the source document",
            },
            "headers_to_split_on": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "description": "List of header patterns and their names",
                "default": [
                    ["#", "Header 1"],
                    ["##", "Header 2"],
                    ["###", "Header 3"],
                ],
            },
            "strip_headers": {
                "type": "boolean",
                "description": "Whether to strip headers from text chunks",
                "default": True,
            },
            "return_each_line": {
                "type": "boolean",
                "description": "Whether to return each line as a separate document",
                "default": False,
            },
            "chunk_size": {
                "type": "integer",
                "description": "Maximum size of each chunk in characters (for further splitting)",
                "default": None,
            },
            "chunk_overlap": {
                "type": "integer",
                "description": "Number of characters to overlap between chunks",
                "default": 200,
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata to associate with all chunks",
                "default": {},
            },
        },
        "required": ["text", "document_id"],
    }

    def __init__(self, workspace_dir: str, collection: chromadb.Collection):
        super().__init__(workspace_dir)
        self.collection = collection

    async def _split_text_markdown(
        self, text: str, document_id: str, params: dict
    ) -> list[TextChunk]:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        headers_to_split_on = params.get(
            "headers_to_split_on",
            [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
        )

        strip_headers = params.get("strip_headers", True)
        return_each_line = params.get("return_each_line", False)

        # Initialize markdown splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=strip_headers,
            return_each_line=return_each_line,
        )

        # Split by headers
        splits = markdown_splitter.split_text(text)

        # Further split by chunk size if specified
        chunk_size = params.get("chunk_size")
        if chunk_size:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=params.get("chunk_overlap", 200)
            )
            splits = text_splitter.split_documents(splits)

        return [
            TextChunk(
                text=doc.page_content,
                source_id=f"{document_id}:{i}",
                start_index=doc.metadata.get("start_index", 0),
            )
            for i, doc in enumerate(splits)
        ]

    def _generate_document_id(self, source_id: str) -> str:
        """Generate a unique document ID from the source ID."""
        import hashlib

        return f"{source_id}-{hashlib.md5(source_id.encode()).hexdigest()[:8]}"

    async def process(self, context: ProcessingContext, params: dict) -> dict[str, Any]:
        text = params["text"]
        document_id = params["document_id"]
        base_metadata = params.get("metadata", {})

        if not text.strip():
            return {"error": "The text cannot be empty"}

        if not document_id.strip():
            return {"error": "The document ID cannot be empty"}

        # Split the text
        try:
            chunks = await self._split_text_markdown(text, document_id, params)
        except Exception as e:
            return {"error": f"Text splitting failed: {str(e)}"}

        # Index each chunk
        indexed_ids = []
        try:
            for i, chunk in enumerate(chunks):
                # Generate a unique ID for this chunk
                unique_id = self._generate_document_id(f"{chunk.source_id}:{i}")

                # Combine base metadata with chunk-specific metadata
                metadata = {**base_metadata}
                if hasattr(chunk, "start_index"):
                    metadata["start_index"] = chunk.start_index

                # Index the chunk
                self.collection.add(
                    ids=[unique_id],
                    documents=[chunk.text],
                    metadatas=[metadata],
                )
                indexed_ids.append(unique_id)

        except Exception as e:
            return {
                "error": f"Indexing failed: {str(e)}",
                "indexed_count": len(indexed_ids),
                "total_chunks": len(chunks),
            }

        return {
            "status": "success",
            "indexed_count": len(indexed_ids),
            "document_id": document_id,
            "message": f"Successfully indexed {len(indexed_ids)} chunks from document {document_id}",
        }


class ChromaBatchIndexTool(Tool):
    name = "chroma_batch_index"
    description = "Index a batch of text chunks into a ChromaDB collection"
    input_schema = {
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source_id": {"type": "string"},
                        "metadata": {"type": "object", "default": {}},
                    },
                    "required": ["text", "source_id"],
                },
                "description": "List of text chunks to index",
            },
            "base_metadata": {
                "type": "object",
                "description": "Base metadata to add to all chunks",
                "default": {},
            },
        },
        "required": ["chunks"],
    }

    def __init__(self, workspace_dir: str, collection: chromadb.Collection):
        super().__init__(workspace_dir)
        self.collection = collection

    def _generate_document_id(self, source_id: str) -> str:
        """Generate a unique document ID from the source ID."""
        import hashlib

        return f"{source_id}-{hashlib.md5(source_id.encode()).hexdigest()[:8]}"

    async def process(self, context: ProcessingContext, params: dict) -> dict[str, Any]:
        chunks = params["chunks"]
        base_metadata = params.get("base_metadata", {})

        if not chunks:
            return {"error": "No chunks provided"}

        # Prepare batch for indexing
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            if not chunk.get("text") or not chunk.get("source_id"):
                continue

            # Generate a unique ID for this chunk
            chunk_id = self._generate_document_id(chunk["source_id"])

            # Combine base metadata with chunk-specific metadata
            combined_metadata = {**base_metadata, **(chunk.get("metadata", {}))}

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(combined_metadata)

        if not ids:
            return {"error": "No valid chunks to index"}

        try:
            # Batch add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

            return {
                "status": "success",
                "indexed_count": len(ids),
                "message": f"Successfully indexed {len(ids)} chunks",
            }
        except Exception as e:
            return {"error": f"Indexing failed: {str(e)}"}
