"""Collection management tools.

These tools provide functionality for managing Chroma vector database collections.
"""

from __future__ import annotations

from typing import Any

from nodetool.models.workflow import Workflow as WorkflowModel
# NOTE: ChromaDB imports are done lazily in methods to avoid
# heavy initialization of chromadb/langchain during CLI startup


class CollectionTools:
    """Collection management tools."""

    @staticmethod
    async def list_collections(limit: int = 50) -> dict[str, Any]:
        """
        List all vector database collections.

        Args:
            limit: Maximum number of collections to return (default: 50, max: 100)

        Returns:
            Dictionary with collections list and total count
        """
        if limit > 100:
            limit = 100

        from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
            get_async_chroma_client,
        )

        client = await get_async_chroma_client()
        collections = await client.list_collections()

        async def get_workflow_name(metadata: dict[str, Any]) -> str | None:
            if workflow_id := metadata.get("workflow"):
                workflow = await WorkflowModel.get(workflow_id)
                if workflow:
                    return workflow.name
            return None

        collections = collections[:limit]

        import asyncio

        counts = await asyncio.gather(*(col.count() for col in collections))
        workflows = await asyncio.gather(*(get_workflow_name(col.metadata) for col in collections))

        return {
            "collections": [
                {
                    "name": col.name,
                    "metadata": col.metadata,
                    "workflow_name": wf,
                    "count": count,
                }
                for col, wf, count in zip(collections, workflows, counts, strict=False)
            ],
            "count": len(collections),
        }

    @staticmethod
    async def get_collection(name: str) -> dict[str, Any]:
        """
        Get details about a specific collection.

        Args:
            name: Name of collection

        Returns:
            Collection details including metadata and document count
        """
        from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
            get_async_collection,
        )

        collection = await get_async_collection(name=name)
        count = await collection.count()

        return {
            "name": collection.name,
            "metadata": collection.metadata,
            "count": count,
        }

    @staticmethod
    async def query_collection(
        name: str,
        query_texts: list[str],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Query a collection for similar documents using semantic search.

        Args:
            name: Name of collection to query
            query_texts: List of query texts to search for
            n_results: Number of results to return per query (default: 10, max: 50)
            where: Optional metadata filter (e.g., {"source": "pdf"})

        Returns:
            Query results with ids, documents, distances, and metadatas
        """
        if n_results > 50:
            n_results = 50

        from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
            get_async_collection,
        )

        collection = await get_async_collection(name=name)

        results = await collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
        )

        return {
            "ids": results.get("ids", []),
            "documents": results.get("documents", []),
            "distances": results.get("distances", []),
            "metadatas": results.get("metadatas", []),
        }

    @staticmethod
    async def get_documents_from_collection(
        name: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get documents from a collection by IDs or metadata filter.

        Args:
            name: Name of collection
            ids: Optional list of document IDs to retrieve
            where: Optional metadata filter (e.g., {"source": "pdf"})
            limit: Maximum number of documents to return (default: 50, max: 100)

        Returns:
            Documents with their IDs, texts, and metadatas
        """
        if limit > 100:
            limit = 100

        from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
            get_async_collection,
        )

        collection = await get_async_collection(name=name)

        results = await collection.get(
            ids=ids,
            where=where,
            limit=limit,
        )

        return {
            "ids": results.get("ids", []),
            "documents": results.get("documents", []),
            "metadatas": results.get("metadatas", []),
            "count": len(results.get("ids", [])),
        }

    @staticmethod
    def get_tool_functions() -> dict[str, Any]:
        """Get all collection tool functions."""
        return {
            "list_collections": CollectionTools.list_collections,
            "get_collection": CollectionTools.get_collection,
            "query_collection": CollectionTools.query_collection,
            "get_documents_from_collection": CollectionTools.get_documents_from_collection,
        }
