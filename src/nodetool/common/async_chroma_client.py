"""
Async ChromaDB client wrappers.

This module provides asynchronous wrappers around the blocking ChromaDB
client and collection APIs. All blocking calls are executed on a dedicated
single background thread per async client, ensuring compatibility with
async applications while respecting Chroma's synchronous interfaces.

Key classes:
    - AsyncChromaClient: wraps a Chroma ClientAPI
    - AsyncChromaCollection: wraps a Chroma Collection

Helpers:
    - get_async_chroma_client(user_id): wraps `get_chroma_client`
    - get_async_collection(name): wraps `get_collection` with embedding config
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Optional, Iterable
import asyncio

import chromadb
from chromadb.api import ClientAPI

from .chroma_client import get_chroma_client, get_collection


class _SingleThreadExecutor:
    """
    A dedicated single-thread executor for running blocking calls.
    """

    def __init__(self, thread_name_prefix: str = "chroma-io") -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name_prefix
        )

    async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, partial(func, *args, **kwargs)
        )

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


class AsyncChromaCollection:
    """
    Async wrapper around a Chroma `Collection`.

    All method calls are executed on a dedicated background thread via
    the owning client's executor.
    """

    def __init__(self, collection: Any, executor: _SingleThreadExecutor) -> None:
        self._collection = collection
        self._executor = executor

        # Cache simple metadata for quick access without roundtrips
        self.name: str = collection.name
        # chromadb sets metadata to Optional[dict]
        self.metadata: dict[str, Any] = collection.metadata or {}

    # Core read operations
    async def count(self) -> int:
        return await self._executor.run(self._collection.count)

    async def get(
        self,
        ids: Optional[Iterable[str]] = None,
        where: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[Iterable] = None,
    ) -> dict:
        return await self._executor.run(
            self._collection.get,
            ids=ids,
            where=where,
            limit=limit,
            offset=offset,
            include=include or [],
        )

    async def peek(self, limit: int = 10, include: Optional[Iterable] = None) -> dict:
        return await self._executor.run(
            self._collection.peek, limit=limit, include=include or []
        )

    async def query(
        self,
        query_texts: Optional[Iterable[str]] = None,
        query_embeddings: Optional[Iterable[Iterable[float]]] = None,
        query_images: Optional[Iterable] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[Iterable] = None,
    ) -> dict:
        return await self._executor.run(
            self._collection.query,
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            query_images=query_images,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include or [],
        )

    # Mutations
    async def add(
        self,
        ids: Iterable[str],
        documents: Optional[Iterable[str]] = None,
        embeddings: Optional[Iterable[Iterable[float]]] = None,
        metadatas: Optional[Iterable[dict]] = None,
        images: Optional[Iterable] = None,
    ) -> None:
        await self._executor.run(
            self._collection.add,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            images=images,
        )

    async def upsert(
        self,
        ids: Iterable[str],
        documents: Optional[Iterable[str]] = None,
        embeddings: Optional[Iterable[Iterable[float]]] = None,
        metadatas: Optional[Iterable[dict]] = None,
        images: Optional[Iterable] = None,
    ) -> None:
        await self._executor.run(
            self._collection.upsert,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            images=images,
        )

    async def modify(
        self,
        name: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        await self._executor.run(self._collection.modify, name=name, metadata=metadata)
        # Update cached values on success
        if name:
            self.name = name
        if metadata is not None:
            self.metadata = metadata

    async def delete(
        self, ids: Optional[Iterable[str]] = None, where: Optional[dict] = None
    ) -> None:
        await self._executor.run(self._collection.delete, ids=ids, where=where)


class AsyncChromaClient:
    """
    Async wrapper around a Chroma `ClientAPI` using a dedicated single thread.
    """

    def __init__(
        self, client: ClientAPI, thread_name_prefix: str = "chroma-io"
    ) -> None:
        self._client = client
        self._executor = _SingleThreadExecutor(thread_name_prefix=thread_name_prefix)

    # Lifecycle
    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait)

    # Client operations
    async def list_collections(
        self,
    ) -> list[chromadb.Collection]:
        return await self._executor.run(self._client.list_collections)

    async def count_collections(self) -> int:
        collections = await self.list_collections()
        return len(collections)

    async def get_collection(
        self, name: str, embedding_function: Any | None = None
    ) -> AsyncChromaCollection:
        collection = await self._executor.run(
            self._client.get_collection,
            name=name,
            embedding_function=embedding_function,
        )
        return AsyncChromaCollection(collection, self._executor)

    async def get_or_create_collection(
        self, name: str, metadata: Optional[dict[str, Any]] = None
    ) -> AsyncChromaCollection:
        collection = await self._executor.run(
            self._client.get_or_create_collection, name=name, metadata=metadata
        )
        return AsyncChromaCollection(collection, self._executor)

    async def create_collection(
        self, name: str, metadata: Optional[dict[str, Any]] = None
    ) -> AsyncChromaCollection:
        collection = await self._executor.run(
            self._client.create_collection, name=name, metadata=metadata
        )
        return AsyncChromaCollection(collection, self._executor)

    async def delete_collection(self, name: str) -> None:
        await self._executor.run(self._client.delete_collection, name=name)


async def get_async_chroma_client(user_id: str | None = None) -> AsyncChromaClient:
    """
    Create an AsyncChromaClient wrapping the synchronous client from `get_chroma_client`.
    """
    loop = asyncio.get_running_loop()
    client: ClientAPI = await loop.run_in_executor(
        None, partial(get_chroma_client, user_id)
    )
    return AsyncChromaClient(client)


async def get_async_collection(name: str) -> AsyncChromaCollection:
    """
    Get a collection by name.
    """
    client = await get_async_chroma_client()
    return await client.get_collection(name)
