import pytest

from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    AsyncChromaClient,
    AsyncChromaCollection,
    get_async_chroma_client,
)


@pytest.mark.asyncio
async def test_async_client_create_add_query_delete(tmp_path, monkeypatch):
    # Ensure local, on-disk Chroma
    monkeypatch.delenv("CHROMA_URL", raising=False)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

    client: AsyncChromaClient = await get_async_chroma_client()
    try:
        # Create collection
        collection: AsyncChromaCollection = await client.create_collection(name="test_async_chroma")

        # Add a couple of documents with explicit embeddings (avoid external embedder)
        ids = ["doc-a", "doc-b"]
        docs = ["hello world", "goodbye world"]
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.1, 0.0, 0.3],
        ]

        await collection.add(ids=ids, documents=docs, embeddings=embeddings)

        # Count should reflect inserts
        count = await collection.count()
        assert count == 2

        # Query using embeddings; expect at least one id back
        result = await collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=1)
        assert result["ids"] is not None
        assert len(result["ids"][0]) == 1
        assert result["ids"][0][0] in ids

        # Delete collection
        await client.delete_collection("test_async_chroma")

    finally:
        # Ensure executor thread is stopped
        client.shutdown()


@pytest.mark.asyncio
async def test_list_and_modify(tmp_path, monkeypatch):
    # Isolated on-disk Chroma environment
    monkeypatch.delenv("CHROMA_URL", raising=False)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

    client: AsyncChromaClient = await get_async_chroma_client()
    try:
        # Initial collections count
        initial_count = await client.count_collections()

        # Create then list
        collection = await client.create_collection(name="modify_me")
        after_create_count = await client.count_collections()
        assert after_create_count == initial_count + 1

        # Modify metadata
        await collection.modify(metadata={"stage": "test"})
        assert collection.metadata.get("stage") == "test"

        # Upsert additional doc and query
        await collection.upsert(ids=["d1"], documents=["alpha beta"], embeddings=[[0.0, 0.1, 0.0, 0.2]])
        res = await collection.query(query_embeddings=[[0.0, 0.1, 0.0, 0.2]], n_results=1)
        assert res["ids"] is not None and "d1" in res["ids"][0]

        # Cleanup
        await client.delete_collection("modify_me")
        final_count = await client.count_collections()
        assert final_count == initial_count

    finally:
        client.shutdown()
