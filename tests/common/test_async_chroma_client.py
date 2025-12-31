"""
Comprehensive tests for AsyncChromaClient and AsyncChromaCollection.

These tests use a real ChromaDB instance (on-disk persistent client) to ensure
all functionality works correctly with actual data storage and retrieval.
"""

import chromadb
import pytest
import pytest_asyncio
from packaging import version

from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    AsyncChromaClient,
    AsyncChromaCollection,
    get_async_chroma_client,
)

# Version checks for feature compatibility
CHROMADB_VERSION = version.parse(chromadb.__version__)
PEEK_INCLUDE_SUPPORT = CHROMADB_VERSION >= version.parse("1.5.0")  # peek() include param


@pytest_asyncio.fixture
async def chroma_client(tmp_path, monkeypatch):
    """Create an isolated ChromaDB client for testing."""
    monkeypatch.delenv("CHROMA_URL", raising=False)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

    client: AsyncChromaClient = await get_async_chroma_client()
    try:
        yield client
    finally:
        client.shutdown()


@pytest_asyncio.fixture
async def test_collection(chroma_client):
    """Create a test collection and clean it up after the test."""
    collection_name = "test_collection"
    collection = await chroma_client.create_collection(name=collection_name)
    try:
        yield collection
    finally:
        try:
            await chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass


# =============================================================================
# AsyncChromaClient Tests
# =============================================================================


class TestAsyncChromaClient:
    """Tests for AsyncChromaClient operations."""

    @pytest.mark.asyncio
    async def test_create_collection(self, chroma_client):
        """Test creating a new collection."""
        collection = await chroma_client.create_collection(name="new_collection")
        assert collection is not None
        assert collection.name == "new_collection"
        await chroma_client.delete_collection("new_collection")

    @pytest.mark.asyncio
    async def test_create_collection_with_metadata(self, chroma_client):
        """Test creating a collection with metadata."""
        metadata = {"description": "Test collection", "version": "1.0"}
        collection = await chroma_client.create_collection(name="metadata_collection", metadata=metadata)
        assert collection.name == "metadata_collection"
        assert collection.metadata.get("description") == "Test collection"
        assert collection.metadata.get("version") == "1.0"
        await chroma_client.delete_collection("metadata_collection")

    @pytest.mark.asyncio
    async def test_get_collection(self, chroma_client):
        """Test retrieving an existing collection."""
        await chroma_client.create_collection(name="get_test")
        collection = await chroma_client.get_collection(name="get_test")
        assert collection is not None
        assert collection.name == "get_test"
        await chroma_client.delete_collection("get_test")

    @pytest.mark.asyncio
    async def test_get_or_create_collection_creates(self, chroma_client):
        """Test get_or_create when collection doesn't exist."""
        collection = await chroma_client.get_or_create_collection(name="get_or_create_test")
        assert collection is not None
        assert collection.name == "get_or_create_test"
        await chroma_client.delete_collection("get_or_create_test")

    @pytest.mark.asyncio
    async def test_get_or_create_collection_gets_existing(self, chroma_client):
        """Test get_or_create when collection already exists."""
        metadata = {"original": "true"}
        await chroma_client.create_collection(name="existing_col", metadata=metadata)
        collection = await chroma_client.get_or_create_collection(name="existing_col")
        assert collection.name == "existing_col"
        assert collection.metadata.get("original") == "true"
        await chroma_client.delete_collection("existing_col")

    @pytest.mark.asyncio
    async def test_list_collections(self, chroma_client):
        """Test listing all collections."""
        initial = await chroma_client.list_collections()
        initial_count = len(initial)

        await chroma_client.create_collection(name="list_test_1")
        await chroma_client.create_collection(name="list_test_2")

        collections = await chroma_client.list_collections()
        assert len(collections) == initial_count + 2

        collection_names = [c.name for c in collections]
        assert "list_test_1" in collection_names
        assert "list_test_2" in collection_names

        await chroma_client.delete_collection("list_test_1")
        await chroma_client.delete_collection("list_test_2")

    @pytest.mark.asyncio
    async def test_count_collections(self, chroma_client):
        """Test counting collections."""
        initial_count = await chroma_client.count_collections()

        await chroma_client.create_collection(name="count_test_1")
        await chroma_client.create_collection(name="count_test_2")

        new_count = await chroma_client.count_collections()
        assert new_count == initial_count + 2

        await chroma_client.delete_collection("count_test_1")
        await chroma_client.delete_collection("count_test_2")

    @pytest.mark.asyncio
    async def test_delete_collection(self, chroma_client):
        """Test deleting a collection."""
        await chroma_client.create_collection(name="to_delete")
        initial_count = await chroma_client.count_collections()

        await chroma_client.delete_collection("to_delete")

        final_count = await chroma_client.count_collections()
        assert final_count == initial_count - 1

    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection_raises(self, chroma_client):
        """Test that deleting a non-existent collection raises an error."""
        with pytest.raises(Exception):
            await chroma_client.delete_collection("nonexistent_collection")


# =============================================================================
# AsyncChromaCollection Tests - Basic Operations
# =============================================================================


class TestAsyncChromaCollectionBasicOperations:
    """Tests for basic AsyncChromaCollection operations."""

    @pytest.mark.asyncio
    async def test_collection_count_empty(self, test_collection):
        """Test counting documents in an empty collection."""
        count = await test_collection.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_add_documents_with_embeddings(self, test_collection):
        """Test adding documents with explicit embeddings."""
        ids = ["doc1", "doc2", "doc3"]
        documents = ["First document", "Second document", "Third document"]
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)

        count = await test_collection.count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_add_documents_with_metadata(self, test_collection):
        """Test adding documents with metadata."""
        ids = ["meta1", "meta2"]
        documents = ["Document with metadata", "Another doc with metadata"]
        embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        metadatas = [
            {"source": "file1.txt", "page": 1},
            {"source": "file2.txt", "page": 2},
        ]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

        result = await test_collection.get(ids=["meta1"], include=["metadatas"])
        assert result["metadatas"] is not None
        assert result["metadatas"][0]["source"] == "file1.txt"
        assert result["metadatas"][0]["page"] == 1

    @pytest.mark.asyncio
    async def test_get_by_ids(self, test_collection):
        """Test retrieving documents by IDs."""
        ids = ["get1", "get2", "get3"]
        documents = ["Get doc 1", "Get doc 2", "Get doc 3"]
        embeddings = [[0.1] * 4, [0.2] * 4, [0.3] * 4]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)

        result = await test_collection.get(ids=["get1", "get3"], include=["documents"])
        assert len(result["ids"]) == 2
        assert "get1" in result["ids"]
        assert "get3" in result["ids"]
        assert "Get doc 1" in result["documents"]
        assert "Get doc 3" in result["documents"]

    @pytest.mark.asyncio
    async def test_get_all_documents(self, test_collection):
        """Test retrieving all documents from a collection."""
        ids = ["all1", "all2"]
        documents = ["All doc 1", "All doc 2"]
        embeddings = [[0.1] * 4, [0.2] * 4]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)

        result = await test_collection.get(include=["documents"])
        assert len(result["ids"]) == 2
        assert "all1" in result["ids"]
        assert "all2" in result["ids"]

    @pytest.mark.asyncio
    async def test_get_with_limit_and_offset(self, test_collection):
        """Test retrieving documents with pagination."""
        ids = [f"page{i}" for i in range(10)]
        documents = [f"Document {i}" for i in range(10)]
        embeddings = [[0.1 * i] * 4 for i in range(10)]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)

        # Get first 3
        result1 = await test_collection.get(limit=3, include=["documents"])
        assert len(result1["ids"]) == 3

        # Get next 3
        result2 = await test_collection.get(limit=3, offset=3, include=["documents"])
        assert len(result2["ids"]) == 3
        # Ensure different documents
        assert set(result1["ids"]) != set(result2["ids"])


# =============================================================================
# AsyncChromaCollection Tests - Query Operations
# =============================================================================


class TestAsyncChromaCollectionQueryOperations:
    """Tests for query operations on AsyncChromaCollection."""

    @pytest.mark.asyncio
    async def test_query_by_embeddings(self, test_collection):
        """Test querying by embeddings."""
        ids = ["q1", "q2", "q3"]
        documents = ["Query doc 1", "Query doc 2", "Query doc 3"]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)

        # Query with embedding similar to first document
        result = await test_collection.query(query_embeddings=[[1.0, 0.0, 0.0, 0.0]], n_results=1)
        assert result["ids"][0][0] == "q1"

    @pytest.mark.asyncio
    async def test_query_n_results(self, test_collection):
        """Test querying with different n_results values."""
        ids = [f"nres{i}" for i in range(5)]
        documents = [f"N results doc {i}" for i in range(5)]
        embeddings = [[i * 0.1] * 4 for i in range(5)]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)

        result2 = await test_collection.query(query_embeddings=[[0.1] * 4], n_results=2)
        assert len(result2["ids"][0]) == 2

        result5 = await test_collection.query(query_embeddings=[[0.1] * 4], n_results=5)
        assert len(result5["ids"][0]) == 5

    @pytest.mark.asyncio
    async def test_query_with_include(self, test_collection):
        """Test querying with include options."""
        ids = ["inc1"]
        documents = ["Include test doc"]
        embeddings = [[0.1, 0.2, 0.3, 0.4]]
        metadatas = [{"key": "value"}]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

        # Include documents and metadatas
        result = await test_collection.query(
            query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=1, include=["documents", "metadatas"]
        )
        assert "documents" in result
        assert "metadatas" in result
        assert result["documents"][0][0] == "Include test doc"
        assert result["metadatas"][0][0]["key"] == "value"

    @pytest.mark.asyncio
    async def test_query_with_where_filter(self, test_collection):
        """Test querying with where filter on metadata."""
        ids = ["filter1", "filter2", "filter3"]
        documents = ["Filter doc 1", "Filter doc 2", "Filter doc 3"]
        embeddings = [[0.1] * 4, [0.2] * 4, [0.3] * 4]
        metadatas = [
            {"category": "A", "priority": 1},
            {"category": "B", "priority": 2},
            {"category": "A", "priority": 3},
        ]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

        # Filter by category
        result = await test_collection.query(
            query_embeddings=[[0.1] * 4], n_results=3, where={"category": "A"}, include=["documents"]
        )
        assert len(result["ids"][0]) == 2
        assert "filter1" in result["ids"][0]
        assert "filter3" in result["ids"][0]
        assert "filter2" not in result["ids"][0]

    @pytest.mark.asyncio
    async def test_query_with_where_document_filter(self, test_collection):
        """Test querying with where_document filter."""
        ids = ["wdoc1", "wdoc2", "wdoc3"]
        documents = ["The quick brown fox", "A slow red dog", "A quick blue cat"]
        embeddings = [[0.1] * 4, [0.2] * 4, [0.3] * 4]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)

        result = await test_collection.query(
            query_embeddings=[[0.1] * 4], n_results=3, where_document={"$contains": "quick"}, include=["documents"]
        )
        assert len(result["ids"][0]) == 2
        assert "wdoc1" in result["ids"][0]
        assert "wdoc3" in result["ids"][0]


# =============================================================================
# AsyncChromaCollection Tests - Update Operations
# =============================================================================


class TestAsyncChromaCollectionUpdateOperations:
    """Tests for update operations on AsyncChromaCollection."""

    @pytest.mark.asyncio
    async def test_upsert_new_documents(self, test_collection):
        """Test upserting new documents."""
        ids = ["ups1", "ups2"]
        documents = ["Upsert doc 1", "Upsert doc 2"]
        embeddings = [[0.1] * 4, [0.2] * 4]

        await test_collection.upsert(ids=ids, documents=documents, embeddings=embeddings)

        count = await test_collection.count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_upsert_existing_documents(self, test_collection):
        """Test upserting existing documents updates them."""
        # Add initial document
        await test_collection.add(ids=["ups_exist"], documents=["Original"], embeddings=[[0.1] * 4])

        # Upsert with new content
        await test_collection.upsert(ids=["ups_exist"], documents=["Updated"], embeddings=[[0.2] * 4])

        result = await test_collection.get(ids=["ups_exist"], include=["documents"])
        assert result["documents"][0] == "Updated"

        # Count should remain 1
        count = await test_collection.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_upsert_with_metadata(self, test_collection):
        """Test upserting with metadata."""
        await test_collection.upsert(
            ids=["ups_meta"], documents=["Metadata doc"], embeddings=[[0.1] * 4], metadatas=[{"version": 1}]
        )

        await test_collection.upsert(
            ids=["ups_meta"], documents=["Metadata doc v2"], embeddings=[[0.1] * 4], metadatas=[{"version": 2}]
        )

        result = await test_collection.get(ids=["ups_meta"], include=["documents", "metadatas"])
        assert result["documents"][0] == "Metadata doc v2"
        assert result["metadatas"][0]["version"] == 2

    @pytest.mark.asyncio
    async def test_modify_collection_metadata(self, test_collection):
        """Test modifying collection metadata."""
        original_metadata = test_collection.metadata.copy()

        await test_collection.modify(metadata={"new_key": "new_value"})

        assert test_collection.metadata.get("new_key") == "new_value"


# =============================================================================
# AsyncChromaCollection Tests - Delete Operations
# =============================================================================


class TestAsyncChromaCollectionDeleteOperations:
    """Tests for delete operations on AsyncChromaCollection."""

    @pytest.mark.asyncio
    async def test_delete_by_ids(self, test_collection):
        """Test deleting documents by IDs."""
        ids = ["del1", "del2", "del3"]
        documents = ["Delete doc 1", "Delete doc 2", "Delete doc 3"]
        embeddings = [[0.1] * 4, [0.2] * 4, [0.3] * 4]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)
        assert await test_collection.count() == 3

        await test_collection.delete(ids=["del1", "del3"])

        count = await test_collection.count()
        assert count == 1

        result = await test_collection.get(include=["documents"])
        assert "del2" in result["ids"]
        assert "del1" not in result["ids"]
        assert "del3" not in result["ids"]

    @pytest.mark.asyncio
    async def test_delete_by_where_filter(self, test_collection):
        """Test deleting documents by where filter."""
        ids = ["del_filter1", "del_filter2", "del_filter3"]
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = [[0.1] * 4, [0.2] * 4, [0.3] * 4]
        metadatas = [
            {"to_delete": True},
            {"to_delete": False},
            {"to_delete": True},
        ]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

        await test_collection.delete(where={"to_delete": True})

        count = await test_collection.count()
        assert count == 1

        result = await test_collection.get(include=["documents"])
        assert "del_filter2" in result["ids"]


# =============================================================================
# AsyncChromaCollection Tests - Peek Operations
# =============================================================================


class TestAsyncChromaCollectionPeekOperations:
    """Tests for peek operations on AsyncChromaCollection.

    Note: The current implementation of peek() in AsyncChromaCollection passes
    an 'include' parameter which is not supported in ChromaDB < 1.5.0. These tests
    are automatically skipped based on the ChromaDB version.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PEEK_INCLUDE_SUPPORT, reason="ChromaDB < 1.5.0 does not support include parameter in peek()")
    async def test_peek_default(self, test_collection):
        """Test peeking with default limit."""
        ids = [f"peek{i}" for i in range(15)]
        documents = [f"Peek doc {i}" for i in range(15)]
        embeddings = [[0.1 * i] * 4 for i in range(15)]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)

        result = await test_collection.peek(limit=10)
        assert len(result["ids"]) == 10

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PEEK_INCLUDE_SUPPORT, reason="ChromaDB < 1.5.0 does not support include parameter in peek()")
    async def test_peek_with_limit(self, test_collection):
        """Test peeking with custom limit."""
        ids = [f"peek_lim{i}" for i in range(10)]
        documents = [f"Peek limit doc {i}" for i in range(10)]
        embeddings = [[0.1 * i] * 4 for i in range(10)]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)

        result = await test_collection.peek(limit=5)
        assert len(result["ids"]) == 5

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PEEK_INCLUDE_SUPPORT, reason="ChromaDB < 1.5.0 does not support include parameter in peek()")
    async def test_peek_returns_documents_and_metadata(self, test_collection):
        """Test that peek returns documents and metadata by default."""
        ids = ["peek_inc1"]
        documents = ["Peek include doc"]
        embeddings = [[0.1, 0.2, 0.3, 0.4]]
        metadatas = [{"key": "value"}]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

        result = await test_collection.peek(limit=1)
        assert result["documents"][0] == "Peek include doc"
        assert result["metadatas"][0]["key"] == "value"


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestAsyncChromaEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_collection_query(self, test_collection):
        """Test querying an empty collection."""
        result = await test_collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=5)
        assert len(result["ids"][0]) == 0

    @pytest.mark.asyncio
    async def test_duplicate_id_add_behavior(self, test_collection):
        """Test duplicate ID behavior during add.

        Note: ChromaDB's handling of duplicate IDs may vary by version.
        This test documents the current behavior where duplicates are silently ignored.
        """
        await test_collection.add(ids=["dup1"], documents=["Original"], embeddings=[[0.1] * 4])

        # Add duplicate - behavior may be version-dependent
        await test_collection.add(ids=["dup1"], documents=["Duplicate"], embeddings=[[0.2] * 4])

        # Count should still be 1 (duplicate ignored in current version)
        count = await test_collection.count()
        assert count == 1

        # Original document should be unchanged
        result = await test_collection.get(ids=["dup1"], include=["documents"])
        assert result["documents"][0] == "Original"

    @pytest.mark.asyncio
    async def test_get_nonexistent_ids(self, test_collection):
        """Test getting non-existent IDs returns empty results."""
        result = await test_collection.get(ids=["nonexistent1", "nonexistent2"])
        assert len(result["ids"]) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_ids(self, test_collection):
        """Test deleting non-existent IDs doesn't raise an error."""
        # This should not raise an error
        await test_collection.delete(ids=["nonexistent"])

    @pytest.mark.asyncio
    async def test_large_batch_operations(self, test_collection):
        """Test operations with larger batches of documents."""
        batch_size = 100
        ids = [f"batch{i}" for i in range(batch_size)]
        documents = [f"Batch document {i}" for i in range(batch_size)]
        embeddings = [[i * 0.01] * 4 for i in range(batch_size)]

        await test_collection.add(ids=ids, documents=documents, embeddings=embeddings)

        count = await test_collection.count()
        assert count == batch_size

        # Query should return results
        result = await test_collection.query(query_embeddings=[[0.5] * 4], n_results=10)
        assert len(result["ids"][0]) == 10

    @pytest.mark.asyncio
    async def test_collection_name_property(self, test_collection):
        """Test that collection name property is accessible."""
        assert test_collection.name == "test_collection"

    @pytest.mark.asyncio
    async def test_collection_metadata_property(self, test_collection):
        """Test that collection metadata property is accessible."""
        assert isinstance(test_collection.metadata, dict)


# =============================================================================
# Legacy Tests (kept for backward compatibility)
# =============================================================================


@pytest.mark.asyncio
async def test_async_client_create_add_query_delete(tmp_path, monkeypatch):
    """Legacy test for basic create, add, query, and delete operations."""
    monkeypatch.delenv("CHROMA_URL", raising=False)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

    client: AsyncChromaClient = await get_async_chroma_client()
    try:
        collection: AsyncChromaCollection = await client.create_collection(name="test_async_chroma")

        ids = ["doc-a", "doc-b"]
        docs = ["hello world", "goodbye world"]
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.1, 0.0, 0.3],
        ]

        await collection.add(ids=ids, documents=docs, embeddings=embeddings)

        count = await collection.count()
        assert count == 2

        result = await collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=1)
        assert result["ids"] is not None
        assert len(result["ids"][0]) == 1
        assert result["ids"][0][0] in ids

        await client.delete_collection("test_async_chroma")

    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_list_and_modify(tmp_path, monkeypatch):
    """Legacy test for list and modify operations."""
    monkeypatch.delenv("CHROMA_URL", raising=False)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

    client: AsyncChromaClient = await get_async_chroma_client()
    try:
        initial_count = await client.count_collections()

        collection = await client.create_collection(name="modify_me")
        after_create_count = await client.count_collections()
        assert after_create_count == initial_count + 1

        await collection.modify(metadata={"stage": "test"})
        assert collection.metadata.get("stage") == "test"

        await collection.upsert(ids=["d1"], documents=["alpha beta"], embeddings=[[0.0, 0.1, 0.0, 0.2]])
        res = await collection.query(query_embeddings=[[0.0, 0.1, 0.0, 0.2]], n_results=1)
        assert res["ids"] is not None and "d1" in res["ids"][0]

        await client.delete_collection("modify_me")
        final_count = await client.count_collections()
        assert final_count == initial_count

    finally:
        client.shutdown()
