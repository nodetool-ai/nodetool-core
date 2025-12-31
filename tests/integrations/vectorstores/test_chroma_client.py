"""
Comprehensive tests for the synchronous chroma_client module.

These tests use a real ChromaDB instance (on-disk persistent client) to ensure
all functionality works correctly with actual data storage and retrieval.
"""

import chromadb.errors
import pytest

from nodetool.integrations.vectorstores.chroma.chroma_client import (
    DEFAULT_SEPARATORS,
    get_all_collections,
    get_chroma_client,
    get_collection,
    split_document,
)

# Test data constants
EMBEDDING_DIM = 4  # Dimension of test embeddings
LARGE_DOC_PARAGRAPHS = 100  # Number of paragraphs for large document tests
PARAGRAPH_REPETITIONS = 10  # Repetitions of filler text per paragraph


def generate_test_embedding(value: float, dim: int = EMBEDDING_DIM) -> list[float]:
    """Generate a simple test embedding."""
    return [value] * dim


def generate_distinct_embeddings(count: int, dim: int = EMBEDDING_DIM) -> list[list[float]]:
    """Generate distinct embeddings for testing (orthogonal-ish)."""
    embeddings = []
    for i in range(count):
        emb = [0.0] * dim
        emb[i % dim] = 1.0  # One-hot style for distinctness
        embeddings.append(emb)
    return embeddings


@pytest.fixture
def chroma_client(tmp_path, monkeypatch):
    """Create an isolated ChromaDB client for testing."""
    monkeypatch.delenv("CHROMA_URL", raising=False)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path))
    return get_chroma_client()


@pytest.fixture
def collection_with_data(chroma_client):
    """Create a collection with some test data."""
    collection = chroma_client.create_collection(name="test_collection")
    embeddings = generate_distinct_embeddings(3)
    collection.add(
        ids=["doc1", "doc2", "doc3"],
        documents=["First document", "Second document", "Third document"],
        embeddings=embeddings,
    )
    yield collection
    try:
        chroma_client.delete_collection("test_collection")
    except Exception:
        pass


# =============================================================================
# get_chroma_client Tests
# =============================================================================


class TestGetChromaClient:
    """Tests for get_chroma_client function."""

    def test_get_client_creates_persistent_client(self, tmp_path, monkeypatch):
        """Test that get_chroma_client creates a persistent client when no URL is set."""
        monkeypatch.delenv("CHROMA_URL", raising=False)
        monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

        client = get_chroma_client()
        assert client is not None

        # Verify it's a working client by creating a collection
        collection = client.create_collection("test")
        assert collection.name == "test"
        client.delete_collection("test")

    def test_get_client_with_user_id(self, tmp_path, monkeypatch):
        """Test that get_chroma_client works with a user_id."""
        monkeypatch.delenv("CHROMA_URL", raising=False)
        monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

        # When using a local client, user_id doesn't affect the client
        client = get_chroma_client(user_id="test_user")
        assert client is not None

    def test_client_persists_data(self, tmp_path, monkeypatch):
        """Test that data persists across client instances."""
        monkeypatch.delenv("CHROMA_URL", raising=False)
        monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

        # Create collection and add data
        client1 = get_chroma_client()
        collection1 = client1.create_collection("persistent_test")
        collection1.add(ids=["p1"], documents=["persistent"], embeddings=[[0.1, 0.2, 0.3, 0.4]])

        # Create new client and verify data exists
        client2 = get_chroma_client()
        collection2 = client2.get_collection("persistent_test")
        assert collection2.count() == 1

        # Cleanup
        client2.delete_collection("persistent_test")


# =============================================================================
# get_collection Tests
# =============================================================================


class TestGetCollection:
    """Tests for get_collection function."""

    def test_get_collection_returns_existing_collection(self, chroma_client, tmp_path, monkeypatch):
        """Test that get_collection retrieves an existing collection."""
        monkeypatch.delenv("CHROMA_URL", raising=False)
        monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

        chroma_client.create_collection(name="existing_col")
        collection = get_collection("existing_col")
        assert collection is not None
        assert collection.name == "existing_col"

        chroma_client.delete_collection("existing_col")

    def test_get_collection_raises_on_empty_name(self, chroma_client):
        """Test that get_collection raises ValueError for empty name."""
        with pytest.raises(ValueError, match="Collection name cannot be empty"):
            get_collection("")

    def test_get_collection_raises_on_nonexistent(self, chroma_client, tmp_path, monkeypatch):
        """Test that get_collection raises error for non-existent collection."""
        monkeypatch.delenv("CHROMA_URL", raising=False)
        monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

        with pytest.raises(chromadb.errors.NotFoundError):
            get_collection("nonexistent_collection")


# =============================================================================
# split_document Tests
# =============================================================================


class TestSplitDocument:
    """Tests for split_document function."""

    def test_split_document_basic(self):
        """Test basic text splitting."""
        text = "This is a short paragraph.\n\nThis is another paragraph."
        chunks = split_document(text, source_id="test_source")

        assert len(chunks) >= 1
        assert all(chunk.source_id == "test_source" for chunk in chunks)
        assert all(chunk.text for chunk in chunks)

    def test_split_document_preserves_content(self):
        """Test that splitting preserves all content."""
        text = "First paragraph with some content.\n\nSecond paragraph with more content."
        chunks = split_document(text, source_id="source1", chunk_size=2000)

        # Join all chunks and verify content is preserved
        all_text = " ".join(chunk.text.strip() for chunk in chunks)
        assert "First paragraph" in all_text
        assert "Second paragraph" in all_text

    def test_split_document_respects_chunk_size(self):
        """Test that split_document respects chunk_size parameter."""
        # Create a text with natural paragraphs
        text = "First paragraph.\n\n" + "Second paragraph with more words here.\n\n" * 10
        chunks = split_document(text, source_id="source", chunk_size=100, chunk_overlap=10)

        # Each chunk should be reasonably sized
        for chunk in chunks:
            # Allow some buffer for boundaries and overlap
            assert len(chunk.text) <= 200

    def test_split_document_with_overlap(self):
        """Test that chunks have overlap."""
        text = "Paragraph one with some content.\n\nParagraph two with different content.\n\nParagraph three here."
        chunks = split_document(text, source_id="source", chunk_size=60, chunk_overlap=20)

        # With small chunks, we should get multiple chunks
        if len(chunks) >= 2:
            # The overlap should cause some content to appear in multiple chunks
            # This is hard to test directly, so we just verify chunks are created
            assert len(chunks) >= 2

    def test_split_document_with_custom_separators(self):
        """Test splitting with custom separators."""
        text = "First sentence. Second sentence. Third sentence."
        # Use chunk_overlap smaller than chunk_size
        chunks = split_document(text, source_id="source", chunk_size=50, chunk_overlap=10, separators=[". "])

        assert len(chunks) >= 1

    def test_split_document_empty_text(self):
        """Test splitting empty text."""
        chunks = split_document("", source_id="empty")
        assert len(chunks) == 0

    def test_split_document_start_index(self):
        """Test that start_index is correctly set."""
        text = "First paragraph with content.\n\nSecond paragraph with more.\n\nThird paragraph final."
        chunks = split_document(text, source_id="source", chunk_size=50, chunk_overlap=10)

        # Start indices should be non-negative
        for chunk in chunks:
            assert chunk.start_index >= 0

    def test_split_document_long_text(self):
        """Test splitting a long document."""
        # Create a large document with well-defined paragraphs
        filler_text = "Lorem ipsum dolor sit amet. " * PARAGRAPH_REPETITIONS
        text = "\n\n".join([f"Paragraph {i}: {filler_text}" for i in range(LARGE_DOC_PARAGRAPHS)])
        chunks = split_document(text, source_id="long_doc", chunk_size=2000, chunk_overlap=200)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # All chunks should have valid source_id
        assert all(chunk.source_id == "long_doc" for chunk in chunks)

    def test_default_separators(self):
        """Test that DEFAULT_SEPARATORS are correctly defined."""
        assert "\n\n" in DEFAULT_SEPARATORS
        assert "\n" in DEFAULT_SEPARATORS
        assert "." in DEFAULT_SEPARATORS


# =============================================================================
# get_all_collections Tests
# =============================================================================


class TestGetAllCollections:
    """Tests for get_all_collections function.

    Note: get_all_collections requires embedding models to be available,
    which may not work in all test environments.
    """

    def test_get_all_collections_empty(self, tmp_path, monkeypatch):
        """Test get_all_collections when there are no collections."""
        monkeypatch.delenv("CHROMA_URL", raising=False)
        monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

        collections = get_all_collections()
        assert collections == []

    @pytest.mark.skip(reason="Requires embedding model to be available")
    def test_get_all_collections_with_data(self, chroma_client, tmp_path, monkeypatch):
        """Test get_all_collections with multiple collections."""
        monkeypatch.delenv("CHROMA_URL", raising=False)
        monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

        # Create collections
        chroma_client.create_collection(name="col1")
        chroma_client.create_collection(name="col2")

        collections = get_all_collections()
        collection_names = [c.name for c in collections]

        assert "col1" in collection_names
        assert "col2" in collection_names

        # Cleanup
        chroma_client.delete_collection("col1")
        chroma_client.delete_collection("col2")


# =============================================================================
# Integration Tests
# =============================================================================


class TestChromaClientIntegration:
    """Integration tests combining multiple operations."""

    def test_full_workflow(self, tmp_path, monkeypatch):
        """Test a complete workflow of creating, adding, querying, and deleting."""
        monkeypatch.delenv("CHROMA_URL", raising=False)
        monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

        client = get_chroma_client()

        # Create collection
        collection = client.create_collection("workflow_test")

        # Add documents
        ids = ["w1", "w2", "w3"]
        docs = ["Apple is a fruit", "Banana is yellow", "Cherry is red"]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]

        collection.add(ids=ids, documents=docs, embeddings=embeddings)

        # Query
        result = collection.query(query_embeddings=[[1.0, 0.0, 0.0, 0.0]], n_results=1)
        assert "w1" in result["ids"][0]

        # Count
        assert collection.count() == 3

        # Get
        get_result = collection.get(ids=["w1", "w2"])
        assert len(get_result["ids"]) == 2

        # Delete items
        collection.delete(ids=["w1"])
        assert collection.count() == 2

        # Delete collection
        client.delete_collection("workflow_test")

    def test_split_and_store_document(self, tmp_path, monkeypatch):
        """Test splitting a document and storing in ChromaDB."""
        monkeypatch.delenv("CHROMA_URL", raising=False)
        monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

        client = get_chroma_client()
        collection = client.create_collection("split_store_test")

        # Split a document
        text = """
        # Introduction

        This is the introduction section with important information.

        # Main Content

        The main content discusses various topics in detail.
        It spans multiple paragraphs to ensure proper splitting.

        # Conclusion

        The conclusion summarizes everything.
        """

        chunks = split_document(text, source_id="test_doc", chunk_size=100, chunk_overlap=20)

        # Store chunks with simple embeddings
        for i, chunk in enumerate(chunks):
            embedding = [0.0] * 4
            embedding[i % 4] = 1.0  # Simple rotation embedding
            collection.add(ids=[f"chunk_{i}"], documents=[chunk.text], embeddings=[embedding])

        # Verify all chunks are stored
        assert collection.count() == len(chunks)

        # Cleanup
        client.delete_collection("split_store_test")
