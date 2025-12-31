"""
Unit tests for ChromaDB agent tools.

Tests cover:
- ChromaTextSearchTool: Semantic search in ChromaDB
- ChromaIndexTool: Indexing single text chunks
- ChromaHybridSearchTool: Combined semantic/keyword search
- ChromaRecursiveSplitAndIndexTool: Recursive text splitting and indexing
- ChromaMarkdownSplitAndIndexTool: Markdown splitting and indexing
- ChromaBatchIndexTool: Batch indexing of multiple chunks
"""

import os
from pathlib import Path

import pytest
import pytest_asyncio

from nodetool.agents.tools.chroma_tools import (
    ChromaBatchIndexTool,
    ChromaHybridSearchTool,
    ChromaIndexTool,
    ChromaMarkdownSplitAndIndexTool,
    ChromaRecursiveSplitAndIndexTool,
    ChromaTextSearchTool,
)
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    AsyncChromaClient,
    AsyncChromaCollection,
    get_async_chroma_client,
)
from nodetool.workflows.processing_context import ProcessingContext


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
    collection_name = "test_chroma_tools"

    try:
        await chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = await chroma_client.create_collection(name=collection_name)
    try:
        yield collection
    finally:
        try:
            await chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass


@pytest_asyncio.fixture
async def collection_with_data(chroma_client):
    """Create a test collection with pre-populated data."""
    collection_name = "test_chroma_tools_data"

    try:
        await chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = await chroma_client.create_collection(name=collection_name)

    # Add test documents with embeddings
    await collection.add(
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
        documents=[
            "The quick brown fox jumps over the lazy dog",
            "Python is a programming language",
            "Machine learning uses neural networks",
            "The fox is quick and agile",
            "Programming in Python is fun",
        ],
        embeddings=[
            [1.0, 0.0, 0.0, 0.0],  # fox-related
            [0.0, 1.0, 0.0, 0.0],  # python-related
            [0.0, 0.0, 1.0, 0.0],  # ML-related
            [0.9, 0.1, 0.0, 0.0],  # fox-related
            [0.1, 0.9, 0.0, 0.0],  # python-related
        ],
        metadatas=[
            {"category": "animals", "priority": 1},
            {"category": "programming", "priority": 2},
            {"category": "ai", "priority": 3},
            {"category": "animals", "priority": 1},
            {"category": "programming", "priority": 2},
        ],
    )

    try:
        yield collection
    finally:
        try:
            await chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass


@pytest.fixture
def test_markdown():
    """Sample markdown content for testing."""
    return """
# Document Title

This is the introduction.

## Section 1

Content for the first section. It contains some text that might be split if the chunk size is small enough.

### Subsection 1.1

Details within the first section.

## Section 2

Content for the second section. This section has bullet points:
*   Item 1
*   Item 2

This section is also longer to test splitting within a header section. Let's add more text here to ensure it potentially exceeds the default chunk size and forces a split based on the RecursiveCharacterTextSplitter logic applied after header splitting. More filler text. Even more filler text.

# Another H1 Header (Edge Case)

Testing how it handles multiple H1 headers.
"""


@pytest.fixture
def processing_context(tmp_path):
    """Create a ProcessingContext with a temporary workspace directory."""
    workspace_dir = str(tmp_path / "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    return ProcessingContext(workspace_dir=workspace_dir)


# =============================================================================
# ChromaIndexTool Tests
# =============================================================================


class TestChromaIndexTool:
    """Tests for ChromaIndexTool.

    Note: Most tests are skipped because ChromaIndexTool requires ChromaDB to
    generate embeddings, which requires network access for downloading models.
    """

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_index_single_document(self, test_collection, processing_context):
        """Test indexing a single document."""
        tool = ChromaIndexTool(collection=test_collection)

        params = {
            "text": "This is a test document for indexing.",
            "source_id": "test_source_1",
        }

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"
        assert "document_id" in result
        assert "message" in result

        # Verify the document was indexed
        count = await test_collection.count()
        assert count == 1

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_index_with_metadata(self, test_collection, processing_context):
        """Test indexing a document with metadata."""
        tool = ChromaIndexTool(collection=test_collection)

        params = {
            "text": "Document with metadata.",
            "source_id": "meta_source",
            "metadata": {"author": "test", "version": "1.0"},
        }

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"

        # Verify metadata was stored
        docs = await test_collection.get(include=["metadatas"])
        assert len(docs["ids"]) == 1
        assert docs["metadatas"][0]["author"] == "test"
        assert docs["metadatas"][0]["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_index_empty_source_id(self, test_collection, processing_context):
        """Test that empty source_id returns an error."""
        tool = ChromaIndexTool(collection=test_collection)

        params = {
            "text": "Some text",
            "source_id": "   ",  # Empty after strip
        }

        result = await tool.process(context=processing_context, params=params)
        assert "error" in result

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_index_document_id_generation(self, test_collection, processing_context):
        """Test that document IDs are generated consistently."""
        tool = ChromaIndexTool(collection=test_collection)

        params1 = {"text": "First document", "source_id": "source_a"}
        params2 = {"text": "Second document", "source_id": "source_b"}

        result1 = await tool.process(context=processing_context, params=params1)
        result2 = await tool.process(context=processing_context, params=params2)

        # Different sources should have different IDs
        assert result1["document_id"] != result2["document_id"]

    @pytest.mark.asyncio
    async def test_index_user_message(self, test_collection):
        """Test the user_message method."""
        tool = ChromaIndexTool(collection=test_collection)

        params = {"source_id": "my_source.txt"}
        message = tool.user_message(params)

        assert "Indexing" in message
        assert "my_source.txt" in message

    @pytest.mark.asyncio
    async def test_index_document_id_generator(self, test_collection):
        """Test the document ID generation function."""
        tool = ChromaIndexTool(collection=test_collection)

        id1 = tool._generate_document_id("source_a")
        id2 = tool._generate_document_id("source_b")
        id1_again = tool._generate_document_id("source_a")

        # Different sources should have different IDs
        assert id1 != id2
        # Same source should have same ID
        assert id1 == id1_again


# =============================================================================
# ChromaRecursiveSplitAndIndexTool Tests
# =============================================================================


class TestChromaRecursiveSplitAndIndexTool:
    """Tests for ChromaRecursiveSplitAndIndexTool.

    Note: Most tests are skipped because the tool requires ChromaDB to
    generate embeddings, which requires network access for downloading models.
    """

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_split_and_index_basic(self, test_collection, processing_context):
        """Test basic recursive splitting and indexing."""
        tool = ChromaRecursiveSplitAndIndexTool(collection=test_collection)

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        params = {
            "text": text,
            "document_id": "test_doc",
            "chunk_size": 50,
            "chunk_overlap": 10,
        }

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"
        assert result["indexed_count"] >= 1
        assert result["document_id"] == "test_doc"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_split_and_index_with_metadata(self, test_collection, processing_context):
        """Test splitting with base metadata."""
        tool = ChromaRecursiveSplitAndIndexTool(collection=test_collection)

        params = {
            "text": "A longer paragraph with enough content to potentially create multiple chunks.",
            "document_id": "meta_doc",
            "metadata": {"source": "test_file.txt", "type": "document"},
            "chunk_size": 50,
            "chunk_overlap": 10,
        }

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"

        # Verify metadata on chunks
        docs = await test_collection.get(include=["metadatas"])
        for metadata in docs["metadatas"]:
            assert metadata["source"] == "test_file.txt"
            assert metadata["type"] == "document"

    @pytest.mark.asyncio
    async def test_split_empty_text(self, test_collection, processing_context):
        """Test splitting empty text returns error."""
        tool = ChromaRecursiveSplitAndIndexTool(collection=test_collection)

        params = {
            "text": "   ",
            "document_id": "empty_doc",
        }

        result = await tool.process(context=processing_context, params=params)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_split_empty_document_id(self, test_collection, processing_context):
        """Test splitting with empty document_id returns error."""
        tool = ChromaRecursiveSplitAndIndexTool(collection=test_collection)

        params = {
            "text": "Some content",
            "document_id": "   ",
        }

        result = await tool.process(context=processing_context, params=params)
        assert "error" in result

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_split_custom_separators(self, test_collection, processing_context):
        """Test splitting with custom separators."""
        tool = ChromaRecursiveSplitAndIndexTool(collection=test_collection)

        params = {
            "text": "Sentence one. Sentence two. Sentence three.",
            "document_id": "custom_sep_doc",
            "separators": [". "],
            "chunk_size": 50,
            "chunk_overlap": 10,
        }

        result = await tool.process(context=processing_context, params=params)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_split_user_message(self, test_collection):
        """Test the user_message method."""
        tool = ChromaRecursiveSplitAndIndexTool(collection=test_collection)

        params = {"source_id": "my_document.txt"}
        message = tool.user_message(params)

        assert "splitting" in message.lower() or "indexing" in message.lower()

    @pytest.mark.asyncio
    async def test_split_text_recursive_function(self, test_collection, processing_context):
        """Test the internal _split_text_recursive function."""
        tool = ChromaRecursiveSplitAndIndexTool(collection=test_collection)

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        params = {
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

        chunks = await tool._split_text_recursive(text, "test_doc", params)

        assert len(chunks) >= 1
        # All chunks should have the correct source_id prefix
        for chunk in chunks:
            assert "test_doc" in chunk.source_id


# =============================================================================
# ChromaBatchIndexTool Tests
# =============================================================================


class TestChromaBatchIndexTool:
    """Tests for ChromaBatchIndexTool.

    Note: Most tests are skipped because the tool requires ChromaDB to
    generate embeddings, which requires network access for downloading models.
    """

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_batch_index_basic(self, test_collection, processing_context):
        """Test basic batch indexing."""
        tool = ChromaBatchIndexTool(collection=test_collection)

        params = {
            "chunks": [
                {"text": "First chunk", "source_id": "chunk1"},
                {"text": "Second chunk", "source_id": "chunk2"},
                {"text": "Third chunk", "source_id": "chunk3"},
            ]
        }

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"
        assert result["indexed_count"] == 3

        count = await test_collection.count()
        assert count == 3

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_batch_index_with_base_metadata(self, test_collection, processing_context):
        """Test batch indexing with base metadata."""
        tool = ChromaBatchIndexTool(collection=test_collection)

        params = {
            "chunks": [
                {"text": "Chunk A", "source_id": "a"},
                {"text": "Chunk B", "source_id": "b"},
            ],
            "base_metadata": {"batch_id": "test_batch", "version": "1.0"},
        }

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"

        docs = await test_collection.get(include=["metadatas"])
        for metadata in docs["metadatas"]:
            assert metadata["batch_id"] == "test_batch"
            assert metadata["version"] == "1.0"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_batch_index_with_chunk_metadata(self, test_collection, processing_context):
        """Test batch indexing with chunk-specific metadata."""
        tool = ChromaBatchIndexTool(collection=test_collection)

        params = {
            "chunks": [
                {"text": "Chunk 1", "source_id": "s1", "metadata": {"priority": 1}},
                {"text": "Chunk 2", "source_id": "s2", "metadata": {"priority": 2}},
            ]
        }

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"

        docs = await test_collection.get(include=["metadatas"])
        priorities = {m["priority"] for m in docs["metadatas"]}
        assert priorities == {1, 2}

    @pytest.mark.asyncio
    async def test_batch_index_empty_chunks(self, test_collection, processing_context):
        """Test batch indexing with empty chunks list."""
        tool = ChromaBatchIndexTool(collection=test_collection)

        params = {"chunks": []}

        result = await tool.process(context=processing_context, params=params)
        assert "error" in result

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_batch_index_invalid_chunks(self, test_collection, processing_context):
        """Test batch indexing with invalid chunk data."""
        tool = ChromaBatchIndexTool(collection=test_collection)

        params = {
            "chunks": [
                {"text": "", "source_id": ""},  # Invalid
                {"text": "Valid", "source_id": "valid"},
            ]
        }

        result = await tool.process(context=processing_context, params=params)

        # Should index only the valid chunk
        assert result["status"] == "success"
        assert result["indexed_count"] == 1

    @pytest.mark.asyncio
    async def test_batch_index_user_message(self, test_collection):
        """Test the user_message method."""
        tool = ChromaBatchIndexTool(collection=test_collection)

        params = {"chunks": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}
        message = tool.user_message(params)

        assert "3" in message
        assert "batch" in message.lower()

    @pytest.mark.asyncio
    async def test_batch_index_document_id_generator(self, test_collection):
        """Test the document ID generation function."""
        tool = ChromaBatchIndexTool(collection=test_collection)

        id1 = tool._generate_document_id("source_a")
        id2 = tool._generate_document_id("source_b")
        id1_again = tool._generate_document_id("source_a")

        # Different sources should have different IDs
        assert id1 != id2
        # Same source should have same ID
        assert id1 == id1_again


# =============================================================================
# ChromaHybridSearchTool Tests
# =============================================================================


class TestChromaHybridSearchTool:
    """Tests for ChromaHybridSearchTool.

    Note: Tests that require actual search are skipped because they require
    ChromaDB to generate embeddings, which requires network access.
    """

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_hybrid_search_basic(self, collection_with_data, processing_context):
        """Test basic hybrid search."""
        tool = ChromaHybridSearchTool(collection=collection_with_data)

        params = {
            "text": "quick fox",
            "n_results": 3,
        }

        result = await tool.process(context=processing_context, params=params)

        # Should return results (not an error)
        assert "error" not in result or result.get("error") is None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_text(self, collection_with_data, processing_context):
        """Test hybrid search with empty text."""
        tool = ChromaHybridSearchTool(collection=collection_with_data)

        params = {"text": "   "}

        result = await tool.process(context=processing_context, params=params)
        assert "error" in result

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_hybrid_search_custom_parameters(self, collection_with_data, processing_context):
        """Test hybrid search with custom parameters."""
        tool = ChromaHybridSearchTool(collection=collection_with_data)

        params = {
            "text": "Python programming",
            "n_results": 2,
            "k_constant": 30.0,
            "min_keyword_length": 4,
        }

        result = await tool.process(context=processing_context, params=params)

        # Should return at most 2 results
        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_hybrid_search_keyword_query_generation(self, test_collection):
        """Test keyword query generation."""
        tool = ChromaHybridSearchTool(collection=test_collection)

        # Test with multiple words
        query = tool._get_keyword_query("hello world test", min_length=3)
        assert "$or" in query or "$contains" in query

        # Test with single word
        query_single = tool._get_keyword_query("hello", min_length=3)
        assert "$contains" in query_single

        # Test with all words too short
        query_empty = tool._get_keyword_query("hi a", min_length=3)
        assert query_empty == {}

    @pytest.mark.asyncio
    async def test_hybrid_search_user_message(self, test_collection):
        """Test the user_message method."""
        tool = ChromaHybridSearchTool(collection=test_collection)

        params = {"text": "search query"}
        message = tool.user_message(params)

        assert "hybrid" in message.lower() or "search" in message.lower()

    @pytest.mark.asyncio
    async def test_hybrid_search_keyword_query_with_special_chars(self, test_collection):
        """Test keyword query generation with special characters."""
        tool = ChromaHybridSearchTool(collection=test_collection)

        # Test with various separators
        query = tool._get_keyword_query("hello, world! test-case", min_length=3)
        assert "$or" in query or "$contains" in query

    @pytest.mark.asyncio
    async def test_hybrid_search_keyword_query_min_length_filter(self, test_collection):
        """Test keyword query respects min_length filter."""
        tool = ChromaHybridSearchTool(collection=test_collection)

        # Only "hello" and "world" should pass the 4-char filter
        query = tool._get_keyword_query("hi hello the world a", min_length=4)

        if "$or" in query:
            # Should have 2 conditions
            assert len(query["$or"]) == 2
        elif "$contains" in query:
            # Should be a single word
            assert query["$contains"] in ["hello", "world"]


# =============================================================================
# ChromaMarkdownSplitAndIndexTool Tests
# =============================================================================


class TestChromaMarkdownSplitAndIndexTool:
    """Tests for ChromaMarkdownSplitAndIndexTool."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model download")
    async def test_markdown_split_and_index_basic(self, test_collection, test_markdown, processing_context):
        """Test basic markdown splitting and indexing functionality."""
        tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

        params = {"text": test_markdown}

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"
        assert "indexed_ids" in result
        assert len(result["indexed_ids"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model download")
    async def test_markdown_split_custom_chunk_size(self, test_collection, test_markdown, processing_context):
        """Test markdown splitting with custom chunk size."""
        tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

        params = {
            "text": test_markdown,
            "chunk_size": 200,
            "chunk_overlap": 50,
        }

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"
        assert len(result["indexed_ids"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model download")
    async def test_markdown_split_from_file(self, test_collection, test_markdown, processing_context):
        """Test markdown splitting from a file path."""
        workspace_dir = Path(processing_context.workspace_dir)
        markdown_file = workspace_dir / "test.md"
        markdown_file.write_text(test_markdown)

        tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

        params = {"file_path": "test.md"}

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"
        assert len(result["indexed_ids"]) > 0

    @pytest.mark.asyncio
    async def test_markdown_split_no_text_or_file(self, test_collection, processing_context):
        """Test that the tool raises an error when neither text nor file_path is provided."""
        tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

        params = {}

        with pytest.raises(ValueError, match="Neither file_path nor text is provided"):
            await tool.process(context=processing_context, params=params)

    @pytest.mark.asyncio
    async def test_markdown_split_empty_text(self, test_collection, processing_context):
        """Test handling of empty markdown content."""
        tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

        params = {"text": ""}

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"
        assert "indexed_ids" in result

    @pytest.mark.asyncio
    async def test_markdown_split_user_message(self, test_collection):
        """Test the user_message method."""
        tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

        params = {"source_id": "test_source.md"}
        message = tool.user_message(params)

        assert "Splitting and indexing Markdown" in message
        assert "test_source.md" in message

        # Test truncation for long source_id
        params_long = {"source_id": "a" * 100}
        message_long = tool.user_message(params_long)
        assert len(message_long) <= 80


# =============================================================================
# Integration Tests
# =============================================================================


class TestChromaToolsIntegration:
    """Integration tests combining multiple tools.

    Note: Most integration tests are skipped because they require ChromaDB
    to generate embeddings, which requires network access.
    """

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_index_then_search(self, test_collection, processing_context):
        """Test indexing documents and then searching them."""
        # Index some documents
        # ChromaIndexTool is not used directly here; we use test_collection.add
        docs = [
            ("Apple is a fruit", "doc1"),
            ("Banana is yellow", "doc2"),
            ("Cherry is red", "doc3"),
        ]

        for text, source_id in docs:
            await test_collection.add(ids=[source_id], documents=[text], embeddings=[[0.1, 0.2, 0.3, 0.4]])

        # Search
        search_tool = ChromaHybridSearchTool(collection=test_collection)

        result = await search_tool.process(context=processing_context, params={"text": "fruit", "n_results": 2})

        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_add_documents_with_embeddings_then_query(self, test_collection, processing_context):
        """Test adding documents with explicit embeddings and querying."""
        # Add documents with explicit embeddings (no embedding model needed)
        await test_collection.add(
            ids=["doc1", "doc2", "doc3"],
            documents=["Apple is a fruit", "Banana is yellow", "Cherry is red"],
            embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        )

        # Verify count
        count = await test_collection.count()
        assert count == 3

        # Query using embeddings
        result = await test_collection.query(query_embeddings=[[1.0, 0.0, 0.0, 0.0]], n_results=1)
        assert "doc1" in result["ids"][0]

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_batch_index_then_search(self, test_collection, processing_context):
        """Test batch indexing and then searching."""
        batch_tool = ChromaBatchIndexTool(collection=test_collection)

        params = {
            "chunks": [
                {"text": "Machine learning basics", "source_id": "ml1"},
                {"text": "Deep learning neural networks", "source_id": "ml2"},
                {"text": "Data science fundamentals", "source_id": "ml3"},
            ]
        }

        await batch_tool.process(context=processing_context, params=params)

        # Verify count
        count = await test_collection.count()
        assert count == 3

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access for embedding model")
    async def test_recursive_split_creates_multiple_chunks(self, test_collection, processing_context):
        """Test that recursive splitting creates multiple chunks for long text."""
        tool = ChromaRecursiveSplitAndIndexTool(collection=test_collection)

        # Create a longer text
        long_text = "\n\n".join([f"Paragraph {i}: This is some content for the paragraph." for i in range(20)])

        params = {
            "text": long_text,
            "document_id": "long_doc",
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

        result = await tool.process(context=processing_context, params=params)

        assert result["status"] == "success"
        assert result["indexed_count"] > 1
