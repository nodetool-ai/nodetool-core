"""
Unit tests for ChromaDB agent tools.

Tests cover:
- ChromaMarkdownSplitAndIndexTool: Markdown splitting and indexing functionality
"""

import os
from pathlib import Path

import pytest

from nodetool.agents.tools.chroma_tools import ChromaMarkdownSplitAndIndexTool
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    AsyncChromaClient,
    AsyncChromaCollection,
    get_async_chroma_client,
)
from nodetool.workflows.processing_context import ProcessingContext


@pytest.fixture
async def chroma_client(tmp_path, monkeypatch):
    """Create an isolated ChromaDB client for testing."""
    # Ensure local, on-disk Chroma
    monkeypatch.delenv("CHROMA_URL", raising=False)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

    client: AsyncChromaClient = await get_async_chroma_client()
    try:
        yield client
    finally:
        client.shutdown()


@pytest.fixture
async def test_collection(chroma_client):
    """Create a test collection and clean it up after the test."""
    collection_name = "test_markdown_collection"

    # Clean up any existing collection
    try:
        await chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection doesn't exist, which is fine

    collection = await chroma_client.create_collection(name=collection_name)
    try:
        yield collection
    finally:
        # Clean up
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


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires network access for embedding model download")
async def test_chroma_markdown_split_and_index_basic(test_collection, test_markdown, processing_context):
    """Test basic markdown splitting and indexing functionality."""
    tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

    params = {
        "text": test_markdown,
    }

    # Run the tool
    result = await tool.process(context=processing_context, params=params)

    # Verify result structure
    assert result["status"] == "success"
    assert "indexed_ids" in result
    assert len(result["indexed_ids"]) > 0
    assert "message" in result

    # Verify documents were actually indexed
    indexed_count = len(result["indexed_ids"])
    retrieved_docs = await test_collection.get(include=["metadatas", "documents"])

    assert retrieved_docs["ids"] is not None
    assert len(retrieved_docs["ids"]) == indexed_count

    # Verify document content
    assert retrieved_docs["documents"] is not None
    for i, _doc_id in enumerate(retrieved_docs["ids"]):
        assert retrieved_docs["documents"][i] is not None
        assert len(retrieved_docs["documents"][i]) > 0


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires network access for embedding model download")
async def test_chroma_markdown_split_and_index_custom_chunk_size(test_collection, test_markdown, processing_context):
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

    # With smaller chunk size, we should get more chunks
    retrieved_docs = await test_collection.get(include=["documents"])
    assert len(retrieved_docs["ids"]) > 0

    # Verify chunks are within size limits (allowing some flexibility)
    for doc in retrieved_docs["documents"]:
        assert doc is not None
        # Chunks should generally be smaller than chunk_size + overlap
        assert len(doc) <= 250  # chunk_size + chunk_overlap


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires network access for embedding model download")
async def test_chroma_markdown_split_and_index_from_file(test_collection, test_markdown, processing_context):
    """Test markdown splitting from a file path."""
    # Create a markdown file inside the workspace directory
    # resolve_workspace_path treats paths relative to workspace_dir
    workspace_dir = Path(processing_context.workspace_dir)
    markdown_file = workspace_dir / "test.md"
    markdown_file.write_text(test_markdown)

    tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

    # Use relative path from workspace root
    params = {
        "file_path": "test.md",
    }

    result = await tool.process(context=processing_context, params=params)

    assert result["status"] == "success"
    assert len(result["indexed_ids"]) > 0

    # Verify documents were indexed
    retrieved_docs = await test_collection.get(include=["documents"])
    assert len(retrieved_docs["ids"]) > 0


@pytest.mark.asyncio
async def test_chroma_markdown_split_and_index_no_text_or_file(test_collection, processing_context):
    """Test that the tool raises an error when neither text nor file_path is provided."""
    tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

    params = {}

    with pytest.raises(ValueError, match="Neither file_path nor text is provided"):
        await tool.process(context=processing_context, params=params)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires network access for embedding model download")
async def test_chroma_markdown_split_and_index_multiple_h1_headers(test_collection, processing_context):
    """Test handling of multiple H1 headers."""
    markdown_with_multiple_h1 = """
# First Header

Content under first header.

# Second Header

Content under second header.

# Third Header

Content under third header.
"""

    tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

    params = {
        "text": markdown_with_multiple_h1,
    }

    result = await tool.process(context=processing_context, params=params)

    assert result["status"] == "success"
    assert len(result["indexed_ids"]) > 0

    # Verify all chunks were indexed
    retrieved_docs = await test_collection.get(include=["documents"])
    assert len(retrieved_docs["ids"]) == len(result["indexed_ids"])


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires network access for embedding model download")
async def test_chroma_markdown_split_and_index_nested_headers(test_collection, processing_context):
    """Test handling of nested headers (H1, H2, H3)."""
    markdown_with_nested = """
# Main Title

Introduction text.

## Section A

Section A content.

### Subsection A.1

Subsection content.

## Section B

Section B content.
"""

    tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

    params = {
        "text": markdown_with_nested,
    }

    result = await tool.process(context=processing_context, params=params)

    assert result["status"] == "success"
    assert len(result["indexed_ids"]) > 0

    # Verify chunks contain expected content
    retrieved_docs = await test_collection.get(include=["documents"])
    documents = retrieved_docs["documents"]

    # Check that we have chunks from different sections
    all_content = " ".join(documents)
    assert "Main Title" in all_content or "Introduction" in all_content
    assert "Section A" in all_content or "Subsection A.1" in all_content


@pytest.mark.asyncio
async def test_chroma_markdown_split_and_index_user_message(
    test_collection,
):
    """Test the user_message method."""
    tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

    params = {"source_id": "test_source.md"}
    message = tool.user_message(params)

    assert "Splitting and indexing Markdown" in message
    assert "test_source.md" in message

    # Test with long source_id (should be truncated)
    params_long = {"source_id": "a" * 100}
    message_long = tool.user_message(params_long)
    assert len(message_long) <= 80
    assert "Splitting and indexing Markdown" in message_long


@pytest.mark.asyncio
async def test_chroma_markdown_split_and_index_empty_markdown(test_collection, processing_context):
    """Test handling of empty markdown content."""
    tool = ChromaMarkdownSplitAndIndexTool(collection=test_collection)

    params = {
        "text": "",
    }

    result = await tool.process(context=processing_context, params=params)

    # Empty content should still succeed but may produce no chunks
    assert result["status"] == "success"
    # The result may have 0 chunks or 1 chunk depending on implementation
    assert "indexed_ids" in result
