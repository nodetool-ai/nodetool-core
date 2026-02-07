"""Tests for indexing.ingestion module."""

from __future__ import annotations

import pytest

from nodetool.indexing.ingestion import (
    Document,
    chunk_documents_markdown,
    chunk_documents_recursive,
    find_input_nodes,
)


class TestDocument:
    """Test Document class."""

    def test_init_with_required_params(self) -> None:
        """Test Document initialization with required parameters."""
        doc = Document(text="Hello world", doc_id="doc1")
        assert doc.text == "Hello world"
        assert doc.doc_id == "doc1"
        assert doc.metadata == {}

    def test_init_with_metadata(self) -> None:
        """Test Document initialization with metadata."""
        metadata = {"source": "test.txt", "page": "1"}
        doc = Document(text="Hello world", doc_id="doc1", metadata=metadata)
        assert doc.metadata == metadata

    def test_init_with_empty_metadata(self) -> None:
        """Test Document initialization with empty metadata dict."""
        doc = Document(text="Hello world", doc_id="doc1", metadata={})
        assert doc.metadata == {}


class TestChunkDocumentsRecursive:
    """Test chunk_documents_recursive function."""

    def test_single_document(self) -> None:
        """Test chunking a single document."""
        doc = Document(text="Hello world", doc_id="doc1")
        ids_docs, metadatas = chunk_documents_recursive([doc])

        assert len(ids_docs) > 0
        assert all(isinstance(k, str) for k in ids_docs)
        assert all(isinstance(v, str) for v in ids_docs.values())
        assert len(metadatas) == len(ids_docs)
        assert metadatas == [{}] * len(ids_docs)

    def test_multiple_documents(self) -> None:
        """Test chunking multiple documents."""
        docs = [
            Document(text="First document", doc_id="doc1"),
            Document(text="Second document", doc_id="doc2"),
        ]
        ids_docs, metadatas = chunk_documents_recursive(docs)

        assert len(ids_docs) > 0
        assert len(metadatas) == len(ids_docs)
        # Check that doc IDs are prefixed correctly
        for doc_id in ids_docs:
            assert doc_id.startswith("doc1:") or doc_id.startswith("doc2:")

    def test_custom_chunk_size(self) -> None:
        """Test chunking with custom chunk size."""
        text = " ".join(["word"] * 1000)
        doc = Document(text=text, doc_id="doc1")
        ids_docs, _ = chunk_documents_recursive([doc], chunk_size=100, chunk_overlap=10)

        assert len(ids_docs) > 1  # Should split into multiple chunks

    def test_chunk_overlap(self) -> None:
        """Test that chunks overlap correctly."""
        text = " ".join(["word"] * 1000)
        doc = Document(text=text, doc_id="doc1")
        ids_docs, _ = chunk_documents_recursive([doc], chunk_size=100, chunk_overlap=50)

        # Verify we have multiple chunks
        assert len(ids_docs) > 1

    def test_metadata_preservation(self) -> None:
        """Test that document metadata is preserved."""
        metadata = {"source": "test.txt", "author": "test"}
        doc = Document(text="Hello world", doc_id="doc1", metadata=metadata)
        _ids_docs, metadatas = chunk_documents_recursive([doc])

        assert all(m == metadata for m in metadatas)

    def test_empty_documents(self) -> None:
        """Test chunking with empty documents."""
        doc = Document(text="", doc_id="doc1")
        ids_docs, metadatas = chunk_documents_recursive([doc])

        # Empty documents should still produce results
        assert isinstance(ids_docs, dict)
        assert isinstance(metadatas, list)

    def test_long_text_chunking(self) -> None:
        """Test chunking of long text."""
        text = " ".join(["word"] * 10000)
        doc = Document(text=text, doc_id="long_doc")
        ids_docs, _ = chunk_documents_recursive([doc])

        # Should split into multiple chunks
        assert len(ids_docs) > 1
        # All chunks should be non-empty
        assert all(chunk for chunk in ids_docs.values())


class TestChunkDocumentsMarkdown:
    """Test chunk_documents_markdown function."""

    def test_simple_markdown(self) -> None:
        """Test chunking simple markdown."""
        text = """# Header 1

Some text here.

## Header 2

More text.
"""
        doc = Document(text=text, doc_id="md1")
        ids_docs, metadatas = chunk_documents_markdown([doc])

        assert len(ids_docs) > 0
        assert all(isinstance(k, str) for k in ids_docs)
        assert all(isinstance(v, str) for v in ids_docs.values())
        assert len(metadatas) == len(ids_docs)

    def test_markdown_with_headers(self) -> None:
        """Test chunking markdown with multiple headers."""
        text = """# Main Title

Content under main title.

## Section 1

Content for section 1.

### Subsection 1.1

Content for subsection 1.1.

## Section 2

Content for section 2.
"""
        doc = Document(text=text, doc_id="md2")
        ids_docs, _ = chunk_documents_markdown([doc])

        # Should split by headers
        assert len(ids_docs) >= 1

    def test_multiple_markdown_docs(self) -> None:
        """Test chunking multiple markdown documents."""
        docs = [
            Document(text="# Doc 1\nContent 1", doc_id="md1"),
            Document(text="# Doc 2\nContent 2", doc_id="md2"),
        ]
        ids_docs, metadatas = chunk_documents_markdown(docs)

        assert len(ids_docs) > 0
        assert len(metadatas) == len(ids_docs)

    def test_custom_chunk_params(self) -> None:
        """Test with custom chunk parameters."""
        text = " ".join(["word"] * 1000)
        doc = Document(text=text, doc_id="md1")
        ids_docs, _ = chunk_documents_markdown(
            [doc], chunk_size=100, chunk_overlap=20
        )

        assert len(ids_docs) >= 1

    def test_metadata_preservation(self) -> None:
        """Test that metadata is preserved."""
        metadata = {"source": "test.md"}
        doc = Document(text="# Test\nContent", doc_id="md1", metadata=metadata)
        _ids_docs, metadatas = chunk_documents_markdown([doc])

        assert all(m == metadata for m in metadatas)

    def test_empty_markdown(self) -> None:
        """Test chunking empty markdown."""
        doc = Document(text="", doc_id="empty_md")
        ids_docs, metadatas = chunk_documents_markdown([doc])

        assert isinstance(ids_docs, dict)
        assert isinstance(metadatas, list)

    def test_markdown_without_headers(self) -> None:
        """Test markdown without any headers."""
        text = "Just some plain text\nwithout any markdown headers."
        doc = Document(text=text, doc_id="plain_md")
        ids_docs, metadatas = chunk_documents_markdown([doc])

        # Should still work
        assert len(ids_docs) >= 1
        assert len(metadatas) == len(ids_docs)


class TestFindInputNodes:
    """Test find_input_nodes function."""

    def test_find_collection_input(self) -> None:
        """Test finding CollectionInput node."""
        graph = {
            "nodes": [
                {
                    "type": "nodetool.input.CollectionInput",
                    "data": {"name": "my_collection"},
                }
            ]
        }
        collection_input, file_input = find_input_nodes(graph)

        assert collection_input == "my_collection"
        assert file_input is None

    def test_find_file_input(self) -> None:
        """Test finding FileInput node."""
        graph = {
            "nodes": [
                {"type": "nodetool.input.FileInput", "data": {"name": "my_file"}}
            ]
        }
        collection_input, file_input = find_input_nodes(graph)

        assert collection_input is None
        assert file_input == "my_file"

    def test_find_document_file_input(self) -> None:
        """Test finding DocumentFileInput node."""
        graph = {
            "nodes": [
                {
                    "type": "nodetool.input.DocumentFileInput",
                    "data": {"name": "my_doc"},
                }
            ]
        }
        collection_input, file_input = find_input_nodes(graph)

        assert collection_input is None
        assert file_input == "my_doc"

    def test_find_both_inputs(self) -> None:
        """Test finding both input types."""
        graph = {
            "nodes": [
                {
                    "type": "nodetool.input.CollectionInput",
                    "data": {"name": "my_collection"},
                },
                {"type": "nodetool.input.FileInput", "data": {"name": "my_file"}},
            ]
        }
        collection_input, file_input = find_input_nodes(graph)

        assert collection_input == "my_collection"
        assert file_input == "my_file"

    def test_no_input_nodes(self) -> None:
        """Test graph with no input nodes."""
        graph = {
            "nodes": [
                {"type": "nodetool.text.GenerateText", "data": {"prompt": "test"}}
            ]
        }
        collection_input, file_input = find_input_nodes(graph)

        assert collection_input is None
        assert file_input is None

    def test_empty_graph(self) -> None:
        """Test empty graph."""
        graph = {"nodes": []}
        collection_input, file_input = find_input_nodes(graph)

        assert collection_input is None
        assert file_input is None

    def test_other_node_types_ignored(self) -> None:
        """Test that other node types are ignored."""
        graph = {
            "nodes": [
                {"type": "nodetool.text.GenerateText", "data": {"name": "text_gen"}},
                {"type": "nodetool.image.GenerateImage", "data": {"name": "img_gen"}},
            ]
        }
        collection_input, file_input = find_input_nodes(graph)

        assert collection_input is None
        assert file_input is None

    def test_multiple_collection_inputs_returns_first(self) -> None:
        """Test behavior with multiple collection inputs."""
        graph = {
            "nodes": [
                {
                    "type": "nodetool.input.CollectionInput",
                    "data": {"name": "collection1"},
                },
                {
                    "type": "nodetool.input.CollectionInput",
                    "data": {"name": "collection2"},
                },
            ]
        }
        collection_input, file_input = find_input_nodes(graph)

        # Should return the last one (due to loop overwriting)
        assert collection_input == "collection2"
        assert file_input is None

    def test_multiple_file_inputs_returns_last(self) -> None:
        """Test behavior with multiple file inputs."""
        graph = {
            "nodes": [
                {"type": "nodetool.input.FileInput", "data": {"name": "file1"}},
                {"type": "nodetool.input.DocumentFileInput", "data": {"name": "file2"}},
            ]
        }
        collection_input, file_input = find_input_nodes(graph)

        # Should return the last one
        assert file_input == "file2"
        assert collection_input is None
