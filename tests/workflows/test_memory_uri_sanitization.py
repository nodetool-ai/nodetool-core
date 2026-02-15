"""
Tests for sanitization of memory:// URIs in websocket payloads.

This ensures that OutputUpdate, ToolResultUpdate, and PreviewUpdate messages
never contain memory:// URIs when serialized for clients.
"""

import pytest

from nodetool.workflows.types import (
    OutputUpdate,
    PreviewUpdate,
    ToolResultUpdate,
    sanitize_memory_uris_for_client,
)


class TestSanitizeMemoryUrisForClient:
    """Tests for the sanitize_memory_uris_for_client utility function."""

    def test_sanitize_dict_with_memory_uri_and_data(self):
        """Dict with memory:// URI and data should clear the URI."""
        value = {"type": "image", "uri": "memory://abc123", "data": b"fake_image_data"}
        result = sanitize_memory_uris_for_client(value)
        assert result["uri"] == ""
        assert result["data"] == b"fake_image_data"

    def test_sanitize_dict_with_memory_uri_and_asset_id(self):
        """Dict with memory:// URI and asset_id should use asset:// URI."""
        value = {"type": "image", "uri": "memory://abc123", "asset_id": "asset-456"}
        result = sanitize_memory_uris_for_client(value)
        assert result["uri"] == "asset://asset-456"

    def test_sanitize_dict_with_memory_uri_no_data_no_asset_id(self):
        """Dict with memory:// URI but no data or asset_id should clear the URI."""
        value = {"type": "image", "uri": "memory://abc123"}
        result = sanitize_memory_uris_for_client(value)
        assert result["uri"] == ""

    def test_sanitize_non_memory_uri_unchanged(self):
        """Non-memory URIs should be unchanged."""
        value = {"type": "image", "uri": "https://example.com/image.png"}
        result = sanitize_memory_uris_for_client(value)
        assert result["uri"] == "https://example.com/image.png"

    def test_sanitize_nested_dict(self):
        """Nested dicts should be sanitized recursively."""
        value = {
            "outer": {
                "type": "image",
                "uri": "memory://nested",
                "data": b"data",
            }
        }
        result = sanitize_memory_uris_for_client(value)
        assert result["outer"]["uri"] == ""

    def test_sanitize_list(self):
        """Lists should be sanitized recursively."""
        value = [
            {"type": "image", "uri": "memory://item1"},
            {"type": "audio", "uri": "memory://item2", "asset_id": "audio-123"},
        ]
        result = sanitize_memory_uris_for_client(value)
        assert result[0]["uri"] == ""
        assert result[1]["uri"] == "asset://audio-123"

    def test_sanitize_tuple(self):
        """Tuples should be sanitized recursively."""
        value = (
            {"type": "image", "uri": "memory://item1"},
            {"type": "audio", "uri": "memory://item2"},
        )
        result = sanitize_memory_uris_for_client(value)
        assert result[0]["uri"] == ""
        assert result[1]["uri"] == ""

    def test_sanitize_primitive_values_unchanged(self):
        """Primitive values should be unchanged."""
        assert sanitize_memory_uris_for_client("hello") == "hello"
        assert sanitize_memory_uris_for_client(123) == 123
        assert sanitize_memory_uris_for_client(None) is None
        assert sanitize_memory_uris_for_client(True) is True

    def test_sanitize_complex_nested_structure(self):
        """Complex nested structures should be fully sanitized."""
        value = {
            "results": [
                {
                    "images": [
                        {"type": "image", "uri": "memory://img1", "data": b"png"},
                        {"type": "image", "uri": "https://example.com/img2.png"},
                    ],
                    "audio": {"type": "audio", "uri": "memory://audio1"},
                }
            ],
            "metadata": {"count": 2},
        }
        result = sanitize_memory_uris_for_client(value)
        assert result["results"][0]["images"][0]["uri"] == ""
        assert result["results"][0]["images"][1]["uri"] == "https://example.com/img2.png"
        assert result["results"][0]["audio"]["uri"] == ""
        assert result["metadata"]["count"] == 2


class TestOutputUpdateSanitization:
    """Tests for OutputUpdate memory:// URI sanitization."""

    def test_output_update_sanitize_memory_uri(self):
        """OutputUpdate should sanitize memory:// URIs in value."""
        update = OutputUpdate(
            node_id="node1",
            node_name="Output",
            output_name="result",
            value={"type": "image", "uri": "memory://abc123", "data": b"fake_data"},
            output_type="image",
        )
        assert update.value["uri"] == ""
        assert "memory://" not in str(update.model_dump())

    def test_output_update_sanitize_nested_memory_uri(self):
        """OutputUpdate should sanitize nested memory:// URIs."""
        update = OutputUpdate(
            node_id="node1",
            node_name="Output",
            output_name="result",
            value={
                "items": [
                    {"type": "image", "uri": "memory://nested1"},
                    {"type": "audio", "uri": "memory://nested2", "asset_id": "audio-123"},
                ]
            },
            output_type="list",
        )
        assert update.value["items"][0]["uri"] == ""
        assert update.value["items"][1]["uri"] == "asset://audio-123"
        assert "memory://" not in str(update.model_dump())

    def test_output_update_non_memory_uri_unchanged(self):
        """OutputUpdate should leave non-memory URIs unchanged."""
        update = OutputUpdate(
            node_id="node1",
            node_name="Output",
            output_name="result",
            value={"type": "image", "uri": "https://example.com/image.png"},
            output_type="image",
        )
        assert update.value["uri"] == "https://example.com/image.png"


class TestToolResultUpdateSanitization:
    """Tests for ToolResultUpdate memory:// URI sanitization."""

    def test_tool_result_update_sanitize_memory_uri(self):
        """ToolResultUpdate should sanitize memory:// URIs in result."""
        update = ToolResultUpdate(
            node_id="node1",
            result={"output": {"type": "image", "uri": "memory://abc123", "data": b"data"}},
        )
        assert update.result["output"]["uri"] == ""
        assert "memory://" not in str(update.model_dump())

    def test_tool_result_update_sanitize_nested_memory_uri(self):
        """ToolResultUpdate should sanitize nested memory:// URIs."""
        update = ToolResultUpdate(
            node_id="node1",
            result={
                "images": [
                    {"type": "image", "uri": "memory://img1"},
                    {"type": "image", "uri": "memory://img2", "asset_id": "img-456"},
                ]
            },
        )
        assert update.result["images"][0]["uri"] == ""
        assert update.result["images"][1]["uri"] == "asset://img-456"
        assert "memory://" not in str(update.model_dump())


class TestPreviewUpdateSanitization:
    """Tests for PreviewUpdate memory:// URI sanitization."""

    def test_preview_update_sanitize_memory_uri(self):
        """PreviewUpdate should sanitize memory:// URIs in value."""
        update = PreviewUpdate(
            node_id="node1",
            value={"type": "image", "uri": "memory://abc123", "data": b"fake_data"},
        )
        assert update.value["uri"] == ""
        assert "memory://" not in str(update.model_dump())

    def test_preview_update_sanitize_nested_memory_uri(self):
        """PreviewUpdate should sanitize nested memory:// URIs."""
        update = PreviewUpdate(
            node_id="node1",
            value={
                "preview": {
                    "image": {"type": "image", "uri": "memory://preview1"},
                }
            },
        )
        assert update.value["preview"]["image"]["uri"] == ""
        assert "memory://" not in str(update.model_dump())

    def test_preview_update_non_memory_uri_unchanged(self):
        """PreviewUpdate should leave non-memory URIs unchanged."""
        update = PreviewUpdate(
            node_id="node1",
            value={"type": "image", "uri": "asset://existing-asset"},
        )
        assert update.value["uri"] == "asset://existing-asset"


class TestNoMemoryUriInSerializedOutput:
    """Integration tests verifying no memory:// in serialized JSON."""

    def test_output_update_json_no_memory_uri(self):
        """Serialized OutputUpdate JSON should contain no memory:// strings."""
        update = OutputUpdate(
            node_id="node1",
            node_name="Output",
            output_name="result",
            value={
                "nested": {
                    "items": [
                        {"type": "image", "uri": "memory://deep1", "data": [1, 2, 3]},
                        {"type": "audio", "uri": "memory://deep2", "asset_id": "audio-id"},
                    ]
                }
            },
            output_type="complex",
        )
        json_str = update.model_dump_json()
        assert "memory://" not in json_str

    def test_tool_result_update_json_no_memory_uri(self):
        """Serialized ToolResultUpdate JSON should contain no memory:// strings."""
        update = ToolResultUpdate(
            node_id="node1",
            thread_id="thread1",
            result={
                "outputs": [
                    {"type": "image", "uri": "memory://out1"},
                    {"type": "video", "uri": "memory://out2"},
                ]
            },
        )
        json_str = update.model_dump_json()
        assert "memory://" not in json_str

    def test_preview_update_json_no_memory_uri(self):
        """Serialized PreviewUpdate JSON should contain no memory:// strings."""
        update = PreviewUpdate(
            node_id="node1",
            value={"type": "image", "uri": "memory://preview", "data": b"bytes"},
        )
        json_str = update.model_dump_json()
        assert "memory://" not in json_str
