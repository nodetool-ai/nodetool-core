"""
Test cases for result_for_client special handling of AssetRef objects.
"""

from typing import Any

import numpy as np
import PIL.Image
import pytest
from pydantic import Field

from nodetool.metadata.types import AssetRef, AudioRef, ImageRef, TextRef, VideoRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class TestAssetReturningNode(BaseNode):
    """A test node that returns AssetRef objects."""

    test_mode: str = Field(default="simple", description="Test mode")

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """Return various AssetRef structures for testing."""
        if self.test_mode == "image_memory":
            # Create ImageRef with memory URI
            image = PIL.Image.new("RGB", (10, 10), color="red")
            image_ref = await context.image_from_pil(image)
            return {"image": image_ref}
        elif self.test_mode == "audio_memory":
            # Create AudioRef with memory URI
            audio_data = np.array([100, -100, 200, -200], dtype=np.int16)
            audio_ref = await context.audio_from_numpy(audio_data, sample_rate=22050)
            return {"audio": audio_ref}
        elif self.test_mode == "text_memory":
            # Create TextRef with memory URI
            text_ref = await context.text_from_str("Test content")
            return {"text": text_ref}
        elif self.test_mode == "generic_with_data":
            # Create generic AssetRef with data already set
            asset_ref = AssetRef(data=b"test bytes")
            return {"asset": asset_ref}
        elif self.test_mode == "nested_memory":
            # Create nested structure with memory URIs
            image = PIL.Image.new("RGB", (5, 5), color="blue")
            image_ref = await context.image_from_pil(image)
            return {
                "result": {
                    "image": image_ref,
                    "nested": {
                        "more_images": [image_ref, image_ref],
                    },
                }
            }
        else:
            return {"output": "simple"}


@pytest.mark.asyncio
async def test_result_for_client_image_memory_uri():
    """Test that ImageRef with memory:// URI gets data field populated."""
    node = TestAssetReturningNode(test_mode="image_memory")
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
    )

    # Process node to get result with memory URI
    result = await node.process(context)
    assert "image" in result
    assert isinstance(result["image"], ImageRef)
    assert result["image"].uri.startswith("memory://")
    assert result["image"].data is None  # Not populated yet

    # Call result_for_client to populate data field
    client_result = node.result_for_client(result)

    # Verify structure
    assert "image" in client_result
    assert isinstance(client_result["image"], dict)
    assert client_result["image"]["type"] == "image"
    assert "data" in client_result["image"]
    assert isinstance(client_result["image"]["data"], bytes)
    assert len(client_result["image"]["data"]) > 0
    # Should be PNG bytes (starts with PNG header)
    assert client_result["image"]["data"][:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_result_for_client_audio_memory_uri():
    """Test that AudioRef with memory:// URI gets data field populated."""
    node = TestAssetReturningNode(test_mode="audio_memory")
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
    )

    result = await node.process(context)
    assert "audio" in result
    assert isinstance(result["audio"], AudioRef)
    assert result["audio"].uri.startswith("memory://")

    # Call result_for_client
    client_result = node.result_for_client(result)

    # Verify audio data populated
    assert "audio" in client_result
    assert isinstance(client_result["audio"], dict)
    assert client_result["audio"]["type"] == "audio"
    assert "data" in client_result["audio"]
    assert isinstance(client_result["audio"]["data"], bytes)
    assert len(client_result["audio"]["data"]) > 0


@pytest.mark.asyncio
async def test_result_for_client_text_memory_uri():
    """Test that TextRef with memory:// URI gets data field populated."""
    node = TestAssetReturningNode(test_mode="text_memory")
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
    )

    result = await node.process(context)
    assert "text" in result
    assert isinstance(result["text"], TextRef)
    assert result["text"].uri.startswith("memory://")

    # Call result_for_client
    client_result = node.result_for_client(result)

    # Verify text data populated
    assert "text" in client_result
    assert isinstance(client_result["text"], dict)
    assert client_result["text"]["type"] == "text"
    assert "data" in client_result["text"]
    # Should be UTF-8 encoded bytes
    assert client_result["text"]["data"] == b"Test content"


@pytest.mark.asyncio
async def test_result_for_client_generic_with_data():
    """Test that AssetRef with data already set is preserved."""
    node = TestAssetReturningNode(test_mode="generic_with_data")
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
    )

    result = await node.process(context)
    assert "asset" in result
    assert isinstance(result["asset"], AssetRef)
    assert result["asset"].data == b"test bytes"

    # Call result_for_client
    client_result = node.result_for_client(result)

    # Verify data preserved as-is
    assert "asset" in client_result
    assert isinstance(client_result["asset"], dict)
    assert client_result["asset"]["data"] == b"test bytes"


@pytest.mark.asyncio
async def test_result_for_client_nested_memory_uri():
    """Test that nested AssetRef objects with memory URIs are handled."""
    node = TestAssetReturningNode(test_mode="nested_memory")
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
    )

    result = await node.process(context)

    # Call result_for_client
    client_result = node.result_for_client(result)

    # Verify nested image has data
    assert "result" in client_result
    assert "image" in client_result["result"]
    assert isinstance(client_result["result"]["image"], dict)
    assert "data" in client_result["result"]["image"]
    assert isinstance(client_result["result"]["image"]["data"], bytes)

    # Verify images in list also have data
    assert "nested" in client_result["result"]
    assert "more_images" in client_result["result"]["nested"]
    images_list = client_result["result"]["nested"]["more_images"]
    assert len(images_list) == 2
    for img in images_list:
        assert isinstance(img, dict)
        assert "data" in img
        assert isinstance(img["data"], bytes)


def test_result_for_client_primitives():
    """Test that primitive types pass through unchanged."""
    node = TestAssetReturningNode(test_mode="simple")

    result = {
        "string": "hello",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
    }

    client_result = node.result_for_client(result)

    assert client_result == result


def test_result_for_client_bytes_placeholder():
    """Test that raw bytes get replaced with placeholder."""
    node = TestAssetReturningNode(test_mode="simple")

    result = {"data": b"some bytes"}

    client_result = node.result_for_client(result)

    assert client_result["data"] == "<10 bytes>"


def test_result_for_client_assetref_large_data_is_not_inlined():
    """Large AssetRef.data should not be inlined into websocket updates."""
    node = TestAssetReturningNode(test_mode="simple")

    # 5 MiB payload - larger than BaseNode's MAX_INLINE_ASSET_BYTES (4 MiB)
    big = AssetRef(data=b"x" * (5 * 1024 * 1024))
    client_result = node.result_for_client({"asset": big})

    assert isinstance(client_result["asset"], dict)
    assert client_result["asset"].get("data") is None
    assert isinstance(client_result["asset"].get("metadata"), dict)
    assert client_result["asset"]["metadata"].get("inlined_data") is False


@pytest.mark.asyncio
async def test_result_for_client_no_memory_uri():
    """Test that AssetRef without memory URI works (no data fetch)."""
    node = TestAssetReturningNode(test_mode="simple")

    # Create AssetRef with http URI
    result = {"asset": ImageRef(uri="http://example.com/image.png")}

    client_result = node.result_for_client(result)

    # Should have model dump but no data field (since no memory URI)
    assert "asset" in client_result
    assert isinstance(client_result["asset"], dict)
    assert client_result["asset"]["uri"] == "http://example.com/image.png"
    assert client_result["asset"].get("data") is None
