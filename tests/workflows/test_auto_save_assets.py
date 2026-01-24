"""Tests for automatic asset saving functionality."""

import json
from io import BytesIO

import pandas as pd
import pytest
from PIL import Image

from nodetool.metadata.types import (
    AudioRef,
    DataframeRef,
    DocumentRef,
    ExcelRef,
    ImageRef,
    JSONRef,
    SVGRef,
    TextRef,
    VideoRef,
)
from nodetool.models.asset import Asset
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class TestAutoSaveNode(BaseNode):
    """Test node with auto_save_asset enabled."""

    _auto_save_asset = True

    async def process(self, context: ProcessingContext) -> dict:
        """Return a result with an ImageRef."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Return ImageRef with data
        return {"output": ImageRef(data=img_bytes.getvalue())}


class TestNoAutoSaveNode(BaseNode):
    """Test node without auto_save_asset."""

    _auto_save_asset = False

    async def process(self, context: ProcessingContext) -> dict:
        """Return a result with an ImageRef."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return {"output": ImageRef(data=img_bytes.getvalue())}


# Mark all tests to run in the same worker to avoid database conflicts
pytestmark = pytest.mark.xdist_group(name="database")


@pytest.mark.asyncio
async def test_auto_save_asset_enabled(
    context: ProcessingContext,
    user_id: str,
):
    """Test that assets are automatically saved when auto_save_asset is True."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    # Set up the processing context
    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    # Create a test node
    node = TestAutoSaveNode(id="test_node_auto")

    # Create a minimal actor (we only need it for the _auto_save_assets method)
    class MockRunner:
        job_id = "test_job"
        device = "cpu"

    runner = MockRunner()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    # Execute the node's process method and trigger auto-save
    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    # Check that the ImageRef now has an asset_id
    assert result["output"].asset_id is not None

    # Verify the asset was saved to the database
    saved_asset = await Asset.find(user_id, result["output"].asset_id)
    assert saved_asset is not None
    assert saved_asset.node_id == "test_node_auto"
    assert saved_asset.job_id == "test_job"
    assert saved_asset.workflow_id == "test_workflow"
    assert saved_asset.content_type == "image/png"


@pytest.mark.asyncio
async def test_auto_save_asset_disabled(
    context: ProcessingContext,
    user_id: str,
):
    """Test that auto_save_asset flag can be checked."""
    # Create a test node
    node = TestNoAutoSaveNode(id="test_node_no_auto")

    # Check that auto_save_asset returns False
    assert not node.__class__.auto_save_asset()


@pytest.mark.asyncio
async def test_auto_save_multiple_assets(
    context: ProcessingContext,
    user_id: str,
):
    """Test that multiple assets in a result are all saved."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    # Set up the processing context
    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    # Create a node that returns multiple assets
    class MultiAssetNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            img1 = Image.new("RGB", (50, 50), color="red")
            img1_bytes = BytesIO()
            img1.save(img1_bytes, format="PNG")

            img2 = Image.new("RGB", (50, 50), color="green")
            img2_bytes = BytesIO()
            img2.save(img2_bytes, format="PNG")

            return {
                "output1": ImageRef(data=img1_bytes.getvalue()),
                "output2": ImageRef(data=img2_bytes.getvalue()),
            }

    node = MultiAssetNode(id="test_node_multi")

    # Create actor
    class MockRunner:
        job_id = "test_job"
        device = "cpu"

    runner = MockRunner()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    # Execute and auto-save
    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    # Both assets should be saved
    assert result["output1"].asset_id is not None
    assert result["output2"].asset_id is not None

    # Verify both assets in database
    asset1 = await Asset.find(user_id, result["output1"].asset_id)
    asset2 = await Asset.find(user_id, result["output2"].asset_id)

    assert asset1 is not None
    assert asset2 is not None
    assert asset1.node_id == "test_node_multi"
    assert asset2.node_id == "test_node_multi"


@pytest.mark.asyncio
async def test_auto_save_nested_assets(
    context: ProcessingContext,
    user_id: str,
):
    """Test that nested assets in lists/dicts are also saved."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    # Set up the processing context
    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    # Create a node that returns nested assets
    class NestedAssetNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            img = Image.new("RGB", (30, 30), color="yellow")
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")

            return {
                "images": [
                    ImageRef(data=img_bytes.getvalue()),
                    ImageRef(data=img_bytes.getvalue()),
                ]
            }

    node = NestedAssetNode(id="test_node_nested")

    # Create actor
    class MockRunner:
        job_id = "test_job"
        device = "cpu"

    runner = MockRunner()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    # Execute and auto-save
    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    # Both nested assets should be saved
    assert result["images"][0].asset_id is not None
    assert result["images"][1].asset_id is not None

    # Verify in database
    asset1 = await Asset.find(user_id, result["images"][0].asset_id)
    asset2 = await Asset.find(user_id, result["images"][1].asset_id)

    assert asset1 is not None
    assert asset2 is not None


# ============================================================================
# Comprehensive tests for all asset types with binary representations
# ============================================================================


@pytest.mark.asyncio
async def test_auto_save_image_ref_binary(
    context: ProcessingContext,
    user_id: str,
):
    """Test auto-save with ImageRef containing PNG binary data."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    class ImageNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            # Create a real PNG image with PIL
            img = Image.new("RGB", (64, 64), color=(255, 0, 0))
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")
            return {"image": ImageRef(data=img_bytes.getvalue())}

    node = ImageNode(id="image_node")
    runner = type("MockRunner", (), {"job_id": "test_job", "device": "cpu"})()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    assert result["image"].asset_id is not None
    asset = await Asset.find(user_id, result["image"].asset_id)
    assert asset is not None
    assert asset.content_type == "image/png"
    assert asset.node_id == "image_node"


@pytest.mark.asyncio
async def test_auto_save_audio_ref_binary(
    context: ProcessingContext,
    user_id: str,
):
    """Test auto-save with AudioRef containing MP3 binary data."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    class AudioNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            # Create mock MP3 binary data (simplified for testing)
            # In reality, this would be actual MP3 encoded audio
            audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 100
            return {"audio": AudioRef(data=audio_data)}

    node = AudioNode(id="audio_node")
    runner = type("MockRunner", (), {"job_id": "test_job", "device": "cpu"})()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    assert result["audio"].asset_id is not None
    asset = await Asset.find(user_id, result["audio"].asset_id)
    assert asset is not None
    assert asset.content_type == "audio/mp3"
    assert asset.node_id == "audio_node"


@pytest.mark.asyncio
async def test_auto_save_video_ref_binary(
    context: ProcessingContext,
    user_id: str,
):
    """Test auto-save with VideoRef containing MP4 binary data."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    class VideoNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            # Create mock MP4 binary data (simplified for testing)
            # Real MP4 would have proper ftyp and moov boxes
            video_data = b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00" + b"\x00" * 100
            return {"video": VideoRef(data=video_data)}

    node = VideoNode(id="video_node")
    runner = type("MockRunner", (), {"job_id": "test_job", "device": "cpu"})()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    assert result["video"].asset_id is not None
    asset = await Asset.find(user_id, result["video"].asset_id)
    assert asset is not None
    assert asset.content_type == "video/mp4"
    assert asset.node_id == "video_node"


@pytest.mark.asyncio
async def test_auto_save_text_ref_binary(
    context: ProcessingContext,
    user_id: str,
):
    """Test auto-save with TextRef containing UTF-8 text data."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    class TextNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            # Create UTF-8 encoded text
            text_content = "Hello, World! 🌍\nThis is a test document."
            text_data = text_content.encode("utf-8")
            return {"text": TextRef(data=text_data)}

    node = TextNode(id="text_node")
    runner = type("MockRunner", (), {"job_id": "test_job", "device": "cpu"})()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    assert result["text"].asset_id is not None
    asset = await Asset.find(user_id, result["text"].asset_id)
    assert asset is not None
    assert asset.content_type == "text/plain"
    assert asset.node_id == "text_node"


@pytest.mark.asyncio
async def test_auto_save_document_ref_binary(
    context: ProcessingContext,
    user_id: str,
):
    """Test auto-save with DocumentRef containing PDF binary data."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    class DocumentNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            # Create mock PDF binary data (simplified for testing)
            # Real PDF would have proper structure
            pdf_data = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n" + b"fake pdf content" + b"\n%%EOF"
            return {"document": DocumentRef(data=pdf_data)}

    node = DocumentNode(id="document_node")
    runner = type("MockRunner", (), {"job_id": "test_job", "device": "cpu"})()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    assert result["document"].asset_id is not None
    asset = await Asset.find(user_id, result["document"].asset_id)
    assert asset is not None
    assert asset.content_type == "application/pdf"
    assert asset.node_id == "document_node"


@pytest.mark.asyncio
async def test_auto_save_dataframe_ref_binary(
    context: ProcessingContext,
    user_id: str,
):
    """Test auto-save with DataframeRef containing JSON binary data."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    class DataframeNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            # Create a pandas DataFrame and convert to JSON
            df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
            # Use DataframeRef.from_pandas helper
            dataframe_ref = DataframeRef.from_pandas(df)
            return {"dataframe": dataframe_ref}

    node = DataframeNode(id="dataframe_node")
    runner = type("MockRunner", (), {"job_id": "test_job", "device": "cpu"})()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    assert result["dataframe"].asset_id is not None
    asset = await Asset.find(user_id, result["dataframe"].asset_id)
    assert asset is not None
    assert asset.content_type == "application/json"
    assert asset.node_id == "dataframe_node"


@pytest.mark.asyncio
async def test_auto_save_excel_ref_binary(
    context: ProcessingContext,
    user_id: str,
):
    """Test auto-save with ExcelRef containing XLSX binary data."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    class ExcelNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            # Create a real Excel file with pandas
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            excel_bytes = BytesIO()
            df.to_excel(excel_bytes, index=False, engine="openpyxl")
            return {"excel": ExcelRef(data=excel_bytes.getvalue())}

    node = ExcelNode(id="excel_node")
    runner = type("MockRunner", (), {"job_id": "test_job", "device": "cpu"})()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    assert result["excel"].asset_id is not None
    asset = await Asset.find(user_id, result["excel"].asset_id)
    assert asset is not None
    assert asset.content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert asset.node_id == "excel_node"


@pytest.mark.asyncio
async def test_auto_save_json_ref_binary(
    context: ProcessingContext,
    user_id: str,
):
    """Test auto-save with JSONRef containing JSON binary data."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    class JSONNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            # Create JSON data
            json_data = json.dumps({"key": "value", "number": 42, "list": [1, 2, 3]})
            return {"json": JSONRef(data=json_data)}

    node = JSONNode(id="json_node")
    runner = type("MockRunner", (), {"job_id": "test_job", "device": "cpu"})()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    assert result["json"].asset_id is not None
    asset = await Asset.find(user_id, result["json"].asset_id)
    assert asset is not None
    assert asset.content_type == "application/json"
    assert asset.node_id == "json_node"


@pytest.mark.asyncio
async def test_auto_save_svg_ref_binary(
    context: ProcessingContext,
    user_id: str,
):
    """Test auto-save with SVGRef containing SVG XML binary data."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    class SVGNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            # Create SVG data
            svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="40" fill="red"/>
</svg>"""
            svg_data = svg_content.encode("utf-8")
            return {"svg": SVGRef(data=svg_data)}

    node = SVGNode(id="svg_node")
    runner = type("MockRunner", (), {"job_id": "test_job", "device": "cpu"})()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    assert result["svg"].asset_id is not None
    asset = await Asset.find(user_id, result["svg"].asset_id)
    assert asset is not None
    assert asset.content_type == "image/svg+xml"
    assert asset.node_id == "svg_node"


@pytest.mark.asyncio
async def test_auto_save_mixed_asset_types(
    context: ProcessingContext,
    user_id: str,
):
    """Test auto-save with multiple different asset types in one result."""
    from nodetool.workflows.actor import NodeActor
    from nodetool.workflows.inbox import NodeInbox

    context.workflow_id = "test_workflow"
    context.job_id = "test_job"

    class MixedAssetsNode(BaseNode):
        _auto_save_asset = True

        async def process(self, context: ProcessingContext) -> dict:
            # Create different asset types
            img = Image.new("RGB", (32, 32), color="blue")
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")

            text_data = b"Sample text"

            json_data = json.dumps({"test": "data"}).encode("utf-8")

            return {
                "image": ImageRef(data=img_bytes.getvalue()),
                "text": TextRef(data=text_data),
                "json": JSONRef(data=json_data),
            }

    node = MixedAssetsNode(id="mixed_node")
    runner = type("MockRunner", (), {"job_id": "test_job", "device": "cpu"})()
    inbox = NodeInbox(node._id)
    actor = NodeActor(runner, node, context, inbox)  # type: ignore

    result = await node.process(context)
    await actor._auto_save_assets(node, result, context)

    # All three assets should be saved
    assert result["image"].asset_id is not None
    assert result["text"].asset_id is not None
    assert result["json"].asset_id is not None

    # Verify each asset
    image_asset = await Asset.find(user_id, result["image"].asset_id)
    text_asset = await Asset.find(user_id, result["text"].asset_id)
    json_asset = await Asset.find(user_id, result["json"].asset_id)

    assert image_asset is not None
    assert image_asset.content_type == "image/png"
    assert text_asset is not None
    assert text_asset.content_type == "text/plain"
    assert json_asset is not None
    assert json_asset.content_type == "application/json"

    # All should have the same node_id
    assert image_asset.node_id == "mixed_node"
    assert text_asset.node_id == "mixed_node"
    assert json_asset.node_id == "mixed_node"
