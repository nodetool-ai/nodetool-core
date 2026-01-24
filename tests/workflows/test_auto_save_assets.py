"""Tests for automatic asset saving functionality."""

import pytest
from io import BytesIO

from PIL import Image

from nodetool.metadata.types import ImageRef
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
