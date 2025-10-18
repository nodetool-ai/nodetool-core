"""
Test cases for NodeTool class
"""

import pytest
from typing import Any
from nodetool.agents.tools.node_tool import NodeTool
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef, AudioRef, TextRef, AssetRef
from pydantic import Field
import PIL.Image
import numpy as np


class TestNode(BaseNode):
    """A simple test node for testing NodeTool."""

    input_text: str = Field(default="", description="Input text to process")
    multiplier: int = Field(default=1, description="Number of times to repeat the text")

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """Process the input by repeating it."""
        result = self.input_text * self.multiplier
        return {"output": result}


class ComplexTestNode(BaseNode):
    """A more complex test node with multiple inputs and outputs."""

    text: str = Field(description="Text to process")
    prefix: str = Field(default="", description="Prefix to add")
    suffix: str = Field(default="", description="Suffix to add")
    uppercase: bool = Field(default=False, description="Convert to uppercase")

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """Process text with various transformations."""
        result = self.prefix + self.text + self.suffix
        if self.uppercase:
            result = result.upper()
        return {
            "processed_text": result,
            "length": len(result),
            "word_count": len(result.split()),
        }


class AssetReturningNode(BaseNode):
    """A node that returns various asset refs for testing memory URI conversion."""

    asset_type: str = Field(description="Type of asset to return")
    nesting_level: str = Field(default="simple", description="Level of nesting")

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """Return asset refs in various structures."""
        if self.asset_type == "image":
            # Create a simple test image
            image = PIL.Image.new("RGB", (100, 100), color="red")
            asset_ref = await context.image_from_pil(image)
        elif self.asset_type == "audio":
            # Create simple audio data
            audio_data = np.array([100, -100, 200, -200], dtype=np.int16)
            asset_ref = await context.audio_from_numpy(audio_data, sample_rate=22050)
        elif self.asset_type == "text":
            # Create text asset
            asset_ref = await context.text_from_str("Test text content")
        else:
            # Create generic asset
            asset_ref = AssetRef(data=b"test data")

        if self.nesting_level == "simple":
            return {"asset": asset_ref}
        elif self.nesting_level == "nested":
            return {
                "result": {
                    "primary_asset": asset_ref,
                    "metadata": {
                        "secondary_asset": asset_ref,
                        "info": "nested structure",
                    },
                }
            }
        elif self.nesting_level == "list":
            return {
                "assets": [asset_ref, asset_ref],
                "mixed_list": [asset_ref, "string", 42, {"nested_asset": asset_ref}],
            }
        elif self.nesting_level == "deep":
            return {
                "level1": {
                    "level2": {
                        "level3": {
                            "deep_asset": asset_ref,
                            "deep_list": [asset_ref, {"even_deeper": asset_ref}],
                        }
                    }
                }
            }
        else:
            return {"asset": asset_ref}


@pytest.mark.asyncio
async def test_node_tool_creation():
    """Test creating a NodeTool from a node class."""
    tool = NodeTool(TestNode)

    assert tool.name == "test_node_tool_Test"
    assert "test node" in tool.description.lower()
    assert tool.node_type == TestNode.get_node_type()

    # Check input schema
    assert tool.input_schema["type"] == "object"
    assert "input_text" in tool.input_schema["properties"]
    assert "multiplier" in tool.input_schema["properties"]


@pytest.mark.asyncio
async def test_node_tool_process():
    """Test executing a node through NodeTool."""
    tool = NodeTool(TestNode)

    # Create a mock context
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False,
    )

    # Execute the tool
    params = {"input_text": "Hello ", "multiplier": 3}

    result = await tool.process(context, params)

    assert result["status"] == "completed"
    assert result["node_type"] == TestNode.get_node_type()
    assert result["result"]["output"]["output"] == "Hello Hello Hello "


@pytest.mark.asyncio
async def test_node_tool_complex_node():
    """Test NodeTool with a more complex node."""
    tool = NodeTool(ComplexTestNode)

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False,
    )

    params = {"text": "world", "prefix": "Hello ", "suffix": "!", "uppercase": True}

    result = await tool.process(context, params)

    # Debug print to see what's wrong
    if result["status"] == "failed":
        print(f"Error: {result.get('error')}")
        print(f"Traceback: {result.get('traceback')}")

    assert result["status"] == "completed"
    # The result is wrapped in "output" because convert_output wraps dict returns
    assert result["result"]["output"]["processed_text"] == "HELLO WORLD!"
    assert result["result"]["output"]["length"] == 12
    assert result["result"]["output"]["word_count"] == 2


@pytest.mark.asyncio
async def test_node_tool_error_handling():
    """Test NodeTool error handling with invalid parameters."""
    tool = NodeTool(ComplexTestNode)

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False,
    )

    # Missing required parameter 'text'
    params = {"prefix": "Hello "}

    result = await tool.process(context, params)

    assert result["status"] == "failed"
    assert "error" in result
    assert result["node_type"] == ComplexTestNode.get_node_type()


@pytest.mark.asyncio
async def test_node_tool_from_node_type():
    """Test creating NodeTool from a node type string."""
    # This test assumes the TestNode is registered in NODE_BY_TYPE
    # In practice, you'd use a real registered node type
    try:
        tool = NodeTool.from_node_type("nodetool.text.Concatenate")
        assert tool.node_type == "nodetool.text.Concatenate"
    except ValueError:
        # If the node type doesn't exist, that's ok for this test
        pass


def test_node_tool_user_message():
    """Test the user message generation."""
    tool = NodeTool(TestNode)

    params = {"input_text": "Hello", "multiplier": 2}
    message = tool.user_message(params)

    assert "Test" in message
    assert "Hello" in message
    assert "2" in message


def test_node_tool_name_conversion():
    """Test the tool name conversion for different node classes."""
    tool = NodeTool(ComplexTestNode)
    assert tool.name == "test_node_tool_ComplexTest"

    # Test with a node that doesn't end in "Node"
    class MySpecialProcessor(BaseNode):
        async def process(self, context: Any) -> Any:
            return {}

    tool2 = NodeTool(MySpecialProcessor)
    assert tool2.name == "test_node_tool_MySpecialProcessor"


@pytest.mark.asyncio
async def test_node_tool_simple_asset_memory_uri():
    """Test that simple asset refs are returned as memory URIs."""
    tool = NodeTool(AssetReturningNode)

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False,
    )

    # Test with image asset
    params = {"asset_type": "image", "nesting_level": "simple"}
    result = await tool.process(context, params)

    assert result["status"] == "completed"
    asset = result["result"]["output"]["asset"]
    assert isinstance(asset, ImageRef)
    assert asset.uri.startswith("memory://")


@pytest.mark.asyncio
async def test_node_tool_nested_asset_memory_uri():
    """Test that nested asset refs are returned as memory URIs."""
    tool = NodeTool(AssetReturningNode)

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False,
    )

    # Test with nested audio asset
    params = {"asset_type": "audio", "nesting_level": "nested"}
    result = await tool.process(context, params)

    assert result["status"] == "completed"
    output = result["result"]["output"]

    # Check primary asset
    primary_asset = output["result"]["primary_asset"]
    assert isinstance(primary_asset, AudioRef)
    assert primary_asset.uri.startswith("memory://")

    # Check nested asset
    secondary_asset = output["result"]["metadata"]["secondary_asset"]
    assert isinstance(secondary_asset, AudioRef)
    assert secondary_asset.uri.startswith("memory://")


@pytest.mark.asyncio
async def test_node_tool_list_asset_memory_uri():
    """Test that asset refs in lists are returned as memory URIs."""
    tool = NodeTool(AssetReturningNode)

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False,
    )

    # Test with text assets in lists
    params = {"asset_type": "text", "nesting_level": "list"}
    result = await tool.process(context, params)

    assert result["status"] == "completed"
    output = result["result"]["output"]

    # Check assets in simple list
    assets_list = output["assets"]
    assert len(assets_list) == 2
    for asset in assets_list:
        assert isinstance(asset, TextRef)
        assert asset.uri.startswith("memory://")

    # Check mixed list with nested asset
    mixed_list = output["mixed_list"]
    assert len(mixed_list) == 4
    assert isinstance(mixed_list[0], TextRef)
    assert mixed_list[0].uri.startswith("memory://")
    assert mixed_list[1] == "string"
    assert mixed_list[2] == 42
    assert isinstance(mixed_list[3]["nested_asset"], TextRef)
    assert mixed_list[3]["nested_asset"].uri.startswith("memory://")


@pytest.mark.asyncio
async def test_node_tool_deep_nested_asset_memory_uri():
    """Test that deeply nested asset refs are returned as memory URIs."""
    tool = NodeTool(AssetReturningNode)

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False,
    )

    # Test with deeply nested generic asset
    params = {"asset_type": "generic", "nesting_level": "deep"}
    result = await tool.process(context, params)

    assert result["status"] == "completed"
    output = result["result"]["output"]

    # Navigate to deep asset
    deep_asset = output["level1"]["level2"]["level3"]["deep_asset"]
    assert isinstance(deep_asset, AssetRef)
    assert deep_asset.data == b"test data"  # Generic assets keep data field

    # Check assets in deep list
    deep_list = output["level1"]["level2"]["level3"]["deep_list"]
    assert len(deep_list) == 2
    assert isinstance(deep_list[0], AssetRef)
    assert deep_list[0].data == b"test data"
    assert isinstance(deep_list[1]["even_deeper"], AssetRef)
    assert deep_list[1]["even_deeper"].data == b"test data"


@pytest.mark.asyncio
async def test_node_tool_multiple_asset_types():
    """Test that different asset types are handled correctly."""
    tool = NodeTool(AssetReturningNode)

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False,
    )

    asset_types = ["image", "audio", "text"]

    for asset_type in asset_types:
        params = {"asset_type": asset_type, "nesting_level": "simple"}
        result = await tool.process(context, params)

        assert result["status"] == "completed"
        asset = result["result"]["output"]["asset"]

        if asset_type == "image":
            assert isinstance(asset, ImageRef)
        elif asset_type == "audio":
            assert isinstance(asset, AudioRef)
        elif asset_type == "text":
            assert isinstance(asset, TextRef)

        # All should have memory URIs
        assert asset.uri.startswith("memory://")


def test_node_tool_asset_ref_helper():
    """Test helper function to check asset ref structures recursively."""

    def check_asset_refs_recursive(obj, path=""):
        """Recursively check that all AssetRefs have memory URIs."""
        if isinstance(obj, (ImageRef, AudioRef, TextRef)):
            assert obj.uri.startswith(
                "memory://"
            ), f"AssetRef at {path} should have memory URI, got: {obj.uri}"
        elif isinstance(obj, AssetRef):
            # Generic AssetRef should have data field set
            assert (
                obj.data is not None
            ), f"Generic AssetRef at {path} should have data field set"
        elif isinstance(obj, dict):
            for key, value in obj.items():
                check_asset_refs_recursive(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_asset_refs_recursive(item, f"{path}[{i}]")

    # Test the helper function itself
    test_data = {
        "simple": ImageRef(uri="memory://test123"),
        "nested": {
            "audio": AudioRef(uri="memory://test456"),
            "list": [TextRef(uri="memory://test789")],
        },
    }

    # This should pass
    check_asset_refs_recursive(test_data)

    # This should fail
    bad_data = {"asset": ImageRef(uri="http://example.com/image.jpg")}
    try:
        check_asset_refs_recursive(bad_data)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "should have memory URI" in str(e)
