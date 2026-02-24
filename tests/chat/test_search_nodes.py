
import pytest
from unittest.mock import MagicMock, patch
from nodetool.chat.search_nodes import search_nodes
from nodetool.metadata.node_metadata import NodeMetadata, Property, OutputSlot
from nodetool.metadata.type_metadata import TypeMetadata

@pytest.fixture
def mock_registry():
    with patch("nodetool.chat.search_nodes.get_registry") as mock_get_registry:
        registry = MagicMock()
        mock_get_registry.return_value = registry

        # Create some sample nodes
        nodes = [
            NodeMetadata(
                node_type="test.TextToImage",
                title="Text to Image Generator",
                description="Generates an image from text prompt using Stable Diffusion.",
                namespace="test",
                properties=[Property(name="prompt", type=TypeMetadata(type="string"))],
                outputs=[OutputSlot(name="image", type=TypeMetadata(type="image"))]
            ),
            NodeMetadata(
                node_type="test.ImageToText",
                title="Image to Text",
                description="Describes an image.",
                namespace="test",
                properties=[Property(name="image", type=TypeMetadata(type="image"))],
                outputs=[OutputSlot(name="text", type=TypeMetadata(type="string"))]
            ),
             NodeMetadata(
                node_type="test.ComplexNode",
                title="Complex Node",
                description="A node that does text processing and image manipulation.",
                namespace="test",
                properties=[],
                outputs=[]
            )
        ]
        registry.get_all_installed_nodes.return_value = nodes
        yield registry

def test_search_nodes_basic(mock_registry):
    results = search_nodes(["text", "image"])
    assert len(results) > 0
    # Check if TextToImage is in results
    assert any(n.node_type == "test.TextToImage" for n in results)

def test_search_nodes_phrase_match(mock_registry):
    # "text to image" should match "Text to Image Generator" strongly
    results = search_nodes(["text", "to", "image"])
    # Should be the first result because of phrase match bonus
    assert results[0].node_type == "test.TextToImage"

def test_search_nodes_empty(mock_registry):
    results = search_nodes(["nonexistenttoken"])
    assert len(results) == 0

def test_search_nodes_repeated(mock_registry):
    # Calling multiple times to ensure caching doesn't break things
    results1 = search_nodes(["text", "to", "image"])
    results2 = search_nodes(["text", "to", "image"])
    assert len(results1) == len(results2)
    assert results1[0].node_type == results2[0].node_type
