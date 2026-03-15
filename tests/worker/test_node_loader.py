import pytest

from nodetool.worker.node_loader import load_nodes, node_to_metadata


def test_node_to_metadata_from_mock_node():
    """Test that we can extract metadata from a BaseNode subclass."""
    from pydantic import Field

    from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
    from nodetool.workflows.processing_context import ProcessingContext

    class MockTestNode(BaseNode):
        """A test node for unit testing."""
        prompt: str = Field(default="hello")

        @classmethod
        def get_node_type(cls) -> str:
            return "test_ns.MockTestNode"

        async def process(self, context: ProcessingContext) -> str:
            return self.prompt

    NODE_BY_TYPE["test_ns.MockTestNode"] = MockTestNode

    try:
        meta = node_to_metadata(MockTestNode)
        assert meta["node_type"] == "test_ns.MockTestNode"
        assert any(p["name"] == "prompt" for p in meta["properties"])
    finally:
        del NODE_BY_TYPE["test_ns.MockTestNode"]


def test_load_nodes_filters_by_namespace():
    from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
    from nodetool.workflows.processing_context import ProcessingContext

    class AllowedNode(BaseNode):
        @classmethod
        def get_node_type(cls) -> str:
            return "huggingface.AllowedNode"
        async def process(self, context: ProcessingContext) -> str:
            return ""

    class DeniedNode(BaseNode):
        @classmethod
        def get_node_type(cls) -> str:
            return "other.DeniedNode"
        async def process(self, context: ProcessingContext) -> str:
            return ""

    NODE_BY_TYPE["huggingface.AllowedNode"] = AllowedNode
    NODE_BY_TYPE["other.DeniedNode"] = DeniedNode

    try:
        nodes = load_nodes(namespaces=["huggingface"])
        types = [n["node_type"] for n in nodes]
        assert "huggingface.AllowedNode" in types
        assert "other.DeniedNode" not in types
    finally:
        NODE_BY_TYPE.pop("huggingface.AllowedNode", None)
        NODE_BY_TYPE.pop("other.DeniedNode", None)
