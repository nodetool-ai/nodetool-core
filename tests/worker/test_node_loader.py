import pytest
from nodetool.worker.node_loader import _discover_namespaces, load_nodes, node_to_metadata


def test_node_to_metadata_from_mock_node():
    """Test that we can extract metadata from a BaseNode subclass."""
    from nodetool.workflows.base_node import BaseNode, NODE_BY_TYPE
    from nodetool.workflows.processing_context import ProcessingContext
    from pydantic import Field

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


def test_node_to_metadata_uses_streaming_output_type():
    from typing import AsyncGenerator, TypedDict

    from nodetool.metadata.types import AudioRef, Chunk
    from nodetool.workflows.base_node import BaseNode, NODE_BY_TYPE
    from nodetool.workflows.processing_context import ProcessingContext

    class StreamingAudioNode(BaseNode):
        class OutputType(TypedDict):
            audio: AudioRef | None
            chunk: Chunk

        @classmethod
        def get_node_type(cls) -> str:
            return "test_ns.StreamingAudioNode"

        async def gen_process(
            self, context: ProcessingContext
        ) -> AsyncGenerator["StreamingAudioNode.OutputType", None]:
            yield {
                "audio": AudioRef(uri="blob://audio_test"),
                "chunk": Chunk(content="", done=True, content_type="audio"),
            }

    NODE_BY_TYPE["test_ns.StreamingAudioNode"] = StreamingAudioNode

    try:
        meta = node_to_metadata(StreamingAudioNode)
        assert [output["name"] for output in meta["outputs"]] == ["audio", "chunk"]
        assert meta["outputs"][0]["type"]["type"] == "audio"
        assert meta["outputs"][1]["type"]["type"] == "chunk"
    finally:
        del NODE_BY_TYPE["test_ns.StreamingAudioNode"]


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


def test_discover_namespaces_falls_back_to_namespace_paths(tmp_path, monkeypatch):
    pkg_a = tmp_path / "pkg_a" / "nodetool" / "nodes" / "huggingface"
    pkg_b = tmp_path / "pkg_b" / "nodetool" / "nodes" / "mlx"
    pkg_a.mkdir(parents=True)
    pkg_b.mkdir(parents=True)

    monkeypatch.syspath_prepend(str(tmp_path / "pkg_a"))
    monkeypatch.syspath_prepend(str(tmp_path / "pkg_b"))
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [])

    namespaces = _discover_namespaces()

    assert "huggingface" in namespaces
    assert "mlx" in namespaces
