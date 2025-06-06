import pytest
from nodetool.workflows.base_node import BaseNode, split_camel_case
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Edge
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.common.environment import Environment

class SourceNode(BaseNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value

class TargetNode(BaseNode):
    a: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.a

def test_split_camel_case():
    assert split_camel_case("CamelCase") == "Camel Case"
    # acronyms should stay intact
    assert split_camel_case("HTTPResponseCode") == "HTTPResponse Code"
    assert split_camel_case("wifi2Connection") == "wifi 2 Connection"

def test_generate_node_cache_key():
    node = SourceNode(id="n1", value=1.0)
    ctx = ProcessingContext(user_id="user", auth_token="token", graph=Graph())
    key = ctx.generate_node_cache_key(node)
    expected = f"user:{SourceNode.get_node_type()}:{hash(repr(node.model_dump()))}"
    assert key == expected

def test_cache_result_and_get_cached_result():
    node = SourceNode(id="n1", value=1.0)
    ctx = ProcessingContext(user_id="user", auth_token="token", graph=Graph())
    Environment.get_node_cache().clear()
    ctx.cache_result(node, 42)
    assert ctx.get_cached_result(node) == 42

def test_get_node_input_types():
    node1 = SourceNode(id="1", value=1.0)
    node2 = TargetNode(id="2", a=0.0)
    edges = [Edge(source="1", target="2", sourceHandle="output", targetHandle="a")]
    graph = Graph(nodes=[node1, node2], edges=edges)
    ctx = ProcessingContext(user_id="u", auth_token="t", graph=graph)
    types = ctx.get_node_input_types("2")
    assert types == {"a": TypeMetadata(type="float")}

def test_find_node_raises():
    node1 = SourceNode(id="1")
    ctx = ProcessingContext(user_id="u", auth_token="t", graph=Graph(nodes=[node1], edges=[]))
    with pytest.raises(ValueError):
        ctx.find_node("missing")
