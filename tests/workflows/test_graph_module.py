from nodetool.types.graph import Edge
from nodetool.workflows.graph import Graph
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode


class InNode(InputNode):
    pass


class AddNode(BaseNode):
    a: int = 0
    b: int = 0

    async def process(self, context):
        return self.a + self.b


class OutNode(OutputNode):
    pass


def build_graph():
    n1 = InNode(id="1", name="a", value=1)
    n2 = AddNode(id="2", a=0, b=0)
    n3 = OutNode(id="3", name="out")
    edges = [
        Edge(id="e1", source="1", sourceHandle="output", target="2", targetHandle="a"),
        Edge(id="e2", source="2", sourceHandle="output", target="3", targetHandle="value"),
    ]
    return Graph(nodes=[n1, n2, n3], edges=edges)


def test_find_and_edges():
    g = build_graph()
    assert g.find_node("2").id == "2"  # type: ignore
    edges = g.find_edges("1", "output")
    assert len(edges) == 1 and edges[0].target == "2"


def test_topological_sort():
    g = build_graph()
    assert g.topological_sort() == [["1"], ["2"], ["3"]]


def test_from_dict_round_trip():
    g = build_graph()
    data = {
        "nodes": [n.to_dict() for n in g.nodes],
        "edges": [e.model_dump() for e in g.edges],
    }
    new_graph = Graph.from_dict(data)
    assert len(new_graph.nodes) == 3
    assert new_graph.find_node("1") is not None
