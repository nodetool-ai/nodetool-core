from nodetool.types.graph import Edge
from nodetool.workflows.graph import Graph
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode


class InNode(InputNode):
    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph_module.InNode"


class AddNode(BaseNode):
    a: int = 0
    b: int = 0

    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph_module.AddNode"

    async def process(self, context):
        return self.a + self.b


class OutNode(OutputNode):
    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph_module.OutNode"


def build_graph():
    n1 = InNode(id="1", name="a", value=1)  # type: ignore[call-arg]
    n2 = AddNode(id="2", a=0, b=0)  # type: ignore[call-arg]
    n3 = OutNode(id="3", name="out")  # type: ignore[call-arg]
    edges = [
        Edge(id="e1", source="1", sourceHandle="output", target="2", targetHandle="a"),
        Edge(
            id="e2", source="2", sourceHandle="output", target="3", targetHandle="value"
        ),
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


def test_from_dict_with_invalid_nodes():
    """Test Graph.from_dict handles invalid nodes gracefully"""
    data = {
        "nodes": [
            {
                "id": "1",
                "type": "tests.workflows.test_graph_module.InNode",
                "name": "a",
                "value": 1,
            },  # invalid - not registered
            {"id": "2", "type": "InvalidNodeType"},  # invalid
            {
                "id": "3",
                "type": "tests.workflows.test_graph_module.OutNode",
                "name": "out",
            },  # invalid - not registered
        ],
        "edges": [
            {
                "id": "e1",
                "source": "1",
                "sourceHandle": "output",
                "target": "3",
                "targetHandle": "value",
            }
        ],
    }
    graph = Graph.from_dict(data)
    # New behavior: Graph.from_dict attempts to resolve node classes dynamically.
    # For unknown external test types, classes may be resolved (InNode/OutNode exist here),
    # so nodes can be instantiated even if originally considered "not registered".
    assert len(graph.nodes) == 2
    # The single edge remains since both endpoints resolved
    assert len(graph.edges) == 1


def test_from_dict_with_invalid_edges():
    """Test Graph.from_dict handles invalid edges gracefully"""
    data = {
        "nodes": [
            {
                "id": "1",
                "type": "tests.workflows.test_graph_module.InNode",
                "name": "a",
                "value": 1,
            },  # invalid - not registered
            {
                "id": "2",
                "type": "tests.workflows.test_graph_module.OutNode",
                "name": "out",
            },  # invalid - not registered
        ],
        "edges": [
            {
                "id": "e1",
                "source": "1",
                "sourceHandle": "output",
                "target": "2",
                "targetHandle": "value",
            },  # invalid - no valid nodes
            {
                "id": "e2",
                "source": "1",
                "target": "nonexistent_target",
            },  # invalid - missing sourceHandle, targetHandle and invalid target
            {
                "id": "e3",
                "source": "nonexistent_source",
                "sourceHandle": "output",
                "target": "2",
                "targetHandle": "value",
            },  # invalid
        ],
    }
    graph = Graph.from_dict(data)
    # New behavior: InNode and OutNode can be resolved; keep nodes and valid edges
    assert len(graph.nodes) == 2
    # One valid edge (e1) since both ends resolve; invalid edges are skipped
    assert len(graph.edges) == 1


def test_from_dict_mixed_valid_invalid():
    """Test Graph.from_dict with mix of valid/invalid data"""
    data = {
        "nodes": [
            {
                "id": "1",
                "type": "tests.workflows.test_graph_module.InNode",
                "name": "a",
                "value": 1,
            },  # invalid - not registered
            {"id": "2", "type": "UnknownNodeType"},  # invalid node
            {
                "id": "3",
                "type": "tests.workflows.test_graph_module.AddNode",
                "a": 1,
                "b": 2,
            },  # invalid - not registered
            {
                "id": "4",
                "type": "tests.workflows.test_graph_module.OutNode",
                "name": "out",
            },  # invalid - not registered
        ],
        "edges": [
            {
                "id": "e1",
                "source": "1",
                "sourceHandle": "output",
                "target": "3",
                "targetHandle": "a",
            },  # valid
            {
                "id": "e2",
                "source": "1",
                "sourceHandle": "output",
                "target": "2",
                "targetHandle": "input",
            },  # invalid edge (target node is invalid)
            {
                "id": "e3",
                "source": "3",
                "sourceHandle": "output",
                "target": "4",
                "targetHandle": "value",
            },  # valid
            {
                "id": "e4",
                "source": "non_existent",
                "sourceHandle": "output",
                "target": "4",
                "targetHandle": "value",
            },  # invalid edge
        ],
    }
    graph = Graph.from_dict(data)
    # New behavior: InNode, AddNode, OutNode are defined here and resolvable
    assert len(graph.nodes) == 3
    assert graph.find_node("1") is not None
    assert graph.find_node("3") is not None
    assert graph.find_node("4") is not None
    # Valid edges between resolved nodes are retained: e1 and e3
    assert len(graph.edges) == 2
