from nodetool.types.api_graph import (
    Edge,
    Graph,
    Node,
    get_input_schema,
    get_output_schema,
    remove_connected_slots,
)


def test_node_get_node_type():
    node = Node(id="test_id", type="custom.NodeType")
    assert node.get_node_type() == "custom.NodeType"


def test_remove_connected_slots():
    node1 = Node(id="1", type="nodetool.input.IntegerInput", data={"value": 1})
    node2 = Node(id="2", type="custom", data={"slot1": "a", "slot2": "b"})
    edge = Edge(id="e1", source="1", sourceHandle="output", target="2", targetHandle="slot1")
    graph = Graph(nodes=[node1, node2], edges=[edge])

    result = remove_connected_slots(graph)

    assert "slot1" not in result.nodes[1].data
    assert "slot2" in result.nodes[1].data
    # ensure the function returns the same graph instance
    assert result is graph


def test_get_input_schema_basic():
    int_node = Node(
        id="int",
        type="nodetool.input.IntegerInput",
        data={"name": "num", "min": 0, "max": 10, "value": 3},
    )
    img_node = Node(id="img", type="nodetool.input.ImageInput", data={"name": "pic"})
    schema = get_input_schema(Graph(nodes=[int_node, img_node], edges=[]))

    assert set(schema["required"]) == {"num", "pic"}
    assert schema["properties"]["num"]["type"] == "integer"
    assert schema["properties"]["num"]["default"] == 3
    img_schema = schema["properties"]["pic"]
    assert img_schema["type"] == "object" and "uri" in img_schema["properties"]


def test_get_output_schema_basic():
    output_node = Node(id="output", type="nodetool.output.Output", data={"name": "result"})
    schema = get_output_schema(Graph(nodes=[output_node], edges=[]))

    assert set(schema["required"]) == {"result"}
    assert schema["properties"]["result"]["type"] == "any"


def test_edge_is_control():
    # Test default (data)
    edge_default = Edge(source="1", sourceHandle="out", target="2", targetHandle="in")
    assert edge_default.is_control() is False

    # Test explicit data
    edge_data = Edge(source="1", sourceHandle="out", target="2", targetHandle="in", edge_type="data")
    assert edge_data.is_control() is False

    # Test control
    edge_control = Edge(source="1", sourceHandle="out", target="2", targetHandle="in", edge_type="control")
    assert edge_control.is_control() is True
