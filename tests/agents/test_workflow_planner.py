import pytest
from nodetool.agents.workflow_planner import WorkflowPlanner
from nodetool.chat.providers.base import MockProvider
from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import OutputSlot
from nodetool.workflows.property import Property


def make_node_metadata(node_type: str, namespace: str, props: list[Property], outputs: list[OutputSlot]):
    return NodeMetadata(
        title=node_type,
        description=node_type,
        namespace=namespace,
        node_type=node_type,
        layout="",
        properties=props,
        outputs=outputs,
        the_model_info={},
        recommended_models=[],
        basic_fields=[p.name for p in props],
        is_dynamic=False,
    )


def create_planner(tmp_path):
    node_input = make_node_metadata(
        "InputNode",
        "ns1",
        [Property(name="value", type=TypeMetadata(type="int"))],
        [OutputSlot(type=TypeMetadata(type="int"), name="output")],
    )
    node_add = make_node_metadata(
        "AddNode",
        "ns2",
        [
            Property(name="a", type=TypeMetadata(type="int")),
            Property(name="b", type=TypeMetadata(type="int")),
        ],
        [OutputSlot(type=TypeMetadata(type="int"), name="output")],
    )
    node_output = make_node_metadata(
        "OutputNode",
        "ns2",
        [Property(name="value", type=TypeMetadata(type="int"))],
        [],
    )
    provider = MockProvider([])
    planner = WorkflowPlanner(
        provider=provider,
        model="gpt-4o",
        objective="test",
        workspace_dir=str(tmp_path),
        node_types=[node_input, node_add, node_output],
    )
    return planner, node_input, node_add, node_output


def test_available_namespaces_and_node_types(tmp_path):
    planner, node_input, node_add, node_output = create_planner(tmp_path)
    assert planner._get_available_namespaces() == ["ns1", "ns2"]
    assert planner._get_available_node_types_list() == "InputNode, AddNode, OutputNode"


def test_validate_property_type(tmp_path):
    planner, *_ = create_planner(tmp_path)
    prop = Property(name="value", type=TypeMetadata(type="int"))
    # Should not raise for correct type
    planner._validate_property_type(prop, 1, "1", "value")
    # Invalid type should raise
    with pytest.raises(ValueError):
        planner._validate_property_type(prop, "a", "1", "value")


def test_validate_workflow_success(tmp_path):
    planner, node_input, node_add, node_output = create_planner(tmp_path)
    nodes = [
        {"id": "1", "type": node_input.node_type, "data": {"value": 1}},
        {
            "id": "2",
            "type": node_add.node_type,
            "data": {
                "a": {"source": "1", "sourceHandle": "output"},
                "b": 2,
            },
        },
        {
            "id": "3",
            "type": node_output.node_type,
            "data": {"value": {"source": "2", "sourceHandle": "output"}},
        },
    ]
    errors = planner._validate_workflow(nodes)
    assert errors == []


def test_validate_workflow_invalid_reference(tmp_path):
    planner, node_input, node_add, _ = create_planner(tmp_path)
    nodes = [
        {"id": "1", "type": node_input.node_type, "data": {"value": 1}},
        {
            "id": "2",
            "type": node_add.node_type,
            "data": {"a": {"source": "X", "sourceHandle": "output"}},
        },
    ]
    errors = planner._validate_workflow(nodes)
    assert any("non-existent node" in e for e in errors)


def test_validate_workflow_invalid_node_type(tmp_path):
    planner, node_input, *_ = create_planner(tmp_path)
    nodes = [
        {"id": "1", "type": "Unknown", "data": {}},
        {"id": "2", "type": node_input.node_type, "data": {"value": 1}},
    ]
    errors = planner._validate_workflow(nodes)
    assert "Invalid node type: Unknown" in errors
