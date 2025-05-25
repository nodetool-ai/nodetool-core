import re
from nodetool.models.workflow import Workflow
from nodetool.types.graph import Node, Edge
from nodetool.workflows.export_dsl import workflow_to_dsl
from nodetool.workflows.base_node import InputNode, OutputNode


def test_workflow_to_dsl_basic():
    nodes = [
        Node(
            id="1",
            type=InputNode.get_node_type(),
            data={"name": "a", "value": 1},
        ),
        Node(
            id="2",
            type=OutputNode.get_node_type(),
            data={"name": "out"},
        ),
    ]
    edges = [
        Edge(
            id="e1",
            source="1",
            sourceHandle="output",
            target="2",
            targetHandle="value",
        )
    ]
    wf = Workflow(
        id="w1",
        user_id="u1",
        name="test",
        graph={
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges],
        },
    )

    code = workflow_to_dsl(wf)
    assert "graph(" in code
    assert re.search(r"n0 = .*Input", code)
    assert "(n0, 'output')" in code
