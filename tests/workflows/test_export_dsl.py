from nodetool.types.graph import Graph, Node, Edge
from nodetool.types.workflow import Workflow
from nodetool.workflows.export_dsl import workflow_to_dsl_code


def test_workflow_to_dsl_code_basic():
    graph = Graph(
        nodes=[
            Node(id="1", type="nodetool.input.StringInput", data={"name": "text"})
        ],
        edges=[],
    )
    wf = Workflow(
        id="wf1",
        name="Test",
        description="",
        access="public",
        created_at="now",
        updated_at="now",
        graph=graph,
    )

    code = workflow_to_dsl_code(wf, websocket_url="ws://example.com/chat")
    assert "StringInput" in code
    assert "websockets.connect('ws://example.com/chat')" in code
