import msgpack
from fastapi.testclient import TestClient
import pytest

from nodetool.api.worker import app
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.types.graph import Node, Edge, Graph


class FloatInput(InputNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:  # pragma: no cover
        return self.value


class FloatOutput(OutputNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:  # pragma: no cover
        return self.value


class Add(BaseNode):
    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:  # pragma: no cover
        return self.a + self.b


@pytest.fixture()
def client():
    return TestClient(app)


def test_system_stats(client: TestClient):
    res = client.get("/system_stats")
    assert res.status_code == 200
    data = res.json()
    assert "cpu_percent" in data


def test_run_simple_workflow(client: TestClient):
    nodes = [
        Node(
            id="1", type=FloatInput.get_node_type(), data={"name": "in1", "value": 1.0}
        ),
        Node(
            id="2", type=FloatInput.get_node_type(), data={"name": "in2", "value": 2.0}
        ),
        Node(id="3", type=Add.get_node_type(), data={}),
        Node(id="4", type=FloatOutput.get_node_type(), data={"name": "output"}),
    ]
    edges = [
        Edge(source="1", target="3", sourceHandle="output", targetHandle="a"),
        Edge(source="2", target="3", sourceHandle="output", targetHandle="b"),
        Edge(source="3", target="4", sourceHandle="output", targetHandle="value"),
    ]

    graph = Graph(nodes=nodes, edges=edges)

    req = RunJobRequest(
        workflow_id="wf",
        user_id="u",
        auth_token="",
        params={},
        graph=graph,
    )

    with client.websocket_connect("/predict") as ws:
        ws.send_bytes(msgpack.packb({"command": "run_job", "data": req.model_dump()}))
        msg = msgpack.unpackb(ws.receive_bytes())
        assert msg.get("message") == "Job started"

        result = None
        for _ in range(10):
            m = msgpack.unpackb(ws.receive_bytes())
            if m.get("type") == "job_update" and m.get("status") == "completed":
                result = m
                break
        assert result is not None
        assert result["result"]["output"] == [3.0]
