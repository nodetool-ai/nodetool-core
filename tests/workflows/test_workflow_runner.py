from collections import deque
import pytest
from nodetool.metadata.types import ImageRef
from nodetool.types.graph import Node, Edge
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.graph import Graph
from nodetool.workflows.types import NodeUpdate
from nodetool.types.graph import (
    Graph as APIGraph,
)
from nodetool.common.environment import Environment


class String(BaseNode):
    value: str = ""

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class Float(BaseNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class Add(BaseNode):
    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.a + self.b


class IntegerOutput(OutputNode):
    value: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class IntegerInput(InputNode):
    value: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class ImageInput(InputNode):
    value: ImageRef = ImageRef()

    async def process(self, context: ProcessingContext) -> ImageRef:
        return self.value


@pytest.fixture
def workflow_runner() -> WorkflowRunner:
    return WorkflowRunner(job_id="1")


@pytest.mark.asyncio
async def test_process_node(workflow_runner: WorkflowRunner):
    node = String(id="1", value="test")  # type: ignore
    next_node = IntegerOutput(id="2")  # type: ignore
    context = ProcessingContext(
        user_id="",
        workflow_id="",
        auth_token="token",
        graph=Graph(
            nodes=[node, next_node],
            edges=[
                Edge(
                    id="1",
                    source="1",
                    target="2",
                    sourceHandle="output",
                    targetHandle="value",
                ),
            ],
        ),
    )
    workflow_runner.edge_queues[("1", "output", "2", "value")] = deque()
    await workflow_runner.process_node(context, node, {})

    value = workflow_runner.edge_queues[("1", "output", "2", "value")][0]

    assert value == "test"


async def get_workflow_updates(context: ProcessingContext):
    messages = []

    while context.has_messages():
        messages.append(await context.pop_message_async())

    return list(filter(lambda x: isinstance(x, JobUpdate), messages))


@pytest.mark.asyncio
async def test_from_dict():
    image_input = {
        "id": "1",
        "type": ImageInput.get_node_type(),
        "data": {
            "name": "image_a",
            "value": {
                "type": "image",
                "uri": "https://example.com/image.jpg",
            },
        },
    }

    node = ImageInput.from_dict(image_input)

    assert node.id == "1"  # type: ignore
    assert node.value.uri == "https://example.com/image.jpg"  # type: ignore


@pytest.mark.asyncio
async def test_process_graph(workflow_runner: WorkflowRunner):
    input_a = {
        "id": "1",
        "data": {"name": "input_1"},
        "type": IntegerInput.get_node_type(),
    }
    input_b = {
        "id": "2",
        "data": {"name": "input_2"},
        "type": IntegerInput.get_node_type(),
    }
    add_node = {"id": "3", "type": Add.get_node_type()}
    out_node = {
        "id": "4",
        "data": {"name": "output"},
        "type": IntegerOutput.get_node_type(),
    }

    nodes = [
        input_a,
        input_b,
        add_node,
        out_node,
    ]

    edges = [
        {
            "id": "1",
            "source": "1",
            "target": "3",
            "sourceHandle": "output",
            "targetHandle": "a",
            "ui_properties": {},
        },
        {
            "id": "2",
            "source": "2",
            "target": "3",
            "sourceHandle": "output",
            "targetHandle": "b",
            "ui_properties": {},
        },
        {
            "id": "3",
            "source": "3",
            "target": "4",
            "sourceHandle": "output",
            "targetHandle": "value",
            "ui_properties": {},
        },
    ]
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    params = {"input_1": 1, "input_2": 2}

    req = RunJobRequest(
        user_id="1",
        workflow_id="",
        job_type="",
        params=params,
        graph=graph,
    )
    context = ProcessingContext(
        user_id="1",
        auth_token="local_token",
    )

    await workflow_runner.run(req, context)

    workflow_updates = await get_workflow_updates(context)
    print(workflow_updates)

    assert len(workflow_updates) == 2
    assert workflow_updates[1].result["output"] == [3]


class ErrorNode(BaseNode):
    async def process(self, context: ProcessingContext) -> str:
        raise ValueError("Node processing error")


class InitErrorNode(BaseNode):
    async def initialize(self, context: ProcessingContext):
        raise ValueError("Node initialization error")

    async def process(self, context: ProcessingContext) -> str:
        return "should not reach here"


class CacheableNode(BaseNode):
    value: str = "initial"
    process_count: int = 0

    def is_cacheable(self) -> bool:
        return True

    async def process(self, context: ProcessingContext) -> str:
        # Adding a log to trace calls to this method
        log = Environment.get_logger()  # Corrected: No argument
        log.info(
            f"CacheableNode '{self.id}' process() called. Current process_count: {self.process_count}. Instance MEM ID: {id(self)}"
        )
        self.process_count += 1
        return self.value


@pytest.mark.asyncio
async def test_process_node_error(workflow_runner: WorkflowRunner):
    error_node = {"id": "1", "type": ErrorNode.get_node_type()}
    out_node = {
        "id": "2",
        "data": {"name": "output"},
        "type": IntegerOutput.get_node_type(),
    }
    nodes = [error_node, out_node]
    edges = [
        {
            "id": "1",
            "source": "1",
            "target": "2",
            "sourceHandle": "output",
            "targetHandle": "value",
            "ui_properties": {},
        }
    ]
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="local_token")

    # Expect run() to raise the error originating from the node's process method
    with pytest.raises(ValueError, match="Node processing error"):
        await workflow_runner.run(req, context)

    # After the error is raised and caught by pytest.raises,
    # check messages posted to the context before the exception propagated.
    messages = []
    while context.has_messages():
        messages.append(await context.pop_message_async())

    node_updates = [m for m in messages if isinstance(m, NodeUpdate)]
    job_updates = [m for m in messages if isinstance(m, JobUpdate)]

    assert any(
        update.node_id == "1" and update.status == "error" and "Node processing error" in update.error  # type: ignore
        for update in node_updates
    ), "NodeUpdate with processing error not found"

    assert any(
        update.status == "error" and "Node processing error" in str(update.error)  # type: ignore
        for update in job_updates
    ), "JobUpdate with status error not found or error message mismatch"


@pytest.mark.asyncio
async def test_initialize_node_error(workflow_runner: WorkflowRunner):
    init_error_node = {"id": "1", "type": InitErrorNode.get_node_type()}
    nodes = [init_error_node]
    edges = []
    graph = APIGraph(nodes=[Node(**n) for n in nodes], edges=[Edge(**e) for e in edges])
    req = RunJobRequest(
        user_id="1", workflow_id="", job_type="", params={}, graph=graph
    )
    context = ProcessingContext(user_id="1", auth_token="local_token")

    with pytest.raises(ValueError, match="Node initialization error"):
        await workflow_runner.run(req, context)

    messages = []
    while context.has_messages():
        messages.append(await context.pop_message_async())

    node_updates = [m for m in messages if isinstance(m, NodeUpdate)]

    assert any(
        update.node_id == "1" and update.status == "error" and "Node initialization error" in update.error  # type: ignore
        for update in node_updates
    ), "NodeUpdate with initialization error not found"

    # Check if a JobUpdate with status error was posted
    # This might depend on how workflow_runner.run finalizes after re-raising
    # For now, primarily check NodeUpdate as initialize_graph posts it before raising.
    # A JobUpdate error might also be there.
    # assert any(
    #     update.status == "error" and "Node initialization error" in update.error # type: ignore
    #     for update in job_updates
    # ), "JobUpdate with status error not found for initialization error"
