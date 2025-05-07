import PIL.Image
import PIL.ImageChops
import pytest
from nodetool.metadata.types import ImageRef
from nodetool.types.graph import Node, Edge
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.graph import Graph
from nodetool.types.graph import (
    Graph as APIGraph,
)


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
    await workflow_runner.process_node(context, node, {})

    assert len(workflow_runner.messages) == 1
    message = workflow_runner.messages[0]

    assert message.target.id == "2"
    assert message.slot == "value"
    assert message.value == "test"


@pytest.mark.asyncio
async def test_process_node_with_input_edges(workflow_runner: WorkflowRunner):
    input_a = Float(id="1", value=1)  # type: ignore
    input_b = Float(id="2", value=2)  # type: ignore
    add_node = Add(id="3")  # type: ignore
    output_node = IntegerOutput(id="4", name="sum")  # type: ignore
    nodes = [input_a, input_b, add_node, output_node]
    edges = [
        Edge(
            id="1",
            source="1",
            target="3",
            sourceHandle="output",
            targetHandle="a",
        ),
        Edge(
            id="2",
            source="2",
            target="3",
            sourceHandle="output",
            targetHandle="b",
        ),
        Edge(
            id="3",
            source="3",
            target="4",
            sourceHandle="output",
            targetHandle="value",
        ),
    ]

    context = ProcessingContext(
        user_id="",
        workflow_id="",
        auth_token="token",
        graph=Graph(nodes=nodes, edges=edges),
    )

    await workflow_runner.process_graph(context, context.graph)

    assert workflow_runner.outputs["sum"] == 3


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

    assert len(workflow_updates) == 2
    assert workflow_updates[1].result["output"] == 3


# @pytest.mark.asyncio
# async def test_loop_node(user: User, workflow_runner: WorkflowRunner):
#     list_node = {
#         "id": "list",
#         "type": List.get_node_type(),
#         "data": {"value": [1, 2, 3]},
#     }
#     loop_node = {
#         "id": "loop",
#         "type": Loop.get_node_type(),
#     }
#     input_node = {
#         "id": "in",
#         "parent_id": "loop",
#         "type": GroupInput.get_node_type(),
#     }
#     output_node = {
#         "id": "out",
#         "parent_id": "loop",
#         "type": GroupOutput.get_node_type(),
#     }
#     multiply_node = {
#         "id": "mul",
#         "parent_id": "loop",
#         "type": Multiply.get_node_type(),
#         "data": {},
#     }
#     nodes = [list_node, loop_node, input_node, multiply_node, output_node]
#     edges = [
#         Edge(
#             id="0",
#             source="list",
#             target="loop",
#             sourceHandle="output",
#             targetHandle="input",
#         ),
#         Edge(
#             id="1",
#             source="in",
#             target="mul",
#             sourceHandle="output",
#             targetHandle="a",
#         ),
#         Edge(
#             id="2",
#             source="in",
#             target="mul",
#             sourceHandle="output",
#             targetHandle="b",
#         ),
#         Edge(
#             id="3",
#             source="mul",
#             target="out",
#             sourceHandle="output",
#             targetHandle="input",
#         ),
#     ]
#     graph = Graph.from_dict({"nodes": nodes, "edges": edges})
#     context = ProcessingContext(
#         user_id="", workflow_id="", auth_token="token", graph=graph
#     )

#     await workflow_runner.process_graph(context, graph)

#     assert context.get_result("loop", "output") == [1, 4, 9]
