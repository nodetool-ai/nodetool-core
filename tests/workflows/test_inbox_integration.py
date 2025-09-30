import asyncio
from typing import Any, AsyncGenerator, TypedDict

import pytest

from nodetool.types.graph import Edge, Node as APINode, Graph as APIGraph
from nodetool.workflows.base_node import BaseNode, OutputNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.workflow_runner import WorkflowRunner


class StringOutput(OutputNode):
    value: str = ""

    async def process(self, context: ProcessingContext) -> str:
        return self.value


@pytest.mark.asyncio
async def test_streaming_producer_to_streaming_consumer():
    """A streaming producer feeds a streaming-input consumer using iter_input()."""

    class StreamingProducer(BaseNode):
        @classmethod
        def get_node_type(cls) -> str:
            return "test.inbox.StreamingProducer"

        class OutputType(TypedDict):
            out: int

        async def gen_process(
            self, context: ProcessingContext
        ) -> AsyncGenerator[OutputType, None]:
            for i in range(3):
                yield {"out": i}
                await asyncio.sleep(0.01)

    class StreamingConsumer(BaseNode):
        # Declare the handle name to opt-in for inbox streaming
        in_a: int | None = None

        @classmethod
        def get_node_type(cls) -> str:
            return "test.inbox.StreamingConsumer"

        @classmethod
        def is_streaming_input(cls) -> bool:
            return True

        class OutputType(TypedDict):
            out: str

        async def gen_process(
            self, context: ProcessingContext
        ) -> AsyncGenerator[OutputType, None]:
            async for item in self.iter_input("in_a"):
                yield {"out": f"got:{item}"}

    prod = {"id": "p", "type": StreamingProducer.get_node_type()}
    cons = {"id": "c", "type": StreamingConsumer.get_node_type()}
    out = {"id": "o", "type": StringOutput.get_node_type(), "data": {"name": "sink"}}

    nodes = [prod, cons, out]
    edges = [
        {
            "id": "e1",
            "source": "p",
            "target": "c",
            "sourceHandle": "out",
            "targetHandle": "in_a",
            "ui_properties": {},
        },
        {
            "id": "e2",
            "source": "c",
            "target": "o",
            "sourceHandle": "out",
            "targetHandle": "value",
            "ui_properties": {},
        },
    ]

    graph = APIGraph(
        nodes=[APINode(**n) for n in nodes], edges=[Edge(**e) for e in edges]
    )
    req = RunJobRequest(
        user_id="u", workflow_id="wf", job_type="t", params={}, graph=graph
    )
    ctx = ProcessingContext(user_id="u", auth_token="token")
    runner = WorkflowRunner(job_id="job_stream_in_1")

    routed: list[tuple[str, Any]] = []

    original_send = runner.send_messages

    async def capture_send_async(node, result, context):
        for key, value in result.items():
            for edge in context.graph.find_edges(node.id, key):
                routed.append((edge.targetHandle, value))
        await original_send(node, result, context)

    runner.send_messages = capture_send_async  # type: ignore

    await runner.run(req, ctx)

    sink_values = [value for handle, value in routed if handle == "value"]
    assert sink_values == [
        "got:0",
        "got:1",
        "got:2",
    ]


@pytest.mark.asyncio
async def test_streaming_multi_input_fanin_iter_any():
    """Two producers feed one consumer that merges with iter_any_input()."""

    class P(BaseNode):
        name: str = "p"

        @classmethod
        def get_node_type(cls) -> str:
            return f"test.inbox.{cls.__name__}"

        class OutputType(TypedDict):
            out: str

        async def gen_process(
            self, context: ProcessingContext
        ) -> AsyncGenerator[OutputType, None]:
            for i in range(2):
                yield {"out": f"{self.name}{i}"}
                await asyncio.sleep(0.01)

    class FanIn(BaseNode):
        a: str | None = None
        b: str | None = None

        @classmethod
        def get_node_type(cls) -> str:
            return "test.inbox.FanIn"

        @classmethod
        def is_streaming_input(cls) -> bool:  # type: ignore[override]
            return True

        class OutputType(TypedDict):
            out: str

        async def gen_process(
            self, context: ProcessingContext
        ) -> AsyncGenerator[OutputType, None]:
            async for handle, item in self.iter_any_input():
                yield {"out": f"{handle}:{item}"}

    p1 = {"id": "p1", "type": P.get_node_type(), "data": {"name": "a"}}
    p2 = {"id": "p2", "type": P.get_node_type(), "data": {"name": "b"}}
    fan = {"id": "f", "type": FanIn.get_node_type()}
    out = {"id": "o", "type": StringOutput.get_node_type(), "data": {"name": "merged"}}

    nodes = [p1, p2, fan, out]
    edges = [
        {
            "id": "e1",
            "source": "p1",
            "target": "f",
            "sourceHandle": "out",
            "targetHandle": "a",
            "ui_properties": {},
        },
        {
            "id": "e2",
            "source": "p2",
            "target": "f",
            "sourceHandle": "out",
            "targetHandle": "b",
            "ui_properties": {},
        },
        {
            "id": "e3",
            "source": "f",
            "target": "o",
            "sourceHandle": "out",
            "targetHandle": "value",
            "ui_properties": {},
        },
    ]

    graph = APIGraph(
        nodes=[APINode(**n) for n in nodes], edges=[Edge(**e) for e in edges]
    )
    req = RunJobRequest(
        user_id="u", workflow_id="wf", job_type="t", params={}, graph=graph
    )
    ctx = ProcessingContext(user_id="u", auth_token="token")
    runner = WorkflowRunner(job_id="job_stream_in_2")

    routed: list[tuple[str, Any]] = []

    original_send = runner.send_messages

    async def capture_send_async(node, result, context):
        for key, value in result.items():
            for edge in context.graph.find_edges(node.id, key):
                routed.append((edge.targetHandle, value))
        await original_send(node, result, context)

    runner.send_messages = capture_send_async  # type: ignore

    await runner.run(req, ctx)

    merged_values = [value for handle, value in routed if handle == "value"]
    assert merged_values == ["a:a0", "b:b0", "a:a1", "b:b1"]


@pytest.mark.asyncio
async def test_non_streaming_node_unchanged_edge_queue_path():
    """Non-streaming nodes still receive inputs via property assignment (edge queues path)."""

    class PassThrough(BaseNode):
        value: str = ""

        @classmethod
        def get_node_type(cls) -> str:
            return "test.inbox.PassThrough"

        async def process(self, context: ProcessingContext) -> str:
            return self.value

    src = {"id": "s", "type": PassThrough.get_node_type(), "data": {"value": "hello"}}
    out = {"id": "o", "type": StringOutput.get_node_type(), "data": {"name": "sink"}}

    nodes = [src, out]
    edges = [
        {
            "id": "e1",
            "source": "s",
            "target": "o",
            "sourceHandle": "output",
            "targetHandle": "value",
            "ui_properties": {},
        },
    ]

    graph = APIGraph(
        nodes=[APINode(**n) for n in nodes], edges=[Edge(**e) for e in edges]
    )
    req = RunJobRequest(
        user_id="u", workflow_id="wf", job_type="t", params={}, graph=graph
    )
    ctx = ProcessingContext(user_id="u", auth_token="token")
    runner = WorkflowRunner(job_id="job_stream_in_4")

    routed: list[tuple[str, Any]] = []

    original_send = runner.send_messages

    async def capture_send_async(node, result, context):
        for key, value in result.items():
            for edge in context.graph.find_edges(node.id, key):
                routed.append((edge.targetHandle, value))
        await original_send(node, result, context)

    runner.send_messages = capture_send_async  # type: ignore

    await runner.run(req, ctx)

    sink_values = [value for handle, value in routed if handle == "value"]
    assert sink_values == ["hello"]
