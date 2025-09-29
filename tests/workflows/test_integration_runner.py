from __future__ import annotations

import asyncio
import queue

from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.types import EdgeUpdate, JobUpdate
from nodetool.types.graph import Edge as APIEdge, Node as APINode, Graph as APIGraph


ASYNC_TEST_TIMEOUT = 3.0


class NumberInput(InputNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class NumberOutput(OutputNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class Multiply(BaseNode):
    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.a * self.b


def _drain_msgs(ctx: ProcessingContext):
    items = []
    while not ctx.message_queue.empty():
        items.append(ctx.message_queue.get_nowait())
    return items


def test_workflow_runner_completes_and_produces_output(event_loop):
    # Ensure test nodes are registered
    from nodetool.workflows import test_nodes  # noqa: F401

    # Build a small graph: (5 * 3) -> output
    nodes = [
        APINode(
            id="in1",
            type=NumberInput.get_node_type(),
            data={"name": "in1", "value": 5.0},
        ),
        APINode(
            id="in2",
            type=NumberInput.get_node_type(),
            data={"name": "in2", "value": 3.0},
        ),
        APINode(id="mul", type=Multiply.get_node_type(), data={}),
        APINode(
            id="out",
            type=NumberOutput.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        APIEdge(
            id="e1", source="in1", sourceHandle="output", target="mul", targetHandle="a"
        ),
        APIEdge(
            id="e2", source="in2", sourceHandle="output", target="mul", targetHandle="b"
        ),
        APIEdge(
            id="e3",
            source="mul",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    api_graph = APIGraph(nodes=nodes, edges=edges)

    from nodetool.workflows.run_job_request import RunJobRequest

    req = RunJobRequest(graph=api_graph)
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="job-int-1")

    try:
        event_loop.run_until_complete(
            asyncio.wait_for(runner.run(req, ctx), timeout=ASYNC_TEST_TIMEOUT)
        )
    except asyncio.TimeoutError:
        pytest.fail(
            f"WorkflowRunner.run timed out after {ASYNC_TEST_TIMEOUT}s for job job-int-1"
        )

    assert runner.status == "completed"
    assert runner.outputs.get("result") and runner.outputs["result"][-1] == 15.0

    msgs = _drain_msgs(ctx)
    statuses = [m.status for m in msgs if isinstance(m, JobUpdate)]
    assert "completed" in statuses


def test_workflow_cancellation_drains_edges_and_finalizes(event_loop):

    finalize_calls: list[str] = []

    class SlowNode(BaseNode):
        async def process(self, context):  # type: ignore[override]
            await asyncio.sleep(2.0)
            return 1

        async def finalize(self, context):  # type: ignore[override]
            finalize_calls.append(self.id)

    # Build a graph: slow -> output
    slow_type = SlowNode.get_node_type()
    nodes = [
        APINode(id="slow", type=slow_type, data={}),
        APINode(
            id="out",
            type=NumberOutput.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        APIEdge(
            id="eS",
            source="slow",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    api_graph = APIGraph(nodes=nodes, edges=edges)

    from nodetool.workflows.run_job_request import RunJobRequest

    req = RunJobRequest(graph=api_graph)
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="job-int-2")

    async def _run_and_cancel():
        async def run_task():
            async for msg in run_workflow(req):
                print(msg)

        task = asyncio.create_task(run_task())
        await asyncio.sleep(1.0)
        task.cancel()
        try:
            await task
        except Exception:
            pass

    event_loop.run_until_complete(
        asyncio.wait_for(_run_and_cancel(), timeout=ASYNC_TEST_TIMEOUT)
    )

    assert runner.status == "cancelled"

    msgs = _drain_msgs(ctx)
    has_cancel = any(isinstance(m, JobUpdate) and m.status == "cancelled" for m in msgs)
    assert has_cancel
    # We no longer emit synthetic per-edge "drained" updates on cancellation.
    # finalize called for slow node
    assert "slow" in finalize_calls
