from __future__ import annotations

import asyncio
import queue

import pytest

from nodetool.types.graph import Edge as APIEdge
from nodetool.types.graph import Graph as APIGraph
from nodetool.types.graph import Node as APINode
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import JobUpdate
from nodetool.workflows.workflow_runner import WorkflowRunner

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


@pytest.mark.asyncio
async def test_workflow_runner_completes_and_produces_output():
    # Ensure test nodes are registered
    from nodetool.workflows import test_nodes

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
        await asyncio.wait_for(runner.run(req, ctx), timeout=ASYNC_TEST_TIMEOUT)
    except TimeoutError:
        pytest.fail(
            f"WorkflowRunner.run timed out after {ASYNC_TEST_TIMEOUT}s for job job-int-1"
        )

    assert runner.status == "completed"
    assert runner.outputs.get("result") and runner.outputs["result"][-1] == 15.0

    msgs = _drain_msgs(ctx)
    statuses = [m.status for m in msgs if isinstance(m, JobUpdate)]
    assert "completed" in statuses


@pytest.mark.asyncio
async def test_workflow_cancellation_drains_edges_and_finalizes():
    finalize_calls: list[str] = []

    class SlowNode(BaseNode):
        async def process(self, context):  # type: ignore[override]
            await asyncio.sleep(5.0)
            return 1

        async def finalize(self, context):  # type: ignore[override]
            finalize_calls.append(self.id)

    # Build a simple graph with just the slow node
    slow_type = SlowNode.get_node_type()
    nodes = [
        APINode(id="slow", type=slow_type, data={}),
    ]
    edges = []
    api_graph = APIGraph(nodes=nodes, edges=edges)

    from nodetool.workflows.run_job_request import RunJobRequest

    req = RunJobRequest(graph=api_graph)
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="job-int-2")

    async def _run_and_cancel():
        run_task = asyncio.create_task(runner.run(req, ctx))
        # Give the workflow time to start running
        await asyncio.sleep(0.1)
        # Cancel the workflow
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

    await asyncio.wait_for(_run_and_cancel(), timeout=ASYNC_TEST_TIMEOUT)

    assert runner.status == "cancelled"
    # finalize called for slow node
    assert "slow" in finalize_calls
