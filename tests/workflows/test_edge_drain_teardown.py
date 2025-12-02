from __future__ import annotations

import asyncio
import queue

from nodetool.types.graph import Edge
from nodetool.workflows.graph import Graph
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import EdgeUpdate
from nodetool.workflows.workflow_runner import WorkflowRunner


def _collect_messages(ctx: ProcessingContext) -> list:
    msgs = []
    while not ctx.message_queue.empty():
        msgs.append(ctx.message_queue.get_nowait())
    return msgs


def test_drain_active_edges_posts_drained_updates():
    # Build a simple graph with two edges into the same target
    edges = [
        Edge(
            id="e1",
            source="A",
            sourceHandle="out",
            target="B",
            targetHandle="in1",
        ),
        Edge(
            id="e2",
            source="A",
            sourceHandle="out2",
            target="B",
            targetHandle="in2",
        ),
    ]
    graph = Graph(nodes=[], edges=edges)

    runner = WorkflowRunner(job_id="job-x")

    async def _run():
        # Attach an inbox for target B with one buffered item on in1 and an open stream on in2
        inbox_b = NodeInbox()
        inbox_b.add_upstream("in1", 1)
        inbox_b.add_upstream("in2", 1)
        inbox_b.put("in1", "value")  # buffered
        # in2: leave empty but still open

        runner.node_inboxes["B"] = inbox_b

        ctx = ProcessingContext(message_queue=queue.Queue())

        runner.drain_active_edges(ctx, graph)

        msgs = _collect_messages(ctx)
        drained_ids = {
            m.edge_id
            for m in msgs
            if isinstance(m, EdgeUpdate) and m.status == "drained"
        }
        assert drained_ids == {"e1", "e2"}

    asyncio.run(_run())
