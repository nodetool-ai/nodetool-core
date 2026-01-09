import asyncio
import queue

import pytest

from nodetool.types.api_graph import Edge
from nodetool.workflows.actor import NodeActor
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.graph import Graph
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import EdgeUpdate, NodeUpdate


class _SlowNode(BaseNode):
    async def run(self, context, inputs, outputs) -> None:  # type: ignore[override]
        await asyncio.sleep(5.0)  # intentionally long

    def get_timeout_seconds(self) -> float | None:  # type: ignore[override]
        return 0.05


async def _run_actor_and_catch(actor: NodeActor):
    try:
        await actor.run()
    except Exception:
        # timeout raises to caller; we swallow for assertions
        pass


@pytest.mark.asyncio
async def test_node_timeout_emits_error_and_drains_edges():
    # Build graph with one outbound edge from node S -> T
    s_id = "S"
    t_id = "T"
    edge = Edge(id="eST", source=s_id, sourceHandle="output", target=t_id, targetHandle="in")
    graph = Graph(nodes=[], edges=[edge])

    ctx = ProcessingContext(graph=graph, message_queue=queue.Queue())

    # Prepare runner stub with node inboxes
    class _RunnerStub:
        def __init__(self):
            self.node_inboxes = {}

    runner = _RunnerStub()
    runner.node_inboxes[t_id] = NodeInbox()
    runner.node_inboxes[t_id].add_upstream("in", 1)

    node = _SlowNode(id=s_id)  # type: ignore[call-arg]
    inbox = NodeInbox()

    actor = NodeActor(runner, node, ctx, inbox)  # type: ignore[arg-type]
    await _run_actor_and_catch(actor)

    # Collect messages
    msgs = []
    while not ctx.message_queue.empty():
        msgs.append(ctx.message_queue.get_nowait())

    statuses = [m.status for m in msgs if isinstance(m, NodeUpdate)]
    assert "error" in statuses  # timeout posted as error

    drained = any(isinstance(m, EdgeUpdate) and m.edge_id == "eST" and m.status == "drained" for m in msgs)
    assert drained, "Downstream edge should be drained on timeout"
