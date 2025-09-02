import asyncio
import pytest

from nodetool.workflows.graph import Graph
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import PreviewUpdate


def _graph_dict(sync_mode_add: str = "on_any") -> dict:
    return {
        "edges": [
            {
                "id": "e1",
                "source": "seq",
                "sourceHandle": "output",
                "target": "add",
                "targetHandle": "b",
                "ui_properties": {"className": "int"},
            },
            {
                "id": "e2",
                "source": "seq",
                "sourceHandle": "output",
                "target": "neg",
                "targetHandle": "input",
                "ui_properties": {"className": "int"},
            },
            {
                "id": "e3",
                "source": "neg",
                "sourceHandle": "output",
                "target": "add",
                "targetHandle": "a",
                "ui_properties": {"className": "union"},
            },
            {
                "id": "e4",
                "source": "add",
                "sourceHandle": "output",
                "target": "prev",
                "targetHandle": "value",
                "ui_properties": {"className": "union"},
            },
        ],
        "nodes": [
            {
                "id": "seq",
                "type": "nodetool.list.SequenceIterator",
                "data": {"start": 0, "stop": 16, "step": 1},
                "ui_properties": {"position": {"x": 0, "y": 0}},
                "dynamic_properties": {},
                "dynamic_outputs": {},
                "sync_mode": "on_any",
            },
            {
                "id": "add",
                "type": "lib.math.Add",
                "data": {},
                "ui_properties": {"position": {"x": 0, "y": 0}},
                "dynamic_properties": {},
                "dynamic_outputs": {},
                "sync_mode": sync_mode_add,
            },
            {
                "id": "neg",
                "type": "lib.math.MathFunction",
                "data": {"operation": "negate"},
                "ui_properties": {"position": {"x": 0, "y": 0}},
                "dynamic_properties": {},
                "dynamic_outputs": {},
                "sync_mode": "on_any",
            },
            {
                "id": "prev",
                "type": "nodetool.workflows.base_node.Preview",
                "data": {},
                "ui_properties": {"position": {"x": 0, "y": 0}},
                "dynamic_properties": {},
                "dynamic_outputs": {},
                "sync_mode": "on_any",
            },
        ],
    }


async def _run_and_collect_values(graph: Graph, preview_id: str) -> list[int]:
    ctx = ProcessingContext(user_id="u", auth_token="t", graph=graph)
    runner = WorkflowRunner(job_id="job-sync-test")
    # Initialize inboxes explicitly since we call process_graph directly
    runner._initialize_inboxes(ctx, graph)
    await runner.process_graph(ctx, graph)
    values: list[int] = []
    while ctx.has_messages():
        msg = await ctx.pop_message_async()
        if isinstance(msg, PreviewUpdate) and msg.node_id == preview_id:
            values.append(msg.value)  # type: ignore[arg-type]
    return values


@pytest.mark.asyncio
async def test_on_any_mode_produces_mismatched_additions():
    """SequenceIterator → (negate,input) → Add should mismatch under on_any.

    Expect at least one non-zero value and more events than source length
    because Add fires on each arrival.
    """
    graph = Graph.from_dict(
        {
            "nodes": _graph_dict("on_any")["nodes"],
            "edges": _graph_dict("on_any")["edges"],
        }
    )
    values = await _run_and_collect_values(graph, preview_id="prev")

    assert len(values) > 16  # fired on both 'a' and 'b' arrivals
    assert any(v != 0 for v in values)  # mismatch exists (e.g., 1)


@pytest.mark.asyncio
async def test_zip_all_mode_aligns_pairs_and_sums_to_zero():
    """With zip_all, Add should align pairs (n + -n) == 0 exactly once per n."""
    graph = Graph.from_dict(
        {
            "nodes": _graph_dict("zip_all")["nodes"],
            "edges": _graph_dict("zip_all")["edges"],
        }
    )
    values = await _run_and_collect_values(graph, preview_id="prev")

    assert len(values) == 16
    assert all(v == 0 for v in values)
