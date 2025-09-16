import asyncio
import os
from nodetool.metadata.types import Any
from nodetool.workflows.run_job_request import RunJobRequest
import pytest

from nodetool.types.graph import Graph as ApiGraph, Edge, Node
from nodetool.workflows.graph import Graph
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import PreviewUpdate
from nodetool.workflows.run_workflow import run_workflow

pytestmark = pytest.mark.skipif(
    bool(os.getenv("GITHUB_ACTIONS")),
    reason="Skip sync mode integration tests on GitHub Actions",
)


def _graph(sync_mode_add: str = "on_any") -> ApiGraph:
    return ApiGraph(
        edges=[
            Edge(
                id="e1",
                source="seq",
                sourceHandle="output",
                target="add",
                targetHandle="b",
                ui_properties={"className": "int"},
            ),
            Edge(
                id="e2",
                source="seq",
                sourceHandle="output",
                target="neg",
                targetHandle="input",
                ui_properties={"className": "int"},
            ),
            Edge(
                id="e3",
                source="neg",
                sourceHandle="output",
                target="add",
                targetHandle="a",
                ui_properties={"className": "union"},
            ),
            Edge(
                id="e4",
                source="add",
                sourceHandle="output",
                target="prev",
                targetHandle="value",
                ui_properties={"className": "union"},
            ),
        ],
        nodes=[
            Node(
                id="seq",
                type="nodetool.list.SequenceIterator",
                data={"start": 0, "stop": 16, "step": 1},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
            Node(
                id="add",
                type="lib.math.Add",
                data={},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode=sync_mode_add,
            ),
            Node(
                id="neg",
                type="lib.math.MathFunction",
                data={"operation": "negate"},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
            Node(
                id="prev",
                type="nodetool.workflows.base_node.Preview",
                data={},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
        ],
    )


async def _run_and_collect_values(graph: ApiGraph, preview_id: str) -> list[Any]:
    values: list[Any] = []
    async for msg in run_workflow(RunJobRequest(graph=graph), use_thread=False):
        if isinstance(msg, PreviewUpdate) and msg.node_id == preview_id:
            values.append(msg.value)
    return values


@pytest.mark.asyncio
async def test_on_any_mode_waits_for_first_pair_then_mismatches():
    """Non-streaming Add with on_any waits for initial pair, then fires on each arrival.

    With direct b-path outrunning a-path (through negate), the initial fire
    uses the latest b (15) and first a (-0), producing 15. Subsequent a updates
    drive values down to 0.
    """
    graph = _graph("on_any")
    values = await _run_and_collect_values(graph, preview_id="prev")

    assert len(values) == 16
    assert values[0] == 15
    assert values[-1] == 0
    # Should be strictly decreasing sequence from 15 to 0
    assert values == list(range(15, -1, -1))


@pytest.mark.asyncio
async def test_zip_all_mode_aligns_pairs_and_sums_to_zero():
    """With zip_all, Add should align pairs (n + -n) == 0 exactly once per n."""
    graph = _graph("zip_all")
    values = await _run_and_collect_values(graph, preview_id="prev")

    assert len(values) == 16
    assert all(v == 0 for v in values)


@pytest.mark.asyncio
async def test_on_any_mode_with_deeper_a_path():
    """One branch has extra hops: seq → neg → neg → Add(a) vs seq → Add(b).

    Non-streaming Add waits for the first availability from both inputs; since the
    direct b-path outruns the deeper a-path, the latest b (15) combines with the
    first a (0) to produce 15. As the deeper a-path catches up (a == n due to double
    negation), the sum increases deterministically to 30.
    """
    graph = ApiGraph(
        edges=[
            Edge(
                id="e1",
                source="seq",
                sourceHandle="output",
                target="add",
                targetHandle="b",
                ui_properties={"className": "int"},
            ),
            Edge(
                id="e2",
                source="seq",
                sourceHandle="output",
                target="neg1",
                targetHandle="input",
                ui_properties={"className": "int"},
            ),
            Edge(
                id="e3",
                source="neg1",
                sourceHandle="output",
                target="neg2",
                targetHandle="input",
                ui_properties={"className": "int"},
            ),
            Edge(
                id="e4",
                source="neg2",
                sourceHandle="output",
                target="add",
                targetHandle="a",
                ui_properties={"className": "union"},
            ),
            Edge(
                id="e5",
                source="add",
                sourceHandle="output",
                target="prev",
                targetHandle="value",
                ui_properties={"className": "union"},
            ),
        ],
        nodes=[
            Node(
                id="seq",
                type="nodetool.list.SequenceIterator",
                data={"start": 0, "stop": 16, "step": 1},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
            Node(
                id="neg1",
                type="lib.math.MathFunction",
                data={"operation": "negate"},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
            Node(
                id="neg2",
                type="lib.math.MathFunction",
                data={"operation": "negate"},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
            Node(
                id="add",
                type="lib.math.Add",
                data={},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
            Node(
                id="prev",
                type="nodetool.workflows.base_node.Preview",
                data={},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
        ],
    )

    values = await _run_and_collect_values(graph, preview_id="prev")

    assert len(values) == 16
    assert values[0] == 15
    assert values[-1] == 30
    assert values == list(range(15, 31))


@pytest.mark.asyncio
async def test_zip_all_mode_with_deeper_a_path_aligns_indexed_pairs():
    """Deeper a-branch with zip_all should align by index: sum == 2n for n in 0..15."""
    graph = ApiGraph(
        nodes=[
            Node(
                id="seq",
                type="nodetool.list.SequenceIterator",
                data={"start": 0, "stop": 16, "step": 1},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
            Node(
                id="neg1",
                type="lib.math.MathFunction",
                data={"operation": "negate"},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
            Node(
                id="neg2",
                type="lib.math.MathFunction",
                data={"operation": "negate"},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
            Node(
                id="add",
                type="lib.math.Add",
                data={},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="zip_all",
            ),
            Node(
                id="prev",
                type="nodetool.workflows.base_node.Preview",
                data={},
                ui_properties={"position": {"x": 0, "y": 0}},
                dynamic_properties={},
                dynamic_outputs={},
                sync_mode="on_any",
            ),
        ],
        edges=[
            Edge(
                id="e1",
                source="seq",
                sourceHandle="output",
                target="add",
                targetHandle="b",
                ui_properties={"className": "int"},
            ),
            Edge(
                id="e2",
                source="seq",
                sourceHandle="output",
                target="neg1",
                targetHandle="input",
                ui_properties={"className": "int"},
            ),
            Edge(
                id="e3",
                source="neg1",
                sourceHandle="output",
                target="neg2",
                targetHandle="input",
                ui_properties={"className": "int"},
            ),
            Edge(
                id="e4",
                source="neg2",
                sourceHandle="output",
                target="add",
                targetHandle="a",
                ui_properties={"className": "union"},
            ),
            Edge(
                id="e5",
                source="add",
                sourceHandle="output",
                target="prev",
                targetHandle="value",
                ui_properties={"className": "union"},
            ),
        ],
    )

    values = await _run_and_collect_values(graph, preview_id="prev")

    assert len(values) == 16
    assert values == list(range(0, 31, 2))
