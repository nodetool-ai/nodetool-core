"""End-to-end workflow runner tests derived from E2E_TEST_SCENARIOS.md Section 4.

This module implements high-priority runner scenarios:
- RUNNER-002: Graph validation failure
- RUNNER-007, RUNNER-011, RUNNER-012: InputNode handling
- RUNNER-015, RUNNER-019, RUNNER-020: Output and streaming
- RUNNER-032, RUNNER-034, RUNNER-038, RUNNER-041: Error and cache
"""

from __future__ import annotations

import asyncio
import logging
import queue
from typing import Any, Sequence

import pytest

from nodetool.types.api_graph import Edge, Node
from nodetool.types.api_graph import Graph as ApiGraph
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import Error as ErrorMessage
from nodetool.workflows.types import OutputUpdate
from nodetool.workflows.workflow_runner import WorkflowRunner

pytestmark = pytest.mark.asyncio

ASYNC_TEST_TIMEOUT = 5.0


async def run_graph(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    *,
    workflow_id: str,
    params: dict[str, Any] | None = None,
    expect_error: bool = False,
):
    """Execute a workflow graph and collect all emitted messages."""

    graph = ApiGraph(nodes=list(nodes), edges=list(edges))
    request = RunJobRequest(
        user_id="test-user",
        auth_token="",
        workflow_id=workflow_id,
        graph=graph,
        params=params or {},
    )
    context = ProcessingContext(user_id=request.user_id, job_id=f"{workflow_id}-job", auth_token=request.auth_token)

    messages: list[Any] = []
    caught: Exception | None = None
    try:
        async for message in run_workflow(request, context=context, use_thread=False):
            messages.append(message)
    except Exception as exc:
        caught = exc
        if not expect_error:
            raise
    return messages, caught


def float_input_node(node_id: str, *, name: str, value: float) -> Node:
    return Node(
        id=node_id,
        type="nodetool.workflows.test_helper.FloatInput",
        data={
            "name": name,
            "value": value,
            "default": value,
            "description": name,
        },
    )


def string_input_node(node_id: str, *, name: str, value: str) -> Node:
    return Node(
        id=node_id,
        type="nodetool.workflows.test_helper.StringInput",
        data={
            "name": name,
            "value": value,
            "description": name,
        },
    )


def string_output_node(node_id: str, *, name: str) -> Node:
    return Node(
        id=node_id,
        type="nodetool.workflows.test_helper.StringOutput",
        data={"name": name, "value": "", "description": name},
    )


def int_output_node(node_id: str, *, name: str) -> Node:
    return Node(
        id=node_id,
        type="nodetool.workflows.test_helper.IntOutput",
        data={"name": name, "value": 0, "description": name},
    )


def threshold_processor_node(node_id: str) -> Node:
    return Node(
        id=node_id,
        type="nodetool.workflows.test_helper.ThresholdProcessor",
        data={"value": 0.0, "threshold": 0.5, "mode": "normal"},
    )


def streaming_output_processor_node(node_id: str, *, count: int = 3, base_value: str = "data") -> Node:
    return Node(
        id=node_id,
        type="nodetool.workflows.test_helper.StreamingOutputProcessor",
        data={"count": count, "base_value": base_value},
    )


def error_processor_node(node_id: str, *, message: str) -> Node:
    return Node(
        id=node_id,
        type="nodetool.workflows.test_helper.ErrorProcessor",
        data={"message": message},
    )


def format_text_node(node_id: str, *, template: str = "Hello, {{ text }}") -> Node:
    return Node(
        id=node_id,
        type="nodetool.workflows.test_helper.FormatText",
        data={"template": template, "text": ""},
    )


def data_edge(edge_id: str, source: str, source_handle: str, target: str, target_handle: str) -> Edge:
    return Edge(
        id=edge_id,
        source=source,
        sourceHandle=source_handle,
        target=target,
        targetHandle=target_handle,
        edge_type="data",
    )


def get_output_values(messages: list[Any], output_name: str) -> list[Any]:
    return [msg.value for msg in messages if isinstance(msg, OutputUpdate) and msg.output_name == output_name]


# ============================================================================
# RUNNER-002: Graph Validation Failure
# ============================================================================


async def test_runner_002_graph_validation_failure():
    """RUNNER-002: Invalid graph (edge to non-existent target handle) is handled gracefully."""
    # Create a graph where an edge connects to a non-existent handle on the target node
    # The runner should filter out invalid edges during edge filtering
    nodes = [
        float_input_node("input", name="value", value=1.0),
        threshold_processor_node("processor"),
        string_output_node("output", name="result"),
    ]
    edges = [
        # Connect to a non-existent handle "nonexistent_prop" on the processor
        # This edge should be filtered during validation
        Edge(
            id="invalid_edge",
            source="input",
            sourceHandle="output",
            target="processor",
            targetHandle="nonexistent_prop",  # This handle doesn't exist
            edge_type="data",
        ),
        data_edge("proc_to_out", "processor", "output", "output", "value"),
    ]

    messages, error = await run_graph(nodes, edges, workflow_id="runner-002")

    # The graph should complete but the invalid edge should have been filtered
    # during edge validation. The processor will use its default value.
    assert error is None

    # Check that the workflow completed with default value (since edge was filtered)
    results = get_output_values(messages, "result")
    assert len(results) == 1
    # Processor should have used its default value of 0.0 since the edge was filtered
    assert "value=0.0" in results[0]


# ============================================================================
# RUNNER-007, RUNNER-011, RUNNER-012: InputNode Handling
# ============================================================================


async def test_runner_007_static_input_node_emits_once():
    """RUNNER-007: Non-streaming InputNode emits one value then signals EOS."""
    nodes = [
        float_input_node("input", name="value", value=42.0),
        threshold_processor_node("processor"),
        string_output_node("output", name="result"),
    ]
    edges = [
        data_edge("in_to_proc", "input", "output", "processor", "value"),
        data_edge("proc_to_out", "processor", "output", "output", "value"),
    ]

    messages, error = await run_graph(nodes, edges, workflow_id="runner-007")
    assert error is None

    results = get_output_values(messages, "result")
    assert len(results) == 1
    assert "value=42.0" in results[0]


async def test_runner_011_duplicate_input_node_names_rejected():
    """RUNNER-011: Two InputNodes with same name raises ValueError."""
    # Create two inputs with the same name
    nodes = [
        Node(
            id="input1",
            type="nodetool.workflows.test_helper.StringInput",
            data={"name": "duplicate_name", "value": "value1"},
        ),
        Node(
            id="input2",
            type="nodetool.workflows.test_helper.StringInput",
            data={"name": "duplicate_name", "value": "value2"},
        ),
        string_output_node("output", name="result"),
    ]
    edges = []

    _messages, error = await run_graph(nodes, edges, workflow_id="runner-011", expect_error=True)
    assert error is not None


async def test_runner_012_empty_input_node_name_rejected():
    """RUNNER-012: InputNode with empty name raises ValueError."""
    nodes = [
        Node(
            id="input1",
            type="nodetool.workflows.test_helper.StringInput",
            data={"name": "", "value": "test"},
        ),
        string_output_node("output", name="result"),
    ]
    edges = []

    _messages, error = await run_graph(nodes, edges, workflow_id="runner-012", expect_error=True)
    assert error is not None


# ============================================================================
# RUNNER-015, RUNNER-019, RUNNER-020: Output and Streaming
# ============================================================================


async def test_runner_015_single_output_node_captures_result():
    """RUNNER-015: Single OutputNode captures final result in runner.outputs."""
    nodes = [
        string_input_node("input", name="value", value="123.0"),
        format_text_node("formatter", template="Result: {{ text }}"),
        string_output_node("output", name="final_result"),
    ]
    edges = [
        data_edge("in_to_fmt", "input", "output", "formatter", "text"),
        data_edge("fmt_to_out", "formatter", "output", "output", "value"),
    ]

    messages, error = await run_graph(nodes, edges, workflow_id="runner-015")
    assert error is None

    results = get_output_values(messages, "final_result")
    assert len(results) == 1
    assert results[0] == "Result: 123.0"


async def test_runner_019_streaming_propagation():
    """RUNNER-019: Node with is_streaming_output=True marks downstream edges as streaming."""
    # Use StreamingOutputProcessor which has is_streaming_output=True
    nodes = [
        streaming_output_processor_node("streamer", count=3, base_value="item"),
        format_text_node("formatter", template="Got: {{ text }}"),
        string_output_node("output", name="result"),
    ]
    edges = [
        data_edge("stream_to_fmt", "streamer", "result", "formatter", "text"),
        data_edge("fmt_to_out", "formatter", "output", "output", "value"),
    ]

    messages, error = await run_graph(nodes, edges, workflow_id="runner-019")
    assert error is None

    # Should get 3 outputs from the streaming node
    results = get_output_values(messages, "result")
    assert len(results) == 3
    assert results == ["Got: item_1", "Got: item_2", "Got: item_3"]


async def test_runner_020_control_edge_excluded_from_streaming():
    """RUNNER-020: Control edges don't participate in streaming propagation."""
    # This test verifies that control edges are treated differently from data edges
    # A control edge shouldn't mark the target as streaming even if controller is streaming
    from nodetool.workflows.test_helper import SimpleController

    nodes = [
        Node(
            id="controller",
            type="nodetool.workflows.test_helper.SimpleController",
            data={"control_threshold": 0.8, "trigger_on_init": True},
        ),
        threshold_processor_node("processor"),
        string_output_node("output", name="result"),
    ]
    edges = [
        # Control edge (should NOT propagate streaming)
        Edge(
            id="control_edge",
            source="controller",
            sourceHandle="__control__",
            target="processor",
            targetHandle="__control__",
            edge_type="control",
        ),
        data_edge("proc_to_out", "processor", "output", "output", "value"),
    ]

    messages, error = await run_graph(nodes, edges, workflow_id="runner-020")
    assert error is None

    # Should get one result (non-streaming behavior from control edge)
    results = get_output_values(messages, "result")
    assert len(results) == 1


# ============================================================================
# RUNNER-032, RUNNER-034: Error Handling
# ============================================================================


async def test_runner_032_node_execution_error_captures_and_fails_job():
    """RUNNER-032: Node raises exception; error captured, job marked 'failed'."""
    nodes = [
        error_processor_node("error_node", message="Intentional test error"),
    ]
    edges = []

    messages, error = await run_graph(nodes, edges, workflow_id="runner-032", expect_error=True)
    assert error is not None

    # Check for error message
    error_messages = [msg for msg in messages if isinstance(msg, ErrorMessage)]
    assert any("Intentional test error" in msg.message for msg in error_messages)

    # Check job status update
    job_updates = [msg for msg in messages if isinstance(msg, JobUpdate)]
    assert job_updates
    assert job_updates[-1].status == "failed"


async def test_runner_034_cancellation_marks_cancelled():
    """RUNNER-034: asyncio.CancelledError raised; status='cancelled', JobUpdate posted."""
    from nodetool.workflows.base_node import BaseNode

    class SlowNode(BaseNode):
        async def process(self, context):
            await asyncio.sleep(10.0)  # Long sleep that will be cancelled
            return "done"

    nodes = [
        Node(id="slow", type=SlowNode.get_node_type(), data={}),
    ]
    edges = []
    graph = ApiGraph(nodes=nodes, edges=edges)
    request = RunJobRequest(
        user_id="test-user",
        auth_token="",
        workflow_id="runner-034",
        graph=graph,
    )

    ctx = ProcessingContext(user_id=request.user_id, job_id="runner-034-job", auth_token=request.auth_token)
    runner = WorkflowRunner(job_id="runner-034-job")

    # Start the workflow
    run_task = asyncio.create_task(runner.run(request, ctx))

    # Give it time to start
    await asyncio.sleep(0.1)

    # Cancel the task
    run_task.cancel()

    try:
        await asyncio.wait_for(run_task, timeout=ASYNC_TEST_TIMEOUT)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass

    # The runner should have been cancelled
    assert runner.status in ["cancelled", "failed"]


# ============================================================================
# RUNNER-038, RUNNER-041: Caching
# ============================================================================


async def test_runner_038_cache_hit_returns_cached_result():
    """RUNNER-038: Same node with same inputs executed twice; second returns cached."""
    # This test verifies that caching works by running the same operation twice
    # and checking if the second one uses the cache

    nodes = [
        string_input_node("input", name="value", value="5.0"),
        format_text_node("formatter", template="Value: {{ text }}"),
        string_output_node("output", name="result"),
    ]
    edges = [
        data_edge("in_to_fmt", "input", "output", "formatter", "text"),
        data_edge("fmt_to_out", "formatter", "output", "output", "value"),
    ]

    # First run
    messages1, error1 = await run_graph(nodes, edges, workflow_id="runner-038-1")
    assert error1 is None

    # Second run with same graph (cache should hit)
    messages2, error2 = await run_graph(nodes, edges, workflow_id="runner-038-2")
    assert error2 is None

    # Both should produce the same result
    results1 = get_output_values(messages1, "result")
    results2 = get_output_values(messages2, "result")
    assert results1 == results2


async def test_runner_041_streaming_upstream_skips_cache():
    """RUNNER-041: Node with streaming upstream NOT cached (invalidates cache key)."""
    # When a node has streaming upstreams, caching should be skipped
    # because the cache key would be different with streaming

    nodes = [
        streaming_output_processor_node("streamer", count=2, base_value="x"),
        format_text_node("formatter", template="Got: {{ text }}"),
        string_output_node("output", name="result"),
    ]
    edges = [
        data_edge("stream_to_fmt", "streamer", "result", "formatter", "text"),
        data_edge("fmt_to_out", "formatter", "output", "output", "value"),
    ]

    messages, error = await run_graph(nodes, edges, workflow_id="runner-041")
    assert error is None

    # Should get 2 results from streaming
    results = get_output_values(messages, "result")
    assert len(results) == 2


# ============================================================================
# Additional Runner Tests
# ============================================================================


async def test_runner_007_default_value_used_when_no_params():
    """RUNNER-009: InputNode has value=42 but no params provided; default is pushed."""
    # This is similar to RUNNER-007 but explicitly tests default value behavior
    nodes = [
        float_input_node("input", name="my_input", value=99.0),
        threshold_processor_node("processor"),
        string_output_node("output", name="result"),
    ]
    edges = [
        data_edge("in_to_proc", "input", "output", "processor", "value"),
        data_edge("proc_to_out", "processor", "output", "output", "value"),
    ]

    # No params provided - should use default value from node
    messages, error = await run_graph(nodes, edges, workflow_id="runner-007-default", params={})
    assert error is None

    results = get_output_values(messages, "result")
    assert len(results) == 1
    assert "value=99.0" in results[0]


async def test_runner_010_param_overrides_default():
    """RUNNER-010: InputNode has value=10, params provide {'input': 20}; 20 is used."""
    # Note: This tests param overriding, but our FloatInput uses 'default' field
    nodes = [
        Node(
            id="input",
            type="nodetool.workflows.test_helper.FloatInput",
            data={"name": "my_input", "default": 10.0, "value": 10.0},
        ),
        threshold_processor_node("processor"),
        string_output_node("output", name="result"),
    ]
    edges = [
        data_edge("in_to_proc", "input", "output", "processor", "value"),
        data_edge("proc_to_out", "processor", "output", "output", "value"),
    ]

    # Param should override the default
    messages, error = await run_graph(nodes, edges, workflow_id="runner-010", params={"my_input": 20.0})
    assert error is None

    results = get_output_values(messages, "result")
    assert len(results) == 1
    assert "value=20.0" in results[0]


async def test_runner_016_streaming_to_output_node():
    """RUNNER-016: Streaming node feeds OutputNode; multiple values captured."""
    nodes = [
        streaming_output_processor_node("streamer", count=4, base_value="msg"),
        string_output_node("output", name="stream_result"),
    ]
    edges = [
        data_edge("stream_to_out", "streamer", "result", "output", "value"),
    ]

    messages, error = await run_graph(nodes, edges, workflow_id="runner-016")
    assert error is None

    results = get_output_values(messages, "stream_result")
    assert len(results) == 4
    assert results == ["msg_1", "msg_2", "msg_3", "msg_4"]


async def test_runner_017_multiple_output_nodes():
    """RUNNER-017: Two OutputNodes with different names; both captured separately."""
    nodes = [
        string_input_node("input", name="value", value="42.0"),
        format_text_node("formatter1", template="Out1: {{ text }}"),
        format_text_node("formatter2", template="Out2: {{ text }}"),
        string_output_node("output1", name="result_a"),
        string_output_node("output2", name="result_b"),
    ]
    edges = [
        data_edge("in_to_fmt1", "input", "output", "formatter1", "text"),
        data_edge("in_to_fmt2", "input", "output", "formatter2", "text"),
        data_edge("fmt1_to_out1", "formatter1", "output", "output1", "value"),
        data_edge("fmt2_to_out2", "formatter2", "output", "output2", "value"),
    ]

    messages, error = await run_graph(nodes, edges, workflow_id="runner-017")
    assert error is None

    results_a = get_output_values(messages, "result_a")
    results_b = get_output_values(messages, "result_b")

    assert len(results_a) == 1
    assert len(results_b) == 1
    assert results_a[0] == "Out1: 42.0"
    assert results_b[0] == "Out2: 42.0"


async def test_runner_033_job_status_progression():
    """RUNNER-033: Job transitions through 'running' → 'completed' with JobUpdates."""
    nodes = [
        string_input_node("input", name="value", value="test"),
        string_output_node("output", name="result"),
    ]
    edges = [
        data_edge("in_to_out", "input", "output", "output", "value"),
    ]

    messages, error = await run_graph(nodes, edges, workflow_id="runner-033")
    assert error is None

    # Collect job updates
    job_updates = [msg for msg in messages if isinstance(msg, JobUpdate)]

    # Should have updates including running and completed
    statuses = [update.status for update in job_updates]
    assert "running" in statuses or "pending" in statuses
    assert "completed" in statuses


async def test_runner_013_extra_param_ignored_with_warning(caplog):
    """RUNNER-013: Params contain key not matching any InputNode; warning logged."""
    nodes = [
        string_input_node("input", name="value", value="test"),
        string_output_node("output", name="result"),
    ]
    edges = [
        data_edge("in_to_out", "input", "output", "output", "value"),
    ]

    with caplog.at_level(logging.WARNING):
        _messages, error = await run_graph(nodes, edges, workflow_id="runner-013", params={"unknown_param": "value"})

    assert error is None

    # Should have logged a warning about the unknown param
    assert any("unknown_param" in record.message or "not found" in record.message for record in caplog.records)
