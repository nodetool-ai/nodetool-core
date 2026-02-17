import json
from pathlib import Path

import pytest

from nodetool.types.api_graph import Graph as ApiGraph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate


@pytest.mark.asyncio
async def test_control_workflow_fixture():
    """Test the control edges workflow fixture runs correctly."""
    example_path = Path(__file__).parent / "Test Control Workflow.json"
    assert example_path.exists(), f"Missing test workflow fixture: {example_path}"

    with example_path.open("r", encoding="utf-8") as f:
        wf = json.load(f)

    graph = ApiGraph(**wf["graph"])  # type: ignore[arg-type]
    req = RunJobRequest(
        user_id="test",
        auth_token="",
        workflow_id="test_control_workflow",
        graph=graph,
        params={"value": 0.75},
    )
    ctx = ProcessingContext(user_id=req.user_id, job_id="job", auth_token=req.auth_token)

    outputs: dict[str, object] = {}
    async for msg in run_workflow(req, context=ctx, use_thread=False):
        if isinstance(msg, OutputUpdate):
            outputs[msg.output_name] = msg.value

    # The processor should receive control event from controller
    # Controller sets threshold=0.8, mode=strict
    # With value=0.75, threshold=0.8, mode=strict: 0.75 > 0.8 = False
    result = outputs.get("result")
    assert result is not None
    # Result should show threshold was overridden to 0.8 (from default 0.5)
    assert "threshold=0.8" in str(result)
    assert "mode=strict" in str(result)
    assert "value=0.75" in str(result)


@pytest.mark.asyncio
async def test_control_workflow_with_different_params():
    """Test control workflow with different input values."""
    example_path = Path(__file__).parent / "Test Control Workflow.json"

    with example_path.open("r", encoding="utf-8") as f:
        wf = json.load(f)

    graph = ApiGraph(**wf["graph"])  # type: ignore[arg-type]

    # Test with value that exceeds threshold
    req = RunJobRequest(
        user_id="test",
        auth_token="",
        workflow_id="test_control_workflow",
        graph=graph,
        params={"value": 0.9},  # 0.9 > 0.8 (strict threshold)
    )
    ctx = ProcessingContext(user_id=req.user_id, job_id="job", auth_token=req.auth_token)

    outputs: dict[str, object] = {}
    async for msg in run_workflow(req, context=ctx, use_thread=False):
        if isinstance(msg, OutputUpdate):
            outputs[msg.output_name] = msg.value

    result = outputs.get("result")
    assert result is not None
    # With value=0.9, threshold=0.8, mode=strict: 0.9 > 0.8 = True
    assert "exceeds=True" in str(result)


@pytest.mark.asyncio
async def test_control_edge_e2e_with_graph():
    """E2E test for control edge system with programmatically created graph."""
    from nodetool.types.api_graph import Edge, Graph as ApiGraph, Node
    from nodetool.workflows.run_job_request import RunJobRequest
    from nodetool.workflows.run_workflow import run_workflow

    # Build graph using API types
    nodes = [
        Node(
            id="input_value",
            type="nodetool.workflows.test_helper.FloatInput",
            data={"name": "value", "value": 0.6, "default": 0.6, "description": "", "label": "Value"},
        ),
        Node(
            id="controller",
            type="nodetool.workflows.test_helper.SimpleController",
            data={"control_threshold": 0.7, "control_mode": "strict", "trigger_on_init": True},
        ),
        Node(
            id="processor",
            type="nodetool.workflows.test_helper.ThresholdProcessor",
            data={"value": 0.0, "threshold": 0.5, "mode": "normal"},
        ),
        Node(
            id="output",
            type="nodetool.workflows.test_helper.StringOutput",
            data={"name": "result", "value": "", "description": ""},
        ),
    ]

    edges = [
        Edge(id="data1", source="input_value", sourceHandle="output", target="processor", targetHandle="value"),
        Edge(
            id="control1",
            source="controller",
            sourceHandle="__control__",
            target="processor",
            targetHandle="__control__",
            edge_type="control",
        ),
        Edge(id="output1", source="processor", sourceHandle="output", target="output", targetHandle="value"),
    ]

    graph = ApiGraph(nodes=nodes, edges=edges)

    # Run workflow
    req = RunJobRequest(
        user_id="test",
        auth_token="",
        workflow_id="test_e2e_control",
        graph=graph,
        params={},
    )
    ctx = ProcessingContext(user_id=req.user_id, job_id="job", auth_token=req.auth_token)

    outputs: dict[str, object] = {}
    async for msg in run_workflow(req, context=ctx, use_thread=False):
        if isinstance(msg, OutputUpdate):
            outputs[msg.output_name] = msg.value

    result = outputs.get("result")
    assert result is not None
    # Controller should have overridden threshold to 0.7 and mode to strict
    # value=0.6, threshold=0.7, mode=strict: 0.6 > 0.7 = False
    assert "threshold=0.7" in str(result)
    assert "mode=strict" in str(result)


@pytest.mark.asyncio
async def test_control_edge_multiple_triggers():
    """Test controller emitting multiple control events."""
    from nodetool.types.api_graph import Edge, Graph as ApiGraph, Node
    from nodetool.workflows.run_job_request import RunJobRequest
    from nodetool.workflows.run_workflow import run_workflow

    # Build graph using API types - use SimpleController with comma-separated values
    # Set include_properties=False since IntAccumulator doesn't have threshold/mode properties
    nodes = [
        Node(
            id="controller",
            type="nodetool.workflows.test_helper.SimpleController",
            data={"control_threshold": 0.5, "control_mode": "strict", "trigger_on_init": True, "include_properties": False},
        ),
        Node(id="accumulator", type="nodetool.workflows.test_helper.IntAccumulator", data={"value": 0}),
    ]

    edges = [
        Edge(
            id="control1",
            source="controller",
            sourceHandle="__control__",
            target="accumulator",
            targetHandle="__control__",
            edge_type="control",
        ),
    ]

    graph = ApiGraph(nodes=nodes, edges=edges)

    req = RunJobRequest(
        user_id="test",
        auth_token="",
        workflow_id="test_multi_trigger",
        graph=graph,
        params={},
    )
    ctx = ProcessingContext(user_id=req.user_id, job_id="job", auth_token=req.auth_token)

    async for msg in run_workflow(req, context=ctx, use_thread=False):
        pass  # Just consume messages

    # The test passes if the workflow completes without hanging
    # (The SimpleController currently emits one RunEvent, so accumulator runs once)
