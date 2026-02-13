import json
from pathlib import Path

import pytest

from nodetool.types.api_graph import Graph as ApiGraph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate


@pytest.mark.asyncio
async def test_test_workflow_fixture():
    example_path = Path(__file__).parent / "Test Workflow.json"
    assert example_path.exists(), f"Missing test workflow fixture: {example_path}"

    with example_path.open("r", encoding="utf-8") as f:
        wf = json.load(f)

    graph = ApiGraph(**wf["graph"])  # type: ignore[arg-type]
    req = RunJobRequest(
        user_id="test",
        auth_token="",
        workflow_id="test_workflow_fixture",
        graph=graph,
        params={"text": "hello"},
    )
    ctx = ProcessingContext(user_id=req.user_id, job_id="job", auth_token=req.auth_token)

    outputs: dict[str, object] = {}
    async for msg in run_workflow(req, context=ctx, use_thread=False):
        if isinstance(msg, OutputUpdate):
            outputs[msg.output_name] = msg.value

    assert outputs.get("result") == "Test output: hello"
