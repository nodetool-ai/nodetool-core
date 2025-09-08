import json
import os
import pytest
from pathlib import Path

from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Graph as ApiGraph
from nodetool.workflows.types import OutputUpdate

if not os.environ.get("NODETOOL_INTEGRATION_TESTS"):
    pytest.skip("Skipping integration tests")


"""
Summarize RSS

This workflow fetches an RSS feed and summarizes the content.

It uses the following nodes:
- FetchRSSFeed
- Summarizer
- Collect

Requirements:
- Ollama running with gemma3:1b
- Internet access
"""


@pytest.mark.asyncio
async def test_summarize_rss():
    # Load the golden-path workflow JSON copied into core examples
    example_path = Path(__file__).parent / "Summarize RSS.json"
    assert example_path.exists(), f"Missing example: {example_path}"

    with example_path.open("r", encoding="utf-8") as f:
        wf = json.load(f)

    # Build API Graph and request
    graph = ApiGraph(**wf["graph"])  # type: ignore[arg-type]

    req = RunJobRequest(
        user_id="test",
        auth_token="",
        workflow_id="summarize_rss",
        graph=graph,
    )

    ctx = ProcessingContext(
        user_id=req.user_id, job_id="job", auth_token=req.auth_token
    )

    found_summary = False
    async for msg in run_workflow(req, context=ctx, use_thread=False):
        if isinstance(msg, OutputUpdate):
            if msg.output_name == "summary":
                found_summary = True
                break

    assert found_summary, "Expected summarizer node to produce a summary text"
