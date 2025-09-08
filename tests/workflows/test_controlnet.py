import json
import os
import pytest
from pathlib import Path

from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Graph as ApiGraph
from nodetool.workflows.types import PreviewUpdate

"""
Controlnet

This workflow applies ControlNet-guided image generation using a canny edge
condition and previews the generated output.

Nodes used include:
- Image (constant)
- Fit (resize)
- Canny (edge detection)
- ImageToText (captioning)
- StableDiffusionControlNet (generation)
- Preview (sink)

Requirements:
- Internet access and model downloads (large); skipped unless integration tests are enabled
"""


@pytest.mark.skipif(
    os.environ.get("NODETOOL_INTEGRATION_TESTS", "") == "",
    reason="Skipping integration tests",
)
@pytest.mark.asyncio
async def test_controlnet():
    # Load the workflow JSON copied into core tests
    example_path = Path(__file__).parent / "Controlnet.json"
    assert example_path.exists(), f"Missing example: {example_path}"

    with example_path.open("r", encoding="utf-8") as f:
        wf = json.load(f)

    # Build API Graph and request
    graph = ApiGraph(**wf["graph"])  # type: ignore[arg-type]

    req = RunJobRequest(
        user_id="test",
        auth_token="",
        workflow_id="controlnet",
        graph=graph,
    )

    ctx = ProcessingContext(
        user_id=req.user_id, job_id="job", auth_token=req.auth_token
    )

    # Find the Preview node id to assert the correct source emits a preview
    preview_node_id = None
    for n in wf["graph"]["nodes"]:
        if n.get("type") == "nodetool.workflows.base_node.Preview":
            preview_node_id = n.get("id")
            break
    assert preview_node_id, "Expected a Preview node in the Controlnet workflow"

    found_preview = False
    async for msg in run_workflow(req, context=ctx, use_thread=False):
        if isinstance(msg, PreviewUpdate) and msg.node_id == preview_node_id:
            found_preview = True
            break

    assert found_preview, "Expected Preview node to produce a preview update"
