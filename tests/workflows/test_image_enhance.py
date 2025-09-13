import json
import os
import pytest
from pathlib import Path

from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Graph as ApiGraph
from nodetool.workflows.types import OutputUpdate

"""
Image Enhance

This workflow enhances image quality with basic enhancement tools like sharpening, contrast and color adjustment.

It uses the following nodes:
- ImageInput
- Sharpen
- AutoContrast
- ImageOutput

Requirements:
- Internet access (for image download)
"""


@pytest.mark.skipif(
    os.environ.get("NODETOOL_INTEGRATION_TESTS", "") == "",
    reason="Skipping integration tests",
)
@pytest.mark.asyncio
async def test_image_enhance():
    # Load the golden-path workflow JSON copied into core examples
    example_path = Path(__file__).parent / "Image Enhance.json"
    assert example_path.exists(), f"Missing example: {example_path}"

    with example_path.open("r", encoding="utf-8") as f:
        wf = json.load(f)

    # Build API Graph and request
    graph = ApiGraph(**wf["graph"])  # type: ignore[arg-type]

    req = RunJobRequest(
        user_id="test",
        auth_token="",
        workflow_id="image_enhance",
        graph=graph,
    )

    ctx = ProcessingContext(
        user_id=req.user_id, job_id="job", auth_token=req.auth_token
    )

    found_enhanced_image = False
    async for msg in run_workflow(req, context=ctx, use_thread=False):
        if isinstance(msg, OutputUpdate):
            if msg.output_name == "enhanced":
                found_enhanced_image = True
                break

    assert (
        found_enhanced_image
    ), "Expected ImageOutput node to produce an enhanced image"
