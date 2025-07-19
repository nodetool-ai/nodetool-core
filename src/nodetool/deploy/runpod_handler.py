# import dotenv
import json
import os
from nodetool.types.job import JobUpdate
import runpod
from nodetool.types.graph import Graph
from nodetool.workflows.run_workflow import run_workflow
from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import OutputUpdate

"""
For local testing, run:

export NODETOOL_WORKFLOW_PATH=concat.json
python src/nodetool/deploy/runpod_handler.py --rp_serve_api
"""


# dotenv.load_dotenv()


log = Environment.get_logger()


async def workflow_handler(job):
    """
    Workflow handler for RunPod serverless workflow execution.
    
    This function processes NodeTool workflow jobs on RunPod infrastructure.

    The workflow file must be defined in the NODETOOL_WORKFLOW_PATH environment variable.
    
    Args:
        job (dict): RunPod job dictionary containing input data and job ID

    Returns:
        dict: Results of the workflow execution
    """
    # Extract workflow data from the job input
    job_id = job.get("id")
    req = RunJobRequest(params=job.get("input", {}))
    workflow_path = "/app/workflow.json"

    if not os.path.exists(workflow_path):
        raise Exception(f"Workflow file not found at {workflow_path}")
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    req.graph = Graph.model_validate(workflow["graph"])

    print(req)

    # Create processing context for workflow execution
    # This context manages the workflow runtime environment
    context = ProcessingContext(
        upload_assets_to_s3=True,  # Store generated assets in S3 for persistence
    )

    # Collect all messages from workflow execution
    results = {}
    async for msg in run_workflow(req, context=context, use_thread=True):
        print(msg)
        if isinstance(msg, JobUpdate) and msg.status == "error":
            raise Exception(msg.error)
        if isinstance(msg, OutputUpdate):
            value = context.upload_assets_to_temp(msg.value)
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            results[msg.node_name] = value

    return results


if __name__ == "__main__": 
    runpod.serverless.start({"handler": workflow_handler})