# import dotenv
import json
import os
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from nodetool.types.job import JobUpdate
import runpod
from nodetool.types.graph import Graph
from nodetool.workflows.run_workflow import run_workflow
from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import OutputUpdate
from nodetool.deploy.download_models import download_models_from_spec

"""
For local testing, run:

export NODETOOL_WORKFLOW_PATH=concat.json
python src/nodetool/deploy/runpod_handler.py --rp_serve_api
"""


# dotenv.load_dotenv()


log = Environment.get_logger()


def download_models_on_startup() -> None:
    """
    Download missing models on container startup.
    
    Reads model specifications from /app/models.json and downloads
    any models that are not available in the network volume.
    """
    models_file = "/app/models.json"
    
    if not os.path.exists(models_file):
        log.info("No models.json file found, skipping model downloads")
        return
    
    try:
        with open(models_file, 'r') as f:
            models = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log.error(f"Failed to read models file: {e}")
        return
    
    # Use consolidated download function
    hf_cache_dir = "/runpod-volume/.cache/huggingface/hub"
    download_models_from_spec(models, hf_cache_dir, log)


async def workflow_handler(job):
    """
    Workflow handler for RunPod serverless workflow execution.
    
    This function processes NodeTool workflow jobs on RunPod infrastructure.
    Downloads missing models on first run from network volume.

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
            value = context.encode_assets_as_uri(msg.value)
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            results[msg.node_name] = value

    return results


# Global flag to track if models have been downloaded
_models_downloaded = False


if __name__ == "__main__":
    # Download models on startup (only once)
    log.info("Starting RunPod workflow handler...")
    download_models_on_startup()
    
    runpod.serverless.start({"handler": workflow_handler})