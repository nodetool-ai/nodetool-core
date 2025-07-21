"""Hugging Face Inference Endpoint handler for NodeTool workflows."""

import asyncio
import json
from typing import Any, Dict

from nodetool.types.graph import Graph
from nodetool.types.job import JobUpdate
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import OutputUpdate

with open("/app/workflow.json", "r") as f:
    _workflow_data = json.load(f)

_graph = Graph.model_validate(_workflow_data["graph"])


async def _run_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    req = RunJobRequest(params=params)
    req.graph = _graph

    context = ProcessingContext(upload_assets_to_s3=True)
    results: Dict[str, Any] = {}
    async for msg in run_workflow(req, context=context, use_thread=True):
        if isinstance(msg, JobUpdate) and msg.status == "error":
            raise Exception(msg.error)
        if isinstance(msg, OutputUpdate):
            value = context.encode_assets_as_uri(msg.value)
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            results[msg.node_name] = value
    return results


def predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous entry point used by Hugging Face Inference Endpoints."""
    return asyncio.run(_run_workflow(inputs or {}))
