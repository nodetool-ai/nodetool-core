"""
Workflow routes and registry for the lightweight NodeTool FastAPI server.

This module encapsulates:
- Loading workflows from disk
- A simple in-memory workflow registry
- Public endpoints to list and execute workflows (with optional SSE streaming)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from nodetool.common.environment import Environment
from nodetool.types.job import JobUpdate
from nodetool.types.workflow import Workflow
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate


log = Environment.get_logger()

# Internal registry
_workflow_registry: Dict[str, Workflow] = {}


def load_workflow(path: str) -> Workflow:
    with open(path, "r") as f:
        workflow = json.load(f)
    return Workflow.model_validate(workflow)


def load_workflows_from_directory(workflows_dir: str = "/app/workflows") -> Dict[str, Workflow]:
    """Load all workflow JSON files from the specified directory.

    Returns a mapping from workflow_id to Workflow.
    """
    workflows: Dict[str, Workflow] = {}

    if not os.path.exists(workflows_dir):
        log.warning(f"Workflows directory not found: {workflows_dir}")
        return workflows

    for filename in os.listdir(workflows_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(workflows_dir, filename)
        try:
            workflow = load_workflow(filepath)
            workflow_id = workflow.id if getattr(workflow, "id", None) else filename[:-5]
            workflows[workflow_id] = workflow
            log.info(f"Loaded workflow '{workflow_id}' from {filename}")
        except Exception as e:  # noqa: BLE001
            log.error(f"Failed to load workflow from {filename}: {str(e)}")

    return workflows


def initialize_workflow_registry() -> None:
    global _workflow_registry
    _workflow_registry = load_workflows_from_directory()
    log.info(f"Initialized workflow registry with {len(_workflow_registry)} workflows")


def get_workflow_by_id(workflow_id: str) -> Workflow:
    if workflow_id not in _workflow_registry:
        raise ValueError(
            f"Workflow '{workflow_id}' not found. Available workflows: {list(_workflow_registry.keys())}"
        )
    return _workflow_registry[workflow_id]


def get_workflow_registry() -> Dict[str, Workflow]:
    return _workflow_registry


def get_aggregated_workflows(additional_workflows: List[Workflow] | None = None) -> List[Workflow]:
    """Aggregate workflows from registry, optional /workflows dir, and provided list."""
    aggregated: List[Workflow] = list(_workflow_registry.values())
    if os.path.exists("/workflows"):
        try:
            extra = load_workflows_from_directory("/workflows").values()
            aggregated.extend(list(extra))
        except Exception as e:  # noqa: BLE001
            log.warning(f"Failed loading workflows from /workflows: {e}")
    if additional_workflows:
        aggregated.extend(additional_workflows)
    return aggregated


def create_workflow_router() -> APIRouter:
    router = APIRouter()

    @router.get("/workflows")
    async def list_workflows():
        return {
            "workflows": [
                {
                    "id": workflow_id,
                    "name": workflow.name if hasattr(workflow, "name") else workflow_id,
                }
                for workflow_id, workflow in _workflow_registry.items()
            ]
        }

    @router.post("/workflows/execute")
    async def execute_workflow(request: Request):
        try:
            data = await request.json()
            workflow_id = data.get("workflow_id")
            params = data.get("params", {})

            if not workflow_id:
                raise HTTPException(status_code=400, detail="workflow_id is required")

            workflow = get_workflow_by_id(workflow_id)
            req = RunJobRequest(params=params)
            req.graph = workflow.graph

            context = ProcessingContext(upload_assets_to_s3=True)

            results: Dict[str, object] = {}
            async for msg in run_workflow(req, context=context, use_thread=True):
                if isinstance(msg, JobUpdate) and msg.status == "error":
                    raise HTTPException(status_code=500, detail=msg.error)
                if isinstance(msg, OutputUpdate):
                    value = context.encode_assets_as_uri(msg.value)
                    if hasattr(value, "model_dump"):
                        value = value.model_dump()
                    results[msg.node_name] = value

            return {"results": results}

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:  # noqa: BLE001
            print(f"Workflow execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/workflows/execute/stream")
    async def execute_workflow_stream(request: Request):
        try:
            data = await request.json()
            workflow_id = data.get("workflow_id")
            params = data.get("params", {})

            if not workflow_id:
                raise HTTPException(status_code=400, detail="workflow_id is required")

            workflow = get_workflow_by_id(workflow_id)
            req = RunJobRequest(params=params)
            req.graph = workflow.graph

            context = ProcessingContext(upload_assets_to_s3=True)

            async def generate_sse():
                results: Dict[str, object] = {}
                try:
                    async for msg in run_workflow(req, context=context, use_thread=True):
                        if isinstance(msg, JobUpdate):
                            event_data = {"type": "job_update", "data": msg.model_dump()}
                            yield f"data: {json.dumps(event_data)}\n\n"
                            if msg.status == "error":
                                error_data = {"type": "error", "error": msg.error}
                                yield f"data: {json.dumps(error_data)}\n\n"
                                return
                        elif isinstance(msg, OutputUpdate):
                            value = context.encode_assets_as_uri(msg.value)
                            if hasattr(value, "model_dump"):
                                value = value.model_dump()
                            results[msg.node_name] = value
                            event_data = {
                                "type": "output_update",
                                "node_name": msg.node_name,
                                "value": value,
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"

                    final_data = {"type": "complete", "results": results}
                    yield f"data: {json.dumps(final_data)}\n\n"
                    yield "data: [DONE]\n\n"

                except Exception as e:  # noqa: BLE001
                    error_data = {"type": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                },
            )

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:  # noqa: BLE001
            print(f"Workflow streaming error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router


