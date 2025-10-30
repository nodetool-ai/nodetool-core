"""
Workflow routes and registry for the lightweight NodeTool FastAPI server.

This module encapsulates:
- Loading workflows from disk
- A simple in-memory workflow registry
- Public endpoints to list and execute workflows (with optional SSE streaming)
"""

from __future__ import annotations

import json
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from nodetool.config.logging_config import get_logger
from nodetool.types.job import JobUpdate
from nodetool.types.workflow import Workflow
from nodetool.workflows.processing_context import AssetOutputMode, ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.api.workflow import WorkflowList, from_model, WorkflowRequest
from nodetool.api.utils import current_user


log = get_logger(__name__)

# Simple in-memory registry to support tests that patch it
_workflow_registry: dict[str, Workflow] = {}


def get_workflow_by_id(workflow_id: str) -> Workflow:
    """Deprecated: Use WorkflowModel.get and from_model instead."""
    workflow_model = WorkflowModel.get(workflow_id)
    if not workflow_model:
        raise ValueError("not found")
    return from_model(workflow_model)


def create_workflow_router() -> APIRouter:
    router = APIRouter()

    @router.get("/workflows")
    async def list_workflows(user: str = Depends(current_user)) -> WorkflowList:
        """List all workflows in the database."""
        # List all workflows without user restriction (admin mode)
        # Use paginate to get all workflows
        workflows, next_key = await WorkflowModel.paginate(
            user_id=user,
            limit=1000,  # Large page size to get all workflows
        )
        return WorkflowList(workflows=[from_model(w) for w in workflows], next=next_key)

    @router.put("/workflows/{id}")
    async def update_workflow(
        id: str,
        workflow_request: WorkflowRequest,
        user: str = Depends(current_user),
    ) -> Workflow:
        workflow = await WorkflowModel.get(id)
        if workflow and workflow.user_id != user:
            raise HTTPException(status_code=403, detail="Workflow access denied")
        if not workflow:
            workflow = WorkflowModel(id=id, user_id=user)
        if workflow_request.graph is None:
            raise HTTPException(status_code=400, detail="Invalid workflow")
        workflow.name = workflow_request.name
        workflow.description = workflow_request.description
        workflow.tags = workflow_request.tags
        workflow.package_name = workflow_request.package_name
        if workflow_request.thumbnail is not None:
            workflow.thumbnail = workflow_request.thumbnail
        workflow.access = workflow_request.access
        workflow.graph = workflow_request.graph.model_dump()
        workflow.settings = workflow_request.settings
        workflow.run_mode = workflow_request.run_mode
        workflow.updated_at = workflow.updated_at
        await workflow.save()
        updated_workflow = from_model(workflow)

        return updated_workflow

    @router.delete("/workflows/{id}")
    async def delete_workflow(id: str, user: str = Depends(current_user)):
        """Delete a workflow from the database."""
        workflow = await WorkflowModel.get(id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if workflow.user_id != user:
            raise HTTPException(status_code=403, detail="Workflow access denied")
        await workflow.delete()
        return {"status": "ok", "message": f"Workflow {id} deleted"}

    @router.post("/workflows/{id}/run")
    async def execute_workflow(
        id: str, request: Request, user: str = Depends(current_user)
    ):
        try:
            params = await request.json()
            req = RunJobRequest(params=params, workflow_id=id, user_id=user)

            context = ProcessingContext(
                user_id=user, asset_output_mode=AssetOutputMode.DATA_URI
            )

            results: Dict[str, object] = {}
            async for msg in run_workflow(req, context=context, use_thread=True):
                if isinstance(msg, JobUpdate) and msg.status == "error":
                    raise HTTPException(status_code=500, detail=msg.error)
                if isinstance(msg, OutputUpdate):
                    value = msg.value
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

    @router.post("/workflows/{id}/run/stream")
    async def execute_workflow_stream(
        id: str, request: Request, user: str = Depends(current_user)
    ):
        try:
            params = await request.json()
            req = RunJobRequest(params=params, workflow_id=id, user_id=user)

            context = ProcessingContext(
                user_id=user, asset_output_mode=AssetOutputMode.DATA_URI
            )

            async def generate_sse():
                results: Dict[str, object] = {}
                try:
                    async for msg in run_workflow(
                        req, context=context, use_thread=True
                    ):
                        if isinstance(msg, JobUpdate):
                            event_data = {
                                "type": "job_update",
                                "data": msg.model_dump(),
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                            if msg.status == "error":
                                error_data = {"type": "error", "error": msg.error}
                                yield f"data: {json.dumps(error_data)}\n\n"
                                return
                        elif isinstance(msg, OutputUpdate):
                            value = msg.value
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
