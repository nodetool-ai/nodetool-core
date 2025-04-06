#!/usr/bin/env python

from datetime import datetime
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from nodetool.types.graph import Edge, Node, remove_connected_slots
from nodetool.types.workflow import WorkflowList, Workflow, WorkflowRequest
from nodetool.api.utils import current_user, User
from nodetool.common.environment import Environment
from typing import Any, Optional
from nodetool.workflows.examples import load_examples, save_example
from nodetool.workflows.read_graph import read_graph
from nodetool.models.workflow import Workflow as WorkflowModel
import base64
from nodetool.workflows.http_stream_runner import HTTPStreamRunner
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.types.graph import Graph, get_input_schema, get_output_schema
from nodetool.packages.registry import Registry
from nodetool.chat.providers import get_provider
from nodetool.metadata.types import Provider
from nodetool.chat.workflow_planner import WorkflowPlanner
from nodetool.chat.workspace_manager import WorkspaceManager
import asyncio

log = Environment.get_logger()
router = APIRouter(prefix="/api/workflows", tags=["workflows"])


def find_thumbnail(workflow: WorkflowModel) -> str | None:
    if workflow.thumbnail:
        return Environment.get_asset_storage().get_url(workflow.thumbnail)
    else:
        return None


def from_model(workflow: WorkflowModel):
    api_graph = workflow.get_api_graph()

    return Workflow(
        id=workflow.id,
        access=workflow.access,
        created_at=workflow.created_at.isoformat(),
        updated_at=workflow.updated_at.isoformat(),
        name=workflow.name,
        tags=workflow.tags,
        description=workflow.description or "",
        thumbnail=workflow.thumbnail or "",
        thumbnail_url=find_thumbnail(workflow),
        graph=api_graph,
        input_schema=get_input_schema(api_graph),
        output_schema=get_output_schema(api_graph),
        settings=workflow.settings,
    )


@router.post("/")
async def create(
    workflow_request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(current_user),
) -> Workflow:
    if workflow_request.graph:
        workflow = from_model(
            WorkflowModel.create(
                name=workflow_request.name,
                description=workflow_request.description,
                thumbnail=workflow_request.thumbnail,
                thumbnail_url=workflow_request.thumbnail_url,
                tags=workflow_request.tags,
                access=workflow_request.access,
                graph=remove_connected_slots(workflow_request.graph).model_dump(),
                user_id=user.id,
            )
        )
    elif workflow_request.comfy_workflow:
        try:
            edges, nodes = read_graph(workflow_request.comfy_workflow)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        workflow = from_model(
            WorkflowModel.create(
                name=workflow_request.name,
                description=workflow_request.description,
                thumbnail=workflow_request.thumbnail,
                thumbnail_url=workflow_request.thumbnail_url,
                tags=workflow_request.tags,
                access=workflow_request.access,
                user_id=user.id,
                graph={
                    "nodes": [node.model_dump() for node in nodes],
                    "edges": [edge.model_dump() for edge in edges],
                },
            )
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid workflow")

    return workflow


@router.get("/")
async def index(
    user: User = Depends(current_user),
    cursor: Optional[str] = None,
    limit: int = 100,
    columns: Optional[str] = None,
) -> WorkflowList:
    column_list = columns.split(",") if columns else None

    workflows, cursor = WorkflowModel.paginate(
        user_id=user.id, limit=limit, start_key=cursor, columns=column_list
    )
    return WorkflowList(
        workflows=[from_model(workflow) for workflow in workflows], next=cursor
    )


@router.get("/public")
async def public(
    limit: int = 100,
    cursor: Optional[str] = None,
    columns: Optional[str] = None,
) -> WorkflowList:
    column_list = columns.split(",") if columns else None

    workflows, cursor = WorkflowModel.paginate(
        limit=limit, start_key=cursor, columns=column_list
    )
    return WorkflowList(
        workflows=[from_model(workflow) for workflow in workflows], next=cursor
    )


@router.get("/public/{id}")
async def get_public_workflow(id: str) -> Workflow:
    workflow = WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public":
        raise HTTPException(status_code=404, detail="Workflow not found")
    return from_model(workflow)


@router.get("/user/{user_id}")
async def user_workflows(
    user_id: str,
    limit: int = 100,
    cursor: Optional[str] = None,
    columns: Optional[str] = None,
) -> WorkflowList:
    column_list = columns.split(",") if columns else None

    workflows, cursor = WorkflowModel.paginate(
        user_id=user_id, limit=limit, start_key=cursor, columns=column_list
    )
    workflows = [from_model(workflow) for workflow in workflows]
    return WorkflowList(workflows=workflows, next=cursor)


@router.get("/examples")
async def examples() -> WorkflowList:
    return WorkflowList(workflows=load_examples(), next=None)


@router.get("/{id}")
async def get_workflow(id: str, user: User = Depends(current_user)) -> Workflow:
    workflow = WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public" and workflow.user_id != user.id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return from_model(workflow)


@router.put("/{id}")
async def update_workflow(
    id: str,
    workflow_request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(current_user),
) -> Workflow:
    workflow = WorkflowModel.get(id)
    if not workflow:
        workflow = WorkflowModel(id=id, user_id=user.id)
    if workflow.user_id != user.id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow_request.graph is None:
        raise HTTPException(status_code=400, detail="Invalid workflow")
    workflow.name = workflow_request.name
    workflow.description = workflow_request.description
    workflow.tags = workflow_request.tags
    if workflow_request.thumbnail is not None:
        workflow.thumbnail = workflow_request.thumbnail
    workflow.access = workflow_request.access
    workflow.graph = remove_connected_slots(workflow_request.graph).model_dump()
    workflow.settings = workflow_request.settings
    workflow.updated_at = datetime.now()
    workflow.save()
    updated_workflow = from_model(workflow)

    return updated_workflow


# Endpoint to delete a specific workflow by ID
@router.delete("/{id}")
async def delete_workflow(
    id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(current_user),
) -> None:
    workflow = WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user.id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    workflow.delete()


@router.put("/examples/{id}")
async def save_example_workflow(
    id: str,
    workflow_request: WorkflowRequest,
) -> Workflow:
    if Environment.is_production():
        raise HTTPException(
            status_code=403,
            detail="Saving example workflows is only allowed in dev mode",
        )

    if workflow_request.graph is None:
        raise HTTPException(status_code=400, detail="Invalid workflow")

    examples = load_examples()
    for example in examples:
        if example.name == workflow_request.name:
            workflow_request.thumbnail_url = example.thumbnail_url
            break

    # remove "example" from tags
    if workflow_request.tags:
        workflow_request.tags = [
            tag for tag in workflow_request.tags if tag != "example"
        ]

    workflow = Workflow(
        id=id,
        name=workflow_request.name,
        description=workflow_request.description or "",
        tags=workflow_request.tags,
        thumbnail_url=workflow_request.thumbnail_url,
        access="public",
        graph=remove_connected_slots(workflow_request.graph),
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )

    saved_workflow = save_example(id, workflow)
    return saved_workflow


class RunWorkflowRequest(BaseModel):
    params: dict[str, Any] = Field(default={})


@router.post("/{id}/run")
async def run_workflow_by_id(
    id: str,
    run_workflow_request: RunWorkflowRequest,
    request: Request,
    stream: bool = False,
    user: User = Depends(current_user),
):
    """
    Run a specific workflow by ID.
    """
    server_protocol = request.headers.get("x-forwarded-proto", "http")
    server_host_name = request.headers.get("host", "localhost")
    server_port = request.headers.get("x-server-port", "8000")

    job_request = RunJobRequest(
        workflow_id=id,
        user_id=user.id,
        params=run_workflow_request.params,
    )

    if job_request.api_url == "" or job_request.api_url is None:
        job_request.api_url = f"{server_protocol}://{server_host_name}:{server_port}"

    if job_request.auth_token == "":
        job_request.auth_token = user.auth_token or ""

    if stream:
        runner = HTTPStreamRunner()
        return StreamingResponse(
            runner.run_job(job_request), media_type="application/x-ndjson"
        )
    else:
        result = {}
        async for msg in run_workflow(job_request):
            if msg.get("type") == "job_update":
                if msg.get("status") == "completed":
                    result = msg.get("result", {})
                    for key, value in result.items():
                        if isinstance(value, dict) and value.get("data"):
                            data = value.get("data")
                            if isinstance(data, bytes):
                                value["uri"] = (
                                    f"data:application/octet-stream;base64,{base64.b64encode(data).decode('utf-8')}"
                                )
                            elif isinstance(data, list):
                                # TODO: handle multiple assets
                                value["uri"] = (
                                    f"data:application/octet-stream;base64,{base64.b64encode(data[0]).decode('utf-8')}"
                                )
                            value["data"] = None
                elif msg.get("status") == "failed":
                    raise HTTPException(status_code=500, detail=msg.get("error"))
        return result


class SmartWorkflowCreateRequest(BaseModel):
    prompt: str


class SmartWorkflowResponse(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


@router.post("/create-smart")
async def create_smart_workflow(
    workflow_request: SmartWorkflowCreateRequest,
    user: User = Depends(current_user),
) -> SmartWorkflowResponse:
    """
    Create a workflow automatically using AI based on a description.

    This endpoint uses WorkflowPlanner to generate a workflow structure based on
    natural language description provided in the request.
    """
    # Initialize registry and get available node types
    registry = Registry()
    installed_packages = registry.list_installed_packages()
    node_types = []
    for package in installed_packages:
        if package.nodes:
            node_types.extend(package.nodes)

    # Get the chat provider
    provider_type = Provider.OpenAI
    provider = get_provider(provider_type)

    # Create a WorkflowPlanner instance
    workspace_manager = WorkspaceManager()
    workspace_dir = workspace_manager.get_current_directory()
    planner = WorkflowPlanner(
        provider=provider,
        model="gpt-4o",
        objective=workflow_request.prompt,
        workspace_dir=workspace_dir,
        node_types=node_types,
        enable_tracing=True,
    )

    try:
        # Generate workflow
        workflow_graph = await planner.create_workflow()

        return SmartWorkflowResponse(
            nodes=workflow_graph["nodes"], edges=workflow_graph["edges"]
        )

    except Exception as e:
        log.error(f"Error creating smart workflow: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create workflow: {str(e)}"
        )
