#!/usr/bin/env python

from datetime import datetime
import traceback
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Header, Request
from fastapi.responses import StreamingResponse
from nodetool.types.job import JobUpdate
from nodetool.workflows.types import Error, OutputUpdate
from pydantic import BaseModel, Field
from nodetool.types.graph import Edge, Node, remove_connected_slots
from nodetool.types.workflow import WorkflowList, Workflow, WorkflowRequest
from nodetool.api.utils import current_user
from nodetool.common.environment import Environment
import logging
from typing import Any, Optional
from nodetool.workflows.read_graph import read_graph
from nodetool.models.workflow import Workflow as WorkflowModel
import base64
from nodetool.workflows.http_stream_runner import HTTPStreamRunner
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.types.graph import get_input_schema, get_output_schema
from nodetool.packages.registry import Registry
from nodetool.chat.providers import get_provider
from nodetool.metadata.types import Provider
from nodetool.chat.workspace_manager import WorkspaceManager
import asyncio

log = logging.getLogger(__name__)
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
        package_name=workflow.package_name,
        tags=workflow.tags,
        description=workflow.description or "",
        thumbnail=workflow.thumbnail or "",
        thumbnail_url=find_thumbnail(workflow),
        graph=api_graph,
        input_schema=get_input_schema(api_graph),
        output_schema=get_output_schema(api_graph),
        settings=workflow.settings,
        run_mode=workflow.run_mode,
    )


@router.post("/")
async def create(
    workflow_request: WorkflowRequest,
    user: str = Depends(current_user),
    from_example_package: Optional[str] = None,
    from_example_name: Optional[str] = None,
) -> Workflow:
    # If creating from an example
    if from_example_package and from_example_name:
        example_registry = Registry()
        try:
            example_workflow = example_registry.load_example(
                from_example_package, from_example_name
            )
            if not example_workflow:
                raise HTTPException(
                    status_code=404,
                    detail=f"Example '{from_example_name}' not found in package '{from_example_package}'",
                )

            # Create a new workflow based on the example
            workflow = from_model(
                await WorkflowModel.create(
                    name=workflow_request.name,
                    package_name=workflow_request.package_name,
                    description=workflow_request.description
                    or example_workflow.description,
                    thumbnail=workflow_request.thumbnail,
                    thumbnail_url=workflow_request.thumbnail_url
                    or example_workflow.thumbnail_url,
                    tags=workflow_request.tags or example_workflow.tags,
                    access=workflow_request.access,
                    graph=example_workflow.graph.model_dump(),
                    user_id=user,
                    run_mode=workflow_request.run_mode,
                )
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    elif workflow_request.graph:
        workflow = from_model(
            await WorkflowModel.create(
                name=workflow_request.name,
                package_name=workflow_request.package_name,
                description=workflow_request.description,
                thumbnail=workflow_request.thumbnail,
                thumbnail_url=workflow_request.thumbnail_url,
                tags=workflow_request.tags,
                access=workflow_request.access,
                graph=remove_connected_slots(workflow_request.graph).model_dump(),
                user_id=user,
                run_mode=workflow_request.run_mode,
            )
        )
    elif workflow_request.comfy_workflow:
        try:
            edges, nodes = read_graph(workflow_request.comfy_workflow)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        workflow = from_model(
            await WorkflowModel.create(
                name=workflow_request.name,
                description=workflow_request.description,
                thumbnail=workflow_request.thumbnail,
                thumbnail_url=workflow_request.thumbnail_url,
                tags=workflow_request.tags,
                access=workflow_request.access,
                user_id=user,
                graph={
                    "nodes": [node.model_dump() for node in nodes],
                    "edges": [edge.model_dump() for edge in edges],
                },
                run_mode=workflow_request.run_mode,
            )
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid workflow")

    return workflow


@router.get("/")
async def index(
    user: str = Depends(current_user),
    cursor: Optional[str] = None,
    limit: int = 100,
    columns: Optional[str] = None,
) -> WorkflowList:
    column_list = columns.split(",") if columns else None

    workflows, cursor = await WorkflowModel.paginate(
        user_id=user, limit=limit, start_key=cursor, columns=column_list
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

    workflows, cursor = await WorkflowModel.paginate(
        limit=limit, start_key=cursor, columns=column_list
    )
    return WorkflowList(
        workflows=[from_model(workflow) for workflow in workflows], next=cursor
    )


@router.get("/public/{id}")
async def get_public_workflow(id: str) -> Workflow:
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public":
        raise HTTPException(status_code=404, detail="Workflow not found")
    return from_model(workflow)


@router.get("/tools")
async def get_workflow_tools(
    user: str = Depends(current_user),
    cursor: Optional[str] = None,
    limit: int = 100,
    columns: Optional[str] = None,
) -> WorkflowList:
    """
    Get all workflows that have run_mode set to "tool".

    These workflows can be used as tools by agents and other workflows.

    Args:
        user: The authenticated user
        cursor: Pagination cursor
        limit: Maximum number of workflows to return
        columns: Comma-separated list of columns to return

    Returns:
        WorkflowList: List of tool workflows with pagination info
    """
    column_list = columns.split(",") if columns else None

    # Get all user workflows
    workflows, cursor = await WorkflowModel.paginate(
        user_id=user, limit=limit, start_key=cursor, columns=column_list
    )

    # Filter for workflows with run_mode = "tool"
    tool_workflows = [w for w in workflows if w.run_mode == "tool"]

    return WorkflowList(
        workflows=[from_model(workflow) for workflow in tool_workflows],
        next=cursor if len(workflows) == limit else None,
    )


@router.get("/examples")
async def examples() -> WorkflowList:
    example_registry = Registry()
    examples = await asyncio.to_thread(example_registry.list_examples)
    return WorkflowList(workflows=examples, next=None)


@router.get("/examples/search")
async def search_examples(query: str) -> WorkflowList:
    """
    Search for example workflows by searching through node titles, descriptions, and types.

    Args:
        query: The search string to find in node properties

    Returns:
        WorkflowList: A list of workflows that contain nodes matching the query
    """
    example_registry = Registry()
    matching_workflows = await asyncio.to_thread(
        example_registry.search_example_workflows, query
    )
    return WorkflowList(workflows=matching_workflows, next=None)


@router.get("/examples/{package_name}/{example_name}")
async def get_example(package_name: str, example_name: str) -> Workflow:
    """
    Load a specific example workflow from disk by package name and example name.

    Args:
        package_name: The name of the package containing the example
        example_name: The name of the example workflow to load

    Returns:
        Workflow: The loaded example workflow with full graph data

    Raises:
        HTTPException: If the package or example is not found
    """
    example_registry = Registry()
    try:
        workflow = example_registry.load_example(package_name, example_name)
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Example '{example_name}' not found in package '{package_name}'",
            )
        return workflow
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{id}")
async def get_workflow(id: str, user: str = Depends(current_user)) -> Workflow:
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public" and workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return from_model(workflow)


@router.put("/{id}")
async def update_workflow(
    id: str,
    workflow_request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(current_user),
) -> Workflow:
    print(workflow_request.settings)
    workflow = await WorkflowModel.get(id)
    if not workflow:
        workflow = WorkflowModel(id=id, user_id=user)
    if workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow_request.graph is None:
        raise HTTPException(status_code=400, detail="Invalid workflow")
    workflow.name = workflow_request.name
    workflow.description = workflow_request.description
    workflow.tags = workflow_request.tags
    workflow.package_name = workflow_request.package_name
    if workflow_request.thumbnail is not None:
        workflow.thumbnail = workflow_request.thumbnail
    workflow.access = workflow_request.access
    workflow.graph = remove_connected_slots(workflow_request.graph).model_dump()
    workflow.settings = workflow_request.settings
    workflow.run_mode = workflow_request.run_mode
    workflow.updated_at = datetime.now()
    await workflow.save()
    updated_workflow = from_model(workflow)

    return updated_workflow


# Endpoint to delete a specific workflow by ID
@router.delete("/{id}")
async def delete_workflow(
    id: str,
    user: str = Depends(current_user),
) -> None:
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")
    await workflow.delete()


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

    workflow = Workflow(
        id=id,
        name=workflow_request.name,
        description=workflow_request.description or "",
        tags=workflow_request.tags,
        package_name=workflow_request.package_name,
        path=workflow_request.path,
        thumbnail_url=workflow_request.thumbnail_url,
        access="public",
        graph=remove_connected_slots(workflow_request.graph),
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    example_registry = Registry()

    # remove "example" from tags
    if workflow.tags:
        workflow.tags = [tag for tag in workflow.tags if tag != "example"]

    try:
        saved_workflow = await asyncio.to_thread(
            example_registry.save_example, workflow
        )
        return saved_workflow
    except ValueError as e:
        log.error(f"Error saving example workflow: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


class RunWorkflowRequest(BaseModel):
    params: dict[str, Any] = Field(default={})


@router.post("/{id}/run")
async def run_workflow_by_id(
    id: str,
    run_workflow_request: RunWorkflowRequest,
    request: Request,
    stream: bool = False,
    user: str = Depends(current_user),
    authentication: Optional[str] = Header(None),
):
    """
    Run a specific workflow by ID.
    """
    server_protocol = request.headers.get("x-forwarded-proto", "http")
    server_host_name = request.headers.get("host", "localhost")
    server_port = request.headers.get("x-server-port", "8000")

    token = authentication.split(" ")[1] if authentication else "local_token"

    job_request = RunJobRequest(
        workflow_id=id,
        user_id=user,
        params=run_workflow_request.params,
        auth_token=token,
    )

    if job_request.api_url == "" or job_request.api_url is None:
        job_request.api_url = f"{server_protocol}://{server_host_name}:{server_port}"

    if stream:
        runner = HTTPStreamRunner()
        return StreamingResponse(
            runner.run_job(job_request), media_type="application/x-ndjson"
        )
    else:
        result = {}
        async for msg in run_workflow(job_request):
            # Ensure msg is a dictionary-like object for uniform access
            if isinstance(msg, BaseModel):
                msg = msg.model_dump()

            if isinstance(msg, OutputUpdate):
                name = msg.node_name
                value = msg.value
                if isinstance(value, dict):
                    if "data" in value:
                        data = value["data"]
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
                elif isinstance(msg, Error):
                    raise HTTPException(status_code=500, detail=msg.error)
                result[name] = value
        return result
