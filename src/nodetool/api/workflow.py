#!/usr/bin/env python

import asyncio
import base64
import traceback
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nodetool.api.utils import current_user
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.models.workflow_version import WorkflowVersion as WorkflowVersionModel
from nodetool.packages.registry import Registry
from nodetool.runtime.resources import require_scope
from nodetool.types.graph import Graph, get_input_schema, get_output_schema, remove_connected_slots
from nodetool.types.workflow import (
    CreateWorkflowVersionRequest,
    Workflow,
    WorkflowList,
    WorkflowRequest,
    WorkflowTool,
    WorkflowToolList,
    WorkflowVersion,
    WorkflowVersionList,
)
from nodetool.workflows.http_stream_runner import HTTPStreamRunner
from nodetool.workflows.read_graph import read_graph
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import Error, OutputUpdate

log = get_logger(__name__)
router = APIRouter(prefix="/api/workflows", tags=["workflows"])


async def find_thumbnail(workflow: WorkflowModel) -> str | None:
    if workflow.thumbnail:
        return await require_scope().get_asset_storage().get_url(workflow.thumbnail)
    else:
        return None


async def from_model(
    workflow: WorkflowModel,
    *,
    api_graph: Graph | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
):
    if api_graph is None:
        api_graph = workflow.get_api_graph()

    if input_schema is None:
        input_schema = get_input_schema(api_graph)

    if output_schema is None:
        output_schema = get_output_schema(api_graph)

    return Workflow(
        id=workflow.id,
        access=workflow.access,
        created_at=workflow.created_at.isoformat(),
        updated_at=workflow.updated_at.isoformat(),
        name=workflow.name,
        tool_name=workflow.tool_name,
        package_name=workflow.package_name,
        tags=workflow.tags,
        description=workflow.description or "",
        thumbnail=workflow.thumbnail or "",
        thumbnail_url=await find_thumbnail(workflow),
        graph=api_graph,
        input_schema=input_schema,
        output_schema=output_schema,
        settings=workflow.settings,
        run_mode=workflow.run_mode,
    )


def _graph_has_input_and_output(graph: Graph):
    has_input = False
    has_output = False

    for node in graph.nodes:
        node_type = node.type

        if node_type.startswith("nodetool.input."):
            has_input = True
        elif node_type.startswith("nodetool.output."):
            has_output = True

        if has_input and has_output:
            return True

    return False


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
            example_workflow = example_registry.load_example(from_example_package, from_example_name)
            if not example_workflow:
                raise HTTPException(
                    status_code=404,
                    detail=f"Example '{from_example_name}' not found in package '{from_example_package}'",
                )

            # Create a new workflow based on the example
            workflow = await from_model(
                await WorkflowModel.create(
                    name=workflow_request.name,
                    package_name=workflow_request.package_name,
                    description=workflow_request.description or example_workflow.description,
                    thumbnail=workflow_request.thumbnail,
                    thumbnail_url=workflow_request.thumbnail_url or example_workflow.thumbnail_url,
                    tags=workflow_request.tags or example_workflow.tags,
                    access=workflow_request.access,
                    graph=example_workflow.graph.model_dump(),
                    user_id=user,
                    run_mode=workflow_request.run_mode,
                )
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
    elif workflow_request.graph:
        workflow = await from_model(
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
            raise HTTPException(status_code=400, detail=str(e)) from e
        workflow = await from_model(
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
    run_mode: Optional[str] = None,
) -> WorkflowList:
    column_list = columns.split(",") if columns else None

    workflows, cursor = await WorkflowModel.paginate(
        user_id=user,
        limit=limit,
        start_key=cursor,
        columns=column_list,
        run_mode=run_mode,
    )
    workflow_responses = await asyncio.gather(*[from_model(workflow) for workflow in workflows])
    return WorkflowList(workflows=workflow_responses, next=cursor)


@router.get("/public")
async def public(
    limit: int = 100,
    cursor: Optional[str] = None,
    columns: Optional[str] = None,
) -> WorkflowList:
    column_list = columns.split(",") if columns else None

    workflows, cursor = await WorkflowModel.paginate(limit=limit, start_key=cursor, columns=column_list)
    workflow_responses = await asyncio.gather(*[from_model(workflow) for workflow in workflows])
    return WorkflowList(workflows=workflow_responses, next=cursor)


@router.get("/public/{id}")
async def get_public_workflow(id: str) -> Workflow:
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public":
        raise HTTPException(status_code=404, detail="Workflow not found")
    return await from_model(workflow)


@router.get("/tools")
async def get_workflow_tools(
    user: str = Depends(current_user),
    cursor: Optional[str] = None,
    limit: int = 100,
) -> WorkflowToolList:
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
    # Get all user workflows
    workflows, cursor = await WorkflowModel.paginate_tools(
        user_id=user,
        limit=limit,
        start_key=cursor,
    )

    return WorkflowToolList(
        workflows=[
            WorkflowTool(
                name=workflow.name,
                tool_name=workflow.tool_name,
                description=workflow.description,
            )
            for workflow in workflows
        ],
        next=cursor if len(workflows) == limit else None,
    )


@router.get("/examples")
async def examples() -> WorkflowList:
    """
    List example workflows enriched with required providers and models.

    Provider detection rules:
    - If a node's namespace matches a known provider namespace
      (gemini, openai, replicate, huggingface, huggingface_hub, fal, aime)
    - Or if any node property is a LanguageModel (type == 'language_model')
      which has a 'provider' field

    Model detection rules:
    - Collect ids from LanguageModel (id)
    - Collect repo_id from HuggingFaceModel-like types (types starting with 'hf.')
    - Collect model_id from InferenceProvider* types
    """
    example_registry = Registry()

    # Base list (lightweight graph), we will load full example per item to analyze
    examples = await asyncio.to_thread(example_registry.list_examples)

    provider_namespaces = {
        "gemini",
        "openai",
        "replicate",
        "huggingface",
        "huggingface_hub",
        "fal",
        "aime",
    }

    def parse_namespace(node_type: str) -> str:
        parts = node_type.split(".")
        if not parts:
            return ""
        return parts[0]

    def collect_from_value(val: Any, providers: set[str], models: set[str]):
        # Recursively collect provider/model info from nested dict/list values
        if isinstance(val, dict):
            t = val.get("type")
            if t == "language_model":
                # Provider and id from LanguageModel
                provider = val.get("provider")
                if isinstance(provider, str) and provider:
                    providers.add(provider)
                model_id = val.get("id")
                if isinstance(model_id, str) and model_id:
                    models.add(model_id)
            elif isinstance(t, str) and t.startswith("hf."):
                # HuggingFace model types (repo_id)
                repo_id = val.get("repo_id")
                if isinstance(repo_id, str) and repo_id:
                    models.add(repo_id)
            elif isinstance(t, str) and t.startswith("inference_provider_"):
                # Inference provider types: collect model ids and providers
                model_id = val.get("model_id")
                if isinstance(model_id, str) and model_id:
                    models.add(model_id)
                provider = "huggingface_hub"

            # Recurse into nested values
            for v in val.values():
                collect_from_value(v, providers, models)
        elif isinstance(val, list):
            for item in val:
                collect_from_value(item, providers, models)

    # Load full examples in parallel to speed up detection
    load_tasks = []
    indices: list[int] = []
    for i, ex in enumerate(examples):
        if ex.package_name and ex.name:
            load_tasks.append(asyncio.to_thread(example_registry.load_example, ex.package_name, ex.name))
            indices.append(i)

    loaded_map = {}
    if load_tasks:
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        for pos, res in enumerate(results):
            idx = indices[pos]
            if isinstance(res, Exception):
                log.warning(f"Error loading example {idx}: {res}")
                loaded_map[idx] = None
            else:
                loaded_map[idx] = res

    enriched: list[Workflow] = []
    for i, ex in enumerate(examples):
        required_providers: set[str] = set()
        required_models: set[str] = set()

        full_example = loaded_map.get(i)
        if full_example and full_example.graph and full_example.graph.nodes:
            for node in full_example.graph.nodes:
                ns = parse_namespace(node.type)
                if ns in provider_namespaces:
                    required_providers.add(ns)
                collect_from_value(getattr(node, "data", {}), required_providers, required_models)

        enriched.append(
            Workflow(
                id=ex.id,
                access=ex.access,
                created_at=ex.created_at,
                updated_at=ex.updated_at,
                name=ex.name,
                tool_name=ex.tool_name,
                package_name=ex.package_name,
                tags=ex.tags,
                description=ex.description,
                thumbnail=ex.thumbnail,
                thumbnail_url=ex.thumbnail_url,
                graph=ex.graph,
                input_schema=ex.input_schema,
                output_schema=ex.output_schema,
                settings=ex.settings,
                run_mode=ex.run_mode,
                path=ex.path,
                required_providers=sorted(required_providers) or None,
                required_models=sorted(required_models) or None,
            )
        )

    return WorkflowList(workflows=enriched, next=None)


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
    matching_workflows = await asyncio.to_thread(example_registry.search_example_workflows, query)
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
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/{id}")
async def get_workflow(id: str, user: str = Depends(current_user)) -> Workflow:
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public" and workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return await from_model(workflow)


@router.put("/{id}")
async def update_workflow(
    id: str,
    workflow_request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(current_user),
) -> Workflow:
    log.debug(f"Updating workflow {id} with settings: {workflow_request.settings}")
    workflow = await WorkflowModel.get(id)
    if not workflow:
        workflow = WorkflowModel(id=id, user_id=user)
    if workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow_request.graph is None:
        raise HTTPException(status_code=400, detail="Invalid workflow")
    workflow.name = workflow_request.name
    workflow.tool_name = workflow_request.tool_name
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
    updated_workflow = await from_model(workflow)

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
        saved_workflow = await asyncio.to_thread(example_registry.save_example, workflow)
        return saved_workflow
    except ValueError as e:
        log.error(f"Error saving example workflow: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e)) from e


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
    server_port = request.headers.get("x-server-port", "7777")

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
        return StreamingResponse(runner.run_job(job_request), media_type="application/x-ndjson")
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


# Workflow Version Endpoints


def from_version_model(version: WorkflowVersionModel) -> WorkflowVersion:
    """Convert a WorkflowVersionModel to a WorkflowVersion API type."""
    return WorkflowVersion(
        id=version.id,
        workflow_id=version.workflow_id,
        version=version.version,
        created_at=version.created_at.isoformat(),
        name=version.name,
        description=version.description,
        graph=Graph(
            nodes=version.graph.get("nodes", []),
            edges=version.graph.get("edges", []),
        ),
    )


@router.post("/{id}/versions")
async def create_version(
    id: str,
    version_request: CreateWorkflowVersionRequest,
    user: str = Depends(current_user),
) -> WorkflowVersion:
    """
    Create a new version of a workflow.

    This saves the current state of the workflow as a version snapshot.
    """
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")

    version = await WorkflowVersionModel.create(
        workflow_id=id,
        user_id=user,
        graph=workflow.graph,
        name=version_request.name or f"Version {await WorkflowVersionModel.get_next_version(id)}",
        description=version_request.description or "",
    )

    return from_version_model(version)


@router.get("/{id}/versions")
async def list_versions(
    id: str,
    user: str = Depends(current_user),
    cursor: Optional[str] = None,
    limit: int = 100,
) -> WorkflowVersionList:
    """
    List all versions of a workflow.
    """
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user and workflow.access != "public":
        raise HTTPException(status_code=404, detail="Workflow not found")

    versions, next_cursor = await WorkflowVersionModel.paginate(
        workflow_id=id,
        limit=limit,
        start_key=cursor,
    )

    return WorkflowVersionList(
        versions=[from_version_model(v) for v in versions],
        next=next_cursor if next_cursor else None,
    )


@router.get("/{id}/versions/{version}")
async def get_version(
    id: str,
    version: int,
    user: str = Depends(current_user),
) -> WorkflowVersion:
    """
    Get a specific version of a workflow.
    """
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user and workflow.access != "public":
        raise HTTPException(status_code=404, detail="Workflow not found")

    version_model = await WorkflowVersionModel.get_by_version(id, version)
    if not version_model:
        raise HTTPException(status_code=404, detail="Version not found")

    return from_version_model(version_model)


@router.post("/{id}/versions/{version}/restore")
async def restore_version(
    id: str,
    version: int,
    user: str = Depends(current_user),
) -> Workflow:
    """
    Restore a workflow to a specific version.

    This replaces the current workflow graph with the graph from the specified version.
    The current state is NOT automatically saved as a new version before restoring.
    """
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")

    version_model = await WorkflowVersionModel.get_by_version(id, version)
    if not version_model:
        raise HTTPException(status_code=404, detail="Version not found")

    # Restore the workflow graph from the version
    workflow.graph = version_model.graph
    workflow.updated_at = datetime.now()
    await workflow.save()

    return await from_model(workflow)
