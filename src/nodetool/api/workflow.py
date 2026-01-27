#!/usr/bin/env python

import asyncio
import base64
import traceback
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from nodetool.api.utils import current_user
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Message, Provider
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.models.workflow_version import WorkflowVersion as WorkflowVersionModel
from nodetool.packages.registry import Registry
from nodetool.providers import get_provider
from nodetool.runtime.resources import require_scope
from nodetool.types.api_graph import Graph, get_input_schema, get_output_schema, remove_connected_slots
from nodetool.types.workflow import (
    AutosaveResponse,
    AutosaveWorkflowRequest,
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


class WorkflowGenerateNameRequest(BaseModel):
    """Request model for generating a workflow name using an LLM."""

    provider: str
    model: str


# Constants for workflow name generation
MAX_WORKFLOW_NAME_LENGTH = 60
MAX_NODES_IN_DESCRIPTION = 10


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
        workspace_id=workflow.workspace_id,
        html_app=workflow.html_app,
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
                    workspace_id=workflow_request.workspace_id,
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
                workspace_id=workflow_request.workspace_id,
                html_app=workflow_request.html_app,
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
                workspace_id=workflow_request.workspace_id,
                html_app=workflow_request.html_app,
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


@router.get("/{id}/app", response_class=HTMLResponse)
async def get_workflow_app(id: str, user: str = Depends(current_user)) -> HTMLResponse:
    """
    Serve the HTML app for a workflow as a website.

    Returns the stored html_app content as an HTML response that can be
    rendered directly in a browser.
    """
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public" and workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if not workflow.html_app:
        raise HTTPException(status_code=404, detail="No HTML app configured for this workflow")
    return HTMLResponse(content=workflow.html_app, status_code=200)


@router.put("/{id}")
async def update_workflow(
    id: str,
    workflow_request: WorkflowRequest,
    user: str = Depends(current_user),
) -> Workflow:
    """
    Update an existing workflow.

    Note: This endpoint does NOT create a version. Use POST /versions for manual
    version creation, or POST /autosave for automatic version saving.
    """
    log.debug(f"Updating workflow {id} with settings: {workflow_request.settings}")
    workflow = await WorkflowModel.get(id)
    if not workflow:
        workflow = WorkflowModel(id=id, user_id=user)
    if workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow_request.graph is None:
        raise HTTPException(status_code=400, detail="Invalid workflow")

    new_graph = remove_connected_slots(workflow_request.graph).model_dump()
    workflow.name = workflow_request.name
    workflow.tool_name = workflow_request.tool_name
    workflow.description = workflow_request.description
    workflow.tags = workflow_request.tags
    workflow.package_name = workflow_request.package_name
    if workflow_request.thumbnail is not None:
        workflow.thumbnail = workflow_request.thumbnail
    workflow.access = workflow_request.access
    workflow.graph = new_graph
    workflow.settings = workflow_request.settings
    workflow.run_mode = workflow_request.run_mode
    workflow.workspace_id = workflow_request.workspace_id
    workflow.html_app = workflow_request.html_app
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
                    raise HTTPException(status_code=500, detail=msg.message)
                result[name] = value
        return result


@router.post("/{id}/generate-name")
async def generate_workflow_name(
    id: str,
    req: WorkflowGenerateNameRequest,
    user: str = Depends(current_user),
) -> Workflow:
    """
    Generate a name for a workflow using an LLM based on its content.

    This endpoint analyzes the workflow's nodes and structure to generate
    a descriptive name (maximum 60 characters). Similar to chat thread
    auto-titling functionality.

    Args:
        id: The workflow ID
        req: Request containing provider and model to use for generation

    Returns:
        The updated workflow with the generated name
    """
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Build a description of the workflow from its graph
    graph = workflow.get_api_graph()
    node_descriptions = []
    for node in graph.nodes:
        node_info = node.type.split(".")[-1]  # Get the last part of the node type
        if node.data and isinstance(node.data, dict) and "label" in node.data:
            node_info += f" ({node.data['label']})"
        node_descriptions.append(node_info)

    workflow_content = (
        f"Workflow with {len(graph.nodes)} nodes: {', '.join(node_descriptions[:MAX_NODES_IN_DESCRIPTION])}"
        + (
            f"... and {len(graph.nodes) - MAX_NODES_IN_DESCRIPTION} more"
            if len(graph.nodes) > MAX_NODES_IN_DESCRIPTION
            else ""
        )
    )
    if workflow.description:
        workflow_content = f"Description: {workflow.description}\n{workflow_content}"

    # Use the provided provider and model for LLM call
    provider = await get_provider(Provider(req.provider), user_id=user)
    log.debug(f"Generating name for workflow {id} using provider: {provider}")

    # Make the LLM call
    response = await provider.generate_message(
        model=req.model,
        messages=[
            Message(
                role="system",
                content=f"You are a helpful assistant that creates concise, descriptive names for workflows based on their content (maximum {MAX_WORKFLOW_NAME_LENGTH} characters). Return only the name, nothing else.",
            ),
            Message(
                role="user",
                content="Create a workflow name for: " + workflow_content,
            ),
        ],
    )
    log.debug(f"Name generation response: {response}")

    if response.content:
        new_name = str(response.content)
        # Clean up the name (remove quotes if present)
        new_name = new_name.strip("\"'")

        # Update the workflow name
        workflow.name = new_name[:MAX_WORKFLOW_NAME_LENGTH]
        workflow.updated_at = datetime.now()
        await workflow.save()

        log.info(f"Updated workflow {id} name to: {new_name}")

    return await from_model(workflow)


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
        save_type=version.save_type,
        autosave_metadata=version.autosave_metadata,
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

    # Get the next version number once before creating
    next_version = await WorkflowVersionModel.get_next_version(id)
    version_name = version_request.name or f"Version {next_version}"

    version = await WorkflowVersionModel.create(
        workflow_id=id,
        user_id=user,
        graph=workflow.graph,
        name=version_name,
        description=version_request.description,
    )

    return from_version_model(version)


@router.get("/{id}/versions")
async def list_versions(
    id: str,
    user: str = Depends(current_user),
    cursor: Optional[int] = None,
    limit: int = 100,
) -> WorkflowVersionList:
    """
    List all versions of a workflow.

    Args:
        id: Workflow ID
        cursor: Version number to start pagination after (for next page, use the
                version number from the 'next' field in the response)
        limit: Maximum number of versions to return
    """
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user and workflow.access != "public":
        raise HTTPException(status_code=404, detail="Workflow not found")

    versions, next_cursor = await WorkflowVersionModel.paginate(
        workflow_id=id,
        limit=limit,
        start_version=cursor,
    )

    return WorkflowVersionList(
        versions=[from_version_model(v) for v in versions],
        next=str(next_cursor) if next_cursor else None,
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


@router.delete("/{id}/versions/{version_id}")
async def delete_version(
    id: str,
    version_id: str,
    user: str = Depends(current_user),
) -> dict[str, bool]:
    """
    Delete a specific workflow version.

    Args:
        id: Workflow ID
        version_id: Version ID to delete

    Returns:
        Success status
    """
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")

    deleted = await WorkflowVersionModel.delete_by_id(version_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Version not found")

    return {"success": True}

@router.post("/{id}/autosave")
async def autosave_workflow(
    id: str,
    autosave_request: AutosaveWorkflowRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(current_user),
) -> AutosaveResponse:
    """
    Create an autosave version of a workflow.

    The backend decides whether to create a version based on:
    - Rate limiting (minimum interval between saves)
    - Max versions per workflow limit

    Args:
        id: Workflow ID
        autosave_request: Autosave request with save_type, description, force flag
        background_tasks: Background tasks for cleanup operations
        user: Current authenticated user

    Returns:
        AutosaveResponse with version info or skipped status
    """
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if autosave_request.save_type == "autosave" and not autosave_request.force:
        latest_autosave = await WorkflowVersionModel.get_latest_autosave(id)
        if latest_autosave:
            min_interval = 30
            elapsed = (datetime.now() - latest_autosave.created_at).total_seconds()
            if elapsed < min_interval:
                log.debug(f"Autosave skipped for workflow {id}: too soon ({elapsed:.1f}s < {min_interval}s)")
                return AutosaveResponse(version=None, message="skipped (too soon)", skipped=True)

    # Use user's max_versions setting, default to 50
    max_versions = autosave_request.max_versions or 50

    # FIFO: Delete oldest autosaves to make room before creating new one
    autosave_count = await WorkflowVersionModel.count_autosaves(id)
    if autosave_count >= max_versions:
        # Delete oldest autosaves, keeping max_versions - 1 to make room for new one
        deleted = await WorkflowVersionModel.delete_old_autosaves(
            workflow_id=id,
            keep_count=max_versions - 1,
        )
        if deleted > 0:
            log.debug(f"Deleted {deleted} old autosaves for workflow {id} (FIFO)")

    next_version = await WorkflowVersionModel.get_next_version(id)
    version_name = f"Autosave {next_version}"

    autosave_metadata: dict[str, Any] = {
        "client_id": autosave_request.client_id,
        "trigger_reason": autosave_request.save_type,
    }

    # Convert dict to Graph if provided
    graph_to_save = workflow.graph
    if autosave_request.graph:
        try:
            graph_to_save = Graph(**autosave_request.graph).model_dump()
        except Exception as e:
            log.warning(f"Failed to parse graph from request, using database graph: {e}")
            graph_to_save = workflow.graph

    # Skip empty workflows (no nodes)
    nodes = graph_to_save.get("nodes", []) if isinstance(graph_to_save, dict) else []
    if not nodes:
        log.debug(f"Autosave skipped for workflow {id}: empty workflow")
        return AutosaveResponse(version=None, message="skipped (empty workflow)", skipped=True)

    version = await WorkflowVersionModel.create(
        workflow_id=id,
        user_id=user,
        graph=graph_to_save,
        name=version_name,
        description=autosave_request.description,
        save_type=autosave_request.save_type,
        autosave_metadata=autosave_metadata,
    )

    log.info(f"Autosave created for workflow {id}: version {version.version}")

    background_tasks.add_task(
        cleanup_old_autosaves,
        workflow_id=id,
        max_versions=max_versions,
        keep_days=7,
    )

    return AutosaveResponse(
        version=from_version_model(version),
        message="autosaved",
        skipped=False,
    )


async def cleanup_old_autosaves(
    workflow_id: str,
    max_versions: int = 20,
    keep_days: int = 7,
) -> None:
    """
    Cleanup old autosave versions for a workflow.

    Args:
        workflow_id: The workflow ID to clean up
        max_versions: Maximum autosaves to keep per workflow
        keep_days: Number of days to keep autosaves
    """
    try:
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=keep_days)
        deleted_count = await WorkflowVersionModel.delete_old_autosaves(
            workflow_id=workflow_id,
            keep_count=max_versions,
            older_than=cutoff_date,
        )
        if deleted_count > 0:
            log.info(f"Cleaned up {deleted_count} old autosaves for workflow {workflow_id}")
    except Exception as e:
        log.error(f"Error cleaning up autosaves for workflow {workflow_id}: {e}")


class GradioExportRequest(BaseModel):
    """Request model for Gradio export configuration."""

    app_title: str = Field(default="NodeTool Workflow", description="Title for the Gradio app")
    theme: Optional[str] = Field(default=None, description="Gradio theme to use")
    description: Optional[str] = Field(default=None, description="Description for the Gradio app")
    allow_flagging: bool = Field(default=False, description="Allow flagging in the Gradio app")
    queue: bool = Field(default=True, description="Enable request queuing")


@router.get("/{id}/dsl-export", response_class=PlainTextResponse)
async def dsl_export(
    id: str,
    user: str = Depends(current_user),
) -> str:
    """
    Export a workflow to Python DSL code.

    Returns Python code that reconstructs the workflow using DSL node wrappers
    and connections. The generated code can be saved to a .py file and executed
    to recreate the workflow graph.

    Args:
        id: Workflow ID

    Returns:
        Python source code as a plain text response
    """
    from nodetool.dsl.export import graph_to_dsl_py

    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public" and workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")

    api_graph = workflow.get_api_graph()
    if api_graph is None:
        raise HTTPException(status_code=400, detail="Workflow has no associated graph")

    try:
        code = graph_to_dsl_py(api_graph)
    except Exception as e:
        log.error(f"Error exporting workflow {id} to DSL: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting workflow: {e}") from e

    return code


@router.post("/{id}/gradio-export", response_class=PlainTextResponse)
async def gradio_export(
    id: str,
    config: GradioExportRequest,
    user: str = Depends(current_user),
) -> str:
    """
    Export a workflow to a Gradio app Python script.

    Returns Python code that reconstructs the workflow using DSL node wrappers
    and wraps it in a Gradio app for interactive execution.

    Args:
        id: Workflow ID
        config: Gradio app configuration options

    Returns:
        Python source code as a plain text response
    """
    from nodetool.dsl.export import graph_to_gradio_py

    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public" and workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")

    api_graph = workflow.get_api_graph()
    if api_graph is None:
        raise HTTPException(status_code=400, detail="Workflow has no associated graph")

    try:
        code = graph_to_gradio_py(
            api_graph,
            app_title=config.app_title,
            theme=config.theme,
            description=config.description,
            allow_flagging=config.allow_flagging,
            queue=config.queue,
        )
    except Exception as e:
        log.error(f"Error exporting workflow {id} to Gradio: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting workflow: {e}") from e

    return code

