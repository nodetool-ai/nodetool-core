#!/usr/bin/env python
"""
FastMCP server for NodeTool API

This module provides MCP (Model Context Protocol) server integration for NodeTool,
allowing AI assistants to interact with NodeTool workflows, nodes, and assets.
"""

from fastmcp import FastMCP, Context
from typing import Any, Optional
from huggingface_hub import ModelInfo
from nodetool.types.job import JobUpdate
from nodetool.workflows.types import Error, LogUpdate, OutputUpdate, PreviewUpdate, SaveUpdate, NodeUpdate, NodeProgress
from pydantic import BaseModel, Field
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.types.graph import Graph, get_input_schema, get_output_schema
from nodetool.packages.registry import Registry
from nodetool.config.logging_config import get_logger
from nodetool.chat.search_nodes import search_nodes as search_nodes_tool
from nodetool.models.asset import Asset as AssetModel
from nodetool.types.asset import Asset
from nodetool.config.environment import Environment
from nodetool.models.job import Job as JobModel
from nodetool.workflows.job_execution_manager import JobExecutionManager
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.api.model import (
    get_all_models,
    recommended_models,
    get_language_models,
)
from nodetool.types.model import UnifiedModel
from nodetool.metadata.types import LanguageModel, ImageModel, Provider, TTSModel, ASRModel
from nodetool.ml.models.image_models import get_all_image_models as get_all_image_models_func
from nodetool.ml.models.tts_models import get_all_tts_models as get_all_tts_models_func
from nodetool.ml.models.asr_models import get_all_asr_models as get_all_asr_models_func
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    get_async_chroma_client,
    get_async_collection,
)
from nodetool.integrations.huggingface.huggingface_models import read_cached_hf_models
from huggingface_hub.constants import HF_HUB_CACHE
from nodetool.indexing.service import index_file_to_collection
from nodetool.indexing.ingestion import find_input_nodes
from nodetool.models.thread import Thread
from nodetool.models.message import Message as DBMessage
from nodetool.metadata.types import Message as ApiMessage
from nodetool.providers import get_provider
from io import BytesIO
import base64
from dataclasses import asdict
from nodetool.agents.agent import Agent
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.agents.tools.email_tools import SearchEmailTool
from nodetool.workflows.types import Chunk

log = get_logger(__name__)

# Initialize FastMCP server
mcp = FastMCP("NodeTool API Server")


class WorkflowRunParams(BaseModel):
    """Parameters for running a workflow"""
    workflow_id: str = Field(..., description="The ID of the workflow to run")
    params: dict[str, Any] = Field(default_factory=dict, description="Input parameters for the workflow")


class NodeSearchParams(BaseModel):
    """Parameters for searching nodes"""
    query: str = Field(..., description="Search query for nodes")
    namespace: Optional[str] = Field(None, description="Optional namespace to filter nodes")




@mcp.tool()
async def get_workflow(workflow_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific workflow.

    Args:
        workflow_id: The ID of the workflow

    Returns:
        Workflow details including graph structure, input/output schemas
    """
    workflow = await WorkflowModel.find("1", workflow_id)
    if not workflow:
        raise ValueError(f"Workflow {workflow_id} not found")

    api_graph = workflow.get_api_graph()
    input_schema = get_input_schema(api_graph)
    output_schema = get_output_schema(api_graph)

    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description or "",
        "tags": workflow.tags,
        "graph": api_graph.model_dump(),
        "input_schema": input_schema,
        "output_schema": output_schema,
        "created_at": workflow.created_at.isoformat(),
        "updated_at": workflow.updated_at.isoformat(),
    }


@mcp.tool()
async def run_workflow_tool(workflow_id: str, ctx: Context, params: dict[str, Any] = {}) -> dict[str, Any]:
    """
    Execute a NodeTool workflow with given parameters.

    Args:
        workflow_id: The ID of the workflow to run
        params: Dictionary of input parameters for the workflow

    Returns:
        Workflow execution results
    """
    workflow = await WorkflowModel.find("1", workflow_id)
    if not workflow:
        raise ValueError(f"Workflow {workflow_id} not found")

    # Create run request
    request = RunJobRequest(
        workflow_id=workflow_id,
        params=params,
    )

    # Run workflow
    result = {}
    preview = {}
    save = {}
    context = ProcessingContext()
    async for msg in run_workflow(request):
        if isinstance(msg, PreviewUpdate):
            preview[msg.node_id] = await context.upload_assets_to_temp(msg.value)
        elif isinstance(msg, SaveUpdate):
            save[msg.name] = await context.upload_assets_to_temp(msg.value)
        elif isinstance(msg, OutputUpdate):
            result[msg.node_name] = await context.upload_assets_to_temp(msg.value)
        elif isinstance(msg, JobUpdate):
            if msg.status == "error":
                raise Exception(msg.error)
        elif isinstance(msg, NodeUpdate):
            await ctx.info(f"{msg.node_name} {msg.status}")
        elif isinstance(msg, NodeProgress):
            await ctx.report_progress(msg.progress, msg.total)
        elif isinstance(msg, LogUpdate):
            await ctx.info(msg.content)
        elif isinstance(msg, Error):
            raise Exception(msg.error)

    return {
        "workflow_id": workflow_id,
        "status": "completed",
        "result": result,
        "preview": preview,
        "save": save,
    }


@mcp.tool()
async def run_graph(graph: dict[str, Any], params: dict[str, Any] = {}) -> dict[str, Any]:
    """
    Execute a workflow graph directly without saving it as a workflow.

    This is useful for testing workflow graphs or running one-off executions
    without persisting the workflow to the database.

    Args:
        graph: Workflow graph structure with nodes and edges
        params: Dictionary of input parameters for the workflow

    Returns:
        Workflow execution results

    Example:
        result = await run_graph(
            graph={
                "nodes": [
                    {
                        "id": "input1",
                        "type": "nodetool.input.StringInput",
                        "data": {"name": "text", "value": ""}
                    },
                    {
                        "id": "agent1",
                        "type": "nodetool.agents.Agent",
                        "data": {
                            "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"}
                        }
                    }
                ],
                "edges": [
                    {
                        "source": "input1",
                        "sourceHandle": "output",
                        "target": "agent1",
                        "targetHandle": "prompt"
                    }
                ]
            },
            params={"text": "Hello, how are you?"}
        )
    """
    from nodetool.types.graph import remove_connected_slots

    # Parse and validate graph
    graph_obj = Graph.model_validate(graph)
    cleaned_graph = remove_connected_slots(graph_obj)

    # Create temporary run request without workflow_id
    request = RunJobRequest(
        user_id="1",
        params=params,
        graph=cleaned_graph,
    )

    # Run workflow
    result = {}
    async for msg in run_workflow(request):
        if isinstance(msg, OutputUpdate):
            result[msg.node_name] = msg.value
        elif isinstance(msg, JobUpdate):
            if msg.status == "error":
                raise Exception(msg.error)
        elif isinstance(msg, Error):
            raise Exception(msg.error)

    return {
        "status": "completed",
        "result": result,
    }

@mcp.tool()
async def list_nodes(namespace: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
    """
    List available nodes in NodeTool registry.

    Args:
        namespace: Optional namespace to filter nodes (e.g., 'nodetool.text', 'nodetool.image')
        limit: Maximum number of nodes to return (default: 100)

    Returns:
        List of node metadata including type, description, and properties
    """
    registry = Registry.get_instance()
    all_nodes = registry.get_all_installed_nodes()

    result = []
    count = 0

    for node in all_nodes:
        if namespace and not node.namespace.startswith(namespace):
            continue

        if count >= limit:
            break

        result.append({
            "type": node.node_type,
        })
        count += 1

    return result


@mcp.tool()
async def search_nodes(query: list[str], n_results: int = 10, input_type: Optional[str] = None, output_type: Optional[str] = None, exclude_namespaces: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """
    Search for nodes by name, description, or tags.

    Args:
        query: Search query strings
        namespace: Optional namespace to filter nodes

    Returns:
        List of matching nodes
    """
    nodes = search_nodes_tool(
        query=query,
        input_type=input_type,
        output_type=output_type,
        n_results=n_results,
        exclude_namespaces=exclude_namespaces or [],
    )

    result = []
    for node in nodes:
        result.append({
            "type": node.node_type,
            "title": node.title,
            "description": node.description,
            "namespace": node.namespace,
        })

    return result


@mcp.tool()
async def get_node_info(node_type: str) -> dict[str, Any]:
    """
    Get detailed information about a specific node type.

    Args:
        node_type: The fully qualified node type (e.g., 'nodetool.text.Split')

    Returns:
        Detailed node metadata including properties, inputs, outputs
    """
    registry = Registry.get_instance()
    node = registry.find_node_by_type(node_type)

    if node:
        return node

    # If not found in registry, try to dynamically resolve the node class
    # This handles core nodes like Preview, Comment, etc. that aren't in packages
    from nodetool.workflows.base_node import get_node_class

    node_class = get_node_class(node_type)
    if not node_class:
        raise ValueError(f"Node type {node_type} not found")

    # Generate metadata from the node class
    metadata = node_class.get_metadata()
    return metadata.model_dump()


@mcp.tool()
async def save_workflow(
    workflow_id: str,
    name: str,
    graph: dict[str, Any],
    description: str = "",
    tags: list[str] | None = None,
    access: str = "private",
    run_mode: str | None = None,
) -> dict[str, Any]:
    """
    Save or update a workflow.

    Args:
        workflow_id: The ID of the workflow to save (creates new if doesn't exist)
        name: Workflow name
        graph: Workflow graph structure with nodes and edges
        description: Workflow description
        tags: List of tags for categorization
        access: Access level ("private" or "public")
        run_mode: Run mode ("workflow" or "tool")

    Returns:
        Saved workflow details
    """
    from nodetool.types.graph import remove_connected_slots
    from datetime import datetime

    # Try to get existing workflow
    workflow = await WorkflowModel.get(workflow_id)

    if not workflow:
        # Create new workflow
        workflow = WorkflowModel(
            id=workflow_id,
            user_id="1",  # Default user for MCP
            name=name,
            description=description,
            tags=tags or [],
            access=access,
            run_mode=run_mode,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    else:
        # Update existing workflow
        workflow.name = name
        workflow.description = description
        workflow.tags = tags or []
        workflow.access = access
        if run_mode is not None:
            workflow.run_mode = run_mode
        workflow.updated_at = datetime.now()

    # Parse and clean the graph
    graph_obj = Graph.model_validate(graph)
    workflow.graph = remove_connected_slots(graph_obj).model_dump()

    # Save workflow
    await workflow.save()

    # Get schemas
    api_graph = workflow.get_api_graph()
    input_schema = get_input_schema(api_graph)
    output_schema = get_output_schema(api_graph)

    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description or "",
        "tags": workflow.tags,
        "access": workflow.access,
        "run_mode": workflow.run_mode,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "created_at": workflow.created_at.isoformat(),
        "updated_at": workflow.updated_at.isoformat(),
        "message": "Workflow saved successfully"
    }


@mcp.tool()
async def validate_workflow(workflow_id: str) -> dict[str, Any]:
    """
    Validate a workflow's structure, connectivity, and type compatibility.

    Checks:
    - All node types exist in registry
    - No circular dependencies (DAG structure)
    - All required inputs are connected
    - Type compatibility of all edges
    - Proper input/output node configuration

    Args:
        workflow_id: The ID of the workflow to validate

    Returns:
        Validation report with errors, warnings, and suggestions
    """
    workflow = await WorkflowModel.find("1", workflow_id)
    if not workflow:
        raise ValueError(f"Workflow {workflow_id} not found")

    graph = workflow.get_api_graph()
    registry = Registry.get_instance()

    errors = []
    warnings = []
    suggestions = []

    # Track node IDs for uniqueness check
    node_ids = set()
    node_types_found = {}

    # Validate nodes
    for node in graph.nodes:
        # Check unique IDs
        if node.id in node_ids:
            errors.append(f"Duplicate node ID: {node.id}")
        node_ids.add(node.id)

        # Check node type exists
        node_metadata = registry.find_node_by_type(node.type)
        if not node_metadata:
            errors.append(f"Node type not found: {node.type} (node: {node.id})")
        else:
            node_types_found[node.id] = node_metadata

    # Build adjacency map for cycle detection
    adjacency = {node.id: [] for node in graph.nodes}
    edges_by_target = {}

    for edge in graph.edges:
        if edge.source not in node_ids:
            errors.append(f"Edge references non-existent source node: {edge.source}")
            continue
        if edge.target not in node_ids:
            errors.append(f"Edge references non-existent target node: {edge.target}")
            continue

        adjacency[edge.source].append(edge.target)

        # Track edges by target for input validation
        if edge.target not in edges_by_target:
            edges_by_target[edge.target] = []
        edges_by_target[edge.target].append(edge)

    # Check for cycles (DAG validation)
    def has_cycle():
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node_id: WHITE for node_id in node_ids}

        def dfs(node_id):
            if color[node_id] == GRAY:
                return True  # Back edge found - cycle detected
            if color[node_id] == BLACK:
                return False

            color[node_id] = GRAY
            for neighbor in adjacency[node_id]:
                if dfs(neighbor):
                    return True
            color[node_id] = BLACK
            return False

        for node_id in node_ids:
            if color[node_id] == WHITE:
                if dfs(node_id):
                    return True
        return False

    if has_cycle():
        errors.append("Workflow contains circular dependencies - must be a DAG (Directed Acyclic Graph)")

    # Validate node inputs and type compatibility
    for node in graph.nodes:
        if node.id not in node_types_found:
            continue

        metadata = node_types_found[node.id]

        # Check required inputs are connected
        if hasattr(metadata, 'properties'):
            required_inputs = [
                prop_name for prop_name, prop_data in metadata.properties.items()
                if isinstance(prop_data, dict) and prop_data.get('required', False)
            ]

            connected_inputs = set()
            if node.id in edges_by_target:
                for edge in edges_by_target[node.id]:
                    if edge.targetHandle:
                        connected_inputs.add(edge.targetHandle)

            # Check node properties for static values
            if hasattr(node.data, 'properties'):
                for prop_name in node.data.properties:
                    connected_inputs.add(prop_name)

            for required_input in required_inputs:
                if required_input not in connected_inputs:
                    warnings.append(
                        f"Required input '{required_input}' may not be connected on node '{node.id}' ({node.type})"
                    )

    # Check for orphaned nodes
    nodes_with_inputs = set(edges_by_target.keys())
    nodes_with_outputs = set()
    for edge in graph.edges:
        nodes_with_outputs.add(edge.source)

    for node in graph.nodes:
        # Skip input/output/constant nodes from orphan check
        if any(keyword in node.type.lower() for keyword in ['input', 'output', 'constant', 'preview']):
            continue

        if node.id not in nodes_with_inputs and node.id not in nodes_with_outputs:
            warnings.append(f"Orphaned node (not connected): {node.id} ({node.type})")
        elif node.id not in nodes_with_outputs:
            suggestions.append(f"Node '{node.id}' has no outputs - consider adding Preview or Output node")

    # Summary
    is_valid = len(errors) == 0

    return {
        "valid": is_valid,
        "workflow_id": workflow_id,
        "workflow_name": workflow.name,
        "summary": {
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "errors": len(errors),
            "warnings": len(warnings),
            "suggestions": len(suggestions),
        },
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "message": "Workflow is valid and ready to run" if is_valid else "Workflow has validation errors - please fix before running"
    }


@mcp.tool()
async def export_workflow_digraph(workflow_id: str) -> dict[str, Any]:
    """
    Export a workflow as a simple Graphviz Digraph (DOT format) for LLM parsing and visualization.

    Args:
        workflow_id: The ID of the workflow to export

    Returns:
        Dictionary with DOT format string and workflow metadata
    """
    workflow = await WorkflowModel.find("1", workflow_id)
    if not workflow:
        raise ValueError(f"Workflow {workflow_id} not found")

    graph = workflow.get_api_graph()

    # Start building DOT string
    dot_lines = [
        f"digraph workflow {{",
    ]

    # Helper function to sanitize node IDs for DOT format
    def sanitize_id(node_id: str) -> str:
        import re
        return re.sub(r'[^a-zA-Z0-9_]', '_', node_id)

    # Add nodes with simple labels
    for node in graph.nodes:
        sanitized_id = sanitize_id(node.id)
        # Simple label: node_id (type)
        label = f"{node.id} ({node.type})"
        dot_lines.append(f'  {sanitized_id} [label="{label}"];')

    # Add edges
    for edge in graph.edges:
        source_id = sanitize_id(edge.source)
        target_id = sanitize_id(edge.target)
        dot_lines.append(f"  {source_id} -> {target_id};")

    dot_lines.append("}")

    dot_content = "\n".join(dot_lines)

    return {
        "workflow_id": workflow_id,
        "workflow_name": workflow.name,
        "dot": dot_content,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
    }


@mcp.tool()
async def list_workflows(limit: int = 100, start_key: str | None = None) -> dict[str, Any]:
    """
    List all workflows with pagination support.

    Args:
        limit: Maximum number of workflows to return (default: 100)
        start_key: Pagination key for fetching next page (optional)

    Returns:
        Dictionary with workflows list and next pagination key
    """
    workflows, next_key = await WorkflowModel.paginate(
        user_id="1",
        limit=limit,
        start_key=start_key
    )

    result = []
    for workflow in workflows:
        result.append({
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description or "",
            "tags": workflow.tags,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
        })

    return {
        "workflows": result,
        "next": next_key,
    }


def _asset_to_dict(asset: AssetModel) -> dict[str, Any]:
    """Convert asset model to dictionary with URLs."""
    storage = Environment.get_asset_storage()

    if asset.content_type != "folder":
        get_url = storage.get_url(asset.file_name)
    else:
        get_url = None

    if asset.has_thumbnail:
        thumb_url = storage.get_url(asset.thumb_file_name)
    else:
        thumb_url = None

    return {
        "id": asset.id,
        "user_id": asset.user_id,
        "workflow_id": asset.workflow_id,
        "parent_id": asset.parent_id,
        "name": asset.name,
        "content_type": asset.content_type,
        "size": asset.size,
        "metadata": asset.metadata,
        "created_at": asset.created_at.isoformat(),
        "get_url": get_url,
        "thumb_url": thumb_url,
        "duration": asset.duration,
    }


@mcp.tool()
async def list_assets(
    parent_id: str | None = None,
    content_type: str | None = None,
    limit: int = 100,
    start_key: str | None = None,
) -> dict[str, Any]:
    """
    List assets with optional filtering by parent folder or content type.

    Args:
        parent_id: Optional parent folder ID to list assets from (if None, lists root assets)
        content_type: Optional content type filter (e.g., "image", "video", "audio", "text", "folder")
        limit: Maximum number of assets to return (default: 100)
        start_key: Pagination key for fetching next page (optional)

    Returns:
        Dictionary with assets list and next pagination key
    """
    # Use default user "1" for MCP
    user_id = "1"

    if content_type is None and parent_id is None:
        parent_id = user_id

    assets, next_cursor = await AssetModel.paginate(
        user_id=user_id,
        parent_id=parent_id,
        content_type=content_type,
        limit=limit,
        start_key=start_key,
    )

    return {
        "assets": [_asset_to_dict(asset) for asset in assets],
        "next": next_cursor,
    }


@mcp.tool()
async def search_assets(
    query: str,
    content_type: str | None = None,
    limit: int = 100,
    start_key: str | None = None,
) -> dict[str, Any]:
    """
    Search assets by name across all folders.

    Args:
        query: Search query (minimum 2 characters)
        content_type: Optional content type filter (e.g., "image", "video", "audio")
        limit: Maximum number of results to return (default: 100)
        start_key: Pagination key for fetching next page (optional)

    Returns:
        Dictionary with search results including folder path information
    """
    if len(query.strip()) < 2:
        raise ValueError("Search query must be at least 2 characters long")

    # Use default user "1" for MCP
    user_id = "1"

    assets, next_cursor, folder_paths = await AssetModel.search_assets_global(
        user_id=user_id,
        query=query.strip(),
        content_type=content_type,
        limit=limit,
        start_key=start_key,
    )

    results = []
    for i, asset in enumerate(assets):
        asset_dict = _asset_to_dict(asset)
        folder_info = (
            folder_paths[i]
            if i < len(folder_paths)
            else {
                "folder_name": "Unknown",
                "folder_path": "Unknown",
                "folder_id": "",
            }
        )
        asset_dict["folder_name"] = folder_info["folder_name"]
        asset_dict["folder_path"] = folder_info["folder_path"]
        asset_dict["folder_id"] = folder_info["folder_id"]
        results.append(asset_dict)

    return {
        "assets": results,
        "next": next_cursor,
        "total_count": len(results),
    }


@mcp.tool()
async def get_asset(asset_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific asset.

    Args:
        asset_id: The ID of the asset

    Returns:
        Asset details including URLs and metadata
    """
    # Use default user "1" for MCP
    user_id = "1"

    asset = await AssetModel.find(user_id, asset_id)
    if not asset:
        raise ValueError(f"Asset {asset_id} not found")

    return _asset_to_dict(asset)


@mcp.tool()
async def create_folder(
    name: str,
    parent_id: str | None = None,
) -> dict[str, Any]:
    """
    Create a new folder asset.

    Args:
        name: Name of the folder
        parent_id: Optional parent folder ID (if None, creates in root)

    Returns:
        Created folder asset details
    """
    # Use default user "1" for MCP
    user_id = "1"

    if parent_id is None:
        parent_id = user_id

    asset = await AssetModel.create(
        user_id=user_id,
        parent_id=parent_id,
        name=name,
        content_type="folder",
        metadata={},
        size=None,
    )

    return _asset_to_dict(asset)


@mcp.tool()
async def update_asset(
    asset_id: str,
    name: str | None = None,
    parent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Update an existing asset's properties.

    Args:
        asset_id: The ID of the asset to update
        name: New name for the asset (optional)
        parent_id: New parent folder ID to move the asset (optional)
        metadata: New metadata dictionary (optional)

    Returns:
        Updated asset details
    """
    # Use default user "1" for MCP
    user_id = "1"

    asset = await AssetModel.find(user_id, asset_id)
    if not asset:
        raise ValueError(f"Asset {asset_id} not found")

    if name:
        asset.name = name.strip()
    if parent_id:
        asset.parent_id = parent_id
    if metadata is not None:
        asset.metadata = metadata

    await asset.save()
    return _asset_to_dict(asset)


@mcp.tool()
async def delete_asset(asset_id: str) -> dict[str, Any]:
    """
    Delete an asset. If it's a folder, deletes all contents recursively.

    Args:
        asset_id: The ID of the asset to delete

    Returns:
        Dictionary with list of deleted asset IDs
    """
    # Use default user "1" for MCP
    user_id = "1"

    asset = await AssetModel.find(user_id, asset_id)
    if not asset:
        raise ValueError(f"Asset {asset_id} not found")

    deleted_ids = []

    async def delete_folder_recursive(folder_id: str) -> list[str]:
        deleted = []
        assets, _ = await AssetModel.paginate(
            user_id=user_id, parent_id=folder_id, limit=10000
        )

        for child in assets:
            if child.content_type == "folder":
                deleted.extend(await delete_folder_recursive(child.id))
            else:
                await child.delete()
                deleted.append(child.id)

        folder_asset = await AssetModel.find(user_id, folder_id)
        if folder_asset:
            await folder_asset.delete()
            deleted.append(folder_id)

        return deleted

    if asset.content_type == "folder":
        deleted_ids = await delete_folder_recursive(asset_id)
    else:
        await asset.delete()
        deleted_ids = [asset_id]

    return {"deleted_asset_ids": deleted_ids}


@mcp.tool()
async def list_package_assets(package_name: str | None = None) -> list[dict[str, Any]]:
    """
    List assets from installed NodeTool packages.

    Args:
        package_name: Optional package name to filter assets (if None, lists all package assets)

    Returns:
        List of package assets with their metadata
    """
    registry = Registry.get_instance()
    all_assets = registry.list_assets()

    if package_name:
        all_assets = [a for a in all_assets if a.package_name == package_name]

    return [
        {
            "id": f"pkg:{asset.package_name}/{asset.name}",
            "name": asset.name,
            "package_name": asset.package_name,
            "virtual_path": f"/api/assets/packages/{asset.package_name}/{asset.name}",
        }
        for asset in all_assets
    ]


@mcp.tool()
async def list_jobs(
    workflow_id: str | None = None,
    limit: int = 100,
    start_key: str | None = None,
) -> dict[str, Any]:
    """
    List jobs with optional filtering by workflow.

    Args:
        workflow_id: Optional workflow ID to filter jobs by
        limit: Maximum number of jobs to return (default: 100)
        start_key: Pagination key for fetching next page (optional)

    Returns:
        Dictionary with jobs list and next pagination key
    """
    # Use default user "1" for MCP
    user_id = "1"

    jobs, next_cursor = await JobModel.paginate(
        user_id=user_id,
        workflow_id=workflow_id,
        limit=limit,
        start_key=start_key,
    )

    return {
        "jobs": [
            {
                "id": job.id,
                "user_id": job.user_id,
                "job_type": job.job_type,
                "status": job.status,
                "workflow_id": job.workflow_id,
                "started_at": job.started_at.isoformat() if job.started_at else "",
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                "error": job.error,
                "cost": job.cost,
            }
            for job in jobs
        ],
        "next": next_cursor,
    }


@mcp.tool()
async def get_job(job_id: str, include_logs: bool = False) -> dict[str, Any]:
    """
    Get detailed information about a specific job.

    Args:
        job_id: The ID of the job
        include_logs: Whether to include log count information (default: False)

    Returns:
        Job details including status, timing, and error information
    """
    # Use default user "1" for MCP
    user_id = "1"

    job = await JobModel.find(user_id=user_id, job_id=job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    result = {
        "id": job.id,
        "user_id": job.user_id,
        "job_type": job.job_type,
        "status": job.status,
        "workflow_id": job.workflow_id,
        "started_at": job.started_at.isoformat() if job.started_at else "",
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "error": job.error,
        "cost": job.cost,
    }

    if include_logs:
        result["log_count"] = len(job.logs) if job.logs else 0
        result["has_logs"] = bool(job.logs)

    return result


@mcp.tool()
async def list_running_jobs() -> list[dict[str, Any]]:
    """
    List all currently running background jobs.

    Returns:
        List of running jobs with their current status
    """
    # Use default user "1" for MCP
    user_id = "1"

    job_manager = JobExecutionManager.get_instance()
    bg_jobs = job_manager.list_jobs(user_id=user_id)

    return [
        {
            "job_id": job.job_id,
            "status": job.status,
            "workflow_id": job.request.workflow_id,
            "created_at": job.created_at.isoformat(),
            "is_running": job.is_running(),
            "is_completed": job.is_completed(),
        }
        for job in bg_jobs
    ]


@mcp.tool()
async def cancel_job(job_id: str) -> dict[str, Any]:
    """
    Cancel a running job.

    Args:
        job_id: The ID of the job to cancel

    Returns:
        Dictionary with cancellation result message
    """
    # Use default user "1" for MCP
    user_id = "1"

    # Verify the job exists
    job = await JobModel.find(user_id=user_id, job_id=job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    # Cancel the background job if it's running
    job_manager = JobExecutionManager.get_instance()
    cancelled = await job_manager.cancel_job(job_id)

    if cancelled:
        return {
            "success": True,
            "message": "Job cancelled successfully",
            "job_id": job_id,
        }
    else:
        return {
            "success": False,
            "message": "Job not found in background manager or already completed",
            "job_id": job_id,
        }


@mcp.tool()
async def get_job_logs(
    job_id: str,
    limit: int | None = None,
    include_live: bool = True,
) -> dict[str, Any]:
    """
    Get logs from a job execution.

    For completed jobs, retrieves persisted logs from the database.
    For running jobs, optionally retrieves live logs from the active log handler.

    Args:
        job_id: The ID of the job
        limit: Optional limit on number of logs to return (most recent)
        include_live: Whether to include live logs for running jobs (default: True)

    Returns:
        Dictionary with logs and job status information
    """
    # Use default user "1" for MCP
    user_id = "1"

    # Get job from database
    job = await JobModel.find(user_id=user_id, job_id=job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    logs = []
    is_running = False

    # Check if job is still running and get live logs
    if include_live:
        job_manager = JobExecutionManager.get_instance()
        job_execution = job_manager.get_job(job_id)

        if job_execution and job_execution.is_running():
            is_running = True
            logs = job_execution.get_live_logs(limit=limit)
        else:
            # Job is completed, get persisted logs
            logs = job.logs or []
            if limit and len(logs) > limit:
                logs = logs[-limit:]
    else:
        # Only get persisted logs
        logs = job.logs or []
        if limit and len(logs) > limit:
            logs = logs[-limit:]

    return {
        "job_id": job_id,
        "status": job.status,
        "is_running": is_running,
        "logs": logs,
        "total_logs": len(logs),
    }


@mcp.tool()
async def start_background_job(
    workflow_id: str,
    params: dict[str, Any] = {},
    auth_token: str | None = None,
) -> dict[str, Any]:
    """
    Start a workflow job that runs in the background without streaming results.

    This tool starts a workflow execution that continues running independently.
    Use list_jobs, get_job, or list_running_jobs to check the status later.
    The job will continue even if the MCP connection is closed.

    Args:
        workflow_id: The ID of the workflow to run
        params: Dictionary of input parameters for the workflow
        auth_token: Optional authentication token for external API calls

    Returns:
        Dictionary with job_id and initial status
    """
    # Use default user "1" for MCP
    user_id = "1"

    # Verify workflow exists
    workflow = await WorkflowModel.find(user_id, workflow_id)
    if not workflow:
        raise ValueError(f"Workflow {workflow_id} not found")

    # Create run request
    request = RunJobRequest(
        user_id=user_id,
        workflow_id=workflow_id,
        params=params,
        graph=workflow.get_api_graph(),
    )

    # Create processing context
    context = ProcessingContext(
        user_id=user_id,
        auth_token=auth_token,
        workflow_id=workflow_id,
    )

    # Start job in background via JobExecutionManager
    job_manager = JobExecutionManager.get_instance()
    job_execution = await job_manager.start_job(request, context)

    log.info(
        f"Started background job {job_execution.job_id} for workflow {workflow_id}",
        extra={
            "job_id": job_execution.job_id,
            "workflow_id": workflow_id,
            "user_id": user_id,
        },
    )

    return {
        "job_id": job_execution.job_id,
        "workflow_id": workflow_id,
        "status": job_execution.status,
        "message": "Background job started successfully. Use get_job or list_running_jobs to check status.",
        "created_at": job_execution.created_at.isoformat(),
    }



@mcp.tool()
async def list_all_models(
    provider: str,
    model_type: str | None = None,
    downloaded_only: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    List available models from all providers with mandatory filtering.

    Args:
        provider: Filter by provider (required). Use "all" for all providers, or specific provider like "huggingface", "ollama", "openai", "anthropic"
        model_type: Optional filter by model type (e.g., "language_model", "image_model", "tts_model", "asr_model")
        downloaded_only: Only show models that are downloaded locally (default: False)
        limit: Maximum number of models to return (default: 50, max: 200)

    Returns:
        List of models matching the filters
    """
    if limit > 200:
        limit = 200

    all_models = await get_all_models(user="1")

    # Filter by provider
    if provider.lower() != "all":
        all_models = [m for m in all_models if provider.lower() in m.id.lower() or (m.repo_id and provider.lower() in m.repo_id.lower())]

    # Filter by model type
    if model_type:
        all_models = [m for m in all_models if m.type == model_type]

    # Filter by downloaded status
    if downloaded_only:
        all_models = [m for m in all_models if m.downloaded]

    # Apply limit
    all_models = all_models[:limit]

    return [
        {
            "id": model.id,
            "name": model.name,
            "repo_id": model.repo_id,
            "path": model.path,
            "type": model.type,
            "downloaded": model.downloaded,
            "size_on_disk": model.size_on_disk,
        }
        for model in all_models
    ]


@mcp.tool()
async def list_recommended_models(limit: int = 20) -> list[dict[str, Any]]:
    """
    List recommended models for NodeTool (curated list, typically small).

    Args:
        limit: Maximum number of models to return (default: 20, max: 50)

    Returns:
        List of recommended models
    """
    if limit > 50:
        limit = 50

    models = await recommended_models(user="1")
    models = models[:limit]

    return [
        {
            "id": model.id,
            "name": model.name,
            "repo_id": model.repo_id,
            "path": model.path,
            "type": model.type,
            "downloaded": model.downloaded,
        }
        for model in models
    ]


@mcp.tool()
async def list_language_models(
    provider: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    List available language models (LLMs) with mandatory provider filter.

    Args:
        provider: Filter by provider (required). Options: "openai", "anthropic", "ollama", "google", "groq", "openrouter", "all"
        limit: Maximum number of models to return (default: 50, max: 100)

    Returns:
        List of language models matching the filter
    """
    if limit > 100:
        limit = 100

    all_models = await get_language_models()

    # Filter by provider
    if provider.lower() != "all":
        all_models = [m for m in all_models if m.provider.value.lower() == provider.lower()]

    # Apply limit
    all_models = all_models[:limit]

    return [
        {
            "id": model.id,
            "name": model.name,
            "provider": model.provider.value,
        }
        for model in all_models
    ]


@mcp.tool()
async def list_image_models(
    provider: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    List available image generation models with mandatory provider filter.

    Args:
        provider: Filter by provider (required). Options: "replicate", "comfyui", "huggingface", "openai", "all"
        limit: Maximum number of models to return (default: 50, max: 100)

    Returns:
        List of image models matching the filter
    """
    if limit > 100:
        limit = 100

    all_models = await get_all_image_models_func()

    # Filter by provider
    if provider.lower() != "all":
        all_models = [m for m in all_models if m.provider.value.lower() == provider.lower()]

    # Apply limit
    all_models = all_models[:limit]

    return [
        {
            "id": model.id,
            "name": model.name,
            "provider": model.provider.value,
        }
        for model in all_models
    ]


@mcp.tool()
async def list_tts_models(
    provider: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    List available text-to-speech models with mandatory provider filter.

    Args:
        provider: Filter by provider (required). Options: "openai", "elevenlabs", "huggingface", "all"
        limit: Maximum number of models to return (default: 50, max: 100)

    Returns:
        List of TTS models matching the filter
    """
    if limit > 100:
        limit = 100

    all_models = await get_all_tts_models_func()

    # Filter by provider
    if provider.lower() != "all":
        all_models = [m for m in all_models if m.provider.value.lower() == provider.lower()]

    # Apply limit
    all_models = all_models[:limit]

    return [
        {
            "id": model.id,
            "name": model.name,
            "provider": model.provider.value,
        }
        for model in all_models
    ]


@mcp.tool()
async def list_asr_models(
    provider: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    List available automatic speech recognition models with mandatory provider filter.

    Args:
        provider: Filter by provider (required). Options: "openai", "huggingface", "all"
        limit: Maximum number of models to return (default: 50, max: 100)

    Returns:
        List of ASR models matching the filter
    """
    if limit > 100:
        limit = 100

    all_models = await get_all_asr_models_func()

    # Filter by provider
    if provider.lower() != "all":
        all_models = [m for m in all_models if m.provider.value.lower() == provider.lower()]

    # Apply limit
    all_models = all_models[:limit]

    return [
        {
            "id": model.id,
            "name": model.name,
            "provider": model.provider.value,
        }
        for model in all_models
    ]


@mcp.tool()
async def create_collection(
    name: str,
    embedding_model: str = "all-minilm:latest",
) -> dict[str, Any]:
    """
    Create a new vector database collection for storing document embeddings.

    Args:
        name: Name of the collection to create
        embedding_model: Embedding model to use (default: "all-minilm:latest")

    Returns:
        Created collection details
    """
    client = await get_async_chroma_client()
    metadata = {
        "embedding_model": embedding_model,
    }
    collection = await client.create_collection(name=name, metadata=metadata)
    return {
        "name": collection.name,
        "metadata": collection.metadata,
        "count": 0,
    }


@mcp.tool()
async def list_collections(
    limit: int = 50,
) -> dict[str, Any]:
    """
    List all vector database collections.

    Args:
        limit: Maximum number of collections to return (default: 50, max: 100)

    Returns:
        Dictionary with collections list and total count
    """
    if limit > 100:
        limit = 100

    client = await get_async_chroma_client()
    collections = await client.list_collections()

    async def get_workflow_name(metadata: dict[str, str]) -> str | None:
        if workflow_id := metadata.get("workflow"):
            workflow = await WorkflowModel.get(workflow_id)
            if workflow:
                return workflow.name
        return None

    # Apply limit
    collections = collections[:limit]

    return {
        "collections": [
            {
                "name": col.name,
                "metadata": col.metadata or {},
                "workflow_name": await get_workflow_name(col.metadata or {}),
                "count": col.count(),
            }
            for col in collections
        ],
        "count": len(collections),
    }


@mcp.tool()
async def get_collection(name: str) -> dict[str, Any]:
    """
    Get details about a specific collection.

    Args:
        name: Name of the collection

    Returns:
        Collection details including metadata and document count
    """
    client = await get_async_chroma_client()
    collection = await client.get_collection(name=name)
    count = await collection.count()
    return {
        "name": collection.name,
        "metadata": collection.metadata,
        "count": count,
    }


@mcp.tool()
async def update_collection(
    name: str,
    new_name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Update a collection's name or metadata.

    Args:
        name: Current name of the collection
        new_name: New name for the collection (optional)
        metadata: New metadata to merge with existing metadata (optional)

    Returns:
        Updated collection details
    """
    client = await get_async_chroma_client()
    collection = await client.get_collection(name=name)

    current_metadata = collection.metadata.copy() if collection.metadata else {}
    if metadata:
        current_metadata.update(metadata)

    # Validate workflow if specified
    if workflow_id := current_metadata.get("workflow"):
        workflow = await WorkflowModel.get(workflow_id)
        if not workflow:
            raise ValueError("Workflow not found")

        # Validate workflow input nodes
        graph = workflow.graph
        collection_input, file_input = find_input_nodes(graph)
        if not collection_input:
            raise ValueError("Workflow must have a CollectionInput node")
        if not file_input:
            raise ValueError("Workflow must have a FileInput or DocumentFileInput node")

    await collection.modify(name=new_name, metadata=current_metadata)

    return {
        "name": collection.name,
        "metadata": collection.metadata,
        "count": await collection.count(),
    }


@mcp.tool()
async def delete_collection(name: str) -> dict[str, Any]:
    """
    Delete a collection and all its documents.

    Args:
        name: Name of the collection to delete

    Returns:
        Success message
    """
    client = await get_async_chroma_client()
    await client.delete_collection(name=name)
    return {"message": f"Collection {name} deleted successfully"}


@mcp.tool()
async def query_collection(
    name: str,
    query_texts: list[str],
    n_results: int = 10,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Query a collection for similar documents using semantic search.

    Args:
        name: Name of the collection to query
        query_texts: List of query texts to search for
        n_results: Number of results to return per query (default: 10, max: 50)
        where: Optional metadata filter (e.g., {"source": "pdf"})

    Returns:
        Query results with ids, documents, distances, and metadata
    """
    if n_results > 50:
        n_results = 50

    collection = await get_async_collection(name)

    results = await collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
    )

    return {
        "ids": results.get("ids", []),
        "documents": results.get("documents", []),
        "distances": results.get("distances", []),
        "metadatas": results.get("metadatas", []),
    }


@mcp.tool()
async def add_documents_to_collection(
    name: str,
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
    ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Add documents to a collection.

    Args:
        name: Name of the collection
        documents: List of document texts to add
        metadatas: Optional list of metadata dictionaries (one per document)
        ids: Optional list of document IDs (auto-generated if not provided)

    Returns:
        Success message with count of documents added
    """
    collection = await get_async_collection(name)

    # Generate IDs if not provided
    if ids is None:
        import uuid
        ids = [str(uuid.uuid4()) for _ in documents]

    await collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    return {
        "message": f"Added {len(documents)} documents to collection {name}",
        "count": len(documents),
        "ids": ids,
    }


@mcp.tool()
async def get_documents_from_collection(
    name: str,
    ids: list[str] | None = None,
    where: dict[str, Any] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Get documents from a collection by IDs or metadata filter.

    Args:
        name: Name of the collection
        ids: Optional list of document IDs to retrieve
        where: Optional metadata filter (e.g., {"source": "pdf"})
        limit: Maximum number of documents to return (default: 50, max: 100)

    Returns:
        Documents with their IDs, texts, and metadata
    """
    if limit > 100:
        limit = 100

    collection = await get_async_collection(name)

    results = await collection.get(
        ids=ids,
        where=where,
        limit=limit,
    )

    return {
        "ids": results.get("ids", []),
        "documents": results.get("documents", []),
        "metadatas": results.get("metadatas", []),
        "count": len(results.get("ids", [])),
    }


@mcp.tool()
async def delete_documents_from_collection(
    name: str,
    ids: list[str] | None = None,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Delete documents from a collection by IDs or metadata filter.

    Args:
        name: Name of the collection
        ids: Optional list of document IDs to delete
        where: Optional metadata filter to delete matching documents

    Returns:
        Success message
    """
    collection = await get_async_collection(name)

    await collection.delete(
        ids=ids,
        where=where,
    )

    return {
        "message": f"Deleted documents from collection {name}",
    }


@mcp.tool()
async def create_thread(name: str | None = None) -> dict[str, Any]:
    """
    Create a new chat thread for organizing conversation history.

    Args:
        name: Optional name for the thread (auto-generated if not provided)

    Returns:
        Created thread details
    """
    thread = await Thread.create(user_id="1", title=name)
    return {
        "id": thread.id,
        "user_id": thread.user_id,
        "name": thread.title,
        "created_at": thread.created_at.isoformat(),
        "updated_at": thread.updated_at.isoformat(),
    }


@mcp.tool()
async def list_threads(limit: int = 50) -> dict[str, Any]:
    """
    List all chat threads with pagination.

    Args:
        limit: Maximum number of threads to return (default: 50, max: 100)

    Returns:
        Dictionary with threads list and count
    """
    if limit > 100:
        limit = 100

    threads, _ = await Thread.paginate(user_id="1", limit=limit)

    return {
        "threads": [
            {
                "id": thread.id,
                "user_id": thread.user_id,
                "name": thread.title,
                "created_at": thread.created_at.isoformat(),
                "updated_at": thread.updated_at.isoformat(),
            }
            for thread in threads
        ],
        "count": len(threads),
    }


@mcp.tool()
async def get_thread(thread_id: str) -> dict[str, Any]:
    """
    Get details about a specific chat thread.

    Args:
        thread_id: ID of the thread

    Returns:
        Thread details
    """
    thread = await Thread.find(user_id="1", id=thread_id)
    if not thread:
        raise ValueError(f"Thread {thread_id} not found")

    return {
        "id": thread.id,
        "user_id": thread.user_id,
        "name": thread.title,
        "created_at": thread.created_at.isoformat(),
        "updated_at": thread.updated_at.isoformat(),
    }


@mcp.tool()
async def update_thread(thread_id: str, name: str) -> dict[str, Any]:
    """
    Update a thread's name.

    Args:
        thread_id: ID of the thread to update
        name: New name for the thread

    Returns:
        Updated thread details
    """
    thread = await Thread.find(user_id="1", id=thread_id)
    if not thread:
        raise ValueError(f"Thread {thread_id} not found")

    thread.title = name
    await thread.save()

    return {
        "id": thread.id,
        "user_id": thread.user_id,
        "name": thread.title,
        "created_at": thread.created_at.isoformat(),
        "updated_at": thread.updated_at.isoformat(),
    }


@mcp.tool()
async def delete_thread(thread_id: str) -> dict[str, Any]:
    """
    Delete a chat thread and all its messages.

    Args:
        thread_id: ID of the thread to delete

    Returns:
        Success message
    """
    thread = await Thread.find(user_id="1", id=thread_id)
    if not thread:
        raise ValueError(f"Thread {thread_id} not found")

    await thread.delete()
    return {"message": f"Thread {thread_id} deleted successfully"}


@mcp.tool()
async def get_thread_messages(
    thread_id: str,
    limit: int = 50,
    start_key: str | None = None,
) -> dict[str, Any]:
    """
    Get messages from a chat thread with pagination.

    Args:
        thread_id: ID of the thread
        limit: Maximum number of messages to return (default: 50, max: 100)
        start_key: Pagination key for fetching next page (optional)

    Returns:
        Dictionary with messages list and pagination info
    """
    if limit > 100:
        limit = 100

    # Verify thread exists
    thread = await Thread.find(user_id="1", id=thread_id)
    if not thread:
        raise ValueError(f"Thread {thread_id} not found")

    messages, next_key = await DBMessage.paginate(
        thread_id=thread_id,
        limit=limit,
        start_key=start_key,
    )

    return {
        "messages": [
            {
                "id": msg.id,
                "thread_id": msg.thread_id,
                "role": msg.role,
                "content": msg.content,
                "tool_calls": msg.tool_calls,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
            for msg in messages
        ],
        "count": len(messages),
        "next": next_key,
    }


@mcp.tool()
async def send_chat_message(
    thread_id: str,
    content: str,
    model: str = "gpt-4o",
    provider: Provider = Provider.OpenAI,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Send a message to a chat thread and get an AI response.

    This is a stateless MCP-optimized version that:
    - Saves the user message to the thread
    - Loads conversation history
    - Gets AI response using the specified model
    - Saves the assistant response
    - Returns the complete response

    Args:
        thread_id: ID of the thread to send message to
        content: User message content
        model: Model to use for response (default: "gpt-4o")
        provider: Provider for the model (default: "openai")
        system_prompt: Optional system prompt to override default

    Returns:
        Dictionary with the assistant's response and message metadata
    """
    # Verify thread exists
    thread = await Thread.find(user_id="1", id=thread_id)
    if not thread:
        raise ValueError(f"Thread {thread_id} not found")

    # Save user message
    user_message = await DBMessage.create(
        user_id="1",
        thread_id=thread_id,
        role="user",
        content=content,
        tool_calls=[],
    )

    # Load conversation history
    messages, _ = await DBMessage.paginate(
        thread_id=thread_id,
        limit=100,  # Get recent context
    )

    # Convert to API message format
    api_messages = [
        ApiMessage(
            role=msg.role,
            content=msg.content,
            tool_calls=msg.tool_calls if msg.tool_calls else None,
        )
        for msg in reversed(messages)  # Messages come in reverse order
    ]

    # Add system prompt if provided
    if system_prompt:
        api_messages.insert(0, ApiMessage(role="system", content=system_prompt))

    # Get provider and generate response
    provider_instance = get_provider(provider)

    # Stream response and collect chunks
    response = await provider_instance.generate_message(
        messages=api_messages,
        model=model,
    )

    # Save assistant message
    assistant_message = await DBMessage.create(
        user_id="1",
        thread_id=thread_id,
        role="assistant",
        content=response.content,
        tool_calls=[],
    )

    return {
        "thread_id": thread_id,
        "user_message_id": user_message.id,
        "assistant_message_id": assistant_message.id,
        "content": response.content,
        "model": model,
        "provider": provider,
    }


@mcp.tool()
async def delete_message(message_id: str) -> dict[str, Any]:
    """
    Delete a specific message from a thread.

    Args:
        message_id: ID of the message to delete

    Returns:
        Success message
    """
    message = await DBMessage.get(message_id)
    if not message:
        raise ValueError(f"Message {message_id} not found")

    await message.delete()
    return {"message": f"Message {message_id} deleted successfully"}


@mcp.tool()
async def upload_file_to_storage(
    key: str,
    content: str,
    temp: bool = False,
) -> dict[str, Any]:
    """
    Upload a file to NodeTool storage (asset or temp storage).

    Args:
        key: File key/name (no path separators allowed)
        content: Base64-encoded file content
        temp: If True, upload to temp storage; if False, upload to asset storage (default: False)

    Returns:
        Success message with file details
    """
    # Validate key has no path separators
    if "/" in key or "\\" in key:
        raise ValueError("Invalid key: path separators not allowed")

    # Decode base64 content
    try:
        file_data = base64.b64decode(content)
    except Exception as e:
        raise ValueError(f"Invalid base64 content: {e}")

    # Get appropriate storage
    storage = Environment.get_temp_storage() if temp else Environment.get_asset_storage()

    # Upload file
    await storage.upload(key, BytesIO(file_data))

    # Get file info
    size = await storage.get_size(key)

    return {
        "key": key,
        "size": size,
        "storage": "temp" if temp else "asset",
        "message": f"File uploaded successfully to {'temp' if temp else 'asset'} storage",
    }


@mcp.tool()
async def download_file_from_storage(
    key: str,
    temp: bool = False,
) -> dict[str, Any]:
    """
    Download a file from NodeTool storage.

    Args:
        key: File key/name to download
        temp: If True, download from temp storage; if False, download from asset storage (default: False)

    Returns:
        File content (base64-encoded) and metadata
    """
    # Validate key has no path separators
    if "/" in key or "\\" in key:
        raise ValueError("Invalid key: path separators not allowed")

    # Get appropriate storage
    storage = Environment.get_temp_storage() if temp else Environment.get_asset_storage()

    # Check if file exists
    if not await storage.file_exists(key):
        raise ValueError(f"File not found: {key}")

    # Download file
    stream = BytesIO()
    await storage.download(key, stream)
    file_data = stream.getvalue()

    # Get file metadata
    size = await storage.get_size(key)
    last_modified = await storage.get_mtime(key)

    return {
        "key": key,
        "content": base64.b64encode(file_data).decode("utf-8"),
        "size": size,
        "last_modified": last_modified.isoformat() if last_modified else None,
        "storage": "temp" if temp else "asset",
    }


@mcp.tool()
async def get_file_metadata(
    key: str,
    temp: bool = False,
) -> dict[str, Any]:
    """
    Get metadata about a file in storage without downloading it.

    Args:
        key: File key/name
        temp: If True, check temp storage; if False, check asset storage (default: False)

    Returns:
        File metadata (size, last modified, etc.)
    """
    # Validate key has no path separators
    if "/" in key or "\\" in key:
        raise ValueError("Invalid key: path separators not allowed")

    # Get appropriate storage
    storage = Environment.get_temp_storage() if temp else Environment.get_asset_storage()

    # Check if file exists
    if not await storage.file_exists(key):
        raise ValueError(f"File not found: {key}")

    # Get file metadata
    size = await storage.get_size(key)
    last_modified = await storage.get_mtime(key)

    return {
        "key": key,
        "exists": True,
        "size": size,
        "last_modified": last_modified.isoformat() if last_modified else None,
        "storage": "temp" if temp else "asset",
    }


@mcp.tool()
async def delete_file_from_storage(
    key: str,
    temp: bool = False,
) -> dict[str, Any]:
    """
    Delete a file from NodeTool storage.

    Args:
        key: File key/name to delete
        temp: If True, delete from temp storage; if False, delete from asset storage (default: False)

    Returns:
        Success message
    """
    # Validate key has no path separators
    if "/" in key or "\\" in key:
        raise ValueError("Invalid key: path separators not allowed")

    # Get appropriate storage
    storage = Environment.get_temp_storage() if temp else Environment.get_asset_storage()

    # Check if file exists
    if not await storage.file_exists(key):
        raise ValueError(f"File not found: {key}")

    # Delete file
    await storage.delete(key)

    return {
        "key": key,
        "storage": "temp" if temp else "asset",
        "message": f"File deleted successfully from {'temp' if temp else 'asset'} storage",
    }


@mcp.tool()
async def list_storage_files(
    temp: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """
    List files in storage (note: this may not be supported by all storage backends).

    Args:
        temp: If True, list temp storage; if False, list asset storage (default: False)
        limit: Maximum number of files to return (default: 100, max: 200)

    Returns:
        List of file keys and metadata
    """
    if limit > 200:
        limit = 200

    # Get appropriate storage
    storage = Environment.get_temp_storage() if temp else Environment.get_asset_storage()

    # Try to list files (not all storage backends support this)
    try:
        if hasattr(storage, "list_files"):
            files = await storage.list_files(limit=limit)
            return {
                "files": [
                    {
                        "key": f.get("key"),
                        "size": f.get("size"),
                        "last_modified": f.get("last_modified"),
                    }
                    for f in files[:limit]
                ],
                "count": len(files[:limit]),
                "storage": "temp" if temp else "asset",
            }
        else:
            return {
                "message": "Storage backend does not support listing files",
                "storage": "temp" if temp else "asset",
            }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to list files - storage backend may not support this operation",
            "storage": "temp" if temp else "asset",
        }


@mcp.tool()
async def get_hf_cache_info() -> dict[str, Any]:
    """
    Get information about the HuggingFace cache directory and cached models.

    Returns:
        Cache directory path and summary of cached models
    """
    cached_models = await read_cached_hf_models()

    # Calculate total size
    total_size = sum(model.size_on_disk or 0 for model in cached_models)

    return {
        "cache_dir": str(HF_HUB_CACHE),
        "total_models": len(cached_models),
        "total_size_bytes": total_size,
        "total_size_gb": round(total_size / (1024**3), 2),
        "models": [
            {
                "repo_id": model.repo_id,
                "type": model.type,
                "size_on_disk": model.size_on_disk,
                "path": model.path,
            }
            for model in cached_models[:100]  # Limit to first 100
        ],
    }


@mcp.tool()
async def inspect_hf_cached_model(repo_id: str) -> dict[str, Any]:
    """
    Inspect a specific HuggingFace model in the cache.

    Args:
        repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b-hf")

    Returns:
        Detailed information about the cached model
    """
    cached_models = await read_cached_hf_models()

    # Find matching models
    matching_models = [m for m in cached_models if m.repo_id == repo_id]

    if not matching_models:
        raise ValueError(f"Model {repo_id} not found in cache")

    model = matching_models[0]

    return {
        "repo_id": model.repo_id,
        "name": model.name,
        "type": model.type,
        "path": model.path,
        "size_on_disk": model.size_on_disk,
        "size_on_disk_gb": round((model.size_on_disk or 0) / (1024**3), 2) if model.size_on_disk else None,
        "downloaded": model.downloaded,
    }


@mcp.tool()
async def query_hf_model_files(
    repo_id: str,
    repo_type: str = "model",
    revision: str = "main",
    patterns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Query HuggingFace Hub for files in a repository.

    Args:
        repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b-hf")
        repo_type: Type of repository (default: "model", options: "model", "dataset", "space")
        revision: Git revision (default: "main")
        patterns: Optional list of glob patterns to filter files (e.g., ["*.safetensors", "*.json"])

    Returns:
        List of files in the repository with metadata
    """
    from huggingface_hub import HfApi
    from fnmatch import fnmatch
    from huggingface_hub.hf_api import RepoFile, RepoFolder
    from dataclasses import asdict
    try:
        api = HfApi()
        file_infos = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)

        # Filter by patterns if provided
        if patterns:
            filtered_files = []
            for file_path in file_infos:
                if any(fnmatch(file_path, pattern) for pattern in patterns):
                    filtered_files.append(file_path)
            file_infos = filtered_files

        # Get file info for each file
        files_data = []
        for file_path in file_infos[:100]:  # Limit to 100 files
            try:
                info = api.get_paths_info(repo_id=repo_id, paths=[file_path], repo_type=repo_type, revision=revision)
                if info:
                    file_info: RepoFile | RepoFolder = info[0]
                    files_data.append(asdict(file_info))
            except:
                files_data.append({"path": file_path, "size": 0})

        total_size = sum(f["size"] for f in files_data)

        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "revision": revision,
            "file_count": len(files_data),
            "total_size_bytes": total_size,
            "total_size_gb": round(total_size / (1024**3), 2),
            "files": files_data,
        }
    except Exception as e:
        raise ValueError(f"Failed to query HuggingFace Hub: {str(e)}")


@mcp.tool()
async def search_hf_hub_models(
    query: str,
    limit: int = 20,
    model_filter: str | None = None,
) -> dict[str, Any]:
    """
    Search for models on HuggingFace Hub.

    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 20, max: 50)
        model_filter: Optional filter (e.g., "task:text-generation", "library:transformers")

    Returns:
        List of matching models from HuggingFace Hub
    """
    if limit > 50:
        limit = 50

    from huggingface_hub import HfApi

    api = HfApi()

    # Parse filter
    filter_dict = {}
    if model_filter:
        if ":" in model_filter:
            key, value = model_filter.split(":", 1)
            filter_dict[key] = value

    models = api.list_models(
        search=query,
        limit=limit,
        **filter_dict,
    )

    results = []
    for model in models:
        results.append(asdict(model))

    return {
        "query": query,
        "count": len(results),
        "models": results,
    }


@mcp.tool()
async def get_hf_model_info(repo_id: str) -> dict[str, Any]:
    """
    Get detailed information about a model from HuggingFace Hub.

    Args:
        repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b-hf")

    Returns:
        Detailed model information including README, tags, metrics
    """
    from huggingface_hub import HfApi

    return asdict(HfApi().model_info(repo_id))


# ============================================================================
# Agent Execution Tools
# ============================================================================


@mcp.tool()
async def run_agent(
    objective: str,
    provider: Provider,
    model: str = "gpt-4o",
    tools: list[str] = [],
    output_schema: dict[str, Any] | None = None,
    enable_analysis_phase: bool = False,
    enable_data_contracts_phase: bool = False,
) -> dict[str, Any]:
    """
    Execute a NodeTool agent to perform autonomous task execution.

    Agents can use various tools to accomplish objectives like web search,
    browsing, email access, and more. They autonomously plan and execute tasks
    based on the objective you provide.

    Args:
        objective: The task description for the agent to accomplish
        provider: AI provider (default: "openai"). Options: "openai", "anthropic",
                 "ollama", "gemini", "huggingface_cerebras", etc.
        model: Model to use (default: "gpt-4o")
        tools: List of tool names to enable. Options:
               - "google_search": Search the web using Google
               - "browser": Browse and extract content from web pages
               - "email": Search and read emails
               - [] (empty): Agent runs without external tools
        output_schema: Optional JSON schema to structure the agent's output
        enable_analysis_phase: Enable analysis phase for complex reasoning (default: False)
        enable_data_contracts_phase: Enable data contract validation (default: False)

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - results: The agent's final output (string or structured data if output_schema provided)
        - workspace_dir: Path to workspace directory with any generated files
        - error: Error message if status is "error"

    Example:
        ```python
        # Simple web research task
        result = await run_agent(
            objective="Find 3 trending AI workflows on Reddit and summarize them",
            tools=["google_search", "browser"]
        )

        # Structured output task
        result = await run_agent(
            objective="Find chicken wing recipes from 3 different websites",
            tools=["google_search", "browser"],
            output_schema={
                "type": "object",
                "properties": {
                    "recipes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "url": {"type": "string"},
                                "ingredients": {"type": "array", "items": {"type": "string"}},
                                "instructions": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                }
            }
        )
        ```
    """
    try:
        context = ProcessingContext()

        # Map tool names to tool instances
        tool_instances = []
        tool_map = {
            "google_search": GoogleSearchTool,
            "browser": BrowserTool,
            "email": SearchEmailTool,
        }

        for tool_name in tools:
            if tool_name in tool_map:
                tool_instances.append(tool_map[tool_name]())
            else:
                log.warning(f"Unknown tool: {tool_name}, skipping")

        provider_instance = get_provider(provider)

        # Create and execute agent
        agent = Agent(
            name=f"Agent: {objective[:50]}...",
            objective=objective,
            provider=provider_instance,
            model=model,
            tools=tool_instances,
            output_schema=output_schema,
            enable_analysis_phase=enable_analysis_phase,
            enable_data_contracts_phase=enable_data_contracts_phase,
        )

        # Execute agent and collect output
        output_chunks = []
        events = []
        async for event in agent.execute(context):
            if isinstance(event, Chunk):
                output_chunks.append(event.content)
            else:
                events.append(event.model_dump())


        # Get final results
        results = agent.get_results()

        return {
            "status": "success",
            "results": results,
            "events": events,
            "workspace_dir": str(context.workspace_dir),
        }

    except Exception as e:
        log.error(f"Agent execution failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def run_web_research_agent(
    query: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    num_sources: int = 3,
) -> dict[str, Any]:
    """
    Run a specialized agent for web research tasks.

    This is a convenience tool that configures an agent with Google Search
    and Browser tools for research tasks.

    Args:
        query: The research query or objective
        provider: AI provider (default: "openai")
        model: Model to use (default: "gpt-4o")
        num_sources: Number of sources to research (default: 3)

    Returns:
        Dictionary with research results and workspace directory

    Example:
        ```python
        result = await run_web_research_agent(
            query="What are the latest developments in AI agent frameworks?",
            num_sources=5
        )
        ```
    """
    objective = f"""
    Research the following topic by finding and analyzing {num_sources} relevant web sources:

    {query}

    For each source:
    1. Use Google Search to find relevant URLs
    2. Use Browser tool to extract content from each URL
    3. Summarize key findings

    Provide a comprehensive summary with citations.
    """

    return await run_agent(
        objective=objective,
        provider=provider,
        model=model,
        tools=["google_search", "browser"],
    )


@mcp.tool()
async def run_email_agent(
    task: str,
    provider: str = "openai",
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """
    Run a specialized agent for email-related tasks.

    This agent can search and analyze emails using the SearchEmailTool.

    Args:
        task: The email task description (e.g., "Find emails about AI from last week")
        provider: AI provider (default: "openai")
        model: Model to use (default: "gpt-4o")

    Returns:
        Dictionary with task results and workspace directory

    Example:
        ```python
        result = await run_email_agent(
            task="Search for emails with 'project update' in subject from last 7 days and summarize them"
        )
        ```
    """
    return await run_agent(
        objective=task,
        provider=provider,
        model=model,
        tools=["email"],
    )


@mcp.prompt()
async def build_workflow_guide() -> str:
    """
    Comprehensive guide on how to build NodeTool workflows.

    Returns:
        Step-by-step instructions for creating workflows
    """
    return """# How to Build NodeTool Workflows

## Overview
NodeTool workflows are visual, node-based programs that connect operations together. Each workflow is a **Directed Acyclic Graph (DAG)** where nodes process data and pass results through typed connections.

## Running Workflows

NodeTool provides three ways to execute workflows:

### 1. Synchronous Execution with `run_workflow_tool`
- **Use when**: You want to wait for results and get outputs immediately from a saved workflow
- **Behavior**: Blocks until workflow completes
- **Returns**: Final output values from all output nodes
- **Requires**: Workflow must be saved first with `save_workflow`
- **Example**:
  ```python
  result = await run_workflow_tool(
      workflow_id="abc123",
      params={"input_text": "Hello world"}
  )
  # Returns: {"status": "completed", "result": {"output_1": "..."}}
  ```

### 2. Direct Graph Execution with `run_graph`
- **Use when**: Testing graph structures without saving, or running one-off executions
- **Behavior**: Blocks until graph execution completes
- **Returns**: Final output values from all output nodes
- **Does not require**: Saving the workflow first - executes graph directly
- **Example**:
  ```python
  result = await run_graph(
      graph={
          "nodes": [
              {
                  "id": "input1",
                  "type": "nodetool.input.StringInput",
                  "data": {"name": "text", "value": ""}
              },
              {
                  "id": "agent1",
                  "type": "nodetool.agents.Agent",
                  "data": {
                      "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"}
                  }
              }
          ],
          "edges": [
              {
                  "source": "input1",
                  "sourceHandle": "output",
                  "target": "agent1",
                  "targetHandle": "prompt"
              }
          ]
      },
      params={"text": "Hello, how are you?"}
  )
  # Returns: {"status": "completed", "result": {...}}
  ```

### 3. Background Execution with `start_background_job`
- **Use when**: Running long workflows that you want to monitor asynchronously
- **Behavior**: Returns immediately with a job_id, workflow continues in background
- **Returns**: Job ID for status tracking
- **Requires**: Workflow must be saved first with `save_workflow`
- **Monitor with**: `get_job()`, `list_running_jobs()`, `get_job_logs()`
- **Example**:
  ```python
  job = await start_background_job(
      workflow_id="abc123",
      params={"input_text": "Hello world"}
  )
  # Returns: {"job_id": "xyz789", "status": "running"}

  # Check status later
  status = await get_job(job_id="xyz789")
  logs = await get_job_logs(job_id="xyz789", limit=50)
  ```

### Choosing the Right Execution Method

**Use `run_workflow_tool` for:**
- Quick operations that complete in seconds
- When you need results immediately
- Simple workflows with direct outputs
- Running saved, tested workflows

**Use `run_graph` for:**
- Testing workflow graphs before saving
- One-off executions that don't need to be persisted
- Rapid prototyping and experimentation
- When you don't want to clutter the workflow database

**Use `start_background_job` for:**
- Long-running workflows (minutes to hours)
- Workflows that process large datasets
- Jobs you want to monitor with logs
- When you need to start multiple workflows in parallel

## Working with Assets

Assets are files and folders stored in NodeTool's asset system. They provide a way to manage media files (images, audio, video) and documents that workflows can access.

### Asset Types
- **folder**: Directory containers for organizing assets
- **image**: Image files (JPG, PNG, GIF, WebP, etc.)
- **video**: Video files (MP4, AVI, MOV, etc.)
- **audio**: Audio files (MP3, WAV, FLAC, etc.)
- **text**: Text documents (TXT, MD, etc.)
- **document**: Documents (PDF, DOCX, etc.)
- **dataframe**: Tabular data (CSV, Parquet, etc.)

### Working with Assets via MCP

**List assets in a folder:**
```python
# List root assets
assets = await list_assets()

# List assets in a specific folder
assets = await list_assets(parent_id="folder_id_123")

# Filter by content type
images = await list_assets(content_type="image")
```

**Search for assets:**
```python
results = await search_assets(
    query="screenshot",
    content_type="image",
    limit=20
)
# Returns assets with folder path information
```

**Get asset details:**
```python
asset = await get_asset(asset_id="asset_123")
# Returns: URLs for accessing the file, metadata, size, etc.
```

**Create folders:**
```python
folder = await create_folder(
    name="My Images",
    parent_id="parent_folder_id"  # Optional, defaults to root
)
```

**Update assets:**
```python
updated = await update_asset(
    asset_id="asset_123",
    name="New Name",
    parent_id="new_folder_id",  # Move to different folder
    metadata={"tags": ["important"], "description": "..."}
)
```

**Delete assets:**
```python
result = await delete_asset(asset_id="asset_123")
# For folders, recursively deletes all contents
```

### Using Assets in Workflows

Assets are referenced in workflows using typed references:

**ImageRef example:**
```json
{
  "id": "image_input",
  "type": "nodetool.constant.Image",
  "data": {
    "value": {
      "type": "image",
      "uri": "file:///path/to/image.jpg",
      "asset_id": "asset_123"
    }
  }
}
```

**Common asset workflow patterns:**

1. **Load asset from storage:**
   - Use `get_asset()` to get asset details and URLs
   - Pass asset reference to image/audio/video input nodes

2. **Process and save results:**
   - Workflow outputs create new assets automatically
   - Assets are saved to the workspace and associated with the job

3. **Batch processing:**
   - Use `list_assets(content_type="image")` to get all images
   - Create workflow that processes each image
   - Run multiple jobs with `start_background_job()`

**Example: Processing all images in a folder:**
```python
# Get all images from a folder
assets = await list_assets(
    parent_id="folder_id",
    content_type="image"
)

# Start a job for each image
job_ids = []
for asset in assets["assets"]:
    job = await start_background_job(
        workflow_id="image_processor_workflow",
        params={"image_url": asset["get_url"]}
    )
    job_ids.append(job["job_id"])

# Monitor all jobs
for job_id in job_ids:
    status = await get_job(job_id)
    print(f"Job {job_id}: {status['status']}")
```

## Graph Structure

Workflows consist of **nodes** and **edges** that form a directed acyclic graph.

### Node Structure
```json
{
  "id": "string",                    // Unique identifier for the node (required)
  "parent_id": "string" | null,      // Parent group ID (for grouped nodes, optional)
  "type": "string",                  // Fully qualified node type (e.g., "nodetool.text.Split")
  "data": {},                        // Node configuration and properties
  "ui_properties": {},               // UI-specific properties (position, etc.)
  "dynamic_properties": {},          // Dynamic properties for flexible nodes (FormatText, MakeDictionary)
  "dynamic_outputs": {},             // Dynamic output definitions
  "sync_mode": "on_any"              // Execution sync mode: "on_any" or "zip_all"
}
```

### Common Node Types Reference

**Input Nodes:**
- `nodetool.input.StringInput` - Text input with default value
- `nodetool.input.ImageInput` - Image file input
- `nodetool.input.GroupInput` - Input for Group nodes (automatic iteration)

**Constant Nodes:**
- `nodetool.constant.String` - Static text value
- `nodetool.constant.Image` - Static image reference
- `nodetool.constant.Integer` - Static number

**Text Processing:**
- `nodetool.text.FormatText` - Template formatting with dynamic variables
- `nodetool.text.Concat` - Concatenate two strings
- `nodetool.text.Slice` - Extract substring (start, stop, step)
- `nodetool.text.HtmlToText` - Convert HTML to plain text

**AI/LLM Nodes:**
- `nodetool.agents.Agent` - LLM agent with tool calling support
- `nodetool.generators.ListGenerator` - Generate list of items with LLM

**Data Processing:**
- `nodetool.dictionary.GetValue` - Extract value from dictionary by key
- `nodetool.dictionary.MakeDictionary` - Create dictionary with dynamic keys
- `lib.mail.GmailSearch` - Search Gmail (requires authentication)

**Image Generation:**
- `mlx.mflux.MFlux` - FLUX image generation (Apple Silicon)
- `nodetool.image.Replicate` - Replicate API image generation

**Organization:**
- `nodetool.workflows.base_node.Group` - Container for iterating over lists
- `nodetool.output.GroupOutput` - Output from Group iteration
- `nodetool.workflows.base_node.Preview` - Display intermediate results
- `nodetool.workflows.base_node.Comment` - Documentation/annotations

### Edge Structure
```json
{
  "id": "string" | null,             // Optional edge identifier
  "source": "string",                // Source node ID (required)
  "sourceHandle": "string",          // Output handle name on source node (required)
  "target": "string",                // Target node ID (required)
  "targetHandle": "string",          // Input handle name on target node (required)
  "ui_properties": {}                // UI-specific properties (className for type, etc.)
}
```

### Graph Structure
```json
{
  "nodes": [                         // List of all nodes in the workflow
    { /* Node objects */ }
  ],
  "edges": [                         // List of all edges connecting nodes
    { /* Edge objects */ }
  ]
}
```

### Key Concepts

1. **Node IDs**: Must be unique across the entire workflow
2. **Node Types**: Use exact fully qualified names from registry (e.g., "nodetool.text.Split")
3. **Edges**: Connect outputs to inputs using source/target node IDs and handle names
4. **Parent ID**: Used for grouping nodes (e.g., nodes inside a Group node)
5. **Dynamic Properties**: Some nodes (like FormatText, MakeDictionary) accept arbitrary properties
6. **Data vs Properties**: Node configuration goes in `data` field, edge connections can be in properties

## Streaming Nodes and Iteration

NodeTool implements **iteration through streaming** - nodes emit multiple values one at a time, and downstream nodes automatically process each emitted value in sequence. This is the core mechanism for loops and batch processing in NodeTool.

### How Iteration Works

**Key Concept:** When a streaming node emits multiple values, each downstream node automatically executes once per emitted value. This creates implicit iteration without explicit loop constructs.

**Regular (Non-Streaming) Nodes:**
- Process inputs once and return a single output
- Example: Text nodes that transform one string to another
- Execute once per workflow run

**Streaming Nodes (Iteration Generators):**
- Emit multiple outputs over time, one after another
- Each emission triggers downstream nodes to execute again
- This creates automatic iteration - downstream nodes run once per emitted item
- Common streaming nodes:
  - `nodetool.generators.ListGenerator` - Streams text items from a list (iteration source)
  - `nodetool.generators.DataGenerator` - Streams dataframe records (iteration source)
  - Agent nodes with streaming enabled

**Important:** Outputs can occur multiple times when streaming. For example:
- A ListGenerator emits 3 items
- A downstream Preview node connected to it will output 3 times (once per item)
- When accumulated/collected, these become a list of all outputs

### Streaming Node Examples

**ListGenerator** - Generates a stream of text items:
```json
{
  "id": "list_gen",
  "type": "nodetool.generators.ListGenerator",
  "data": {
    "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"},
    "prompt": "Generate 5 creative movie poster taglines",
    "max_tokens": 2048
  }
}
```
- Outputs: `item` (string) and `index` (int) for each generated item
- Each item is streamed as it's generated by the LLM

**DataGenerator** - Generates structured data as a stream:
```json
{
  "id": "data_gen",
  "type": "nodetool.generators.DataGenerator",
  "data": {
    "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"},
    "prompt": "Generate customer data",
    "columns": {
      "columns": [
        {"name": "name", "data_type": "str"},
        {"name": "email", "data_type": "str"},
        {"name": "age", "data_type": "int"}
      ]
    }
  }
}
```
- Outputs: `record` (dict) for each row, then final `dataframe` (DataframeRef)
- Records are streamed as they're generated, dataframe emitted at the end

**Agent** - Can stream text chunks AND produce final complete text:
```json
{
  "id": "story_agent",
  "type": "nodetool.agents.Agent",
  "data": {
    "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"},
    "prompt": "Write a short story about a robot",
    "max_tokens": 1000
  }
}
```
- Outputs: `chunk` (string) for each token/chunk as it's generated in real-time
- Then outputs: `text` (string) with the complete final text

### Using Streaming for Iteration

**Pattern 1: Simple Iteration**
```
ListGenerator → ProcessNode → Preview
```
- ListGenerator emits items: "A", "B", "C"
- ProcessNode executes 3 times (once for "A", once for "B", once for "C")
- Preview outputs 3 times, creating a list of results when accumulated

**Pattern 2: Nested Iteration**
When a streaming node is connected to another streaming node, you get nested loops:
```
A -> B -> C
A streams 1,2,3
B streams all items for 1 (e.g., "a", "b")
B streams all items for 2 (e.g., "c", "d")
B streams all items for 3 (e.g., "e", "f")
Result at C: processes "a", "b", "c", "d", "e", "f" (6 executions)
```

**Pattern 3: Stream Synchronization**
When a stream gets split into multiple paths, and you want to combine the results, set sync_mode to "zip_all" on the downstream node:
```
A -> B -> D
  -> C -> D

A emits: 1, 2, 3
D with sync_mode="zip_all" receives paired items: (B1,C1), (B2,C2), (B3,C3)
```

**Pattern 4: Group Processing (Explicit Iteration)**
Group nodes provide explicit iteration control:
```
DataSource → Group(ProcessingChain) → CollectResults
```
- Group receives list from DataSource
- GroupInput receives each item sequentially
- Processing chain runs once per item
- GroupOutput accumulates all results

### Important Notes

- **Streaming is automatic iteration** - You don't configure streaming; nodes are either streaming or not
- **All items are processed sequentially** - Downstream nodes execute once per streamed item
- **Order is preserved** - Items arrive and process in the order they're emitted
- **Outputs accumulate** - Each execution adds to the output list
- **Use Preview nodes strategically** - They show accumulated results from all iterations
- **Output nodes capture results** - To capture workflow outputs, you MUST use one of:
  - Output nodes: `nodetool.output.TextOutput`, `nodetool.output.ImageOutput`, etc.
  - Preview nodes: `nodetool.workflows.base_node.Preview`
  - Save to Assets nodes: `nodetool.image.SaveImage`, `nodetool.audio.SaveAudio`, `nodetool.text.SaveText`, `nodetool.video.SaveVideo`, `nodetool.data.SaveDataframe`
  - Save to File nodes: `nodetool.image.SaveImageFile`, `nodetool.audio.SaveAudioFile`, `nodetool.text.SaveTextFile`, `nodetool.video.SaveVideoFile`, `nodetool.data.SaveCSVDataframeFile`
  - Processing nodes alone don't return results to the workflow caller

## Dynamic Properties Pattern

Some nodes accept **dynamic properties** - custom inputs defined at workflow creation time. This is useful for creating flexible, reusable nodes.

### FormatText - Template with Dynamic Variables

The most common dynamic properties pattern is **template formatting** using `FormatText`:

```json
{
  "nodes": [
    {
      "id": "movie_title",
      "type": "nodetool.input.StringInput",
      "data": {
        "name": "Movie Title",
        "value": "Stellar Odyssey"
      }
    },
    {
      "id": "genre",
      "type": "nodetool.input.StringInput",
      "data": {
        "name": "Genre",
        "value": "Sci-Fi"
      }
    },
    {
      "id": "format_prompt",
      "type": "nodetool.text.FormatText",
      "data": {
        "template": "Write a movie poster tagline for {{TITLE}}, a {{GENRE}} film."
      },
      "dynamic_properties": {
        "TITLE": "",
        "GENRE": ""
      }
    }
  ],
  "edges": [
    {
      "source": "movie_title",
      "sourceHandle": "output",
      "target": "format_prompt",
      "targetHandle": "TITLE"
    },
    {
      "source": "genre",
      "sourceHandle": "output",
      "target": "format_prompt",
      "targetHandle": "GENRE"
    }
  ]
}
```

**How it works:**
1. Define template with `{{VARIABLE_NAME}}` placeholders in the `template` field
2. Add each variable to `dynamic_properties` object (values are typically empty strings)
3. Connect edges to the dynamic property names as `targetHandle`
4. FormatText replaces `{{VARIABLE_NAME}}` with the connected input values
5. Output: "Write a movie poster tagline for Stellar Odyssey, a Sci-Fi film."

### Key Points

- **Template variables**: Must match exactly between `{{VAR}}` and `dynamic_properties` key
- **Case sensitive**: `{{TITLE}}` ≠ `{{title}}`
- **No defaults in dynamic_properties**: Values are typically empty strings; actual data comes from edges
- **Edge connections required**: Each dynamic property should have a connected edge
- **Common nodes with dynamic properties**:
  - `nodetool.text.FormatText` - Template formatting
  - `nodetool.dictionary.MakeDictionary` - Dynamic dictionaries
  - Some agent/generator nodes - Custom parameters

## Dynamic Outputs Pattern (Tool Calling)

Agent nodes can define **dynamic outputs** that become **tool calls**. Each dynamic output creates a new output handle on the agent that triggers when the agent decides to call that tool.

### Agent with Tool Calls

```json
{
  "nodes": [
    {
      "id": "weather_agent",
      "type": "nodetool.agents.Agent",
      "data": {
        "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"},
        "prompt": "What's the weather like in Paris?",
        "max_tokens": 1000
      },
      "dynamic_outputs": ["get_weather", "get_forecast"]
    },
    {
      "id": "weather_tool",
      "type": "nodetool.constant.String",
      "data": {
        "value": "Current weather in Paris: 18°C, partly cloudy"
      }
    },
    {
      "id": "tool_result",
      "type": "nodetool.agents.ToolResult",
      "data": {}
    }
  ],
  "edges": [
    {
      "source": "weather_agent",
      "sourceHandle": "get_weather",
      "target": "weather_tool",
      "targetHandle": "trigger"
    },
    {
      "source": "weather_tool",
      "sourceHandle": "output",
      "target": "tool_result",
      "targetHandle": "result"
    },
    {
      "source": "tool_result",
      "sourceHandle": "output",
      "target": "weather_agent",
      "targetHandle": "tool_result"
    }
  ]
}
```

**How tool calling works:**

1. **Define tools in dynamic_outputs**: Each output becomes a tool with the given type.

2. **Agent decides to call tool**: When the agent wants to use a tool, it outputs on that handle
   - Example: Agent outputs `{"city": "Paris"}` on the `get_weather` handle

3. **Tool chain executes**: The tool call messages flow through your processing nodes

4. **ToolResult gathers results**: Collects output from tool processing

5. **Result fed back to agent**: ToolResult connects back to agent's `tool_result` input
   - Agent receives the tool result and continues its reasoning
   - Agent can make more tool calls or produce final answer

### Multi-Tool Example

```json
{
  "id": "research_agent",
  "type": "nodetool.agents.Agent",
  "data": {
    "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"},
    "prompt": "Research the latest news about AI and summarize it",
    "system": "You are a research assistant with access to web search and calculator tools."
  },
  "dynamic_outputs": ["web_search", "calculator"]
}
```

**Tool chain for web_search:**
```
Agent[web_search] → WebSearch → FormatResults → ToolResult
```

**Tool chain for calculator:**
```
Agent[calculator] → Evaluate → FormatResult → ToolResult
```

### Important Notes

- **Each tool needs a chain**: Every dynamic output should have a processing chain ending in ToolResult
- **ToolResult is required**: Always use ToolResult node to feed results back to the agent
- **Agent can call multiple times**: Agent may call tools multiple times before producing final answer
- **Tool schemas are shown to LLM**: The LLM sees tool descriptions and decides when to call them
- **Streaming and tools**: Agents with tools typically don't stream; they work in request/response cycles
- **No circular dependencies**: Tool chains must not create cycles in the workflow graph

## Working with Files and Assets

NodeTool workflows can work with files in two ways: **assets** (managed files) and **direct file access**. For AI agents generating workflows, **use direct file paths** unless you specifically need asset management features.

### For AI Agents: Use Direct File Paths

When generating workflows programmatically, use direct file paths:

```json
{
  "type": "image",
  "uri": "/absolute/path/to/image.jpg"
}
```

**Guidelines for agents:**
- Use absolute paths: `/home/user/image.jpg` (not relative paths)
- Files can be anywhere accessible to the system
- No need to manage asset IDs or import files first
- Simpler and more predictable for automated workflow generation

### Assets vs Direct Files

**Assets** are files imported through NodeTool's UI that get:
- Asset IDs for tracking and management
- Thumbnails and metadata
- Organization in the asset editor
- Version control

**Direct file access** is simpler for agents:
- Reference any accessible file directly
- No import step required
- Works with generated or temporary files
- Less overhead for automated workflows

### File Reference Types

All file references use this structure:
```json
{
  "type": "image|audio|video|document|dataframe",
  "uri": "file:///path/to/file.ext",
  "asset_id": "uuid"  // Optional: only present for imported assets
}
```

**Examples:**
- Image: `{"type": "image", "uri": "file:///data/photo.jpg"}`
- Audio: `{"type": "audio", "uri": "file:///sounds/music.mp3"}`
- Document: `{"type": "document", "uri": "file:///docs/report.pdf"}`
- Data: `{"type": "dataframe", "uri": "file:///data/dataset.csv"}`

### Working with Existing Workflows (Assets)

When **reading or modifying existing workflows**, you may encounter assets:

**Asset references** include an `asset_id`:
```json
{
  "type": "image",
  "uri": "file:///workspace/assets/image.jpg",
  "asset_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Guidelines for agents working with existing workflows:**
- **Preserve asset references** when modifying workflows - don't convert them to direct paths unless necessary
- **Check for asset_id field** to identify assets vs direct file references
- **Assets provide stability** - the same asset can be referenced across multiple workflows
- **Asset paths may use workspace URIs** like `file:///workspace/assets/`

**When to convert assets to direct paths:**
- When you need to work with files outside the workspace
- When creating portable workflows that shouldn't depend on specific assets
- When the asset no longer exists or is inaccessible

## Core Principles

1. **Graph Structure**: Workflows must be DAGs (no circular dependencies)
2. **Data Flow**: Edges represent data flow from inputs → processing → outputs
3. **Type Safety**: Connections must be type-compatible
4. **Complete Connectivity**: Every processing node must receive its required inputs
5. **Valid Node Types**: All nodes must exist in the registry (use `search_nodes` to verify)
6. **Explicit Outputs**: Workflows must have Output, Preview, or Save nodes to capture results - processing nodes alone don't return values to the caller

## Workflow Planning Process

### Phase 1: Understand the Objective
1. **Identify inputs**: What data does the workflow receive?
2. **Identify outputs**: What results should it produce?
3. **Map the transformation**: What processing steps convert inputs to outputs?

### Phase 2: Design the Data Flow
1. **Start with I/O nodes**: Create Input and Output nodes first
2. **Identify processing steps**: What transformations are needed?
3. **Search for nodes efficiently**:
   - Use specific queries with `input_type` and `output_type` filters
   - Batch similar searches together
   - Target relevant namespaces (`nodetool.text`, `nodetool.data`, `lib.*`)
4. **Plan connections**: Trace data flow from inputs through processing to outputs

### Phase 3: Build the Graph
1. **Create node specifications**:
   - Use exact `node_type` from `search_nodes` results
   - Configure properties based on node metadata
   - Define edge connections in properties
2. **Validate connectivity**:
   - Every Input node connects to at least one processing node
   - Every processing node has ALL required inputs connected
   - Every workflow MUST have Output, Preview, or Save nodes to capture results
   - Output/Preview/Save nodes receive data from processing nodes
3. **Check types**: Ensure all connections are type-compatible

## Node Search Strategy

### Efficient Searching
**Minimize search iterations** by being strategic:

- **Plan ahead**: Identify all processing types needed before searching
- **Use type filters**: Specify `input_type` and `output_type` when known
- **Batch searches**: Search for multiple related capabilities together
- **Target namespaces**:
  - `nodetool.text.*`: Text operations (split, concat, format, slice)
  - `nodetool.data.*`: DataFrame/tabular data operations
  - `nodetool.agents.*`: AI agents and generators
  - `lib.browser.*`: Web scraping and fetching
  - `lib.mail.*`: Email operations

### Example Searches
```
# Good: Specific with type filters
search_nodes(query=["dataframe", "group", "aggregate"], input_type="dataframe")

# Good: Targeted functionality
search_nodes(query=["text", "template", "format"], output_type="str")

# Less efficient: Too generic
search_nodes(query=["process"])
```

## Data Types and Compatibility

### Common Types
- **str**: Text data
- **int/float**: Numeric values (interchangeable)
- **bool**: Boolean values
- **list[T]**: Arrays of items
- **dict**: Key-value objects
- **any**: Compatible with all types

### Media Types
- **ImageRef**: Image file references
- **AudioRef**: Audio file references
- **VideoRef**: Video file references
- **DocumentRef**: Document file references
- **DataFrameRef**: Tabular data references

### Type Compatibility Rules
- Exact matches always work: `str → str`, `int → int`
- Numeric conversions allowed: `int ↔ float`
- `any` type accepts everything
- Union types match if any member matches
- Complex types need converter nodes for primitives

## Node Configuration Patterns

### Static Properties
Standard node properties defined in metadata:
```json
{
  "node_id": "text_slice",
  "node_type": "nodetool.text.Slice",
  "properties": "{\"start\": 0, \"stop\": 100}"
}
```

### Edge Connections
Connect node outputs to inputs using edge definitions:
```json
{
  "properties": "{\"text\": {\"type\": \"edge\", \"source\": \"input_1\", \"sourceHandle\": \"output\"}}"
}
```

### Dynamic Properties
Some nodes accept custom properties (check `is_dynamic` in metadata):
```json
{
  "node_id": "make_dict",
  "node_type": "nodetool.dictionary.MakeDictionary",
  "properties": "{\"category\": {\"type\": \"edge\", \"source\": \"agent_1\", \"sourceHandle\": \"text\"}, \"body\": \"example\"}"
}
```

## Common Workflow Patterns

### Sequential Processing
```
Input → Process A → Process B → Output
```
Use when: Each step depends on the previous result

### Parallel Processing
```
       → Process A →
Input →              → Combine → Output
       → Process B →
```
Use when: Multiple independent operations on the same input

### List Processing with Streams
```
Data Source → Streaming Node → Collect Results
```
Use when: Applying operations to each item in a list

### Multi-Agent Pipelines
```
Input → Agent 1 (Strategy) → Agent 2 (Transform) → Generator (Variations) → Output
```
Use when: Complex AI tasks requiring multiple reasoning steps

## Validation Checklist

Before running a workflow, verify:

**Structure**
- [ ] All node types exist (verified via `search_nodes`)
- [ ] All node IDs are unique
- [ ] No circular dependencies (DAG structure)

**Connectivity**
- [ ] Every Input node connects to processing nodes
- [ ] Every processing node has ALL required inputs connected
- [ ] Workflow has Output, Preview, or Save nodes to capture results
- [ ] Output/Preview/Save nodes receive data from processing nodes
- [ ] Template nodes have connections for ALL template variables

**Types**
- [ ] All edge connections are type-compatible
- [ ] Input/Output nodes match the workflow schema types
- [ ] Complex type conversions use appropriate converter nodes

**Properties**
- [ ] Non-dynamic nodes only use properties from their metadata
- [ ] Dynamic nodes have valid property configurations
- [ ] Required properties are set (check node metadata)

## Common Mistakes to Avoid

1. **Missing edge connections**: Forgetting to connect required inputs
2. **Type mismatches**: Connecting incompatible types without converters
3. **Using wrong node_type**: Not using exact string from `search_nodes`
4. **Orphaned nodes**: Nodes not connected to the data flow
5. **Missing Output nodes**: Not creating Output, Preview, or Save nodes to capture results
6. **Missing I/O nodes**: Not creating nodes for schema inputs/outputs
7. **Template variables without connections**: Using variables in templates without edges
8. **Inventing properties**: Adding properties to non-dynamic nodes

## Tools for Building

- **`search_nodes`**: Find nodes by functionality, filter by types
- **`get_node_info`**: Get detailed specifications for a specific node
- **`list_workflows`**: Browse existing workflow examples
- **`get_workflow`**: Examine a specific workflow's structure
- **`run_workflow_tool`**: Execute and test your workflow

## Next Steps

1. **Define objective**: What should the workflow accomplish?
2. **Plan I/O**: What inputs/outputs are needed?
3. **Search nodes**: Find processing nodes using targeted searches
4. **Design flow**: Map out the data transformations
5. **Build graph**: Create node specifications with connections
6. **Validate**: Check structure, connectivity, and types
7. **Test**: Run the workflow and iterate based on results
"""


@mcp.prompt()
async def job_monitoring_guide() -> str:
    """
    Guide for monitoring and debugging workflow jobs.

    Returns:
        Instructions for tracking job execution and accessing logs
    """
    return """# Job Monitoring and Debugging Guide

## Overview

NodeTool provides comprehensive job tracking and logging capabilities for monitoring workflow executions, especially useful for long-running background jobs.

## Job Lifecycle

1. **Starting** - Job is being initialized
2. **Running** - Job is actively executing
3. **Completed** - Job finished successfully
4. **Error** - Job failed with an error
5. **Cancelled** - Job was cancelled by user

## Monitoring Tools

### 1. Get Job Status - `get_job(job_id)`

Get detailed information about a specific job:

```python
job = await get_job(
    job_id="job_123",
    include_logs=True  # Optional: includes log count
)

# Returns:
{
    "id": "job_123",
    "status": "running",
    "workflow_id": "workflow_abc",
    "started_at": "2024-01-15T10:30:00",
    "finished_at": None,  # null if still running
    "error": None,
    "cost": 0.05,
    "log_count": 42,  # if include_logs=True
    "has_logs": True
}
```

### 2. List All Jobs - `list_jobs(workflow_id, limit, start_key)`

List jobs with optional filtering:

```python
# List all jobs for user
jobs = await list_jobs(limit=50)

# List jobs for specific workflow
jobs = await list_jobs(workflow_id="workflow_abc")

# Pagination
jobs = await list_jobs(start_key="last_job_id", limit=20)
```

### 3. List Running Jobs - `list_running_jobs()`

Get all currently active background jobs:

```python
running = await list_running_jobs()

# Returns list of active jobs with status
[
    {
        "job_id": "job_123",
        "status": "running",
        "workflow_id": "workflow_abc",
        "created_at": "2024-01-15T10:30:00",
        "is_running": True,
        "is_completed": False
    }
]
```

### 4. Get Job Logs - `get_job_logs(job_id, limit, include_live)`

Retrieve execution logs from a job:

```python
# Get all logs from a job
logs = await get_job_logs(job_id="job_123")

# Get most recent 50 logs
logs = await get_job_logs(job_id="job_123", limit=50)

# Only get persisted logs (not live)
logs = await get_job_logs(
    job_id="job_123",
    include_live=False
)

# Returns:
{
    "job_id": "job_123",
    "status": "running",
    "is_running": True,
    "logs": [
        {
            "timestamp": "2024-01-15T10:30:01.123",
            "level": "INFO",
            "logger": "nodetool.workflows",
            "message": "Starting workflow execution",
            "module": "workflow_runner",
            "function": "run",
            "line": 42
        },
        {
            "timestamp": "2024-01-15T10:30:02.456",
            "level": "INFO",
            "logger": "nodetool.nodes",
            "message": "Processing node: text_processor",
            "module": "base_node",
            "function": "process",
            "line": 156
        }
    ],
    "total_logs": 42
}
```

### 5. Cancel Job - `cancel_job(job_id)`

Stop a running job:

```python
result = await cancel_job(job_id="job_123")

# Returns:
{
    "success": True,
    "message": "Job cancelled successfully",
    "job_id": "job_123"
}
```

## Common Monitoring Patterns

### Pattern 1: Poll Job Until Complete

```python
import asyncio

job = await start_background_job(
    workflow_id="long_workflow",
    params={"input": "data"}
)

# Poll every 5 seconds
while True:
    status = await get_job(job["job_id"])

    if status["status"] == "completed":
        print("Job completed successfully!")
        break
    elif status["status"] == "error":
        print(f"Job failed: {status['error']}")
        break

    print(f"Job status: {status['status']}")
    await asyncio.sleep(5)
```

### Pattern 2: Monitor Logs in Real-Time

```python
import asyncio

job = await start_background_job(workflow_id="workflow_abc")
last_log_count = 0

while True:
    logs = await get_job_logs(job["job_id"], include_live=True)

    # Print new logs
    if logs["total_logs"] > last_log_count:
        new_logs = logs["logs"][last_log_count:]
        for log in new_logs:
            print(f"[{log['level']}] {log['message']}")
        last_log_count = logs["total_logs"]

    # Check if done
    if not logs["is_running"]:
        break

    await asyncio.sleep(2)
```

### Pattern 3: Batch Processing with Progress Tracking

```python
# Start multiple jobs
workflow_id = "image_processor"
assets = await list_assets(content_type="image")

jobs = []
for asset in assets["assets"]:
    job = await start_background_job(
        workflow_id=workflow_id,
        params={"image_id": asset["id"]}
    )
    jobs.append(job["job_id"])

print(f"Started {len(jobs)} jobs")

# Monitor all jobs
while True:
    statuses = []
    for job_id in jobs:
        status = await get_job(job_id)
        statuses.append(status["status"])

    completed = statuses.count("completed")
    running = statuses.count("running")
    errors = statuses.count("error")

    print(f"Progress: {completed}/{len(jobs)} complete, {running} running, {errors} errors")

    if completed + errors == len(jobs):
        break

    await asyncio.sleep(10)
```

### Pattern 4: Error Investigation

```python
# Find failed jobs
jobs = await list_jobs(workflow_id="workflow_abc", limit=100)
failed_jobs = [j for j in jobs["jobs"] if j["status"] == "error"]

for job in failed_jobs:
    print(f"\nJob {job['id']} failed:")
    print(f"Error: {job['error']}")

    # Get logs to investigate
    logs = await get_job_logs(job["id"])

    # Find error-level logs
    error_logs = [log for log in logs["logs"] if log["level"] in ("ERROR", "CRITICAL")]

    for log in error_logs:
        print(f"  [{log['timestamp']}] {log['message']}")
        if "exc_info" in log:
            print(f"    Exception: {log['exc_info']}")
```

## Log Levels and Filtering

Logs are captured at different levels:
- **DEBUG**: Detailed information for debugging
- **INFO**: General informational messages
- **WARNING**: Warning messages about potential issues
- **ERROR**: Error messages about failures
- **CRITICAL**: Critical failures

When analyzing logs, focus on:
- **ERROR/CRITICAL**: For troubleshooting failures
- **WARNING**: For potential issues
- **INFO**: For understanding execution flow

## Best Practices

1. **Use background jobs for long operations**: Don't block on `run_workflow_tool` for workflows that take minutes
2. **Poll with appropriate intervals**: 2-5 seconds for active monitoring, 30-60 seconds for passive monitoring
3. **Set log limits**: Use `limit` parameter when you only need recent logs
4. **Check logs on failure**: Always inspect logs when a job errors
5. **Clean up completed jobs**: Periodically review and delete old job records
6. **Monitor costs**: Track the `cost` field for AI-powered workflows

## Debugging Tips

**If job is stuck in "starting":**
- Check logs for initialization errors
- Verify workflow ID is correct
- Check if required nodes are available

**If job fails immediately:**
- Check logs for validation errors
- Verify all required parameters are provided
- Check node type compatibility

**If job runs but produces no output:**
- Verify workflow has Output nodes
- Check logs for processing errors
- Ensure edges are properly connected

**If logs are missing:**
- Ensure job was started with `start_background_job` (not `run_workflow_tool`)
- Check if job completed (logs are persisted on completion)
- Verify log handler installed correctly
"""


@mcp.prompt()
async def workflow_examples() -> str:
    """
    Concrete examples of NodeTool workflow structures.

    Returns:
        Example workflow JSON structures with explanations
    """
    return """# NodeTool Workflow Examples

## Example 1: AI Movie Poster Generator (Multi-Stage LLM Pipeline)

**Goal**: Generate cinematic movie posters using a two-stage LLM pipeline followed by image generation

**Pattern**: Input → Template Formatting → Strategy LLM → List Generator → Image Generation → Preview

**Key Techniques Demonstrated:**
- `FormatText` with dynamic properties for template variables
- Multi-stage LLM processing (strategy then prompts)
- `ListGenerator` for creating multiple variations
- Streaming node pattern (ListGenerator outputs multiple items)

```json
{
  "nodes": [
    {
      "id": "movie_title_input",
      "type": "nodetool.input.StringInput",
      "data": {
        "value": "Stellar Odyssey",
        "name": "Movie Title"
      }
    },
    {
      "id": "genre_input",
      "type": "nodetool.input.StringInput",
      "data": {
        "value": "Sci-Fi Action",
        "name": "Genre"
      }
    },
    {
      "id": "audience_input",
      "type": "nodetool.input.StringInput",
      "data": {
        "value": "Adults 25-40, Sci-Fi Fans",
        "name": "Primary Audience"
      }
    },
    {
      "id": "strategy_template",
      "type": "nodetool.constant.String",
      "data": {
        "value": "Create a movie poster strategy for {{MOVIE_TITLE}} (Genre: {{GENRE}}, Audience: {{PRIMARY_AUDIENCE}}). Include visual concept, color palette, and design elements."
      }
    },
    {
      "id": "strategy_formatter",
      "type": "nodetool.text.FormatText",
      "data": {},
      "dynamic_properties": {
        "MOVIE_TITLE": "",
        "GENRE": "",
        "PRIMARY_AUDIENCE": ""
      }
    },
    {
      "id": "strategy_agent",
      "type": "nodetool.agents.Agent",
      "data": {
        "model": {
          "type": "language_model",
          "id": "gpt-4o",
          "provider": "openai"
        },
        "system": "You are a movie poster strategist.",
        "max_tokens": 2048
      }
    },
    {
      "id": "designer_instructions",
      "type": "nodetool.constant.String",
      "data": {
        "value": "Convert this strategy into 3 detailed Stable Diffusion prompts for cinematic movie posters."
      }
    },
    {
      "id": "prompt_generator",
      "type": "nodetool.generators.ListGenerator",
      "data": {
        "model": {
          "type": "language_model",
          "id": "gpt-4o",
          "provider": "openai"
        },
        "max_tokens": 1024
      }
    },
    {
      "id": "image_gen",
      "type": "mlx.mflux.MFlux",
      "data": {
        "model": {
          "type": "hf.flux",
          "repo_id": "dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit"
        },
        "steps": 4,
        "width": 512,
        "height": 768
      }
    },
    {
      "id": "strategy_preview",
      "type": "nodetool.workflows.base_node.Preview",
      "data": {}
    },
    {
      "id": "image_preview",
      "type": "nodetool.workflows.base_node.Preview",
      "data": {}
    }
  ],
  "edges": [
    {
      "source": "strategy_template",
      "sourceHandle": "output",
      "target": "strategy_formatter",
      "targetHandle": "template"
    },
    {
      "source": "movie_title_input",
      "sourceHandle": "output",
      "target": "strategy_formatter",
      "targetHandle": "MOVIE_TITLE"
    },
    {
      "source": "genre_input",
      "sourceHandle": "output",
      "target": "strategy_formatter",
      "targetHandle": "GENRE"
    },
    {
      "source": "audience_input",
      "sourceHandle": "output",
      "target": "strategy_formatter",
      "targetHandle": "PRIMARY_AUDIENCE"
    },
    {
      "source": "strategy_formatter",
      "sourceHandle": "output",
      "target": "strategy_agent",
      "targetHandle": "prompt"
    },
    {
      "source": "strategy_agent",
      "sourceHandle": "text",
      "target": "strategy_preview",
      "targetHandle": "value"
    },
    {
      "source": "strategy_agent",
      "sourceHandle": "text",
      "target": "prompt_generator",
      "targetHandle": "input_text"
    },
    {
      "source": "designer_instructions",
      "sourceHandle": "output",
      "target": "prompt_generator",
      "targetHandle": "prompt"
    },
    {
      "source": "prompt_generator",
      "sourceHandle": "item",
      "target": "image_gen",
      "targetHandle": "prompt"
    },
    {
      "source": "image_gen",
      "sourceHandle": "output",
      "target": "image_preview",
      "targetHandle": "value"
    }
  ]
}
```

**Key Points:**
1. **FormatText dynamic properties**: Define template variables in `dynamic_properties` that match `{{VARIABLE}}` placeholders
2. **Multi-stage LLM**: Strategy agent → List generator creates a refinement pipeline
3. **ListGenerator streaming**: The `item` output emits multiple values, triggering image_gen once per item
4. **Preview nodes**: Show intermediate results (strategy text and final images)

## Example 2: Email Classifier with Group Processing

**Goal**: Fetch emails from Gmail and classify each using AI

**Pattern**: Data Source → Group(Extract → Clean → Process → Structure) → Results

**Key Techniques Demonstrated:**
- Group node for iterating over list items
- GroupInput/GroupOutput pattern
- Dictionary operations (GetValue, MakeDictionary)
- Text preprocessing pipeline

```json
{
  "nodes": [
    {
      "id": "email_address",
      "type": "nodetool.constant.String",
      "data": {
        "value": "your_email@gmail.com"
      }
    },
    {
      "id": "gmail_search",
      "type": "lib.mail.GmailSearch",
      "data": {
        "search_query": "",
        "max_results": 10
      }
    },
    {
      "id": "email_group",
      "type": "nodetool.workflows.base_node.Group",
      "data": {}
    },
    {
      "id": "group_input",
      "parent_id": "email_group",
      "type": "nodetool.input.GroupInput",
      "data": {}
    },
    {
      "id": "get_body",
      "parent_id": "email_group",
      "type": "nodetool.dictionary.GetValue",
      "data": {
        "key": "body"
      }
    },
    {
      "id": "html_to_text",
      "parent_id": "email_group",
      "type": "nodetool.text.HtmlToText",
      "data": {
        "preserve_linebreaks": true
      }
    },
    {
      "id": "concat_prompt",
      "parent_id": "email_group",
      "type": "nodetool.text.Concat",
      "data": {
        "a": "Assign an email category for following email body. REPLY WITH CATEGORY ONLY: "
      }
    },
    {
      "id": "slice_text",
      "parent_id": "email_group",
      "type": "nodetool.text.Slice",
      "data": {
        "start": 0,
        "stop": 128
      }
    },
    {
      "id": "classify_agent",
      "parent_id": "email_group",
      "type": "nodetool.agents.Agent",
      "data": {
        "model": {
          "type": "language_model",
          "id": "gpt-4o-mini",
          "provider": "openai"
        },
        "system": "You are an email classifier.",
        "temperature": 0.3
      }
    },
    {
      "id": "get_id",
      "parent_id": "email_group",
      "type": "nodetool.dictionary.GetValue",
      "data": {
        "key": "id"
      }
    },
    {
      "id": "make_result",
      "parent_id": "email_group",
      "type": "nodetool.dictionary.MakeDictionary",
      "dynamic_properties": {
        "category": "",
        "body": "",
        "id": ""
      }
    },
    {
      "id": "group_output",
      "parent_id": "email_group",
      "type": "nodetool.output.GroupOutput",
      "data": {}
    },
    {
      "id": "results_preview",
      "type": "nodetool.workflows.base_node.Preview",
      "data": {}
    }
  ],
  "edges": [
    {
      "source": "email_address",
      "sourceHandle": "output",
      "target": "gmail_search",
      "targetHandle": "email_address"
    },
    {
      "source": "gmail_search",
      "sourceHandle": "output",
      "target": "email_group",
      "targetHandle": "input"
    },
    {
      "source": "group_input",
      "sourceHandle": "output",
      "target": "get_body",
      "targetHandle": "dictionary"
    },
    {
      "source": "get_body",
      "sourceHandle": "output",
      "target": "html_to_text",
      "targetHandle": "text"
    },
    {
      "source": "html_to_text",
      "sourceHandle": "output",
      "target": "concat_prompt",
      "targetHandle": "b"
    },
    {
      "source": "concat_prompt",
      "sourceHandle": "output",
      "target": "slice_text",
      "targetHandle": "text"
    },
    {
      "source": "slice_text",
      "sourceHandle": "output",
      "target": "classify_agent",
      "targetHandle": "prompt"
    },
    {
      "source": "group_input",
      "sourceHandle": "output",
      "target": "get_id",
      "targetHandle": "dictionary"
    },
    {
      "source": "classify_agent",
      "sourceHandle": "text",
      "target": "make_result",
      "targetHandle": "category"
    },
    {
      "source": "html_to_text",
      "sourceHandle": "output",
      "target": "make_result",
      "targetHandle": "body"
    },
    {
      "source": "get_id",
      "sourceHandle": "output",
      "target": "make_result",
      "targetHandle": "id"
    },
    {
      "source": "make_result",
      "sourceHandle": "output",
      "target": "group_output",
      "targetHandle": "input"
    },
    {
      "source": "email_group",
      "sourceHandle": "output",
      "target": "results_preview",
      "targetHandle": "value"
    }
  ]
}
```

**Key Points:**
1. **Group pattern**: All nodes inside the group have `"parent_id": "email_group"`
2. **GroupInput**: Receives each item from the list automatically
3. **GroupOutput**: Collects results from each iteration
4. **Text preprocessing**: HTML→Text→Concat→Slice creates a clean, sized prompt
5. **MakeDictionary**: Combines multiple values into structured output
6. **Dictionary operations**: GetValue extracts fields from input dictionaries

## Key Patterns from Real Workflows

### Pattern 1: Multi-Stage LLM Refinement
Use multiple LLM nodes in sequence to refine outputs:
- Agent 1: Generate strategy/plan
- Agent 2: Transform into specific format
- Generator: Create multiple variations

### Pattern 2: Group Processing
Process lists item-by-item using Group nodes:
- All processing nodes have `parent_id` set to the group ID
- GroupInput receives each item
- GroupOutput collects results
- Ideal for batch operations on datasets

### Pattern 3: Template Formatting
Use `FormatText` with dynamic properties:
- Define `{{VARIABLE}}` placeholders in template
- Add matching keys to `dynamic_properties`
- Connect inputs to dynamic property handles
- Combine multiple inputs into formatted text

### Pattern 4: Dictionary Manipulation
Work with structured data:
- `GetValue`: Extract fields from dictionaries
- `MakeDictionary`: Combine multiple values into structured output
- Use `dynamic_properties` for flexible key definitions

### Pattern 5: Text Preprocessing Pipeline
Chain text operations for clean inputs:
- HTML→Text: Convert HTML emails/pages to plain text
- Concat: Add instruction prefixes
- Slice: Limit text length for token management

## Building Your Own Workflows

1. **Start simple**: Begin with 2-3 core nodes
2. **Add Preview nodes**: Visualize data at each step
3. **Test incrementally**: Run after each major addition
4. **Use search_nodes**: Find the right node types with filters
5. **Follow patterns**: Adapt examples above to your needs
6. **Check logs**: Use `get_job_logs()` to debug issues
"""


@mcp.prompt()
async def troubleshoot_workflow() -> str:
    """
    Guide for troubleshooting common workflow issues.

    Returns:
        Troubleshooting tips and solutions
    """
    return """# Troubleshooting NodeTool Workflows

## Common Issues and Solutions

### 1. Type Mismatch Errors

**Problem**: "Cannot connect output X to input Y - type mismatch"

**Solutions**:
- Use `get_node_info` to check exact input/output types
- Insert conversion nodes between incompatible types
- Check for list vs single item mismatches (str vs list[str])

**Example Fix**:
```
# Wrong: str → list[str]
TextNode → ListNode

# Right: str → conversion → list[str]
TextNode → MakeList → ListNode
```

### 2. Missing Required Parameters

**Problem**: "Required parameter X not provided"

**Solutions**:
- Check node's `data` field has all required params
- Use `get_node_info` to see required vs optional fields
- Connect an input edge or set a default value

### 3. Circular Dependencies

**Problem**: "Workflow contains circular dependency"

**Solutions**:
- Review edges to find cycles in the graph
- Workflows must be directed acyclic graphs (DAGs)
- Remove edges that create loops

### 4. Node Not Found

**Problem**: "Node type X not found in registry"

**Solutions**:
- Verify node type spelling (case-sensitive)
- Use `search_nodes` to find the correct type
- Check if the required package is installed
- Use fully qualified names (e.g., "nodetool.text.Split")

### 5. Empty Results

**Problem**: Workflow runs but produces no output

**Solutions**:
- Add Output nodes to capture results
- Check that edges connect to output nodes
- Verify intermediate nodes are processing data

### 6. Performance Issues

**Problem**: Workflow runs slowly

**Solutions**:
- Check for unnecessary sequential dependencies
- Use parallel branches where possible
- Optimize heavy operations (large images, long videos)
- Consider batch processing nodes

## Debugging Strategies

### 1. Start Small
- Build with 2-3 nodes first
- Test after each node addition
- Verify outputs at each step

### 2. Check Each Component
- Validate node types exist
- Confirm parameter values are correct
- Verify edge connections are valid

### 3. Use Output Nodes
Add intermediate outputs to inspect data:
```
Process A → [Output] → Process B → [Output] → Process C
```

### 4. Review Examples
- Use `list_workflows` to find similar workflows
- Use `get_workflow` to see working structures
- Adapt proven patterns to your use case

## Validation Checklist

Before running a workflow, verify:
- [ ] All node types exist in registry
- [ ] All required parameters are set
- [ ] All edges connect compatible types
- [ ] No circular dependencies exist
- [ ] Input nodes are defined for parameters
- [ ] Output nodes capture desired results
- [ ] Node IDs are unique

## Getting Help

1. **Search for nodes**: Use `search_nodes` with detailed queries
2. **Inspect nodes**: Use `get_node_info` for full specifications
3. **Review examples**: Use `get_workflow` on existing workflows
4. **Test incrementally**: Use `run_workflow_tool` frequently

## Error Messages Guide

### "Workflow not found"
- Verify workflow_id is correct
- Check workflow exists with `list_workflows`

### "Invalid parameter type"
- Check parameter matches expected type
- Convert strings to numbers where needed
- Ensure refs (ImageRef, AudioRef) are used correctly

### "Node execution failed"
- Check node's specific error message
- Verify input data is valid
- Check file paths exist for file operations

### "Timeout"
- Workflow may be too complex
- Check for infinite loops (shouldn't happen in DAG)
- Consider splitting into smaller workflows
"""


@mcp.prompt()
async def describe_workflow_template() -> str:
    """
    Template for describing NodeTool workflows in a structured, comprehensive format.

    Use this template to create clear, detailed descriptions of workflows that help users
    understand the workflow's purpose, structure, features, and customization options.

    Returns:
        Markdown template for workflow descriptions with sections for overview, flow, features, customization, and use cases
    """
    return """# Workflow Description Template

Use this template to describe NodeTool workflows in a clear, structured format:

## Structure

### 🎨 [Workflow Title]
Brief one-line summary of what the workflow does and its key transformation/output.

### 📋 Pipeline Flow:
1. **Input Stage**: Describe the input(s) the workflow accepts (image, text, audio, video, data, etc.)
2. **Processing Stage(s)**: List each major processing step:
   - What happens to the data
   - Which types of nodes/models are involved
   - Key transformations applied
3. **Output Stage**: Describe the final output(s) produced (text, image, audio, video, data, etc.)

### 💡 Key Features:
- **Feature 1**: Brief description of a key capability
- **Feature 2**: Brief description of another key capability
- **Feature 3**: Brief description of unique aspects
- **Feature 4**: Any special integrations or advanced features

### ⚙️ Customization:
List ways users can customize the workflow:

**Input variations to try:**
- "Example input variation 1"
- "Example input variation 2"
- "Example input variation 3"

**Node/Model options:**
- **Provider 1**: Available models (e.g., OpenAI: GPT-4o, GPT-4-turbo)
- **Provider 2**: Available models (e.g., Anthropic: Claude 3 Opus, Sonnet, Haiku)
- **Provider 3**: Available models (e.g., Local: Ollama models)

**Parameter options:**
- **Parameter 1**: What it controls and suggested values
- **Parameter 2**: What it controls and suggested values

### 🎯 Use Cases:
- Use case 1: Brief explanation
- Use case 2: Brief explanation
- Use case 3: Brief explanation
- Use case 4: Brief explanation
- Use case 5: Brief explanation

## Example: Image-to-Story-to-Speech Pipeline

### 🎨 Image-to-Story-to-Speech Pipeline
Transforms visual art into narrative storytelling and spoken word using multimodal AI.

### 📋 Pipeline Flow:
1. **Image Input**: Any image (artwork, photo, scene, object)
2. **Story Generation**: Vision-capable LLM analyzes the image and creates a creative short story based on visual elements, emotions, and artistic themes
3. **Text Output**: Literary description capturing the essence and narrative of the image
4. **Audio Narration**: Text-to-speech converts the story into spoken word audio

### 💡 Key Features:
- **Multimodal AI**: Combines vision (image analysis) and language (story generation)
- **Creative interpretation**: Goes beyond description to generate original narratives
- **Dual outputs**: Both text story and audio narration
- **Model flexibility**: Works with any vision-capable LLM and TTS service

### ⚙️ Customization:

**Agent prompts to try:**
- "Describe this image as if you're a museum curator"
- "Write a poem inspired by this image"
- "Create a backstory for what's happening in this scene"
- "Explain the emotions and symbolism in this artwork"
- "Tell a children's story based on this picture"

**Vision model options:**
- **OpenAI**: GPT-4o, GPT-4-turbo with vision
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Local**: LLaVA, Qwen-VL via Ollama
- **Google**: Gemini Pro Vision

**TTS provider options:**
- **OpenAI TTS**: Multiple voices and speeds
- **ElevenLabs**: High-quality, customizable voices
- **Local TTS**: Coqui or Bark models

### 🎯 Use Cases:
- Art interpretation and education
- Accessibility (converting visual content to audio)
- Creative writing inspiration
- Storytelling for children's books
- Social media content generation
- Visual art documentation
- Museum audio guides
"""


@mcp.prompt()
async def agent_execution_guide() -> str:
    """
    Guide for using NodeTool agents to perform autonomous task execution.

    Returns:
        Instructions and examples for running agents with various tools and configurations
    """
    return """# NodeTool Agent Execution Guide

## Overview

NodeTool agents are autonomous AI systems that can plan, execute, and complete complex tasks using various tools. Unlike workflows (which are pre-defined node graphs), agents dynamically decide which actions to take based on their objective.

## Available Agent Tools

### 1. `run_agent` - General Purpose Agent
The main agent execution tool with full customization options.

**Parameters:**
- `objective` (required): Task description for the agent
- `provider`: AI provider (default: "openai")
  - Options: "openai", "anthropic", "ollama", "gemini", "huggingface_cerebras", etc.
- `model`: Model name (default: "gpt-4o")
- `tools`: List of tool names to enable:
  - `"google_search"`: Search the web using Google
  - `"browser"`: Browse and extract content from web pages
  - `"email"`: Search and read emails
  - `[]` (empty): Agent runs without external tools (reasoning only)
- `output_schema`: Optional JSON schema for structured output
- `enable_analysis_phase`: Enable analysis phase (default: false)
- `enable_data_contracts_phase`: Enable data contract validation (default: false)

**Example - Simple Research:**
```python
result = await run_agent(
    objective="Find 3 trending AI workflows on Reddit and summarize them",
    tools=["google_search", "browser"]
)
```

**Example - Structured Output:**
```python
result = await run_agent(
    objective="Find chicken wing recipes from 3 different websites",
    tools=["google_search", "browser"],
    output_schema={
        "type": "object",
        "properties": {
            "recipes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "url": {"type": "string"},
                        "ingredients": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "instructions": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
)
```

### 2. `run_web_research_agent` - Specialized Web Research
Pre-configured agent for web research tasks.

**Parameters:**
- `query` (required): Research query
- `provider`: AI provider (default: "openai")
- `model`: Model name (default: "gpt-4o")
- `num_sources`: Number of sources to research (default: 3)

**Example:**
```python
result = await run_web_research_agent(
    query="What are the latest developments in AI agent frameworks?",
    num_sources=5
)
```

### 3. `run_email_agent` - Email Task Agent
Pre-configured agent for email-related tasks.

**Parameters:**
- `task` (required): Email task description
- `provider`: AI provider (default: "openai")
- `model`: Model name (default: "gpt-4o")

**Example:**
```python
result = await run_email_agent(
    task="Search for emails with 'project update' in subject from last 7 days and summarize them"
)
```

## Tool Capabilities

### Google Search Tool
- Searches the web using Google
- Returns titles, URLs, and snippets
- Best for finding relevant web pages

### Browser Tool
- Visits URLs and extracts content
- Can handle JavaScript-rendered pages
- Returns markdown-formatted content
- Useful for reading articles, documentation, Reddit posts, etc.

### Email Tool
- Searches emails by query, sender, date range
- Extracts email content and metadata
- Requires email access configuration

## Common Use Cases

### 1. Reddit Research
```python
result = await run_agent(
    objective=\"\"\"
    Find examples of AI workflows on Reddit.

    Tasks:
    1. Use Google Search to find examples of AI workflows on Reddit
    2. For each URL, append .json to the url and use BrowserTool to fetch the content
    3. Output results as a markdown file

    Report structure:
    ## Workflow: [Workflow Name]
    URL: [Workflow URL]
    Summary: [Summary of the Workflow]
    Comments: [Key Comments from the Workflow]
    \"\"\",
    tools=["google_search", "browser"]
)
```

### 2. Trend Analysis with Structured Output
```python
result = await run_agent(
    objective=\"\"\"
    Identify top 5 trending hashtags on Instagram.
    For each hashtag, find one example post.
    For each post, get the post details.
    Create a summary of the trends, hashtags, and viral content.
    \"\"\",
    tools=["google_search", "browser"],
    output_schema={
        "type": "object",
        "properties": {
            "trends": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hashtag": {"type": "string"},
                        "description": {"type": "string"},
                        "example_post": {
                            "type": "object",
                            "properties": {
                                "post_url": {"type": "string"},
                                "caption": {"type": "string"},
                                "likes": {"type": "integer"},
                                "comments": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        }
    }
)
```

### 3. Email Summarization
```python
result = await run_email_agent(
    task=\"\"\"
    Search for emails with AI in subject from last 2 days.
    Summarize the content of the emails in a markdown format.
    \"\"\"
)
```

### 4. Recipe Collection
```python
result = await run_agent(
    objective=\"\"\"
    1. Identify a list of chicken wing recipe websites
    2. Crawl one website per subtask
    3. Extract the ingredients and instructions for each recipe
    4. Return the results in the format specified by the output_schema
    \"\"\",
    tools=["google_search", "browser"],
    output_schema={
        "type": "object",
        "properties": {
            "recipes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "image_url": {"type": "string"},
                        "ingredients": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "quantity": {"type": "string"},
                                    "unit": {"type": "string"}
                                }
                            }
                        },
                        "instructions": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
)
```

## Response Format

All agent tools return a dictionary with:

```python
{
    "status": "success" | "error",
    "results": "string or structured data (if output_schema provided)",
    "workspace_dir": "/path/to/workspace/with/generated/files",
    "error": "error message (only if status is error)"
}
```

## Tips for Writing Good Objectives

1. **Be specific**: Clearly state what you want the agent to accomplish
2. **Break down steps**: For complex tasks, list the steps in the objective
3. **Specify output format**: Describe how you want the results formatted
4. **Include constraints**: Mention any limitations or preferences
5. **Use output_schema**: For structured data, always provide a JSON schema

## Provider and Model Selection

### OpenAI (default)
- `gpt-4o`: Most capable, good for complex reasoning
- `gpt-4-turbo`: Fast and capable
- `gpt-3.5-turbo`: Faster, lower cost, good for simpler tasks

### Anthropic
- `claude-3-7-sonnet-20250219`: Very capable, good reasoning
- `claude-3-5-sonnet-20241022`: Balanced capability and speed
- `claude-3-haiku-20240307`: Fast, efficient for simpler tasks

### HuggingFace Cerebras (free, fast)
- `openai/gpt-oss-120b`: Good performance, free tier available

### Ollama (local)
- `qwen3:14b`: Good local model
- `gemma3:12b`: Efficient local model
- Requires Ollama running locally

### Google Gemini
- `gemini-2.0-flash`: Fast and capable
- `gemini-pro-vision`: For vision tasks

## Agents vs Workflows

**Use Agents when:**
- Task requires dynamic planning and adaptation
- You don't know exact steps in advance
- Need web search or external data access
- Want autonomous task completion

**Use Workflows when:**
- You have a well-defined process
- Steps are known and repeatable
- Need precise control over execution
- Building reusable pipelines

## Workspace Output

Agents save files to a workspace directory. Check the `workspace_dir` in the response to find:
- Downloaded files
- Generated reports
- Intermediate data
- Any artifacts created during execution
"""


def create_mcp_app():
    """
    Create and configure the FastMCP application.

    Returns:
        Configured FastMCP instance
    """
    return mcp


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
