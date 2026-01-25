#!/usr/bin/env python
"""
FastMCP server for NodeTool API

This module provides MCP (Model Context Protocol) server integration for NodeTool,
allowing AI assistants to interact with NodeTool workflows, nodes, and assets.
"""

from __future__ import annotations

import asyncio
import base64
import os
from contextlib import suppress
from dataclasses import asdict
from io import BytesIO
from typing import TYPE_CHECKING, Any, Optional

from fastmcp import Context, FastMCP
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    from nodetool.metadata.types import Provider


from nodetool.agents.agent import Agent
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.agents.tools.email_tools import SearchEmailTool
from nodetool.api.model import (
    get_all_models,
    get_language_models,
    recommended_models,
)
from nodetool.chat.search_nodes import search_nodes as search_nodes_tool
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.huggingface_models import read_cached_hf_models
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    get_async_chroma_client,
    get_async_collection,
)
from nodetool.metadata.types import Provider
from nodetool.ml.models.asr_models import get_all_asr_models as get_all_asr_models_func
from nodetool.ml.models.image_models import (
    get_all_image_models as get_all_image_models_func,
)
from nodetool.ml.models.tts_models import get_all_tts_models as get_all_tts_models_func
from nodetool.models.asset import Asset as AssetModel
from nodetool.models.condition_builder import Field as ConditionField
from nodetool.models.message import Message as DBMessage
from nodetool.models.thread import Thread
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.packages.registry import Registry
from nodetool.providers import get_provider
from nodetool.runtime.resources import ResourceScope, maybe_scope, require_scope
from nodetool.security.secret_helper import get_secret
from nodetool.types.api_graph import (
    Graph,
    get_input_schema,
    get_output_schema,
    remove_connected_slots,
)
from nodetool.workflows.processing_context import (
    AssetOutputMode,
    ProcessingContext,
)
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import (
    Chunk,
    Error,
    LogUpdate,
    NodeProgress,
    NodeUpdate,
    OutputUpdate,
    PlanningUpdate,
    PreviewUpdate,
    SaveUpdate,
    TaskUpdate,
)

log = get_logger(__name__)


async def get_job_status(job_id: str) -> str:
    """Get the authoritative status for a job from RunState."""
    from nodetool.models.run_state import RunState

    try:
        run_state = await RunState.get(job_id)
        if run_state:
            return run_state.status
    except Exception:
        pass
    return "unknown"


async def get_hf_token(user_id: str | None = None) -> str | None:
    """Get HF_TOKEN from environment variables or database secrets (async).

    Args:
        user_id: Optional user ID. If not provided, will try to get from ResourceScope if available.

    Returns:
        HF_TOKEN if available, None otherwise.
    """
    log.debug(f"get_hf_token (mcp_server): Looking up HF_TOKEN for user_id={user_id}")

    # 1. Check environment variable first (highest priority)
    token = os.environ.get("HF_TOKEN")
    if token:
        log.debug(
            f"get_hf_token (mcp_server): HF_TOKEN found in environment variables (user_id={user_id} was provided but env takes priority)"
        )
        return token

    # 2. Try to get from database if user_id is available
    if user_id is None:
        log.debug("get_hf_token (mcp_server): No user_id provided, checking ResourceScope")
        # Try to get user_id from ResourceScope if available
        with suppress(Exception):
            maybe_scope()
            # Note: ResourceScope doesn't store user_id directly
            # In real usage, user_id would come from authentication context

    if user_id:
        log.debug(f"get_hf_token (mcp_server): Attempting to retrieve HF_TOKEN from database for user_id={user_id}")
        try:
            token = await get_secret("HF_TOKEN", user_id)
            if token:
                log.debug(f"get_hf_token (mcp_server): HF_TOKEN found in database secrets for user_id={user_id}")
                return token
            else:
                log.debug(f"get_hf_token (mcp_server): HF_TOKEN not found in database for user_id={user_id}")
        except Exception as e:
            log.debug(f"get_hf_token (mcp_server): Failed to get HF_TOKEN from database for user_id={user_id}: {e}")
    else:
        log.debug("get_hf_token (mcp_server): No user_id available, skipping database lookup")

    log.debug(f"get_hf_token (mcp_server): HF_TOKEN not found in environment or database secrets (user_id={user_id})")
    return None


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


async def _asset_to_dict(asset: AssetModel) -> dict[str, Any]:
    """
    Convert an Asset model to a dictionary for MCP responses.

    Args:
        asset: The AssetModel instance to convert

    Returns:
        Dictionary with asset information including URLs
    """
    storage = require_scope().get_asset_storage()

    # Generate URLs for non-folder assets
    if asset.content_type != "folder":
        get_url = await storage.get_url(asset.file_name)
    else:
        get_url = None

    # Generate thumbnail URL if applicable
    if asset.has_thumbnail:
        thumb_url = await storage.get_url(asset.thumb_file_name)
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
async def create_workflow(
    name: str,
    graph: dict[str, Any],
    description: str | None = None,
    tags: list[str] | None = None,
    access: str = "private",
    settings: dict[str, Any] | None = None,
    run_mode: str | None = None,
) -> dict[str, Any]:
    """
    Create a new workflow in the database.

    Args:
        name: The workflow name
        graph: Workflow graph structure with nodes and edges
        description: Optional workflow description
        tags: Optional workflow tags
        access: Access level ("private" or "public")
        settings: Optional workflow settings
        run_mode: Optional run mode (e.g., "trigger")

    Returns:
        Workflow details including graph structure, input/output schemas
    """
    api_graph = Graph.model_validate(graph)
    sanitized_graph = remove_connected_slots(api_graph)

    async with ResourceScope():
        workflow = await WorkflowModel.create(
            user_id="1",
            name=name,
            graph=sanitized_graph.model_dump(),
            description=description or "",
            tags=tags or [],
            access=access,
            settings=settings or {},
            run_mode=run_mode,
        )

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
        "run_mode": workflow.run_mode,
    }


@mcp.tool()
async def run_workflow_tool(workflow_id: str, ctx: Context, params: dict[str, Any] | None = None) -> dict[str, Any]:
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

    params = params or {}

    # Create run request
    request = RunJobRequest(
        user_id="1",
        workflow_id=workflow_id,
        params=params,
        graph=workflow.get_api_graph(),
    )

    # Run workflow
    result = {}
    preview = {}
    save = {}
    params = params or {}
    context = ProcessingContext(asset_output_mode=AssetOutputMode.TEMP_URL)
    async for msg in run_workflow(request, context=context):
        if isinstance(msg, PreviewUpdate):
            value = msg.value
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            preview[msg.node_id] = value
        elif isinstance(msg, SaveUpdate):
            value = msg.value
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            save[msg.name] = value
        elif isinstance(msg, OutputUpdate):
            value = msg.value
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            result[msg.node_name] = value

        elif isinstance(msg, NodeUpdate):
            await ctx.info(f"{msg.node_name} {msg.status}")
        elif isinstance(msg, NodeProgress):
            await ctx.report_progress(msg.progress, msg.total)
        elif isinstance(msg, LogUpdate):
            await ctx.info(msg.content)
        elif isinstance(msg, Error):
            raise Exception(msg.message)

    return {
        "workflow_id": workflow_id,
        "status": "completed",
        "result": result,
        "preview": preview,
        "save": save,
    }


@mcp.tool()
async def run_graph(graph: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
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
    from nodetool.types.api_graph import remove_connected_slots

    # Parse and validate graph
    graph_obj = Graph.model_validate(graph)
    cleaned_graph = remove_connected_slots(graph_obj)

    # Create temporary run request without workflow_id
    request = RunJobRequest(
        user_id="1",
        params=params or {},
        graph=cleaned_graph,
    )

    # Run workflow
    params = params or {}
    result = {}
    context = ProcessingContext(asset_output_mode=AssetOutputMode.TEMP_URL)
    async for msg in run_workflow(request, context=context):
        if isinstance(msg, OutputUpdate):
            value = msg.value
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            result[msg.node_name] = value

        elif isinstance(msg, Error):
            raise Exception(msg.message)

    return {
        "status": "completed",
        "result": result,
    }


@mcp.tool()
async def list_nodes(
    namespace: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """
    List available nodes from installed packages.

    Args:
        namespace: Optional namespace prefix filter (e.g. "nodetool.text").
        limit: Maximum number of nodes to return.

    Returns:
        List of nodes with basic info (type/title/description/namespace).
    """
    registry = Registry.get_instance()
    nodes = registry.get_all_installed_nodes()

    if namespace:
        namespace_prefix = namespace.lower()
        nodes = [node for node in nodes if node.node_type.lower().startswith(namespace_prefix)]

    nodes = nodes[: max(0, limit)]
    return [
        {
            "type": node.node_type,
            "title": node.title,
            "description": node.description,
            "namespace": node.namespace,
        }
        for node in nodes
    ]


@mcp.tool()
async def search_nodes(
    query: list[str],
    n_results: int = 10,
    input_type: Optional[str] = None,
    output_type: Optional[str] = None,
    exclude_namespaces: Optional[list[str]] = None,
    include_metadata: bool = False,
) -> list[dict[str, Any]]:
    """
    Search for nodes by name, description, or tags.

    Args:
        query: Search query strings
        n_results: Maximum number of results to return (default: 10)
        input_type: Optional filter by input type
        output_type: Optional filter by output type
        exclude_namespaces: Optional list of namespaces to exclude
        include_metadata: If True, return full node metadata including properties, inputs, outputs.
                         If False (default), return basic info (type/title/description/namespace).

    Returns:
        List of matching nodes with basic info or full metadata based on include_metadata parameter
    """
    nodes = search_nodes_tool(
        query=query,
        input_type=input_type,
        output_type=output_type,
        n_results=n_results,
        exclude_namespaces=exclude_namespaces or [],
    )

    result = []
    registry = Registry.get_instance()

    for node in nodes:
        if include_metadata:
            # Return full metadata
            node_metadata = registry.find_node_by_type(node.node_type)

            if node_metadata:
                result.append(node_metadata)
            else:
                # If not found in registry, try to dynamically resolve the node class
                # This handles core nodes like Preview, Comment, etc. that aren't in packages
                from nodetool.workflows.base_node import get_node_class

                node_class = get_node_class(node.node_type)
                if node_class:
                    metadata = node_class.get_metadata()
                    result.append(metadata.model_dump())
                else:
                    # Fallback to basic info if metadata not available
                    result.append(
                        {
                            "type": node.node_type,
                            "title": node.title,
                            "description": node.description,
                            "namespace": node.namespace,
                        }
                    )
        else:
            # Return basic info (default)
            result.append(
                {
                    "type": node.node_type,
                    "title": node.title,
                    "description": node.description,
                    "namespace": node.namespace,
                }
            )

    return result


@mcp.tool()
async def get_node_info(node_type: str) -> dict[str, Any]:
    """
    Get detailed metadata for a node type.

    Args:
        node_type: Fully-qualified node type (e.g. "nodetool.text.Concat").

    Returns:
        Node metadata, including properties and outputs.
    """
    registry = Registry.get_instance()
    node_metadata = registry.find_node_by_type(node_type)
    if node_metadata is not None:
        return node_metadata

    from nodetool.workflows.base_node import get_node_class

    node_class = get_node_class(node_type)
    if node_class is None:
        raise ValueError(f"Node type {node_type} not found")

    metadata = node_class.get_metadata()
    return metadata.model_dump()


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
        color = dict.fromkeys(node_ids, WHITE)

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

        return any(color[node_id] == WHITE and dfs(node_id) for node_id in node_ids)

    if has_cycle():
        errors.append("Workflow contains circular dependencies - must be a DAG (Directed Acyclic Graph)")

    # Validate node inputs and type compatibility
    for node in graph.nodes:
        if node.id not in node_types_found:
            continue

        metadata = node_types_found[node.id]

        # Check required inputs are connected
        if hasattr(metadata, "properties"):
            required_inputs = [
                prop_name
                for prop_name, prop_data in metadata.properties.items()
                if isinstance(prop_data, dict) and prop_data.get("required", False)
            ]

            connected_inputs = set()
            if node.id in edges_by_target:
                for edge in edges_by_target[node.id]:
                    if edge.targetHandle:
                        connected_inputs.add(edge.targetHandle)

            # Check node properties for static values
            if hasattr(node.data, "properties"):
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
        if any(keyword in node.type.lower() for keyword in ["input", "output", "constant", "preview"]):
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
        "message": "Workflow is valid and ready to run"
        if is_valid
        else "Workflow has validation errors - please fix before running",
    }


@mcp.tool()
async def generate_dot_graph(
    graph: dict[str, Any],
    graph_name: str = "workflow",
) -> dict[str, Any]:
    """
    Generate a Graphviz DOT graph from a workflow graph structure.

    This tool converts a NodeTool workflow graph (with nodes and edges) into a
    visual DOT graph representation for visualization.

    Args:
        graph: Workflow graph structure with nodes and edges
        graph_name: Name of the graph (default: "workflow")

    Returns:
        Dictionary with DOT format string and graph statistics
    """
    import re

    # Parse and validate graph
    graph_obj = Graph.model_validate(graph)

    # Helper function to sanitize node IDs for DOT format
    def sanitize_id(node_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", node_id)

    # Start building DOT string
    dot_lines = [
        f"digraph {sanitize_id(graph_name)} {{",
        "  rankdir=TB;",
        "  node [shape=box, style=rounded];",
        "",
    ]

    # Add nodes with simple labels
    for node in graph_obj.nodes:
        sanitized_id = sanitize_id(node.id)
        # Simple label: node_id (type)
        label = f"{node.id}\\n({node.type})"
        dot_lines.append(f'  {sanitized_id} [label="{label}"];')

    dot_lines.append("")

    # Add edges
    for edge in graph_obj.edges:
        source_id = sanitize_id(edge.source)
        target_id = sanitize_id(edge.target)

        # Add edge label with handle information if available
        edge_parts = []
        if edge.sourceHandle:
            edge_parts.append(edge.sourceHandle)
        if edge.targetHandle:
            edge_parts.append(edge.targetHandle)

        if edge_parts:
            edge_label = " → ".join(edge_parts)
            dot_lines.append(f'  {source_id} -> {target_id} [label="{edge_label}"];')
        else:
            dot_lines.append(f"  {source_id} -> {target_id};")

    dot_lines.append("}")

    dot_content = "\n".join(dot_lines)

    return {
        "graph_name": graph_name,
        "dot": dot_content,
        "node_count": len(graph_obj.nodes),
        "edge_count": len(graph_obj.edges),
    }


@mcp.tool()
async def export_workflow_digraph(workflow_id: str, descriptive_names: bool = True) -> dict[str, Any]:
    """
    Export a workflow as a simple Graphviz Digraph (DOT format) for LLM parsing and visualization.

    This tool checks both saved workflows in the database and example workflows from packages.
    By default, it replaces UUID-based node IDs with descriptive names based on node types.

    Args:
        workflow_id: The ID of the workflow to export
        descriptive_names: Use descriptive node names instead of UUIDs (default: True)

    Returns:
        Dictionary with DOT format string and workflow metadata
    """
    import re

    # First try to find in database
    workflow_model = await WorkflowModel.find("1", workflow_id)

    if workflow_model:
        graph = workflow_model.get_api_graph()
        workflow_name = workflow_model.name
    else:
        # Try to find in example workflows
        example_registry = Registry.get_instance()
        examples = await asyncio.to_thread(example_registry.list_examples)

        # Find matching example by ID
        matching_example = None
        for ex in examples:
            if ex.id == workflow_id:
                matching_example = ex
                break

        if not matching_example:
            raise ValueError(f"Workflow {workflow_id} not found in database or examples")

        # Load the example workflow
        example_workflow = await asyncio.to_thread(
            example_registry.load_example,
            matching_example.package_name or "",
            matching_example.name,
        )

        if not example_workflow or not example_workflow.graph:
            raise ValueError(f"Failed to load example workflow {workflow_id}")

        graph = example_workflow.graph
        workflow_name = example_workflow.name

    # Helper function to sanitize node IDs for DOT format
    def sanitize_id(node_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", node_id)

    # Helper function to create descriptive node ID from type
    def create_descriptive_id(node_type: str, node_data: Any = None) -> str:
        """Create a descriptive ID from node type, handling duplicates"""
        # Extract the last part of the node type (e.g., "StringInput" from "nodetool.input.StringInput")
        type_parts = node_type.split(".")
        base_name = type_parts[-1]

        # Convert PascalCase to snake_case
        base_name = re.sub(r"(?<!^)(?=[A-Z])", "_", base_name).lower()

        # Try to get a more specific name from node data
        if node_data and hasattr(node_data, "name") and node_data.name:
            # Use the node's name field if available
            specific_name = re.sub(r"[^a-zA-Z0-9_]", "_", str(node_data.name).lower())
            return specific_name

        return base_name

    # Helper function to create descriptive label
    def create_descriptive_label(node_type: str, node_data: Any = None) -> str:
        """Create a human-readable label for the node"""
        # Extract the last part of the node type
        type_parts = node_type.split(".")
        base_name = type_parts[-1]

        # If node has a name, include it in the label
        if node_data and hasattr(node_data, "name") and node_data.name:
            return f"{base_name} ({node_data.name})"

        return base_name

    # Start building DOT string
    dot_lines = [
        "digraph workflow {",
    ]

    # Track node ID mappings for descriptive names
    id_map = {}
    id_counter = {}  # Track counts for duplicate types

    # Add nodes with labels
    for node in graph.nodes:
        if descriptive_names:
            # Check if this is a UUID (contains hyphens and hex chars)
            is_uuid = re.match(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                node.id,
                re.IGNORECASE,
            )

            if is_uuid:
                # Create descriptive ID
                base_id = create_descriptive_id(node.type, node.data)

                # Handle duplicates by adding suffix
                if base_id in id_counter:
                    id_counter[base_id] += 1
                    descriptive_id = f"{base_id}_{id_counter[base_id]}"
                else:
                    id_counter[base_id] = 1
                    descriptive_id = base_id

                id_map[node.id] = descriptive_id
                sanitized_id = sanitize_id(descriptive_id)
                label = create_descriptive_label(node.type, node.data)
            else:
                # Keep original ID if it's already descriptive
                id_map[node.id] = node.id
                sanitized_id = sanitize_id(node.id)
                label = f"{node.id} ({node.type})"
        else:
            # Use original IDs
            id_map[node.id] = node.id
            sanitized_id = sanitize_id(node.id)
            label = f"{node.id} ({node.type})"

        dot_lines.append(f'  {sanitized_id} [label="{label}"];')

    # Add edges
    for edge in graph.edges:
        source_id = sanitize_id(id_map[edge.source])
        target_id = sanitize_id(id_map[edge.target])
        dot_lines.append(f"  {source_id} -> {target_id};")

    dot_lines.append("}")

    dot_content = "\n".join(dot_lines)

    return {
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "dot": dot_content,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "descriptive_names": descriptive_names,
    }


@mcp.tool()
async def list_workflows(
    workflow_type: str = "user",
    query: str | None = None,
    limit: int = 100,
    start_key: str | None = None,
) -> dict[str, Any]:
    """
    List workflows with flexible filtering and search options.

    Args:
        workflow_type: Type of workflows to list. Options:
            - "user": User-created workflows (default)
            - "example": Pre-built example workflows from packages
            - "all": Both user and example workflows
        query: Optional search query to filter workflows (searches names, descriptions, node types)
        limit: Maximum number of workflows to return (default: 100)
        start_key: Pagination key for fetching next page (only for user workflows)

    Returns:
        Dictionary with workflows list and optional pagination info

    Examples:
        - list_workflows() # List user workflows
        - list_workflows(workflow_type="example") # List example workflows
        - list_workflows(workflow_type="example", query="image") # Search examples
    """
    result = []
    next_key = None

    # Helper to parse required providers/models from example workflows
    async def enrich_example_metadata(examples: list[Any]) -> list[dict[str, Any]]:
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
            return parts[0] if parts else ""

        def collect_from_value(val: Any, providers: set[str], models: set[str]):
            if isinstance(val, dict):
                t = val.get("type")
                if t == "language_model":
                    provider = val.get("provider")
                    if isinstance(provider, str) and provider:
                        providers.add(provider)
                    model_id = val.get("id")
                    if isinstance(model_id, str) and model_id:
                        models.add(model_id)
                elif isinstance(t, str) and (t.startswith("hf.") or t.startswith("inference_provider_")):
                    model_id = val.get("repo_id") or val.get("model_id")
                    if isinstance(model_id, str) and model_id:
                        models.add(model_id)
                for v in val.values():
                    collect_from_value(v, providers, models)
            elif isinstance(val, list):
                for item in val:
                    collect_from_value(item, providers, models)

        example_registry = Registry.get_instance()

        # Load examples in parallel
        load_tasks = []
        indices = []
        for i, ex in enumerate(examples):
            if ex.package_name and ex.name:
                load_tasks.append(asyncio.to_thread(example_registry.load_example, ex.package_name, ex.name))
                indices.append(i)

        loaded_map = {}
        if load_tasks:
            results = await asyncio.gather(*load_tasks, return_exceptions=True)
            for pos, res in enumerate(results):
                idx = indices[pos]
                if not isinstance(res, Exception):
                    loaded_map[idx] = res

        enriched = []
        for i, ex in enumerate(examples):
            required_providers, required_models = set(), set()
            full_example = loaded_map.get(i)
            if full_example and full_example.graph and full_example.graph.nodes:
                for node in full_example.graph.nodes:
                    ns = parse_namespace(node.type)
                    if ns in provider_namespaces:
                        required_providers.add(ns)
                    collect_from_value(getattr(node, "data", {}), required_providers, required_models)

            enriched.append(
                {
                    "id": ex.id,
                    "name": ex.name,
                    "package_name": ex.package_name,
                    "description": ex.description,
                    "tags": ex.tags,
                    "thumbnail_url": ex.thumbnail_url,
                    "path": ex.path,
                    "required_providers": sorted(required_providers) if required_providers else None,
                    "required_models": sorted(required_models) if required_models else None,
                }
            )
        return enriched

    # Get user workflows
    if workflow_type in ("user", "all"):
        workflows, next_key = await WorkflowModel.paginate(user_id="1", limit=limit, start_key=start_key)
        for workflow in workflows:
            wf_dict = {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description or "",
                "tags": workflow.tags,
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat(),
                "workflow_type": "user",
            }
            # Apply query filter if specified
            if query:
                query_lower = query.lower()
                wf_tags = wf_dict["tags"] or []
                if (
                    query_lower in wf_dict["name"].lower()
                    or query_lower in wf_dict["description"].lower()
                    or any(query_lower in tag.lower() for tag in wf_tags)
                ):
                    result.append(wf_dict)
            else:
                result.append(wf_dict)

    # Get example workflows
    if workflow_type in ("example", "all"):
        example_registry = Registry.get_instance()

        if query:
            # Use search for query
            matching_workflows = await asyncio.to_thread(example_registry.search_example_workflows, query)
            for workflow in matching_workflows:
                result.append(
                    {
                        "id": workflow.id,
                        "name": workflow.name,
                        "package_name": workflow.package_name,
                        "description": workflow.description,
                        "tags": workflow.tags,
                        "thumbnail_url": workflow.thumbnail_url,
                        "path": workflow.path,
                        "workflow_type": "example",
                    }
                )
        else:
            # List all examples with metadata
            examples = await asyncio.to_thread(example_registry.list_examples)
            enriched = await enrich_example_metadata(examples)
            for wf in enriched:
                wf["workflow_type"] = "example"
                result.append(wf)

    # Apply limit to combined results if "all"
    if workflow_type == "all":
        result = result[:limit]

    return {
        "workflows": result,
        "next": next_key if workflow_type == "user" else None,
        "total": len(result),
    }


@mcp.tool()
async def get_example_workflow(package_name: str, example_name: str) -> dict[str, Any]:
    """
    Load a specific example workflow from disk by package name and example name.

    Args:
        package_name: The name of the package containing the example
        example_name: The name of the example workflow to load

    Returns:
        The loaded example workflow with full graph data

    Raises:
        ValueError: If the package or example is not found
    """
    example_registry = Registry.get_instance()
    workflow = example_registry.load_example(package_name, example_name)

    if not workflow:
        raise ValueError(f"Example '{example_name}' not found in package '{package_name}'")

    # Convert to dict format
    api_graph = workflow.graph
    input_schema = get_input_schema(api_graph) if api_graph else {}
    output_schema = get_output_schema(api_graph) if api_graph else {}

    return {
        "id": workflow.id,
        "name": workflow.name,
        "package_name": workflow.package_name,
        "description": workflow.description,
        "tags": workflow.tags,
        "thumbnail_url": workflow.thumbnail_url,
        "path": workflow.path,
        "graph": api_graph.model_dump() if api_graph else None,
        "input_schema": input_schema,
        "output_schema": output_schema,
    }


@mcp.tool()
async def list_assets(
    source: str = "user",
    parent_id: str | None = None,
    query: str | None = None,
    content_type: str | None = None,
    package_name: str | None = None,
    limit: int = 100,
    start_key: str | None = None,
) -> dict[str, Any]:
    """
    List or search assets with flexible filtering options.

    Args:
        source: Asset source. Options:
            - "user": User-uploaded assets (default)
            - "package": Assets from installed NodeTool packages
        parent_id: Filter by parent folder ID (user assets only, None = root)
        query: Search query for asset names (min 2 chars, enables search mode)
        content_type: Filter by type: "image", "video", "audio", "text", "folder"
        package_name: Filter package assets by package name (package source only)
        limit: Maximum number of assets to return (default: 100)
        start_key: Pagination key for next page (user assets only)

    Returns:
        Dictionary with assets list and pagination info

    Examples:
        - list_assets() # List root user assets
        - list_assets(parent_id="folder123") # List assets in folder
        - list_assets(query="vacation", content_type="image") # Search images
        - list_assets(source="package") # List all package assets
        - list_assets(source="package", package_name="nodetool-base") # Filter by package
    """
    user_id = "1"

    if source == "package":
        # List package assets
        registry = Registry.get_instance()
        all_assets = registry.list_assets()

        if package_name:
            all_assets = [a for a in all_assets if a.package_name == package_name]

        # Apply query filter if specified
        if query and len(query.strip()) >= 2:
            query_lower = query.strip().lower()
            all_assets = [a for a in all_assets if query_lower in a.name.lower()]

        # Apply limit
        all_assets = all_assets[:limit]

        results = [
            {
                "id": f"pkg:{asset.package_name}/{asset.name}",
                "name": asset.name,
                "package_name": asset.package_name,
                "virtual_path": f"/api/assets/packages/{asset.package_name}/{asset.name}",
                "source": "package",
            }
            for asset in all_assets
        ]

        return {
            "assets": results,
            "next": None,
            "total": len(results),
        }

    # User assets - search mode
    if query:
        if len(query.strip()) < 2:
            raise ValueError("Search query must be at least 2 characters long")

        assets, next_cursor, folder_paths = await AssetModel.search_assets_global(
            user_id=user_id,
            query=query.strip(),
            content_type=content_type,
            limit=limit,
            start_key=start_key,
        )

        results = []
        for i, asset in enumerate(assets):
            asset_dict = await _asset_to_dict(asset)
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
            asset_dict["source"] = "user"
            results.append(asset_dict)

        return {
            "assets": results,
            "next": next_cursor,
            "total": len(results),
        }

    # User assets - list mode
    if content_type is None and parent_id is None:
        parent_id = user_id

    assets, next_cursor = await AssetModel.paginate(
        user_id=user_id,
        parent_id=parent_id,
        content_type=content_type,
        limit=limit,
        start_key=start_key,
    )

    results = await asyncio.gather(*[_asset_to_dict(asset) for asset in assets])
    for r in results:
        r["source"] = "user"

    return {
        "assets": results,
        "next": next_cursor,
        "total": len(results),
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

    return await _asset_to_dict(asset)


@mcp.tool()
async def list_jobs(
    workflow_id: str | None = None,
    limit: int = 100,
    start_key: str | None = None,
) -> dict[str, Any]:
    """
    List jobs for the default MCP user, optionally filtered by workflow.

    Args:
        workflow_id: Optional workflow ID to filter by.
        limit: Maximum number of jobs to return.
        start_key: Pagination start key.

    Returns:
        Dictionary containing jobs and pagination cursor.
    """
    from nodetool.models.job import Job as JobModel
    from nodetool.models.run_state import RunState

    user_id = "1"
    jobs, next_start_key = await JobModel.paginate(
        user_id=user_id,
        workflow_id=workflow_id,
        limit=limit,
        start_key=start_key,
    )

    # Batch fetch all RunStates to avoid N+1 query problem
    job_ids = [job.id for job in jobs]
    run_state_map: dict[str, Any] = {}
    if job_ids:
        try:
            run_states, _ = await RunState.query(
                condition=ConditionField("run_id").in_list(job_ids),
                limit=len(job_ids),
            )
            run_state_map = {rs.run_id: rs.status for rs in run_states}
        except Exception:
            pass

    return {
        "jobs": [
            {
                "id": job.id,
                "user_id": job.user_id,
                "job_type": job.job_type,
                "status": run_state_map.get(job.id, "unknown"),
                "workflow_id": job.workflow_id,
                "started_at": job.started_at.isoformat() if job.started_at else "",
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                "error": job.error,
                "cost": job.cost,
            }
            for job in jobs
        ],
        "next_start_key": next_start_key,
    }


@mcp.tool()
async def get_job(job_id: str) -> dict[str, Any]:
    """
    Get a job by ID for the default MCP user.
    """
    from nodetool.models.job import Job as JobModel

    user_id = "1"
    job = await JobModel.find(user_id=user_id, job_id=job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    return {
        "id": job.id,
        "user_id": job.user_id,
        "job_type": job.job_type,
        "status": await get_job_status(job_id),
        "workflow_id": job.workflow_id,
        "started_at": job.started_at.isoformat() if job.started_at else "",
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "error": job.error,
        "cost": job.cost,
    }


@mcp.tool()
async def get_job_logs(job_id: str, limit: int = 200) -> dict[str, Any]:
    """
    Get logs for a job, preferring live logs for running jobs.
    """
    from nodetool.models.job import Job as JobModel
    from nodetool.workflows.job_execution_manager import JobExecutionManager

    user_id = "1"
    job = await JobModel.find(user_id=user_id, job_id=job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    manager = JobExecutionManager.get_instance()
    live = manager.get_job(job_id)
    logs = live.get_live_logs(limit=limit) if live is not None else (job.logs or [])[: max(0, limit)]

    return {"job_id": job_id, "logs": logs}


@mcp.tool()
async def start_background_job(
    workflow_id: str,
    params: dict[str, Any] | None = None,
    execution_strategy: str = "threaded",
) -> dict[str, Any]:
    """
    Start running a workflow in the background.
    """
    from nodetool.workflows.job_execution_manager import JobExecutionManager
    from nodetool.workflows.run_job_request import ExecutionStrategy

    workflow = await WorkflowModel.find("1", workflow_id)
    if not workflow:
        raise ValueError(f"Workflow {workflow_id} not found")

    try:
        strategy = ExecutionStrategy(execution_strategy)
    except ValueError as exc:
        raise ValueError(f"Invalid execution_strategy: {execution_strategy}") from exc

    request = RunJobRequest(
        user_id="1",
        workflow_id=workflow_id,
        params=params or {},
        graph=workflow.get_api_graph(),
        execution_strategy=strategy,
    )
    context = ProcessingContext(asset_output_mode=AssetOutputMode.TEMP_URL)

    manager = JobExecutionManager.get_instance()
    job = await manager.start_job(request, context)
    return {"job_id": job.job_id, "status": job.status, "workflow_id": workflow_id}


@mcp.tool()
async def list_models(
    provider: str = "all",
    model_type: str | None = None,
    downloaded_only: bool = False,
    recommended_only: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    List available AI models with flexible filtering options.

    Args:
        provider: Filter by provider. Use "all" for all providers, or specify: "openai", "anthropic", "ollama", "google", "groq", "huggingface", "replicate", "elevenlabs", etc.
        model_type: Filter by type: "language_model", "image_model", "tts_model", "asr_model", or None for all types
        downloaded_only: Only show models downloaded locally (default: False)
        recommended_only: Only show curated recommended models (default: False)
        limit: Maximum number of models to return (default: 50, max: 200)

    Returns:
        List of models matching the filters

    Examples:
        - list_models(provider="openai", model_type="language_model")
        - list_models(recommended_only=True, limit=20)
        - list_models(provider="huggingface", downloaded_only=True)
    """
    if limit > 200:
        limit = 200

    # Get models based on recommended flag
    if recommended_only:
        all_models = await recommended_models("1")
    elif model_type == "language_model":
        # Use specialized function for language models
        lm_models = await get_language_models()
        # Convert to common format
        all_models = [
            type(
                "Model",
                (),
                {
                    "id": m.id,
                    "name": m.name,
                    "repo_id": None,
                    "path": None,
                    "type": "language_model",
                    "downloaded": False,
                    "size_on_disk": None,
                    "provider": m.provider,
                },
            )()
            for m in lm_models
        ]
    elif model_type == "image_model":
        img_models = await get_all_image_models_func("1")
        all_models = [
            type(
                "Model",
                (),
                {
                    "id": m.id,
                    "name": m.name,
                    "repo_id": None,
                    "path": None,
                    "type": "image_model",
                    "downloaded": False,
                    "size_on_disk": None,
                    "provider": m.provider,
                },
            )()
            for m in img_models
        ]
    elif model_type == "tts_model":
        tts_models = await get_all_tts_models_func("1")
        all_models = [
            type(
                "Model",
                (),
                {
                    "id": m.id,
                    "name": m.name,
                    "repo_id": None,
                    "path": None,
                    "type": "tts_model",
                    "downloaded": False,
                    "size_on_disk": None,
                    "provider": m.provider,
                },
            )()
            for m in tts_models
        ]
    elif model_type == "asr_model":
        asr_models = await get_all_asr_models_func("1")
        all_models = [
            type(
                "Model",
                (),
                {
                    "id": m.id,
                    "name": m.name,
                    "repo_id": None,
                    "path": None,
                    "type": "asr_model",
                    "downloaded": False,
                    "size_on_disk": None,
                    "provider": m.provider,
                },
            )()
            for m in asr_models
        ]
    else:
        all_models = await get_all_models("1")

    # Filter by provider
    if provider.lower() != "all":
        filtered = []
        for m in all_models:
            matched = False
            # Check provider attribute if available and meaningful
            if hasattr(m, "provider") and m.provider is not None:
                try:
                    if hasattr(m.provider, "value"):
                        if m.provider.value.lower() == provider.lower():  # type: ignore[union-attr]
                            matched = True
                    elif str(m.provider).lower() == provider.lower():
                        matched = True
                except (AttributeError, TypeError):
                    pass

            # Fallback to checking id and repo_id if provider check didn't match
            if not matched and (
                provider.lower() in m.id.lower()
                or (hasattr(m, "repo_id") and m.repo_id and provider.lower() in m.repo_id.lower())
            ):
                matched = True

            if matched:
                filtered.append(m)
        all_models = filtered

    # Filter by model type if specified and not already filtered
    if model_type and not recommended_only:
        all_models = [m for m in all_models if hasattr(m, "type") and m.type == model_type]

    # Filter by downloaded status
    if downloaded_only:
        all_models = [m for m in all_models if hasattr(m, "downloaded") and m.downloaded]

    # Apply limit
    all_models = all_models[:limit]

    # Build result with available fields
    result = []
    for model in all_models:
        model_dict = {
            "id": model.id,
            "name": model.name,
        }
        if hasattr(model, "repo_id"):
            model_dict["repo_id"] = model.repo_id
        if hasattr(model, "path"):
            model_dict["path"] = model.path
        if hasattr(model, "type"):
            model_dict["type"] = model.type
        if hasattr(model, "downloaded"):
            model_dict["downloaded"] = model.downloaded
        if hasattr(model, "size_on_disk"):
            model_dict["size_on_disk"] = model.size_on_disk
        if hasattr(model, "provider"):
            if hasattr(model.provider, "value"):
                model_dict["provider"] = model.provider.value
            else:
                model_dict["provider"] = str(model.provider)

        result.append(model_dict)

    return result


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

    async def get_workflow_name(metadata: dict[str, Any]) -> str | None:
        if workflow_id := metadata.get("workflow"):
            workflow = await WorkflowModel.get(workflow_id)
            if workflow:
                return workflow.name
        return None

    # Apply limit
    collections = collections[:limit]

    counts = await asyncio.gather(*(col.count() for col in collections))
    workflows = await asyncio.gather(*(get_workflow_name(col.metadata) for col in collections))

    return {
        "collections": [
            {
                "name": col.name,
                "metadata": col.metadata,
                "workflow_name": wf,
                "count": count,
            }
            for col, wf, count in zip(collections, workflows, counts, strict=False)
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
    scope = require_scope()
    storage = scope.get_temp_storage() if temp else scope.get_asset_storage()

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
    scope = require_scope()
    storage = scope.get_temp_storage() if temp else scope.get_asset_storage()

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
    scope = require_scope()
    storage = scope.get_temp_storage() if temp else scope.get_asset_storage()

    # Try to list files (not all storage backends support this)
    try:
        list_files_func = getattr(storage, "list_files", None)
        if callable(list_files_func):
            files = await list_files_func(limit=limit)
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
    from dataclasses import asdict
    from fnmatch import fnmatch

    from huggingface_hub import HfApi

    try:
        # Use HF_TOKEN from secrets if available for gated model downloads
        token = await get_hf_token()
        if token:
            log.debug(
                f"query_hf_model_files: Querying files for {repo_id} with HF_TOKEN (token length: {len(token)} chars)"
            )
            api = HfApi(token=token)
        else:
            log.debug(
                f"query_hf_model_files: Querying files for {repo_id} without HF_TOKEN - gated models may not be accessible"
            )
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
            info = api.get_paths_info(
                repo_id=repo_id,
                paths=[file_path],
                repo_type=repo_type,
                revision=revision,
            )
            if info:
                file_info: RepoFile | RepoFolder = info[0]
                files_data.append(asdict(file_info))

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
        raise ValueError(f"Failed to query HuggingFace Hub: {str(e)}") from e


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

    # Use HF_TOKEN from secrets if available for gated model downloads
    token = await get_hf_token()
    if token:
        log.debug(
            f"search_hf_hub_models: Searching with query '{query}' using HF_TOKEN (token length: {len(token)} chars)"
        )
        api = HfApi(token=token)
    else:
        log.debug(
            f"search_hf_hub_models: Searching with query '{query}' without HF_TOKEN - gated models may not be accessible"
        )
        api = HfApi()

    # Parse filter
    filter_dict = {}
    if model_filter and ":" in model_filter:
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

    # Use HF_TOKEN from secrets if available for gated model downloads
    token = await get_hf_token()
    if token:
        log.debug(
            f"get_hf_model_info: Fetching model info for {repo_id} with HF_TOKEN (token length: {len(token)} chars)"
        )
        api = HfApi(token=token)
    else:
        log.debug(
            f"get_hf_model_info: Fetching model info for {repo_id} without HF_TOKEN - gated models may not be accessible"
        )
        api = HfApi()
    return asdict(api.model_info(repo_id))


# ============================================================================
# Agent Execution Tools
# ============================================================================


async def _run_agent_impl(
    objective: str,
    provider: str,
    model: str = "gpt-4o",
    tools: list[str] | None = None,
    output_schema: dict[str, Any] | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Internal implementation of run_agent."""
    tools = tools or []
    context = ProcessingContext(asset_output_mode=AssetOutputMode.TEMP_URL)

    # Use ResourceScope to ensure HTTP client and other resources are bound to the current context
    # This is required for providers (get_client) and tools that need access to resources
    async with ResourceScope():
        try:
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

            provider_enum = Provider(provider)
            provider_instance = await get_provider(provider_enum)

            # Create and execute agent
            agent = Agent(
                name=f"Agent: {objective[:50]}...",
                objective=objective,
                provider=provider_instance,
                model=model,
                tools=tool_instances,
                output_schema=output_schema,
            )

            # Execute agent and collect output
            output_chunks = []
            events = []
            async for event in agent.execute(context):
                if isinstance(event, Chunk):
                    output_chunks.append(event.content)
                    # We could optionally stream chunks if FastMCP supports it for tools,
                    # but typically tools return a final value.
                    # ctx.info(event.content) might be too noisy.

                elif isinstance(event, PlanningUpdate):
                    if ctx:
                        await ctx.info(f"Plan: {event.phase} - {event.content}")
                    events.append(event.model_dump())

                elif isinstance(event, TaskUpdate):
                    if ctx:
                        task_title = event.task.title if event.task else "Task"
                        await ctx.info(f"Task: {event.event} - {task_title}")
                    events.append(event.model_dump())

                elif isinstance(event, LogUpdate):
                    if ctx:
                        await ctx.info(f"Log: {event.content}")
                    events.append(event.model_dump())

                else:
                    if hasattr(event, "model_dump"):
                        events.append(event.model_dump())
                    else:
                        # Fallback for unexpected types
                        pass

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
async def run_agent(
    objective: str,
    provider: str,
    ctx: Context,
    model: str = "gpt-4o",
    tools: list[str] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute a NodeTool agent to perform autonomous task execution.

    Agents can use various tools to accomplish objectives like web search,
    browsing, email access, and more. They autonomously plan and execute tasks
    based on the objective you provide.

    Args:
        objective: The task description for the agent to accomplish
        provider: AI provider. Options: "openai", "anthropic",
                 "ollama", "gemini", "huggingface_cerebras", etc.
        model: Model to use (default: "gpt-4o")
        tools: List of tool names to enable. Options:
               - "google_search": Search the web using Google
               - "browser": Browse and extract content from web pages
               - "email": Search and read emails
               - [] (empty): Agent runs without external tools
        output_schema: Optional JSON schema to structure the agent's output

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
    return await _run_agent_impl(
        objective=objective,
        provider=provider,
        model=model,
        tools=tools,
        output_schema=output_schema,
        ctx=ctx,
    )


@mcp.tool()
async def run_web_research_agent(
    query: str,
    ctx: Context,
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

    return await _run_agent_impl(
        objective=objective,
        provider=Provider(provider),
        model=model,
        tools=["google_search", "browser"],
        ctx=ctx,
    )


@mcp.tool()
async def run_email_agent(
    task: str,
    ctx: Context,
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
    return await _run_agent_impl(
        objective=task,
        provider=Provider(provider),
        model=model,
        tools=["email"],
        ctx=ctx,
    )


@mcp.prompt()
async def build_workflow_guide() -> str:
    """
    Comprehensive guide on how to build NodeTool workflows.

    Returns:
        Step-by-step instructions for creating workflows
    """
    return """# How to Build NodeTool Workflows

## Rules
- Never invent node types, property names, or IDs.
- Always call `search_nodes` before adding nodes; use `include_properties=true` for exact field names.
- Use `search_examples(query)` to find reference workflows.
- Reply in short bullets; no verbose explanations.

## Core Principles
1. **Data Flows Through Edges**: Nodes connect via typed edges (image→image, text→text, etc.)
2. **Asynchronous Execution**: Nodes execute when dependencies are satisfied
3. **Streaming by Default**: Many nodes support real-time streaming output
4. **Type Safety**: Connections enforce type compatibility
5. **Node Type Resolution**: Nodes are referenced by type string (e.g., `nodetool.image.Resize`); the system auto-resolves classes from the registry

## Node Categories
| Category | Purpose | Key Nodes |
|----------|---------|-----------|
| Input | Accept parameters | `StringInput`, `ImageInput`, `AudioInput`, `ChatInput` |
| Output | Return results | `ImageOutput`, `StringOutput`, `Preview` |
| Agents | LLM-powered | `Agent`, `Summarizer`, `ListGenerator`, `DataGenerator` |
| Control | Flow control | `Collect`, `FormatText`, `If` |
| Storage | Persistence | `CreateTable`, `Insert`, `Query`, `Collection`, `IndexTextChunks`, `HybridSearch` |
| Processing | Transform data | `Resize`, `Filter`, `ExtractText`, `Canny` |
| Realtime | Streaming I/O | `RealtimeAudioInput`, `RealtimeAgent`, `RealtimeWhisper` |

## Special Nodes (Built-in)
These nodes are always available and do NOT require `search_nodes`.

### Input Nodes (`nodetool.input.*`)
| Node Type | Purpose | Key Properties |
|-----------|---------|----------------|
| `nodetool.input.StringInput` | Text input parameter | `value` (str), `name` (str) |
| `nodetool.input.IntegerInput` | Whole number input | `value` (int), `min`, `max`, `name` |
| `nodetool.input.FloatInput` | Decimal number input | `value` (float), `min`, `max`, `name` |
| `nodetool.input.BooleanInput` | True/false toggle | `value` (bool), `name` |
| `nodetool.input.ImageInput` | Image file input | `value` (ImageRef), `name` |
| `nodetool.input.AudioInput` | Audio file input | `value` (AudioRef), `name` |
| `nodetool.input.VideoInput` | Video file input | `value` (VideoRef), `name` |
| `nodetool.input.DocumentInput` | Document file input | `value` (DocumentRef), `name` |
| `nodetool.input.GroupInput` | Receives items inside a Group | `name`; automatically iterates list items |

### Output Nodes (`nodetool.output.*`)
| Node Type | Purpose | Key Properties |
|-----------|---------|----------------|
| `nodetool.output.StringOutput` | Return text result | `value` (str), `name` |
| `nodetool.output.IntegerOutput` | Return integer result | `value` (int), `name` |
| `nodetool.output.FloatOutput` | Return float result | `value` (float), `name` |
| `nodetool.output.BooleanOutput` | Return boolean result | `value` (bool), `name` |
| `nodetool.output.ImageOutput` | Return image result | `value` (ImageRef), `name` |
| `nodetool.output.AudioOutput` | Return audio result | `value` (AudioRef), `name` |
| `nodetool.output.VideoOutput` | Return video result | `value` (VideoRef), `name` |
| `nodetool.output.DocumentOutput` | Return document result | `value` (DocumentRef), `name` |
| `nodetool.output.DataframeOutput` | Return tabular data | `value` (DataframeRef), `name` |
| `nodetool.output.DictionaryOutput` | Return key-value data | `value` (dict), `name` |
| `nodetool.output.ListOutput` | Return list of values | `value` (list), `name` |
| `nodetool.output.GroupOutput` | Collects results from Group | `name`; accumulates iteration outputs |

### Utility Nodes
| Node Type | Purpose | Properties |
|-----------|---------|------------|
| `nodetool.workflows.base_node.Preview` | Display intermediate results | `value` (any), `name` (str) |
| `nodetool.workflows.base_node.Comment` | Add annotations/documentation | `headline` (str), `comment` (any), `comment_color` (str) |
| `nodetool.workflows.base_node.GroupNode` | Container for subgraph iteration | — |

## Data Flow Patterns

**Sequential Pipeline**: Input → Process → Transform → Output
- Each node waits for previous to complete

**Parallel Branches**: Input splits to ProcessA→OutputA and ProcessB→OutputB
- Multiple branches execute simultaneously

**Streaming Pipeline**: Input → StreamingAgent → Collect → Output
- Data flows in chunks for real-time updates
- Use `Collect` to gather stream into list

**Fan-In Pattern**: SourceA + SourceB → Combine → Process → Output
- Multiple inputs combine before processing

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
   - Assets are saved to the workspace

**Example: Processing images:**
```python
# Get all images from a folder
assets = await list_assets(
    parent_id="folder_id",
    content_type="image"
)

# Process each image with run_workflow_tool
for asset in assets["assets"]:
    result = await run_workflow_tool(
        workflow_id="image_processor",
        params={"image_id": asset["id"]}
    )

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

## Workflow Patterns

**Pattern 1: Simple Pipeline** — Input → Process → Transform → Output
- Use for: single input/output transformations
- Example: `ImageInput` → `Sharpen` → `AutoContrast` → `ImageOutput`

**Pattern 2: Agent-Driven Generation** — Input → Agent → Generator → Output
- Use for: creative generation, multimodal transforms (image→text→audio)
- Example: `ImageInput` → `Agent` → `TextToSpeech` → `Preview`
- Key: `Agent` streams LLM responses; `ListGenerator` streams list items

**Pattern 3: RAG (Retrieval-Augmented Generation)**
- **Indexing (with Group per-file)**:
  - `ListFiles` → `Group` (contains: `GroupInput` → `LoadDocumentFile` → `ExtractText` → `SentenceSplitter` → `IndexTextChunks`)
  - `Collection` connects to `IndexTextChunks` inside Group
  - Key nodes: `lib.os.ListFiles`, `lib.pymupdf.ExtractText`, `lib.langchain.SentenceSplitter`, `chroma.index.IndexTextChunks`, `chroma.collections.Collection`
- **Query**: `ChatInput` → `HybridSearch` → `FormatText` → `Agent` → `StringOutput`
- Use for: Q&A over documents, semantic search, reducing hallucinations

**Pattern 4: Database Persistence**
- Flow: Input → `DataGenerator` → `Insert` ← `CreateTable` → `Query` → `Preview`
- Nodes: `CreateTable` (schema), `Insert` (add), `Query` (retrieve), `Update`, `Delete`
- Use for: persistent storage, agent memory, flashcards

**Pattern 5: Realtime Processing**
- Flow: `RealtimeAudioInput` → `RealtimeAgent` → `Preview`
- Use for: voice interfaces, live transcription
- Key nodes: `RealtimeWhisper`, `RealtimeTranscription`

**Pattern 6: Multi-Modal Conversion**
- Audio→Text→Image: `AudioInput` → `Whisper` → `StableDiffusion` → `ImageOutput`
- Image→Text→Audio: `ImageInput` → `ImageToText` → `TextToSpeech` → `AudioOutput`

**Pattern 7: Data Visualization**
- Flow: `GetRequest` → `ImportCSV` → `Filter` → `ChartGenerator` → `Preview`
- Use for: fetching, transforming, visualizing external data

**Pattern 8: Structured Data Generation**
- Flow: `DataGenerator` → `Preview`
- DataGenerator uses LLM with schema to generate structured data (e.g., tables of veggies with name/color columns)
- Configure with: `prompt` (describe data), `columns` (schema with name, data_type, description)
- Use for: synthetic data, test data, structured outputs

**Pattern 9: Email Classification**
- Simple: `GmailSearch` → `Template` → `Classifier` → `AddLabel`
- With Group: `GmailSearch` → `Group` (contains: `GroupInput` → `GetValue` → `HtmlToText` → `Agent` → `MakeDictionary` → `GroupOutput`) → `Preview`
- Use for: automated email organization, per-email processing
- Key nodes: `lib.mail.GmailSearch`, `nodetool.agents.Classifier`, `nodetool.text.Template`

**Pattern 10: Group/ForEach Iteration**
- List source → `Group` node containing subgraph → collected output
- Inside Group: `GroupInput` receives each item, subgraph processes it, `GroupOutput` collects results
- Use for: processing each item in a list with complex multi-node logic
- Key: Group node has `parent_id` for child nodes; children use `GroupInput`/`GroupOutput`

**Pattern 11: Paper2Podcast (Document to Audio)**
- Flow: `GetRequestDocument` → `ExtractText` → `Summarizer` → `TextToSpeech` → `Preview`
- Example: Fetch arxiv PDF → extract first N pages → summarize for TTS → generate speech audio
- Key nodes: `lib.http.GetRequestDocument`, `lib.pymupdf.ExtractText` (with `start_page`/`end_page`), `nodetool.agents.Summarizer`, `elevenlabs.text_to_speech.TextToSpeech`
- Configure Summarizer with TTS-friendly prompt (neutral tone, no intro/conclusion, concise)
- Use for: converting academic papers, reports, or documents into podcast-style audio

**Pattern 12: Pokemon Maker (Creative Batch Generation)**
- Flow: `StringInput` → `FormatText` → `ListGenerator` → `StableDiffusion` → `ImageOutput`
- Example: Enter animal inspirations → format creative prompt → LLM generates multiple character descriptions → each description becomes an image
- Key nodes: `nodetool.input.StringInput`, `nodetool.text.FormatText` (with `{{placeholder}}` syntax), `nodetool.generators.ListGenerator`, `huggingface.text_to_image.StableDiffusion`
- ListGenerator streams items one-by-one; downstream image generation processes each as it arrives
- Use for: batch creative generation (characters, items, concepts) with text + image output

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
- **`search_nodes`**: Find nodes by functionality, get detailed specs with `include_metadata=True`
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
6. **Validate workflows**: Use `validate_workflow()` before running
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
- Use `search_nodes(..., include_metadata=True)` to check exact input/output types
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
- Use `search_nodes(..., include_metadata=True)` to see required vs optional fields
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
2. **Inspect nodes**: Use `search_nodes(..., include_metadata=True)` for full specifications
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


@mcp.tool()
async def get_run_state(run_id: str) -> dict[str, Any]:
    """
    Get the current state of a workflow run.

    This is the authoritative source of truth for run status and execution state.

    Args:
        run_id: The workflow run ID

    Returns:
        Run state including status, timestamps, suspension state, and execution metadata
    """
    from nodetool.models.run_state import RunState

    run_state = await RunState.get(run_id)
    if not run_state:
        raise ValueError(f"Run state not found for run_id: {run_id}")

    return {
        "run_id": run_state.run_id,
        "status": run_state.status,
        "created_at": run_state.created_at.isoformat() if run_state.created_at else None,
        "updated_at": run_state.updated_at.isoformat() if run_state.updated_at else None,
        "completed_at": run_state.completed_at.isoformat() if run_state.completed_at else None,
        "failed_at": run_state.failed_at.isoformat() if run_state.failed_at else None,
        "error_message": run_state.error_message,
        "execution_strategy": run_state.execution_strategy,
        "execution_id": run_state.execution_id,
        "worker_id": run_state.worker_id,
        "heartbeat_at": run_state.heartbeat_at.isoformat() if run_state.heartbeat_at else None,
        "retry_count": run_state.retry_count,
        "max_retries": run_state.max_retries,
        "suspended_node_id": run_state.suspended_node_id,
        "suspension_reason": run_state.suspension_reason,
        "suspension_state": run_state.suspension_state_json,
        "suspension_metadata": run_state.suspension_metadata_json,
        "metadata": run_state.metadata_json,
        "version": run_state.version,
        "is_stale": run_state.is_stale(),
        "is_complete": run_state.is_complete(),
        "is_suspended": run_state.is_suspended(),
        "is_resumable": run_state.is_resumable(),
    }


@mcp.tool()
async def list_run_states(
    status: str | None = None,
    include_stale: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """
    List workflow run states with optional filtering.

    Args:
        status: Filter by status (scheduled, running, suspended, paused, completed, failed, cancelled, recovering)
        include_stale: Include runs with stale heartbeats
        limit: Maximum number of runs to return

    Returns:
        List of run states matching the filters
    """
    from nodetool.models.condition_builder import Field
    from nodetool.models.run_state import RunState

    conditions = []

    if status:
        conditions.append(Field("status").equals(status))

    if not include_stale:
        conditions.append(Field("heartbeat_at") is not None)

    from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, LogicalOperator

    if conditions:
        condition = ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND))
        runs, _ = await RunState.query(condition=condition, limit=limit)
    else:
        runs, _ = await RunState.query(limit=limit)

    return {
        "runs": [
            {
                "run_id": run.run_id,
                "status": run.status,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "updated_at": run.updated_at.isoformat() if run.updated_at else None,
                "worker_id": run.worker_id,
                "execution_strategy": run.execution_strategy,
                "error_message": run.error_message,
                "is_stale": run.is_stale(),
                "is_complete": run.is_complete(),
            }
            for run in runs
        ],
        "count": len(runs),
    }


@mcp.tool()
async def get_run_events(
    run_id: str,
    event_type: str | None = None,
    node_id: str | None = None,
    seq_gt: int | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """
    Get the event log for a workflow run.

    Events provide an audit trail of what happened during execution.
    Events are append-only and best-effort ordered.

    Args:
        run_id: The workflow run ID
        event_type: Optional filter by event type
        node_id: Optional filter by node ID
        seq_gt: Return events with sequence number greater than this
        limit: Maximum number of events to return

    Returns:
        List of events ordered by sequence number
    """
    from nodetool.models.run_event import RunEvent

    events = await RunEvent.get_events(
        run_id=run_id,
        event_type=event_type,
        node_id=node_id,
        seq_gt=seq_gt,
        limit=limit,
    )

    return {
        "run_id": run_id,
        "events": [
            {
                "id": event.id,
                "run_id": event.run_id,
                "seq": event.seq,
                "event_type": event.event_type,
                "event_time": event.event_time.isoformat() if event.event_time else None,
                "node_id": event.node_id,
                "payload": event.payload,
            }
            for event in events
        ],
        "count": len(events),
    }


@mcp.tool()
async def get_run_timeline(run_id: str) -> dict[str, Any]:
    """
    Get a formatted timeline view of a workflow run's execution.

    This provides a human-readable chronological view of all events,
    useful for debugging and understanding execution flow.

    Args:
        run_id: The workflow run ID

    Returns:
        Timeline with events, statistics, and duration info
    """
    from nodetool.models.run_event import RunEvent
    from nodetool.models.run_state import RunState

    run_state = await RunState.get(run_id)
    if not run_state:
        raise ValueError(f"Run state not found for run_id: {run_id}")

    events = await RunEvent.get_events(run_id=run_id, limit=1000)

    if not events:
        return {
            "run_id": run_id,
            "status": run_state.status,
            "message": "No events recorded for this run",
            "timeline": [],
        }

    timeline_entries = []
    event_counts: dict[str, int] = {}
    node_events: dict[str, list[dict]] = {}

    for event in events:
        entry = {
            "seq": event.seq,
            "time": event.event_time.isoformat() if event.event_time else None,
            "event": event.event_type,
            "node_id": event.node_id,
            "details": event.payload,
        }
        timeline_entries.append(entry)

        event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        if event.node_id:
            if event.node_id not in node_events:
                node_events[event.node_id] = []
            node_events[event.node_id].append(entry)

    first_time = events[0].event_time if events else None
    last_time = events[-1].event_time if events else None

    duration_seconds = None
    if first_time and last_time:
        duration_seconds = (last_time - first_time).total_seconds()

    return {
        "run_id": run_id,
        "status": run_state.status,
        "error_message": run_state.error_message,
        "started_at": first_time.isoformat() if first_time else None,
        "completed_at": last_time.isoformat() if last_time else None,
        "duration_seconds": duration_seconds,
        "total_events": len(events),
        "event_counts": event_counts,
        "node_count": len(node_events),
        "nodes_with_events": list(node_events.keys()),
        "timeline": timeline_entries,
    }


@mcp.tool()
async def get_active_jobs() -> dict[str, Any]:
    """
    Get all currently active job executions.

    Returns:
        List of active jobs with their current status and metadata
    """
    from nodetool.models.run_state import RunState
    from nodetool.workflows.job_execution_manager import JobExecutionManager

    manager = JobExecutionManager.get_instance()
    jobs = manager.list_jobs()

    active_jobs = []
    for job in jobs:
        if job.is_running():
            run_state = await RunState.get(job.job_id)
            active_jobs.append(
                {
                    "job_id": job.job_id,
                    "execution_id": job.execution_id,
                    "workflow_id": job.request.workflow_id,
                    "user_id": job.request.user_id,
                    "status": job.status,
                    "execution_strategy": job.request.execution_strategy.value
                    if job.request.execution_strategy
                    else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "worker_id": Environment.get_worker_id(),
                    "run_state_status": run_state.status if run_state else None,
                    "is_stale": run_state.is_stale() if run_state else None,
                }
            )

    return {
        "active_jobs": active_jobs,
        "count": len(active_jobs),
    }


@mcp.tool()
async def cancel_run(run_id: str) -> dict[str, Any]:
    """
    Cancel a running workflow job.

    Args:
        run_id: The job/run ID to cancel

    Returns:
        Cancellation result
    """
    from nodetool.models.run_state import RunState
    from nodetool.workflows.job_execution_manager import JobExecutionManager

    run_state = await RunState.get(run_id)
    if not run_state:
        raise ValueError(f"Run state not found for run_id: {run_id}")

    if run_state.is_complete():
        raise ValueError(f"Run {run_id} is already complete (status: {run_state.status})")

    manager = JobExecutionManager.get_instance()
    cancelled = await manager.cancel_job(run_id)

    if cancelled:
        async with ResourceScope():
            await run_state.mark_cancelled()

        return {
            "run_id": run_id,
            "cancelled": True,
            "message": f"Run {run_id} has been cancelled",
        }

    return {
        "run_id": run_id,
        "cancelled": False,
        "message": f"Run {run_id} could not be cancelled (may not be in memory)",
    }


@mcp.tool()
async def recover_run(run_id: str) -> dict[str, Any]:
    """
    Attempt to recover a stale or failed run.

    Args:
        run_id: The run ID to recover

    Returns:
        Recovery result
    """
    from nodetool.models.run_state import RunState
    from nodetool.workflows.job_execution_manager import JobExecutionManager

    run_state = await RunState.get(run_id)
    if not run_state:
        raise ValueError(f"Run state not found for run_id: {run_id}")

    if not run_state.is_resumable():
        raise ValueError(f"Run {run_id} is not resumable (status: {run_state.status})")

    manager = JobExecutionManager.get_instance()
    success = await manager.resume_run(run_id)

    if success:
        return {
            "run_id": run_id,
            "recovered": True,
            "message": f"Run {run_id} recovery initiated successfully",
        }

    return {
        "run_id": run_id,
        "recovered": False,
        "message": f"Failed to recover run {run_id}",
    }


@mcp.tool()
async def get_node_run_events(run_id: str, node_id: str) -> dict[str, Any]:
    """
    Get all events for a specific node within a run.

    Useful for debugging individual node execution.

    Args:
        run_id: The workflow run ID
        node_id: The node ID to get events for

    Returns:
        All events for the specified node
    """
    from nodetool.models.run_event import RunEvent

    events = await RunEvent.get_events(run_id=run_id, node_id=node_id, limit=500)

    if not events:
        return {
            "run_id": run_id,
            "node_id": node_id,
            "events": [],
            "count": 0,
            "message": f"No events found for node {node_id} in run {run_id}",
        }

    first_time = events[0].event_time if events else None
    last_time = events[-1].event_time if events else None

    duration_seconds = None
    if first_time and last_time:
        duration_seconds = (last_time - first_time).total_seconds()

    return {
        "run_id": run_id,
        "node_id": node_id,
        "events": [
            {
                "seq": event.seq,
                "time": event.event_time.isoformat() if event.event_time else None,
                "event": event.event_type,
                "details": event.payload,
            }
            for event in events
        ],
        "count": len(events),
        "duration_seconds": duration_seconds,
    }


@mcp.tool()
async def analyze_run_errors(run_id: str) -> dict[str, Any]:
    """
    Analyze a run for errors and failures.

    Returns a diagnostic summary of what went wrong.

    Args:
        run_id: The workflow run ID

    Returns:
        Error analysis including failed nodes, error messages, and suggestions
    """
    from nodetool.models.run_event import RunEvent
    from nodetool.models.run_state import RunState

    run_state = await RunState.get(run_id)
    if not run_state:
        raise ValueError(f"Run state not found for run_id: {run_id}")

    events = await RunEvent.get_events(run_id=run_id, limit=1000)

    failed_node_events = [e for e in events if e.event_type == "NodeFailed"]

    run_failed_events = [e for e in events if e.event_type in ("RunFailed", "RunCancelled")]

    errors = []
    for event in failed_node_events:
        errors.append(
            {
                "node_id": event.node_id,
                "error": event.payload.get("error", "Unknown error"),
                "seq": event.seq,
                "time": event.event_time.isoformat() if event.event_time else None,
            }
        )

    for event in run_failed_events:
        errors.append(
            {
                "node_id": None,
                "error": event.payload.get("error", f"Run {event.event_type}"),
                "seq": event.seq,
                "time": event.event_time.isoformat() if event.event_time else None,
            }
        )

    suggestions = []
    if run_state.status == "failed":
        if run_state.error_message:
            suggestions.append(f"Run failed with error: {run_state.error_message}")
        for error in errors:
            if error["node_id"]:
                suggestions.append(f"Node {error['node_id']} failed: {error['error']}")

    return {
        "run_id": run_id,
        "status": run_state.status,
        "error_message": run_state.error_message,
        "total_events": len(events),
        "failed_nodes": len(failed_node_events),
        "errors": errors,
        "suggestions": suggestions,
    }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
