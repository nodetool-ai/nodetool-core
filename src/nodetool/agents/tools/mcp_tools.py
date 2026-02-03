"""
MCP Tool wrappers for the Agent system.

This module provides Tool wrappers for all MCP tools, making the agent
"omnipotent" - capable of controlling all aspects of nodetool including
workflows, nodes, jobs, assets, collections, and storage.
"""

from typing import Any

from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext

# ============================================================================
# Workflow Tools
# ============================================================================


class ListWorkflowsTool(Tool):
    """Tool to list workflows with flexible filtering and search options."""

    name = "list_workflows"
    description = """List workflows with flexible filtering and search options.

Returns user workflows, example workflows, or both. Use this to discover available
workflows before running them.

Args:
    workflow_type: Type of workflows to list ("user", "example", or "all")
    query: Optional search query to filter workflows by name/description
    limit: Maximum number of workflows to return (default: 100)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "workflow_type": {
                "type": "string",
                "description": "Type of workflows to list",
                "enum": ["user", "example", "all"],
                "default": "user",
            },
            "query": {
                "type": "string",
                "description": "Optional search query to filter workflows",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of workflows to return",
                "default": 100,
            },
        },
        "required": [],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        workflow_type = params.get("workflow_type", "user")
        query = params.get("query")
        if query:
            return f"Listing {workflow_type} workflows matching '{query}'"
        return f"Listing {workflow_type} workflows"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import WorkflowTools

        return await WorkflowTools.list_workflows(
            workflow_type=params.get("workflow_type", "user"),
            query=params.get("query"),
            limit=params.get("limit", 100),
            user_id=context.user_id,
        )


class GetWorkflowTool(Tool):
    """Tool to get detailed information about a specific workflow."""

    name = "get_workflow"
    description = """Get detailed information about a specific workflow.

Returns the workflow's graph structure, input/output schemas, metadata, and more.
Use this to understand a workflow before running it.

Args:
    workflow_id: The ID of the workflow to retrieve
"""
    input_schema = {
        "type": "object",
        "properties": {
            "workflow_id": {
                "type": "string",
                "description": "The ID of the workflow",
            },
        },
        "required": ["workflow_id"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Getting workflow {params.get('workflow_id')}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import WorkflowTools

        return await WorkflowTools.get_workflow(
            workflow_id=params["workflow_id"],
            user_id=context.user_id,
        )


class CreateWorkflowTool(Tool):
    """Tool to create a new workflow in the database."""

    name = "create_workflow"
    description = """Create a new workflow in the database.

Creates a workflow with the specified name, graph structure, and optional metadata.
The graph should contain nodes and edges defining the workflow's structure.

Args:
    name: The workflow name
    graph: Workflow graph structure with "nodes" and "edges" arrays
    description: Optional workflow description
    tags: Optional list of tags for organization
    access: Access level ("private" or "public")
"""
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The workflow name",
            },
            "graph": {
                "type": "object",
                "description": "Workflow graph structure with nodes and edges",
                "properties": {
                    "nodes": {"type": "array"},
                    "edges": {"type": "array"},
                },
            },
            "description": {
                "type": "string",
                "description": "Optional workflow description",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional workflow tags",
            },
            "access": {
                "type": "string",
                "enum": ["private", "public"],
                "default": "private",
            },
        },
        "required": ["name", "graph"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Creating workflow '{params.get('name')}'"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import WorkflowTools

        return await WorkflowTools.create_workflow(
            name=params["name"],
            graph=params["graph"],
            description=params.get("description"),
            tags=params.get("tags"),
            access=params.get("access", "private"),
            user_id=context.user_id,
        )


class RunWorkflowTool(Tool):
    """Tool to execute a workflow with given parameters."""

    name = "run_workflow"
    description = """Execute a NodeTool workflow with given parameters.

Runs the specified workflow and returns the results. This is the primary tool
for executing workflows. Provide input parameters as needed by the workflow.

Args:
    workflow_id: The ID of the workflow to run
    params: Dictionary of input parameters for the workflow (optional)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "workflow_id": {
                "type": "string",
                "description": "The ID of the workflow to run",
            },
            "params": {
                "type": "object",
                "description": "Dictionary of input parameters for the workflow",
            },
        },
        "required": ["workflow_id"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Running workflow {params.get('workflow_id')}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import WorkflowTools

        return await WorkflowTools.run_workflow_tool(
            workflow_id=params["workflow_id"],
            params=params.get("params"),
            user_id=context.user_id,
        )


class RunGraphTool(Tool):
    """Tool to execute a workflow graph directly without saving it."""

    name = "run_graph"
    description = """Execute a workflow graph directly without saving it as a workflow.

Useful for testing workflow graphs or running one-off executions. The graph
should contain nodes and edges defining the workflow structure.

Args:
    graph: Workflow graph structure with nodes and edges
    params: Dictionary of input parameters for the workflow (optional)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "graph": {
                "type": "object",
                "description": "Workflow graph structure with nodes and edges",
            },
            "params": {
                "type": "object",
                "description": "Dictionary of input parameters for the workflow",
            },
        },
        "required": ["graph"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return "Running graph directly"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import WorkflowTools

        return await WorkflowTools.run_graph(
            graph=params["graph"],
            params=params.get("params"),
            user_id=context.user_id,
        )


class ValidateWorkflowTool(Tool):
    """Tool to validate a workflow's structure and connectivity."""

    name = "validate_workflow"
    description = """Validate a workflow's structure, connectivity, and type compatibility.

Returns a validation report with errors, warnings, and suggestions. Use this
to debug workflows before running them.

Args:
    workflow_id: The ID of the workflow to validate
"""
    input_schema = {
        "type": "object",
        "properties": {
            "workflow_id": {
                "type": "string",
                "description": "The ID of the workflow to validate",
            },
        },
        "required": ["workflow_id"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Validating workflow {params.get('workflow_id')}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import WorkflowTools

        return await WorkflowTools.validate_workflow(
            workflow_id=params["workflow_id"],
            user_id=context.user_id,
        )


class GetExampleWorkflowTool(Tool):
    """Tool to load an example workflow from a package."""

    name = "get_example_workflow"
    description = """Load a specific example workflow from disk by package name and example name.

Returns the full example workflow including its graph data. Use list_workflows
with workflow_type="example" to discover available examples first.

Args:
    package_name: The name of the package containing the example
    example_name: The name of the example workflow to load
"""
    input_schema = {
        "type": "object",
        "properties": {
            "package_name": {
                "type": "string",
                "description": "The name of the package containing the example",
            },
            "example_name": {
                "type": "string",
                "description": "The name of the example workflow to load",
            },
        },
        "required": ["package_name", "example_name"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Loading example {params.get('package_name')}/{params.get('example_name')}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import WorkflowTools

        return await WorkflowTools.get_example_workflow(
            package_name=params["package_name"],
            example_name=params["example_name"],
        )


class ExportWorkflowDigraphTool(Tool):
    """Tool to export a workflow as a DOT graph for visualization."""

    name = "export_workflow_digraph"
    description = """Export a workflow as a Graphviz Digraph (DOT format).

Returns the workflow as a DOT graph string for visualization or LLM parsing.

Args:
    workflow_id: The ID of the workflow to export
    descriptive_names: Use descriptive node names instead of UUIDs (default: True)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "workflow_id": {
                "type": "string",
                "description": "The ID of the workflow to export",
            },
            "descriptive_names": {
                "type": "boolean",
                "description": "Use descriptive node names instead of UUIDs",
                "default": True,
            },
        },
        "required": ["workflow_id"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Exporting workflow {params.get('workflow_id')} as digraph"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import WorkflowTools

        return await WorkflowTools.export_workflow_digraph(
            workflow_id=params["workflow_id"],
            descriptive_names=params.get("descriptive_names", True),
            user_id=context.user_id,
        )


# ============================================================================
# Node Tools
# ============================================================================


class ListNodesTool(Tool):
    """Tool to list available nodes from installed packages."""

    name = "list_nodes"
    description = """List available nodes from installed packages.

Returns nodes with their type, title, description, and namespace. Use this
to discover available nodes for building workflows.

Args:
    namespace: Optional namespace prefix filter (e.g. "nodetool.text")
    limit: Maximum number of nodes to return (default: 200)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Optional namespace prefix filter",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of nodes to return",
                "default": 200,
            },
        },
        "required": [],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        ns = params.get("namespace")
        if ns:
            return f"Listing nodes in namespace {ns}"
        return "Listing available nodes"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import NodeTools

        return await NodeTools.list_nodes(
            namespace=params.get("namespace"),
            limit=params.get("limit", 200),
        )


class SearchNodesTool(Tool):
    """Tool to search for nodes by name, description, or tags."""

    name = "search_nodes"
    description = """Search for nodes by name, description, or tags.

Returns matching nodes. Use this to find specific nodes for your workflow needs.

Args:
    query: Search query strings (list of keywords)
    n_results: Maximum number of results to return (default: 10)
    input_type: Optional filter by input type
    output_type: Optional filter by output type
    include_metadata: If True, return full node metadata including properties
"""
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Search query strings",
            },
            "n_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10,
            },
            "input_type": {
                "type": "string",
                "description": "Optional filter by input type",
            },
            "output_type": {
                "type": "string",
                "description": "Optional filter by output type",
            },
            "include_metadata": {
                "type": "boolean",
                "description": "If True, return full node metadata",
                "default": False,
            },
        },
        "required": ["query"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        query = params.get("query", [])
        return f"Searching for nodes: {', '.join(query)}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import NodeTools

        return await NodeTools.search_nodes(
            query=params["query"],
            n_results=params.get("n_results", 10),
            input_type=params.get("input_type"),
            output_type=params.get("output_type"),
            include_metadata=params.get("include_metadata", False),
        )


class GetNodeInfoTool(Tool):
    """Tool to get detailed metadata for a node type."""

    name = "get_node_info"
    description = """Get detailed metadata for a node type.

Returns the node's properties, inputs, outputs, and other metadata. Use this
to understand how to use a specific node in a workflow.

Args:
    node_type: Fully-qualified node type (e.g. "nodetool.text.Concat")
"""
    input_schema = {
        "type": "object",
        "properties": {
            "node_type": {
                "type": "string",
                "description": "Fully-qualified node type",
            },
        },
        "required": ["node_type"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Getting info for node type {params.get('node_type')}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import NodeTools

        return await NodeTools.get_node_info(node_type=params["node_type"])


# ============================================================================
# Job Tools
# ============================================================================


class ListJobsTool(Tool):
    """Tool to list workflow execution jobs."""

    name = "list_jobs"
    description = """List jobs (workflow executions) with optional filtering.

Returns job status, timing, and error information. Use this to monitor
workflow executions.

Args:
    workflow_id: Optional workflow ID to filter by
    limit: Maximum number of jobs to return (default: 100)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "workflow_id": {
                "type": "string",
                "description": "Optional workflow ID to filter by",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of jobs to return",
                "default": 100,
            },
        },
        "required": [],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        wf_id = params.get("workflow_id")
        if wf_id:
            return f"Listing jobs for workflow {wf_id}"
        return "Listing jobs"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import JobTools

        return await JobTools.list_jobs(
            workflow_id=params.get("workflow_id"),
            limit=params.get("limit", 100),
            user_id=context.user_id,
        )


class GetJobTool(Tool):
    """Tool to get details about a specific job."""

    name = "get_job"
    description = """Get details about a specific job (workflow execution).

Returns job status, timing, error information, and cost.

Args:
    job_id: The job ID
"""
    input_schema = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "The job ID",
            },
        },
        "required": ["job_id"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Getting job {params.get('job_id')}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import JobTools

        return await JobTools.get_job(
            job_id=params["job_id"],
            user_id=context.user_id,
        )


class GetJobLogsTool(Tool):
    """Tool to get logs for a job."""

    name = "get_job_logs"
    description = """Get logs for a job, preferring live logs for running jobs.

Use this to debug workflow executions and understand what happened.

Args:
    job_id: The job ID
    limit: Maximum number of log entries to return (default: 200)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "The job ID",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of logs to return",
                "default": 200,
            },
        },
        "required": ["job_id"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Getting logs for job {params.get('job_id')}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import JobTools

        return await JobTools.get_job_logs(
            job_id=params["job_id"],
            limit=params.get("limit", 200),
            user_id=context.user_id,
        )


class StartBackgroundJobTool(Tool):
    """Tool to start a workflow running in the background."""

    name = "start_background_job"
    description = """Start running a workflow in the background.

Unlike run_workflow which waits for completion, this starts the workflow
and returns immediately with a job ID for tracking.

Args:
    workflow_id: The workflow ID to run
    params: Optional input parameters
"""
    input_schema = {
        "type": "object",
        "properties": {
            "workflow_id": {
                "type": "string",
                "description": "The workflow ID to run",
            },
            "params": {
                "type": "object",
                "description": "Optional input parameters",
            },
        },
        "required": ["workflow_id"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Starting background job for workflow {params.get('workflow_id')}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import JobTools

        return await JobTools.start_background_job(
            workflow_id=params["workflow_id"],
            params=params.get("params"),
            user_id=context.user_id,
        )


# ============================================================================
# Asset Tools
# ============================================================================


class ListAssetsTool(Tool):
    """Tool to list or search assets."""

    name = "list_assets"
    description = """List or search assets with flexible filtering options.

Returns user assets or package assets with optional filtering.

Args:
    source: Asset source ("user" or "package")
    query: Search query for asset names (min 2 chars)
    content_type: Filter by type ("image", "video", "audio", "text", "folder")
    limit: Maximum number of assets to return (default: 100)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["user", "package"],
                "default": "user",
            },
            "query": {
                "type": "string",
                "description": "Search query for asset names",
            },
            "content_type": {
                "type": "string",
                "description": "Filter by content type",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of assets to return",
                "default": 100,
            },
        },
        "required": [],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        query = params.get("query")
        if query:
            return f"Searching assets for '{query}'"
        return "Listing assets"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import AssetTools

        return await AssetTools.list_assets(
            source=params.get("source", "user"),
            query=params.get("query"),
            content_type=params.get("content_type"),
            limit=params.get("limit", 100),
            user_id=context.user_id,
        )


class GetAssetTool(Tool):
    """Tool to get detailed information about a specific asset."""

    name = "get_asset"
    description = """Get detailed information about a specific asset.

Returns asset details including URLs and metadata.

Args:
    asset_id: The ID of the asset
"""
    input_schema = {
        "type": "object",
        "properties": {
            "asset_id": {
                "type": "string",
                "description": "The ID of the asset",
            },
        },
        "required": ["asset_id"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Getting asset {params.get('asset_id')}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import AssetTools

        return await AssetTools.get_asset(
            asset_id=params["asset_id"],
            user_id=context.user_id,
        )


# ============================================================================
# Collection Tools (Vector Database)
# ============================================================================


class ListCollectionsTool(Tool):
    """Tool to list vector database collections."""

    name = "list_collections"
    description = """List all vector database collections.

Returns collections with their metadata and document counts.

Args:
    limit: Maximum number of collections to return (default: 50)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of collections to return",
                "default": 50,
            },
        },
        "required": [],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return "Listing vector collections"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import CollectionTools

        return await CollectionTools.list_collections(limit=params.get("limit", 50))


class QueryCollectionTool(Tool):
    """Tool to query a collection for similar documents."""

    name = "query_collection"
    description = """Query a collection for similar documents using semantic search.

Returns the most similar documents to the query texts.

Args:
    name: Name of collection to query
    query_texts: List of query texts to search for
    n_results: Number of results to return per query (default: 10)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of collection to query",
            },
            "query_texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Query texts to search for",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results per query",
                "default": 10,
            },
        },
        "required": ["name", "query_texts"],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        return f"Querying collection {params.get('name')}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import CollectionTools

        return await CollectionTools.query_collection(
            name=params["name"],
            query_texts=params["query_texts"],
            n_results=params.get("n_results", 10),
        )


# ============================================================================
# Model Tools
# ============================================================================


class ListModelsTool(Tool):
    """Tool to list available AI models."""

    name = "list_models"
    description = """List available AI models with flexible filtering options.

Returns models from various providers with optional filtering.

Args:
    provider: Filter by provider ("all", "openai", "anthropic", "ollama", etc.)
    model_type: Filter by model type (e.g., "language", "image")
    downloaded_only: Only show downloaded models
    limit: Maximum number of models to return (default: 50)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "description": "Filter by provider",
                "default": "all",
            },
            "model_type": {
                "type": "string",
                "description": "Filter by model type",
            },
            "downloaded_only": {
                "type": "boolean",
                "description": "Only show downloaded models",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of models to return",
                "default": 50,
            },
        },
        "required": [],
    }

    def user_message(self, params: dict[str, Any]) -> str:
        provider = params.get("provider", "all")
        return f"Listing models from {provider}"

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> Any:
        from nodetool.tools import ModelTools

        return await ModelTools.list_models(
            provider=params.get("provider", "all"),
            model_type=params.get("model_type"),
            downloaded_only=params.get("downloaded_only", False),
            limit=params.get("limit", 50),
            user_id=context.user_id,
        )


def get_all_mcp_tools() -> list[Tool]:
    """
    Get all MCP tool instances for the omnipotent agent.

    Returns a list of all tool instances that give the agent full control
    over nodetool's capabilities.
    """
    return [
        # Workflow tools - the bread and butter
        ListWorkflowsTool(),
        GetWorkflowTool(),
        CreateWorkflowTool(),
        RunWorkflowTool(),
        RunGraphTool(),
        ValidateWorkflowTool(),
        GetExampleWorkflowTool(),
        ExportWorkflowDigraphTool(),
        # Node tools - for building workflows
        ListNodesTool(),
        SearchNodesTool(),
        GetNodeInfoTool(),
        # Job tools - for monitoring executions
        ListJobsTool(),
        GetJobTool(),
        GetJobLogsTool(),
        StartBackgroundJobTool(),
        # Asset tools
        ListAssetsTool(),
        GetAssetTool(),
        # Collection tools
        ListCollectionsTool(),
        QueryCollectionTool(),
        # Model tools
        ListModelsTool(),
    ]
