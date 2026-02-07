"""MCP tool registration module.

This module registers all tools from the tools package with FastMCP decorators
so they can be used by MCP-compatible clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import Context, FastMCP

if TYPE_CHECKING:
    from fastmcp import Context as FastMCPContext

from nodetool.api.mcp_server import mcp
from nodetool.tools import (
    AgentTools,
    AssetTools,
    CollectionTools,
    HfTools,
    JobTools,
    ModelTools,
    NodeTools,
    StorageTools,
    WorkflowTools,
)

# Register all workflow tools
if TYPE_CHECKING:
    from fastmcp import Context

    @mcp.tool()
    async def get_workflow(workflow_id: str, user_id: str = "1") -> dict:
        """Get detailed information about a specific workflow."""
        return await WorkflowTools.get_workflow(workflow_id, user_id)

    @mcp.tool()
    async def create_workflow(
        name: str,
        graph: dict,
        description: str | None = None,
        tags: list[str] | None = None,
        access: str = "private",
        settings: dict | None = None,
        run_mode: str | None = None,
        user_id: str = "1",
    ) -> dict:
        """Create a new workflow in the database."""
        return await WorkflowTools.create_workflow(name, graph, description, tags, access, settings, run_mode, user_id)

    @mcp.tool()
    async def run_workflow_tool(workflow_id: str, params: dict | None = None, user_id: str = "1") -> dict:
        """Execute a NodeTool workflow with given parameters."""
        return await WorkflowTools.run_workflow_tool(workflow_id, params, user_id)

    @mcp.tool()
    async def run_graph(graph: dict, params: dict | None = None, user_id: str = "1") -> dict:
        """Execute a workflow graph directly without saving it as a workflow."""
        return await WorkflowTools.run_graph(graph, params, user_id)

    @mcp.tool()
    async def list_workflows(
        workflow_type: str = "user", query: str | None = None, limit: int = 100, user_id: str = "1"
    ) -> dict:
        """List workflows with flexible filtering and search options."""
        return await WorkflowTools.list_workflows(workflow_type, query, limit, user_id)

    @mcp.tool()
    async def get_example_workflow(package_name: str, example_name: str) -> dict:
        """Load a specific example workflow from disk by package name and example name."""
        return await WorkflowTools.get_example_workflow(package_name, example_name)

    @mcp.tool()
    async def validate_workflow(workflow_id: str, user_id: str = "1") -> dict:
        """Validate a workflow's structure, connectivity, and type compatibility."""
        return await WorkflowTools.validate_workflow(workflow_id, user_id)

    @mcp.tool()
    async def generate_dot_graph(graph: dict, graph_name: str = "workflow") -> dict:
        """Generate a Graphviz DOT graph from a workflow graph structure."""
        return await WorkflowTools.generate_dot_graph(graph, graph_name)

    @mcp.tool()
    async def export_workflow_digraph(workflow_id: str, descriptive_names: bool = True, user_id: str = "1") -> dict:
        """Export a workflow as a simple Graphviz Digraph (DOT format) for LLM parsing."""
        return await WorkflowTools.export_workflow_digraph(workflow_id, descriptive_names, user_id)
else:
    @mcp.tool()
    async def get_workflow(workflow_id: str, user_id: str = "1") -> dict:
        """Get detailed information about a specific workflow."""
        return await WorkflowTools.get_workflow(workflow_id, user_id)

    @mcp.tool()
    async def create_workflow(
        name: str,
        graph: dict,
        description: str | None = None,
        tags: list[str] | None = None,
        access: str = "private",
        settings: dict | None = None,
        run_mode: str | None = None,
        user_id: str = "1",
    ) -> dict:
        """Create a new workflow in the database."""
        return await WorkflowTools.create_workflow(name, graph, description, tags, access, settings, run_mode, user_id)

    @mcp.tool()
    async def run_workflow_tool(workflow_id: str, params: dict | None = None, user_id: str = "1") -> dict:
        """Execute a NodeTool workflow with given parameters."""
        return await WorkflowTools.run_workflow_tool(workflow_id, params, user_id)

    @mcp.tool()
    async def run_graph(graph: dict, params: dict | None = None, user_id: str = "1") -> dict:
        """Execute a workflow graph directly without saving it as a workflow."""
        return await WorkflowTools.run_graph(graph, params, user_id)

    @mcp.tool()
    async def list_workflows(
        workflow_type: str = "user", query: str | None = None, limit: int = 100, user_id: str = "1"
    ) -> dict:
        """List workflows with flexible filtering and search options."""
        return await WorkflowTools.list_workflows(workflow_type, query, limit, user_id)

    @mcp.tool()
    async def get_example_workflow(package_name: str, example_name: str) -> dict:
        """Load a specific example workflow from disk by package name and example name."""
        return await WorkflowTools.get_example_workflow(package_name, example_name)

    @mcp.tool()
    async def validate_workflow(workflow_id: str, user_id: str = "1") -> dict:
        """Validate a workflow's structure, connectivity, and type compatibility."""
        return await WorkflowTools.validate_workflow(workflow_id, user_id)

    @mcp.tool()
    async def generate_dot_graph(graph: dict, graph_name: str = "workflow") -> dict:
        """Generate a Graphviz DOT graph from a workflow graph structure."""
        return await WorkflowTools.generate_dot_graph(graph, graph_name)

    @mcp.tool()
    async def export_workflow_digraph(workflow_id: str, descriptive_names: bool = True, user_id: str = "1") -> dict:
        """Export a workflow as a simple Graphviz Digraph (DOT format) for LLM parsing."""
        return await WorkflowTools.export_workflow_digraph(workflow_id, descriptive_names, user_id)


# Register all asset tools
@mcp.tool()
async def list_assets(
    source: str = "user",
    parent_id: str | None = None,
    query: str | None = None,
    content_type: str | None = None,
    package_name: str | None = None,
    limit: int = 100,
    user_id: str = "1",
) -> dict:
    """List or search assets with flexible filtering options."""
    return await AssetTools.list_assets(source, parent_id, query, content_type, package_name, limit, user_id)


@mcp.tool()
async def get_asset(asset_id: str, user_id: str = "1") -> dict:
    """Get detailed information about a specific asset."""
    return await AssetTools.get_asset(asset_id, user_id)


# Register all node tools
@mcp.tool()
async def list_nodes(namespace: str | None = None, limit: int = 200) -> list:
    """List available nodes from installed packages."""
    return await NodeTools.list_nodes(namespace, limit)


@mcp.tool()
async def search_nodes(
    query: list[str],
    n_results: int = 10,
    input_type: str | None = None,
    output_type: str | None = None,
    exclude_namespaces: list[str] | None = None,
    include_metadata: bool = False,
) -> list:
    """Search for nodes by name, description, or tags."""
    return await NodeTools.search_nodes(query, n_results, input_type, output_type, exclude_namespaces, include_metadata)


@mcp.tool()
async def get_node_info(node_type: str) -> dict:
    """Get detailed metadata for a node type."""
    return await NodeTools.get_node_info(node_type)


# Register all model tools
@mcp.tool()
async def list_models(
    provider: str = "all",
    model_type: str | None = None,
    downloaded_only: bool = False,
    recommended_only: bool = False,
    limit: int = 50,
    user_id: str = "1",
) -> list:
    """List available AI models with flexible filtering options."""
    return await ModelTools.list_models(provider, model_type, downloaded_only, recommended_only, limit, user_id)


# Register all collection tools
@mcp.tool()
async def list_collections(limit: int = 50) -> dict:
    """List all vector database collections."""
    return await CollectionTools.list_collections(limit)


@mcp.tool()
async def get_collection(name: str) -> dict:
    """Get details about a specific collection."""
    return await CollectionTools.get_collection(name)


@mcp.tool()
async def query_collection(name: str, query_texts: list[str], n_results: int = 10, where: dict | None = None) -> dict:
    """Query a collection for similar documents using semantic search."""
    return await CollectionTools.query_collection(name, query_texts, n_results, where)


@mcp.tool()
async def get_documents_from_collection(
    name: str, ids: list[str] | None = None, where: dict | None = None, limit: int = 50
) -> dict:
    """Get documents from a collection by IDs or metadata filter."""
    return await CollectionTools.get_documents_from_collection(name, ids, where, limit)


# Register all job tools
@mcp.tool()
async def list_jobs(
    workflow_id: str | None = None, limit: int = 100, start_key: str | None = None, user_id: str = "1"
) -> dict:
    """List jobs for user, optionally filtered by workflow."""
    return await JobTools.list_jobs(workflow_id, limit, start_key, user_id)


@mcp.tool()
async def get_job(job_id: str, user_id: str = "1") -> dict:
    """Get a job by ID for user."""
    return await JobTools.get_job(job_id, user_id)


@mcp.tool()
async def get_job_logs(job_id: str, limit: int = 200, user_id: str = "1") -> dict:
    """Get logs for a job, preferring live logs for running jobs."""
    return await JobTools.get_job_logs(job_id, limit, user_id)


@mcp.tool()
async def start_background_job(
    workflow_id: str,
    params: dict | None = None,
    execution_strategy: str = "threaded",
    user_id: str = "1",
) -> dict:
    """Start running a workflow in background."""
    return await JobTools.start_background_job(workflow_id, params, execution_strategy, user_id)


# Register all agent tools
@mcp.tool()
async def run_agent(
    objective: str,
    provider: str,
    model: str = "gpt-4o",
    tools: list[str] | None = None,
    output_schema: dict | None = None,
    ctx: Context | None = None,
) -> dict:
    """Execute a NodeTool agent to perform autonomous task execution."""
    return await AgentTools.run_agent(objective, provider, model, tools, output_schema, ctx)


@mcp.tool()
async def run_web_research_agent(
    query: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    num_sources: int = 3,
    ctx: Context | None = None,
) -> dict:
    """Run a specialized agent for web research tasks."""
    return await AgentTools.run_web_research_agent(query, provider, model, num_sources, ctx)


@mcp.tool()
async def run_email_agent(
    task: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    ctx: Context | None = None,
) -> dict:
    """Run a specialized agent for email-related tasks."""
    return await AgentTools.run_email_agent(task, provider, model, ctx)


# Register all storage tools
@mcp.tool()
async def download_file_from_storage(key: str, temp: bool = False) -> dict:
    """Download a file from NodeTool storage."""
    return await StorageTools.download_file_from_storage(key, temp)


@mcp.tool()
async def get_file_metadata(key: str, temp: bool = False) -> dict:
    """Get metadata about a file in storage without downloading it."""
    return await StorageTools.get_file_metadata(key, temp)


@mcp.tool()
async def list_storage_files(temp: bool = False, limit: int = 100) -> dict:
    """List files in storage (note: this may not be supported by all storage backends)."""
    return await StorageTools.list_storage_files(temp, limit)


# Register all HuggingFace tools
@mcp.tool()
async def get_hf_cache_info() -> dict:
    """Get information about HuggingFace cache directory and cached models."""
    return await HfTools.get_hf_cache_info()


@mcp.tool()
async def inspect_hf_cached_model(repo_id: str) -> dict:
    """Inspect a specific HuggingFace model in cache."""
    return await HfTools.inspect_hf_cached_model(repo_id)


@mcp.tool()
async def query_hf_model_files(
    repo_id: str, repo_type: str = "model", revision: str = "main", patterns: list[str] | None = None
) -> dict:
    """Query HuggingFace Hub for files in a repository."""
    return await HfTools.query_hf_model_files(repo_id, repo_type, revision, patterns)


@mcp.tool()
async def search_hf_hub_models(query: str, limit: int = 20, model_filter: str | None = None) -> dict:
    """Search for models on HuggingFace Hub."""
    return await HfTools.search_hf_hub_models(query, limit, model_filter)


@mcp.tool()
async def get_hf_model_info(repo_id: str) -> dict:
    """Get detailed information about a model from HuggingFace Hub."""
    return await HfTools.get_hf_model_info(repo_id)


def register_all_tools():
    """
    Register all tools with the MCP server.

    This function can be called when the MCP server starts to ensure
    all tools are properly registered.

    Returns:
        Number of tools registered
    """
    tool_count = 0

    for module in [
        WorkflowTools,
        AssetTools,
        NodeTools,
        ModelTools,
        CollectionTools,
        JobTools,
        AgentTools,
        StorageTools,
        HfTools,
    ]:
        tools = module.get_tool_functions()
        tool_count += len(tools)

    return tool_count
