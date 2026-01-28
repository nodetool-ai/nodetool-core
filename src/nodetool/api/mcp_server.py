#!/usr/bin/env python
"""
FastMCP server for NodeTool API

This module provides MCP (Model Context Protocol) server integration for NodeTool,
allowing AI assistants to interact with NodeTool workflows, nodes, and assets.

Tools are registered in nodetool.api.mcp_tools module to avoid duplication
and ensure a single source of truth for tool implementations.
"""

from __future__ import annotations

from fastmcp import FastMCP

# Initialize FastMCP server BEFORE importing mcp_tools to avoid circular import
# (mcp_tools imports 'mcp' from this module)
mcp = FastMCP("NodeTool API Server")

# Import the tools module which contains all tool implementations
# and registers them with @mcp.tool() decorators
from nodetool.api.mcp_tools import register_all_tools  # noqa: E402

# Re-export all decorated tool functions for backwards compatibility
# (tests access these via mcp_server.function_name.fn)
from nodetool.api.mcp_tools import (  # noqa: E402
    create_workflow,
    get_workflow,
    run_workflow_tool,
    run_graph,
    list_workflows,
    get_example_workflow,
    validate_workflow,
    generate_dot_graph,
    export_workflow_digraph,
    list_assets,
    get_asset,
    list_nodes,
    search_nodes,
    get_node_info,
    list_models,
    list_collections,
    get_collection,
    query_collection,
    get_documents_from_collection,
    list_jobs,
    get_job,
    get_job_logs,
    start_background_job,
    run_agent,
    run_web_research_agent,
    run_email_agent,
    download_file_from_storage,
    get_file_metadata,
    list_storage_files,
    get_hf_cache_info,
    inspect_hf_cached_model,
    query_hf_model_files,
    search_hf_hub_models,
    get_hf_model_info,
)

# Register all tools when this module is imported
# The registration happens in nodetool.api.mcp_tools module
# where all @mcp.tool() decorated functions are defined


if __name__ == "__main__":
    # Register all tools and run the server
    register_all_tools()
    mcp.run()
