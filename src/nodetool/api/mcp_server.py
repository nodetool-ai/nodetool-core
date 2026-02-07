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
# Re-export all decorated tool functions for backwards compatibility
# (tests access these via mcp_server.function_name.fn)
from nodetool.api.mcp_tools import (
    create_workflow,
    download_file_from_storage,
    export_workflow_digraph,
    generate_dot_graph,
    get_asset,
    get_collection,
    get_documents_from_collection,
    get_example_workflow,
    get_file_metadata,
    get_hf_cache_info,
    get_hf_model_info,
    get_job,
    get_job_logs,
    get_node_info,
    get_workflow,
    inspect_hf_cached_model,
    list_assets,
    list_collections,
    list_jobs,
    list_models,
    list_nodes,
    list_storage_files,
    list_workflows,
    query_collection,
    query_hf_model_files,
    register_all_tools,
    run_agent,
    run_email_agent,
    run_graph,
    run_web_research_agent,
    run_workflow_tool,
    search_hf_hub_models,
    search_nodes,
    start_background_job,
    validate_workflow,
)

# Register all tools when this module is imported
# The registration happens in nodetool.api.mcp_tools module
# where all @mcp.tool() decorated functions are defined


if __name__ == "__main__":
    # Register all tools and run the server
    register_all_tools()
    mcp.run()
