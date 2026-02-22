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
    register_all_tools,
)

# Register all tools when this module is imported
# The registration happens in nodetool.api.mcp_tools module
# where all @mcp.tool() decorated functions are defined


if __name__ == "__main__":
    # Register all tools and run the server
    register_all_tools()
    mcp.run()
