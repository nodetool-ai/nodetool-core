#!/usr/bin/env python
"""
Standalone FastMCP server for NodeTool

This script runs the NodeTool MCP server as a standalone service.
It can be run using: fastmcp run nodetool/api/run_mcp_server.py

Or directly with: python -m nodetool.api.run_mcp_server
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodetool.config.logging_config import configure_logging, get_logger
from nodetool.api.mcp_server import mcp

# Configure logging
configure_logging()
log = get_logger(__name__)

if __name__ == "__main__":
    log.info("Starting NodeTool FastMCP Server...")
    log.info(
        "Available tools: list_workflows, get_workflow, run_workflow_tool, list_nodes, search_nodes, get_node_info"
    )
    log.info("Available resources: nodetool://workflows, nodetool://nodes/{namespace}")

    # Run the MCP server
    mcp.run()
