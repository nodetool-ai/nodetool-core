# MCP Server Test Import Fix

**Problem**: The MCP server test file (`tests/api/test_mcp_server.py`) was importing tool functions from `nodetool.api.mcp_server` module, but these functions were not re-exported from that module. The tools were defined in `nodetool.api.mcp_tools.py` with `@mcp.tool()` decorators and only accessible via the FastMCP server's `get_tools()` method.

**Solution**: Changed the test file to import tool functions directly from the `nodetool.tools` classes (e.g., `WorkflowTools.create_workflow`, `AssetTools.get_asset`, etc.) instead of trying to import them from `mcp_server` module. This provides direct access to the underlying tool functions without going through the FastMCP decorator layer.

**Why**: The FastMCP decorator pattern wraps functions and makes them accessible via the MCP server's `get_tools()` method, but they aren't automatically re-exported as module-level attributes. By importing directly from the tool classes, tests get clean access to the actual async functions.

**Files**:
- `tests/api/test_mcp_server.py` - Updated imports to use `nodetool.tools` classes

**Date**: 2026-02-22
