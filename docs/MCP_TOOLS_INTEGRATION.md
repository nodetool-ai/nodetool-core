# MCP Tools Integration for Chat Agents

## Overview

This integration exposes all MCP (Model Context Protocol) server functions as tools that can be used by chat agents in NodeTool. This gives agents complete control over NodeTool functionality through the chat interface.

## Architecture

The integration consists of three main components:

### 1. MCP Tool Wrapper (`src/nodetool/agents/tools/mcp_tools.py`)

This module provides a wrapper that converts MCP functions into Agent Tool format:

- **`MCPToolWrapper`**: A Tool class that wraps individual MCP functions
- **`get_all_mcp_tools()`**: Retrieves all MCP tools from the FastMCP server
- **`get_mcp_tool_by_name()`**: Gets a specific MCP tool by name

The wrapper automatically:
- Introspects the MCP server to discover all registered tools
- Converts MCP function signatures to Agent Tool schemas
- Handles async execution of MCP functions
- Provides user-friendly error handling

### 2. Tool Registry Integration (`src/nodetool/agents/tools/tool_registry.py`)

The tool registry was updated to:
- Include MCP tools in the tool resolution process
- Support async tool lookup
- Provide a `get_all_available_tools()` function that includes both built-in and MCP tools

### 3. CLI Integration (`src/nodetool/chat/chat_cli.py`)

The chat CLI now:
- Loads MCP tools during initialization
- Makes them available through the `/tools` command
- Allows enabling/disabling individual MCP tools
- Displays MCP tools alongside built-in tools

## Available MCP Tools

The integration exposes 46+ MCP tools covering:

### Workflow Management
- `get_workflow` - Get workflow details
- `create_workflow` - Create new workflows
- `run_workflow_tool` - Execute workflows
- `run_graph` - Run workflow graphs directly
- `list_workflows` - List all workflows
- `validate_workflow` - Validate workflow structure
- `export_workflow_digraph` - Export workflow diagrams

### Node Operations
- `list_nodes` - List available nodes
- `search_nodes` - Search for nodes
- `get_node_info` - Get node metadata

### Asset Management
- `list_assets` - List user assets
- `get_asset` - Get asset details
- `download_file_from_storage` - Download files
- `get_file_metadata` - Get file information

### Job & Execution Management
- `list_jobs` - List workflow jobs
- `get_job` - Get job details
- `get_job_logs` - Retrieve job logs
- `start_background_job` - Start background jobs
- `get_run_state` - Get execution state
- `list_run_states` - List execution states
- `get_run_events` - Get execution events
- `cancel_run` - Cancel running jobs
- `recover_run` - Recover failed jobs
- `get_active_jobs` - List active jobs

### Collection & Data Management
- `list_collections` - List vector collections
- `get_collection` - Get collection details
- `query_collection` - Query collections
- `get_documents_from_collection` - Retrieve documents

### Thread & Message Management
- `list_threads` - List conversation threads
- `get_thread` - Get thread details
- `get_thread_messages` - Get thread messages

### Model Information
- `list_models` - List available AI models

## Usage

### In CLI Chat

1. **Start the chat CLI:**
   ```bash
   nodetool chat
   ```

2. **List available tools (including MCP tools):**
   ```
   /tools
   ```

3. **Enable MCP tools:**
   ```
   /enable all              # Enable all tools
   /enable get_workflow     # Enable specific tool
   ```

4. **Use in agent mode:**
   ```
   /agent on
   Create a new workflow that processes images
   ```

### In Regular Chat Mode

MCP tools are automatically available when you enable them and request their use:

```
/enable run_workflow_tool
Can you run workflow abc123 with parameter x=10?
```

### In WebSocket Chat

MCP tools are automatically available through the unified WebSocket endpoint (`/ws`). When a tool is requested by the LLM, the tool registry will resolve it, including MCP tools.

## Technical Details

### Tool Resolution Order

When resolving a tool by name, the system checks in this order:
1. Node-based tools (from installed packages)
2. Built-in agent tools (browser, email, PDF, etc.)
3. Workflow tools (custom workflows exposed as tools)
4. **MCP tools** (all NodeTool API functions)

### Async Execution

MCP tools are async by nature. The wrapper handles this automatically:

```python
# MCP tool execution
result = await mcp_tool.process(context, params)
```

### Error Handling

MCP tools gracefully handle errors and return structured error messages:

```python
{
    "error": "Error message",
    "tool": "tool_name"
}
```

### Caching

MCP tools are cached after first load to improve performance:
- Tools are discovered once on first request
- Subsequent calls use cached tool instances
- Cache is shared across the application

## Configuration

### Enabling/Disabling Tools

In the CLI, tools can be enabled/disabled individually:

```
/enable get_workflow    # Enable one tool
/disable get_workflow   # Disable one tool
/enable all             # Enable all tools
```

Tool state is persisted in `.chat_settings.json`.

### Security Considerations

- MCP tools have full access to NodeTool functionality
- Tools should only be enabled for trusted users
- Consider implementing role-based access control for production use
- Tool execution is logged for audit purposes

## Development

### Adding New MCP Tools

New MCP tools are automatically discovered. Simply add them to `src/nodetool/api/mcp_server.py`:

```python
@mcp.tool()
async def my_new_tool(param: str) -> dict:
    """Description of the tool."""
    # Implementation
    return {"result": "value"}
```

The tool will be automatically available to agents on next restart.

### Testing

Run the MCP tools integration tests:

```bash
pytest tests/agents/tools/test_mcp_tools.py -v
```

### Debugging

Enable debug logging to see tool execution:

```python
import logging
logging.getLogger("nodetool.agents.tools.mcp_tools").setLevel(logging.DEBUG)
```

## Examples

### Example 1: List and Run a Workflow

```
User: /agent on
User: List all my workflows and run the first one with parameter x=5

Agent uses:
1. list_workflows() -> gets workflows
2. run_workflow_tool(workflow_id="...", params={"x": 5}) -> executes
3. get_job(job_id="...") -> checks status
4. get_job_logs(job_id="...") -> retrieves logs
```

### Example 2: Search and Inspect Nodes

```
User: Find all image processing nodes and show me details about the first one

Agent uses:
1. search_nodes(query=["image", "processing"]) -> finds nodes
2. get_node_info(node_type="...") -> gets details
```

### Example 3: Asset Management

```
User: List my recent assets and download the latest image

Agent uses:
1. list_assets(limit=10) -> gets assets
2. download_file_from_storage(file_name="...") -> downloads
```

## Limitations

- MCP tools require FastMCP to be installed and configured
- Some MCP tools may require specific permissions or API keys
- Tool execution is subject to rate limits and timeouts
- Large responses may be truncated by the LLM context window

## Future Enhancements

Potential improvements:
- Role-based access control for MCP tools
- Tool usage analytics and monitoring
- Custom tool grouping and organization
- Tool parameter validation and defaults
- Interactive tool configuration UI
- Tool execution history and replay

## Troubleshooting

### Tools Not Loading

If MCP tools fail to load:

1. Check that FastMCP is installed:
   ```bash
   pip list | grep fastmcp
   ```

2. Verify MCP server imports correctly:
   ```python
   from nodetool.api.mcp_server import mcp
   ```

3. Check logs for error messages:
   ```
   Failed to load MCP tools: <error message>
   ```

### Tool Not Found

If a specific tool cannot be resolved:

1. Verify the tool exists in MCP server:
   ```python
   from nodetool.agents.tools.mcp_tools import get_all_mcp_tools
   tools = await get_all_mcp_tools()
   print([t.name for t in tools])
   ```

2. Check tool is enabled in CLI:
   ```
   /tools <tool_name>
   ```

3. Verify tool name matches exactly (case-sensitive)

### Tool Execution Fails

If tool execution fails:

1. Check tool parameters match expected schema
2. Verify user has necessary permissions
3. Check API keys and credentials are configured
4. Review tool logs for detailed error messages

## References

- [MCP Server Implementation](../api/mcp_server.py)
- [Tool Base Class](base.py)
- [Tool Registry](tool_registry.py)
- [Chat CLI](../../chat/chat_cli.py)
- [Agent Architecture](../../agents/README.md)
