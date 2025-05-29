# Agent System

NodeTool includes an agent framework that breaks complex objectives into smaller subtasks.
Agents use language models and tools to accomplish each step.

Key features include:

- **Smart Task Planning** – objectives are converted into structured plans.
- **Tool Integration** – built in tools for web browsing, file management and more.
- **Independent Subtasks** – each subtask runs in its own context for reliability.
- **Parallel Execution** – independent tasks can run concurrently.
- **Workspace** – subtasks read and write files in a dedicated workspace directory.

See the source [README](../src/nodetool/agents/README.md) for a detailed architecture overview and example usage.

## MCP Tools

The `MCPTool` class makes it easy to call tools hosted on any [Model Context Protocol](https://modelcontextprotocol.io) server.
Configure the endpoint and token via the `MCP_API_URL` and `MCP_TOKEN` environment variables. Tools can be instantiated dynamically:

```python
from nodetool.agents.tools import MCPTool

summarize = MCPTool(
    tool="summarize",
    description="Summarize a block of text using the remote MCP service",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
```

During execution the tool posts the parameters to the configured MCP server and returns the JSON response.
