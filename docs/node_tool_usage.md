[← Back to Docs Index](index.md)

# NodeTool Usage Guide

The `NodeTool` class provides a way to wrap individual `BaseNode` instances as tools for use by agents, enabling
seamless integration between the workflow node system and the agent system.

## Overview

NodeTool allows you to:

- Convert any BaseNode into a tool that agents can use
- Automatically generate tool metadata from node properties
- Execute nodes with proper error handling and result formatting
- Maintain compatibility with the existing workflow system

## Basic Usage

### Creating a NodeTool from a Node Class

```python
from nodetool.agents.tools.node_tool import NodeTool
from nodetool.workflows.base_node import BaseNode
from pydantic import Field

# Define a custom node
class TextProcessorNode(BaseNode):
    text: str = Field(description="Text to process")
    uppercase: bool = Field(default=False, description="Convert to uppercase")

    async def process(self, context):
        result = self.text.upper() if self.uppercase else self.text
        return {"output": result}

# Create a tool from the node
text_tool = NodeTool(TextProcessorNode)
```

### Using NodeTool with Agents

```python
# In an agent's tool list
tools = [
    NodeTool(TextProcessorNode, name="process_text"),
    NodeTool(MathOperationNode, name="calculate"),
    # ... other tools
]

# The agent can then use these tools by name
await agent.use_tool("process_text", {"text": "hello", "uppercase": True})
```

### Creating NodeTool from Node Type String

```python
# Create a tool from a registered node type
tool = NodeTool.from_node_type("nodetool.text.Concatenate")
```

## Features

### Automatic Schema Generation

NodeTool automatically generates JSON schemas from node properties:

```python
tool = NodeTool(MyNode)
print(tool.input_schema)
# Output: {'type': 'object', 'properties': {...}, 'required': [...]}
```

### Error Handling

NodeTool provides comprehensive error handling:

```python
result = await tool.process(context, params)
if result["status"] == "failed":
    print(f"Error: {result['error']}")
else:
    print(f"Result: {result['result']}")
```

### Custom Tool Names

You can specify custom tool names:

```python
tool = NodeTool(MyNode, name="my_custom_tool")
```

## Result Structure

NodeTool returns results in a consistent format:

```python
{
    "node_type": "namespace.NodeName",
    "status": "completed" | "failed",
    "result": {
        # Node output here (format depends on node's return type)
    },
    "error": "Error message if failed",
    "traceback": "Full traceback if failed"
}
```

## Integration with Workflow System

NodeTool maintains full compatibility with the workflow system:

- Proper initialization and finalization of nodes
- Context passing for asset management and API access
- Support for all node types including streaming nodes
- Automatic output conversion based on node return types

## Best Practices

1. **Use descriptive field descriptions**: These become part of the tool's schema
1. **Handle required fields properly**: Ensure nodes can be created with minimal parameters
1. **Implement proper error handling**: Nodes should raise clear exceptions
1. **Follow node conventions**: Use standard return formats for consistency

## Examples

See `examples/node_tool_example.py` for complete working examples demonstrating:

- Basic node wrapping
- Complex nodes with multiple parameters
- Error handling
- Integration patterns

## Implementation Details

NodeTool handles several complexities:

- Dynamic property assignment for nodes with required fields
- Automatic conversion between snake_case tool names and CamelCase node names
- Proper async context management
- Node lifecycle management (initialize, process, finalize)
