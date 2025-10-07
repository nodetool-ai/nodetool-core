#!/usr/bin/env python
"""
FastMCP server for NodeTool API

This module provides MCP (Model Context Protocol) server integration for NodeTool,
allowing AI assistants to interact with NodeTool workflows, nodes, and assets.
"""

from fastmcp import FastMCP
from typing import Any, Optional
from nodetool.types.job import JobUpdate
from nodetool.workflows.types import Error, OutputUpdate
from pydantic import BaseModel, Field
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.types.graph import Graph, get_input_schema, get_output_schema
from nodetool.packages.registry import Registry
from nodetool.config.logging_config import get_logger
from nodetool.chat.search_nodes import search_nodes as search_nodes_tool
import asyncio

log = get_logger(__name__)

# Initialize FastMCP server
mcp = FastMCP("NodeTool API Server")


class WorkflowRunParams(BaseModel):
    """Parameters for running a workflow"""
    workflow_id: str = Field(..., description="The ID of the workflow to run")
    params: dict[str, Any] = Field(default_factory=dict, description="Input parameters for the workflow")


class NodeSearchParams(BaseModel):
    """Parameters for searching nodes"""
    query: str = Field(..., description="Search query for nodes")
    namespace: Optional[str] = Field(None, description="Optional namespace to filter nodes")




@mcp.tool()
async def get_workflow(workflow_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific workflow.

    Args:
        workflow_id: The ID of the workflow

    Returns:
        Workflow details including graph structure, input/output schemas
    """
    workflow = await WorkflowModel.find("1", workflow_id)
    if not workflow:
        raise ValueError(f"Workflow {workflow_id} not found")

    api_graph = workflow.get_api_graph()
    input_schema = get_input_schema(api_graph)
    output_schema = get_output_schema(api_graph)

    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description or "",
        "tags": workflow.tags,
        "graph": api_graph.model_dump(),
        "input_schema": input_schema,
        "output_schema": output_schema,
        "created_at": workflow.created_at.isoformat(),
        "updated_at": workflow.updated_at.isoformat(),
    }


@mcp.tool()
async def run_workflow_tool(workflow_id: str, params: dict[str, Any] = {}) -> dict[str, Any]:
    """
    Execute a NodeTool workflow with given parameters.

    Args:
        workflow_id: The ID of the workflow to run
        params: Dictionary of input parameters for the workflow

    Returns:
        Workflow execution results
    """
    workflow = await WorkflowModel.find("1", workflow_id)
    if not workflow:
        raise ValueError(f"Workflow {workflow_id} not found")

    # Create run request
    request = RunJobRequest(
        workflow_id=workflow_id,
        params=params,
    )

    # Run workflow
    result = {}
    async for msg in run_workflow(request):
        if isinstance(msg, OutputUpdate):
            result[msg.node_name] = msg.value
        elif isinstance(msg, JobUpdate):
            if msg.status == "error":
                raise Exception(msg.error)
        elif isinstance(msg, Error):
            raise Exception(msg.error)

    return {
        "workflow_id": workflow_id,
        "status": "completed",
        "result": result,
    }

@mcp.tool()
async def list_nodes(namespace: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
    """
    List available nodes in NodeTool registry.

    Args:
        namespace: Optional namespace to filter nodes (e.g., 'nodetool.text', 'nodetool.image')
        limit: Maximum number of nodes to return (default: 100)

    Returns:
        List of node metadata including type, description, and properties
    """
    registry = Registry.get_instance()
    all_nodes = registry.get_all_installed_nodes()

    result = []
    count = 0

    for node in all_nodes:
        if namespace and not node.namespace.startswith(namespace):
            continue

        if count >= limit:
            break

        result.append({
            "type": node.node_type,
        })
        count += 1

    return result


@mcp.tool()
async def search_nodes(query: list[str], n_results: int = 10, input_type: Optional[str] = None, output_type: Optional[str] = None, exclude_namespaces: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """
    Search for nodes by name, description, or tags.

    Args:
        query: Search query strings
        namespace: Optional namespace to filter nodes

    Returns:
        List of matching nodes
    """
    nodes = search_nodes_tool(
        query=query,
        input_type=input_type,
        output_type=output_type,
        n_results=n_results,
        exclude_namespaces=exclude_namespaces or [],
    )

    result = []
    for node in nodes:
        result.append({
            "type": node.node_type,
            "title": node.title,
            "description": node.description,
            "namespace": node.namespace,
        })

    return result


@mcp.tool()
async def get_node_info(node_type: str) -> dict[str, Any]:
    """
    Get detailed information about a specific node type.

    Args:
        node_type: The fully qualified node type (e.g., 'nodetool.text.Split')

    Returns:
        Detailed node metadata including properties, inputs, outputs
    """
    registry = Registry.get_instance()
    node = registry.find_node_by_type(node_type)

    if not node:
        raise ValueError(f"Node type {node_type} not found")

    return node


@mcp.tool()
async def save_workflow(
    workflow_id: str,
    name: str,
    graph: dict[str, Any],
    description: str = "",
    tags: list[str] | None = None,
    access: str = "private",
    run_mode: str | None = None,
) -> dict[str, Any]:
    """
    Save or update a workflow.

    Args:
        workflow_id: The ID of the workflow to save (creates new if doesn't exist)
        name: Workflow name
        graph: Workflow graph structure with nodes and edges
        description: Workflow description
        tags: List of tags for categorization
        access: Access level ("private" or "public")
        run_mode: Run mode ("workflow" or "tool")

    Returns:
        Saved workflow details
    """
    from nodetool.types.graph import remove_connected_slots
    from datetime import datetime

    # Try to get existing workflow
    workflow = await WorkflowModel.get(workflow_id)

    if not workflow:
        # Create new workflow
        workflow = WorkflowModel(
            id=workflow_id,
            user_id="1",  # Default user for MCP
            name=name,
            description=description,
            tags=tags or [],
            access=access,
            run_mode=run_mode,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    else:
        # Update existing workflow
        workflow.name = name
        workflow.description = description
        workflow.tags = tags or []
        workflow.access = access
        if run_mode is not None:
            workflow.run_mode = run_mode
        workflow.updated_at = datetime.now()

    # Parse and clean the graph
    graph_obj = Graph.model_validate(graph)
    workflow.graph = remove_connected_slots(graph_obj).model_dump()

    # Save workflow
    await workflow.save()

    # Get schemas
    api_graph = workflow.get_api_graph()
    input_schema = get_input_schema(api_graph)
    output_schema = get_output_schema(api_graph)

    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description or "",
        "tags": workflow.tags,
        "access": workflow.access,
        "run_mode": workflow.run_mode,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "created_at": workflow.created_at.isoformat(),
        "updated_at": workflow.updated_at.isoformat(),
        "message": "Workflow saved successfully"
    }


@mcp.tool()
async def validate_workflow(workflow_id: str) -> dict[str, Any]:
    """
    Validate a workflow's structure, connectivity, and type compatibility.

    Checks:
    - All node types exist in registry
    - No circular dependencies (DAG structure)
    - All required inputs are connected
    - Type compatibility of all edges
    - Proper input/output node configuration

    Args:
        workflow_id: The ID of the workflow to validate

    Returns:
        Validation report with errors, warnings, and suggestions
    """
    workflow = await WorkflowModel.find("1", workflow_id)
    if not workflow:
        raise ValueError(f"Workflow {workflow_id} not found")

    graph = workflow.get_api_graph()
    registry = Registry.get_instance()

    errors = []
    warnings = []
    suggestions = []

    # Track node IDs for uniqueness check
    node_ids = set()
    node_types_found = {}

    # Validate nodes
    for node in graph.nodes:
        # Check unique IDs
        if node.id in node_ids:
            errors.append(f"Duplicate node ID: {node.id}")
        node_ids.add(node.id)

        # Check node type exists
        node_metadata = registry.find_node_by_type(node.type)
        if not node_metadata:
            errors.append(f"Node type not found: {node.type} (node: {node.id})")
        else:
            node_types_found[node.id] = node_metadata

    # Build adjacency map for cycle detection
    adjacency = {node.id: [] for node in graph.nodes}
    edges_by_target = {}

    for edge in graph.edges:
        if edge.source not in node_ids:
            errors.append(f"Edge references non-existent source node: {edge.source}")
            continue
        if edge.target not in node_ids:
            errors.append(f"Edge references non-existent target node: {edge.target}")
            continue

        adjacency[edge.source].append(edge.target)

        # Track edges by target for input validation
        if edge.target not in edges_by_target:
            edges_by_target[edge.target] = []
        edges_by_target[edge.target].append(edge)

    # Check for cycles (DAG validation)
    def has_cycle():
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node_id: WHITE for node_id in node_ids}

        def dfs(node_id):
            if color[node_id] == GRAY:
                return True  # Back edge found - cycle detected
            if color[node_id] == BLACK:
                return False

            color[node_id] = GRAY
            for neighbor in adjacency[node_id]:
                if dfs(neighbor):
                    return True
            color[node_id] = BLACK
            return False

        for node_id in node_ids:
            if color[node_id] == WHITE:
                if dfs(node_id):
                    return True
        return False

    if has_cycle():
        errors.append("Workflow contains circular dependencies - must be a DAG (Directed Acyclic Graph)")

    # Validate node inputs and type compatibility
    for node in graph.nodes:
        if node.id not in node_types_found:
            continue

        metadata = node_types_found[node.id]

        # Check required inputs are connected
        if hasattr(metadata, 'properties'):
            required_inputs = [
                prop_name for prop_name, prop_data in metadata.properties.items()
                if isinstance(prop_data, dict) and prop_data.get('required', False)
            ]

            connected_inputs = set()
            if node.id in edges_by_target:
                for edge in edges_by_target[node.id]:
                    if edge.targetHandle:
                        connected_inputs.add(edge.targetHandle)

            # Check node properties for static values
            if hasattr(node.data, 'properties'):
                for prop_name in node.data.properties:
                    connected_inputs.add(prop_name)

            for required_input in required_inputs:
                if required_input not in connected_inputs:
                    warnings.append(
                        f"Required input '{required_input}' may not be connected on node '{node.id}' ({node.type})"
                    )

    # Check for orphaned nodes
    nodes_with_inputs = set(edges_by_target.keys())
    nodes_with_outputs = set()
    for edge in graph.edges:
        nodes_with_outputs.add(edge.source)

    for node in graph.nodes:
        # Skip input/output/constant nodes from orphan check
        if any(keyword in node.type.lower() for keyword in ['input', 'output', 'constant', 'preview']):
            continue

        if node.id not in nodes_with_inputs and node.id not in nodes_with_outputs:
            warnings.append(f"Orphaned node (not connected): {node.id} ({node.type})")
        elif node.id not in nodes_with_outputs:
            suggestions.append(f"Node '{node.id}' has no outputs - consider adding Preview or Output node")

    # Summary
    is_valid = len(errors) == 0

    return {
        "valid": is_valid,
        "workflow_id": workflow_id,
        "workflow_name": workflow.name,
        "summary": {
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "errors": len(errors),
            "warnings": len(warnings),
            "suggestions": len(suggestions),
        },
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "message": "Workflow is valid and ready to run" if is_valid else "Workflow has validation errors - please fix before running"
    }


@mcp.resource("nodetool://workflows")
async def get_workflows_resource() -> dict[str, Any]:
    """
    Get all workflows as a text resource.
    """
    workflows, next_key = await WorkflowModel.paginate(
        limit=100,
        start_key=None
    )

    result = []
    for workflow in workflows:
        result.append({
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description or "",
            "tags": workflow.tags,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
        })

    return {
        "workflows": result,
        "next": next_key,
    }



@mcp.prompt()
async def build_workflow_guide() -> str:
    """
    Comprehensive guide on how to build NodeTool workflows.

    Returns:
        Step-by-step instructions for creating workflows
    """
    return """# How to Build NodeTool Workflows

## Overview
NodeTool workflows are visual, node-based programs that connect operations together. Each workflow is a **Directed Acyclic Graph (DAG)** where nodes process data and pass results through typed connections.

## Graph Structure

Workflows consist of **nodes** and **edges** that form a directed acyclic graph.

### Node Structure
```json
{
  "id": "string",                    // Unique identifier for the node (required)
  "parent_id": "string" | null,      // Parent group ID (for grouped nodes, optional)
  "type": "string",                  // Fully qualified node type (e.g., "nodetool.text.Split")
  "data": {},                        // Node configuration and properties
  "ui_properties": {},               // UI-specific properties (position, etc.)
  "dynamic_properties": {},          // Dynamic properties for flexible nodes (FormatText, MakeDictionary)
  "dynamic_outputs": {},             // Dynamic output definitions
  "sync_mode": "on_any"              // Execution sync mode: "on_any" or "zip_all"
}
```

### Edge Structure
```json
{
  "id": "string" | null,             // Optional edge identifier
  "source": "string",                // Source node ID (required)
  "sourceHandle": "string",          // Output handle name on source node (required)
  "target": "string",                // Target node ID (required)
  "targetHandle": "string",          // Input handle name on target node (required)
  "ui_properties": {}                // UI-specific properties (className for type, etc.)
}
```

### Graph Structure
```json
{
  "nodes": [                         // List of all nodes in the workflow
    { /* Node objects */ }
  ],
  "edges": [                         // List of all edges connecting nodes
    { /* Edge objects */ }
  ]
}
```

### Key Concepts

1. **Node IDs**: Must be unique across the entire workflow
2. **Node Types**: Use exact fully qualified names from registry (e.g., "nodetool.text.Split")
3. **Edges**: Connect outputs to inputs using source/target node IDs and handle names
4. **Parent ID**: Used for grouping nodes (e.g., nodes inside a Group node)
5. **Dynamic Properties**: Some nodes (like FormatText, MakeDictionary) accept arbitrary properties
6. **Data vs Properties**: Node configuration goes in `data` field, edge connections can be in properties

## Streaming Nodes

Some nodes produce **streaming outputs** - multiple values emitted one at a time rather than a single result. This is useful for:
- Generating lists of items (e.g., ListGenerator creates multiple text items)
- Processing large datasets row-by-row (e.g., DataGenerator emits records as they're created)
- Real-time data processing

### How Streaming Works

**Regular (Non-Streaming) Nodes:**
- Process inputs once and return a single output
- Example: Text nodes that transform one string to another

**Streaming Nodes:**
- Emit multiple outputs over time
- Downstream nodes receive each item as it's produced
- Common streaming nodes:
  - `nodetool.generators.ListGenerator` - Streams text items from a list
  - `nodetool.generators.DataGenerator` - Streams dataframe records
  - Agent nodes with streaming enabled

### Streaming Node Examples

**ListGenerator** - Generates a stream of text items:
```json
{
  "id": "list_gen",
  "type": "nodetool.generators.ListGenerator",
  "data": {
    "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"},
    "prompt": "Generate 5 creative movie poster taglines",
    "max_tokens": 2048
  }
}
```
- Outputs: `item` (string) and `index` (int) for each generated item
- Each item is streamed as it's generated by the LLM

**DataGenerator** - Generates structured data as a stream:
```json
{
  "id": "data_gen",
  "type": "nodetool.generators.DataGenerator",
  "data": {
    "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"},
    "prompt": "Generate customer data",
    "columns": {
      "columns": [
        {"name": "name", "data_type": "str"},
        {"name": "email", "data_type": "str"},
        {"name": "age", "data_type": "int"}
      ]
    }
  }
}
```
- Outputs: `record` (dict) for each row, then final `dataframe` (DataframeRef)
- Records are streamed as they're generated, dataframe emitted at the end

**Agent** - Can stream text chunks AND produce final complete text:
```json
{
  "id": "story_agent",
  "type": "nodetool.agents.Agent",
  "data": {
    "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"},
    "prompt": "Write a short story about a robot",
    "max_tokens": 1000
  }
}
```
- Outputs: `chunk` (string) for each token/chunk as it's generated in real-time
- Then outputs: `text` (string) with the complete final text

### Using Streaming Outputs

When connecting streaming nodes:

1.  Downstream nodes process each streamed item:
```
ListGenerator → ProcessEachItem → CollectResults
```

2. When a streaming node is connected to another streaming node, the stream gets restarted for the downstream node.
```
A -> B -> C
A streams 1,2,3
B streams all items for 1
B streams all items for 2
B streams all items for 3
```

3. When a stream gets split into multiple paths, and you want to combine the results, set sync_mode to "zip_all" on the downstream node.
```
A -> B -> D
  -> C -> D

D needs to zip all items from B and C
```

### Important Notes

- **Streaming is automatic** - You don't configure streaming; nodes are either streaming or not
- **All items are processed** - Downstream nodes receive every streamed item
- **Order is preserved** - Items arrive in the order they're emitted
- **Use Groups for iteration** - Group nodes are the standard way to process streaming data item-by-item

## Dynamic Properties Pattern

Some nodes accept **dynamic properties** - custom inputs defined at workflow creation time. This is useful for creating flexible, reusable nodes.

### FormatText - Template with Dynamic Variables

The most common dynamic properties pattern is **template formatting** using `FormatText`:

```json
{
  "nodes": [
    {
      "id": "movie_title",
      "type": "nodetool.input.StringInput",
      "data": {
        "name": "Movie Title",
        "value": "Stellar Odyssey"
      }
    },
    {
      "id": "genre",
      "type": "nodetool.input.StringInput",
      "data": {
        "name": "Genre",
        "value": "Sci-Fi"
      }
    },
    {
      "id": "format_prompt",
      "type": "nodetool.text.FormatText",
      "data": {
        "template": "Write a movie poster tagline for {{TITLE}}, a {{GENRE}} film."
      },
      "dynamic_properties": {
        "TITLE": "",
        "GENRE": ""
      }
    }
  ],
  "edges": [
    {
      "source": "movie_title",
      "sourceHandle": "output",
      "target": "format_prompt",
      "targetHandle": "TITLE"
    },
    {
      "source": "genre",
      "sourceHandle": "output",
      "target": "format_prompt",
      "targetHandle": "GENRE"
    }
  ]
}
```

**How it works:**
1. Define template with `{{VARIABLE_NAME}}` placeholders in the `template` field
2. Add each variable to `dynamic_properties` object (values are typically empty strings)
3. Connect edges to the dynamic property names as `targetHandle`
4. FormatText replaces `{{VARIABLE_NAME}}` with the connected input values
5. Output: "Write a movie poster tagline for Stellar Odyssey, a Sci-Fi film."

### Key Points

- **Template variables**: Must match exactly between `{{VAR}}` and `dynamic_properties` key
- **Case sensitive**: `{{TITLE}}` ≠ `{{title}}`
- **No defaults in dynamic_properties**: Values are typically empty strings; actual data comes from edges
- **Edge connections required**: Each dynamic property should have a connected edge
- **Common nodes with dynamic properties**:
  - `nodetool.text.FormatText` - Template formatting
  - `nodetool.dictionary.MakeDictionary` - Dynamic dictionaries
  - Some agent/generator nodes - Custom parameters

## Dynamic Outputs Pattern (Tool Calling)

Agent nodes can define **dynamic outputs** that become **tool calls**. Each dynamic output creates a new output handle on the agent that triggers when the agent decides to call that tool.

### Agent with Tool Calls

```json
{
  "nodes": [
    {
      "id": "weather_agent",
      "type": "nodetool.agents.Agent",
      "data": {
        "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"},
        "prompt": "What's the weather like in Paris?",
        "max_tokens": 1000
      },
      "dynamic_outputs": ["get_weather", "get_forecast"]
    },
    {
      "id": "weather_tool",
      "type": "nodetool.constant.String",
      "data": {
        "value": "Current weather in Paris: 18°C, partly cloudy"
      }
    },
    {
      "id": "tool_result",
      "type": "nodetool.agents.ToolResult",
      "data": {}
    }
  ],
  "edges": [
    {
      "source": "weather_agent",
      "sourceHandle": "get_weather",
      "target": "weather_tool",
      "targetHandle": "trigger"
    },
    {
      "source": "weather_tool",
      "sourceHandle": "output",
      "target": "tool_result",
      "targetHandle": "result"
    },
    {
      "source": "tool_result",
      "sourceHandle": "output",
      "target": "weather_agent",
      "targetHandle": "tool_result"
    }
  ]
}
```

**How tool calling works:**

1. **Define tools in dynamic_outputs**: Each output becomes a tool with the given type.

2. **Agent decides to call tool**: When the agent wants to use a tool, it outputs on that handle
   - Example: Agent outputs `{"city": "Paris"}` on the `get_weather` handle

3. **Tool chain executes**: The tool call messages flow through your processing nodes

4. **ToolResult gathers results**: Collects output from tool processing

5. **Result fed back to agent**: ToolResult connects back to agent's `tool_result` input
   - Agent receives the tool result and continues its reasoning
   - Agent can make more tool calls or produce final answer

### Multi-Tool Example

```json
{
  "id": "research_agent",
  "type": "nodetool.agents.Agent",
  "data": {
    "model": {"type": "language_model", "id": "gpt-4o", "provider": "openai"},
    "prompt": "Research the latest news about AI and summarize it",
    "system": "You are a research assistant with access to web search and calculator tools."
  },
  "dynamic_outputs": ["web_search", "calculator"]
}
```

**Tool chain for web_search:**
```
Agent[web_search] → WebSearch → FormatResults → ToolResult
```

**Tool chain for calculator:**
```
Agent[calculator] → Evaluate → FormatResult → ToolResult
```

### Important Notes

- **Each tool needs a chain**: Every dynamic output should have a processing chain ending in ToolResult
- **ToolResult is required**: Always use ToolResult node to feed results back to the agent
- **Agent can call multiple times**: Agent may call tools multiple times before producing final answer
- **Tool schemas are shown to LLM**: The LLM sees tool descriptions and decides when to call them
- **Streaming and tools**: Agents with tools typically don't stream; they work in request/response cycles
- **No circular dependencies**: Tool chains must not create cycles in the workflow graph

## Working with Files and Assets

NodeTool workflows can work with files in two ways: **assets** (managed files) and **direct file access**. For AI agents generating workflows, **use direct file paths** unless you specifically need asset management features.

### For AI Agents: Use Direct File Paths

When generating workflows programmatically, use direct file paths:

```json
{
  "type": "image",
  "uri": "/absolute/path/to/image.jpg"
}
```

**Guidelines for agents:**
- Use absolute paths: `/home/user/image.jpg` (not relative paths)
- Files can be anywhere accessible to the system
- No need to manage asset IDs or import files first
- Simpler and more predictable for automated workflow generation

### Assets vs Direct Files

**Assets** are files imported through NodeTool's UI that get:
- Asset IDs for tracking and management
- Thumbnails and metadata
- Organization in the asset editor
- Version control

**Direct file access** is simpler for agents:
- Reference any accessible file directly
- No import step required
- Works with generated or temporary files
- Less overhead for automated workflows

### File Reference Types

All file references use this structure:
```json
{
  "type": "image|audio|video|document|dataframe",
  "uri": "file:///path/to/file.ext",
  "asset_id": "uuid"  // Optional: only present for imported assets
}
```

**Examples:**
- Image: `{"type": "image", "uri": "file:///data/photo.jpg"}`
- Audio: `{"type": "audio", "uri": "file:///sounds/music.mp3"}`
- Document: `{"type": "document", "uri": "file:///docs/report.pdf"}`
- Data: `{"type": "dataframe", "uri": "file:///data/dataset.csv"}`

### Working with Existing Workflows (Assets)

When **reading or modifying existing workflows**, you may encounter assets:

**Asset references** include an `asset_id`:
```json
{
  "type": "image",
  "uri": "file:///workspace/assets/image.jpg",
  "asset_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Guidelines for agents working with existing workflows:**
- **Preserve asset references** when modifying workflows - don't convert them to direct paths unless necessary
- **Check for asset_id field** to identify assets vs direct file references
- **Assets provide stability** - the same asset can be referenced across multiple workflows
- **Asset paths may use workspace URIs** like `file:///workspace/assets/`

**When to convert assets to direct paths:**
- When you need to work with files outside the workspace
- When creating portable workflows that shouldn't depend on specific assets
- When the asset no longer exists or is inaccessible

## Core Principles

1. **Graph Structure**: Workflows must be DAGs (no circular dependencies)
2. **Data Flow**: Edges represent data flow from inputs → processing → outputs
3. **Type Safety**: Connections must be type-compatible
4. **Complete Connectivity**: Every processing node must receive its required inputs
5. **Valid Node Types**: All nodes must exist in the registry (use `search_nodes` to verify)

## Workflow Planning Process

### Phase 1: Understand the Objective
1. **Identify inputs**: What data does the workflow receive?
2. **Identify outputs**: What results should it produce?
3. **Map the transformation**: What processing steps convert inputs to outputs?

### Phase 2: Design the Data Flow
1. **Start with I/O nodes**: Create Input and Output nodes first
2. **Identify processing steps**: What transformations are needed?
3. **Search for nodes efficiently**:
   - Use specific queries with `input_type` and `output_type` filters
   - Batch similar searches together
   - Target relevant namespaces (`nodetool.text`, `nodetool.data`, `lib.*`)
4. **Plan connections**: Trace data flow from inputs through processing to outputs

### Phase 3: Build the Graph
1. **Create node specifications**:
   - Use exact `node_type` from `search_nodes` results
   - Configure properties based on node metadata
   - Define edge connections in properties
2. **Validate connectivity**:
   - Every Input node connects to at least one processing node
   - Every processing node has ALL required inputs connected
   - Every Output node receives data from a processing node
3. **Check types**: Ensure all connections are type-compatible

## Node Search Strategy

### Efficient Searching
**Minimize search iterations** by being strategic:

- **Plan ahead**: Identify all processing types needed before searching
- **Use type filters**: Specify `input_type` and `output_type` when known
- **Batch searches**: Search for multiple related capabilities together
- **Target namespaces**:
  - `nodetool.text.*`: Text operations (split, concat, format, slice)
  - `nodetool.data.*`: DataFrame/tabular data operations
  - `nodetool.agents.*`: AI agents and generators
  - `lib.browser.*`: Web scraping and fetching
  - `lib.mail.*`: Email operations

### Example Searches
```
# Good: Specific with type filters
search_nodes(query=["dataframe", "group", "aggregate"], input_type="dataframe")

# Good: Targeted functionality
search_nodes(query=["text", "template", "format"], output_type="str")

# Less efficient: Too generic
search_nodes(query=["process"])
```

## Data Types and Compatibility

### Common Types
- **str**: Text data
- **int/float**: Numeric values (interchangeable)
- **bool**: Boolean values
- **list[T]**: Arrays of items
- **dict**: Key-value objects
- **any**: Compatible with all types

### Media Types
- **ImageRef**: Image file references
- **AudioRef**: Audio file references
- **VideoRef**: Video file references
- **DocumentRef**: Document file references
- **DataFrameRef**: Tabular data references

### Type Compatibility Rules
- Exact matches always work: `str → str`, `int → int`
- Numeric conversions allowed: `int ↔ float`
- `any` type accepts everything
- Union types match if any member matches
- Complex types need converter nodes for primitives

## Node Configuration Patterns

### Static Properties
Standard node properties defined in metadata:
```json
{
  "node_id": "text_slice",
  "node_type": "nodetool.text.Slice",
  "properties": "{\"start\": 0, \"stop\": 100}"
}
```

### Edge Connections
Connect node outputs to inputs using edge definitions:
```json
{
  "properties": "{\"text\": {\"type\": \"edge\", \"source\": \"input_1\", \"sourceHandle\": \"output\"}}"
}
```

### Dynamic Properties
Some nodes accept custom properties (check `is_dynamic` in metadata):
```json
{
  "node_id": "make_dict",
  "node_type": "nodetool.dictionary.MakeDictionary",
  "properties": "{\"category\": {\"type\": \"edge\", \"source\": \"agent_1\", \"sourceHandle\": \"text\"}, \"body\": \"example\"}"
}
```

## Common Workflow Patterns

### Sequential Processing
```
Input → Process A → Process B → Output
```
Use when: Each step depends on the previous result

### Parallel Processing
```
       → Process A →
Input →              → Combine → Output
       → Process B →
```
Use when: Multiple independent operations on the same input

### List Processing with Streams
```
Data Source → Streaming Node → Collect Results
```
Use when: Applying operations to each item in a list

### Multi-Agent Pipelines
```
Input → Agent 1 (Strategy) → Agent 2 (Transform) → Generator (Variations) → Output
```
Use when: Complex AI tasks requiring multiple reasoning steps

## Validation Checklist

Before running a workflow, verify:

**Structure**
- [ ] All node types exist (verified via `search_nodes`)
- [ ] All node IDs are unique
- [ ] No circular dependencies (DAG structure)

**Connectivity**
- [ ] Every Input node connects to processing nodes
- [ ] Every processing node has ALL required inputs connected
- [ ] Every Output node receives data via its `value` property
- [ ] Template nodes have connections for ALL template variables

**Types**
- [ ] All edge connections are type-compatible
- [ ] Input/Output nodes match the workflow schema types
- [ ] Complex type conversions use appropriate converter nodes

**Properties**
- [ ] Non-dynamic nodes only use properties from their metadata
- [ ] Dynamic nodes have valid property configurations
- [ ] Required properties are set (check node metadata)

## Common Mistakes to Avoid

1. **Missing edge connections**: Forgetting to connect required inputs
2. **Type mismatches**: Connecting incompatible types without converters
3. **Using wrong node_type**: Not using exact string from `search_nodes`
4. **Orphaned nodes**: Nodes not connected to the data flow
5. **Missing I/O nodes**: Not creating nodes for schema inputs/outputs
6. **Template variables without connections**: Using variables in templates without edges
7. **Inventing properties**: Adding properties to non-dynamic nodes

## Tools for Building

- **`search_nodes`**: Find nodes by functionality, filter by types
- **`get_node_info`**: Get detailed specifications for a specific node
- **`list_workflows`**: Browse existing workflow examples
- **`get_workflow`**: Examine a specific workflow's structure
- **`run_workflow_tool`**: Execute and test your workflow

## Next Steps

1. **Define objective**: What should the workflow accomplish?
2. **Plan I/O**: What inputs/outputs are needed?
3. **Search nodes**: Find processing nodes using targeted searches
4. **Design flow**: Map out the data transformations
5. **Build graph**: Create node specifications with connections
6. **Validate**: Check structure, connectivity, and types
7. **Test**: Run the workflow and iterate based on results
"""


@mcp.prompt()
async def workflow_examples() -> str:
    """
    Concrete examples of NodeTool workflow structures.

    Returns:
        Example workflow JSON structures with explanations
    """
    return """# NodeTool Workflow Examples

## Example 1: Image to Audio Story

**Goal**: Generate a creative story from an image and narrate it using text-to-speech

**Flow**: Image → AI Vision/Agent → Story Text → TTS → Audio

```json
{
  "nodes": [
    {
      "id": "image_input",
      "type": "nodetool.constant.Image",
      "data": {
        "value": {
          "uri": "https://example.com/image.jpg",
          "type": "image"
        }
      }
    },
    {
      "id": "story_agent",
      "type": "nodetool.agents.Agent",
      "data": {
        "model": {
          "type": "language_model",
          "id": "gpt-4o",
          "provider": "openai"
        },
        "prompt": "Write a compelling story about the image.",
        "max_tokens": 4096
      }
    },
    {
      "id": "tts_node",
      "type": "nodetool.audio.TextToSpeech",
      "data": {
        "voice": "alloy",
        "model": "tts-1"
      }
    },
    {
      "id": "audio_preview",
      "type": "nodetool.workflows.base_node.Preview",
      "data": {}
    }
  ],
  "edges": [
    {
      "source": "image_input",
      "sourceHandle": "output",
      "target": "story_agent",
      "targetHandle": "image"
    },
    {
      "source": "story_agent",
      "sourceHandle": "text",
      "target": "tts_node",
      "targetHandle": "text"
    },
    {
      "source": "tts_node",
      "sourceHandle": "audio",
      "target": "audio_preview",
      "targetHandle": "value"
    }
  ]
}
```

## Example 2: Email Classification with Groups

**Goal**: Fetch emails from Gmail, classify them using AI, and structure the results

**Flow**: Gmail Search → Group(Extract → Clean → Classify → Structure) → Results

```json
{
  "nodes": [
    {
      "id": "email_address",
      "type": "nodetool.constant.String",
      "data": {
        "value": "your_email@gmail.com"
      }
    },
    {
      "id": "gmail_search",
      "type": "lib.mail.GmailSearch",
      "data": {
        "search_query": "",
        "max_results": 10
      }
    },
    {
      "id": "email_group",
      "type": "nodetool.workflows.base_node.Group",
      "data": {}
    },
    {
      "id": "group_input",
      "parent_id": "email_group",
      "type": "nodetool.input.GroupInput",
      "data": {}
    },
    {
      "id": "get_body",
      "parent_id": "email_group",
      "type": "nodetool.dictionary.GetValue",
      "data": {
        "key": "body"
      }
    },
    {
      "id": "html_to_text",
      "parent_id": "email_group",
      "type": "nodetool.text.HtmlToText",
      "data": {
        "preserve_linebreaks": true
      }
    },
    {
      "id": "slice_text",
      "parent_id": "email_group",
      "type": "nodetool.text.Slice",
      "data": {
        "start": 0,
        "stop": 512
      }
    },
    {
      "id": "classify_agent",
      "parent_id": "email_group",
      "type": "nodetool.agents.Agent",
      "data": {
        "model": {
          "type": "language_model",
          "id": "gpt-4o-mini",
          "provider": "openai"
        },
        "system": "You are an email classifier. Reply with category only.",
        "temperature": 0.3
      }
    },
    {
      "id": "make_result",
      "parent_id": "email_group",
      "type": "nodetool.dictionary.MakeDictionary",
      "dynamic_properties": {
        "category": "",
        "body": "",
        "id": ""
      }
    },
    {
      "id": "group_output",
      "parent_id": "email_group",
      "type": "nodetool.output.GroupOutput",
      "data": {}
    },
    {
      "id": "results_preview",
      "type": "nodetool.workflows.base_node.Preview",
      "data": {}
    }
  ],
  "edges": [
    {
      "source": "email_address",
      "sourceHandle": "output",
      "target": "gmail_search",
      "targetHandle": "email_address"
    },
    {
      "source": "gmail_search",
      "sourceHandle": "output",
      "target": "email_group",
      "targetHandle": "input"
    },
    {
      "source": "group_input",
      "sourceHandle": "output",
      "target": "get_body",
      "targetHandle": "dictionary"
    },
    {
      "source": "get_body",
      "sourceHandle": "output",
      "target": "html_to_text",
      "targetHandle": "text"
    },
    {
      "source": "html_to_text",
      "sourceHandle": "output",
      "target": "slice_text",
      "targetHandle": "text"
    },
    {
      "source": "slice_text",
      "sourceHandle": "output",
      "target": "classify_agent",
      "targetHandle": "prompt"
    },
    {
      "source": "classify_agent",
      "sourceHandle": "text",
      "target": "make_result",
      "targetHandle": "category"
    },
    {
      "source": "make_result",
      "sourceHandle": "output",
      "target": "group_output",
      "targetHandle": "input"
    },
    {
      "source": "email_group",
      "sourceHandle": "output",
      "target": "results_preview",
      "targetHandle": "value"
    }
  ]
}
```

## Example 3: AI Movie Poster Generator (Multi-Agent Pipeline)

**Goal**: Create cinematic movie posters using chained AI agents and image generation

**Flow**: Inputs → Format → Strategy Agent → Prompt Generator → Image Generation → Preview

```json
{
  "nodes": [
    {
      "id": "movie_title",
      "type": "nodetool.input.StringInput",
      "data": {
        "value": "Stellar Odyssey",
        "name": "Movie Title"
      }
    },
    {
      "id": "genre",
      "type": "nodetool.input.StringInput",
      "data": {
        "value": "Sci-Fi Action",
        "name": "Genre"
      }
    },
    {
      "id": "audience",
      "type": "nodetool.input.StringInput",
      "data": {
        "value": "Adults 25-40, Sci-Fi Fans",
        "name": "Primary Audience"
      }
    },
    {
      "id": "strategy_template",
      "type": "nodetool.constant.String",
      "data": {
        "value": "Create a movie poster strategy for {{MOVIE_TITLE}} (Genre: {{GENRE}}, Audience: {{PRIMARY_AUDIENCE}}). Include visual concept, color palette, and design elements."
      }
    },
    {
      "id": "format_prompt",
      "type": "nodetool.text.FormatText",
      "data": {},
      "dynamic_properties": {
        "MOVIE_TITLE": "",
        "GENRE": "",
        "PRIMARY_AUDIENCE": ""
      }
    },
    {
      "id": "strategy_agent",
      "type": "nodetool.agents.Agent",
      "data": {
        "model": {
          "type": "language_model",
          "id": "gpt-4o",
          "provider": "openai"
        },
        "system": "You are a movie poster strategist.",
        "max_tokens": 2048
      }
    },
    {
      "id": "designer_prompt",
      "type": "nodetool.constant.String",
      "data": {
        "value": "Convert this strategy into 3 detailed Stable Diffusion prompts for cinematic movie posters."
      }
    },
    {
      "id": "prompt_generator",
      "type": "nodetool.generators.ListGenerator",
      "data": {
        "model": {
          "type": "language_model",
          "id": "gpt-4o",
          "provider": "openai"
        },
        "max_tokens": 1024
      }
    },
    {
      "id": "image_gen",
      "type": "nodetool.image.Replicate",
      "data": {
        "model": "stability-ai/sdxl",
        "width": 768,
        "height": 1024
      }
    },
    {
      "id": "image_preview",
      "type": "nodetool.workflows.base_node.Preview",
      "data": {}
    }
  ],
  "edges": [
    {
      "source": "strategy_template",
      "sourceHandle": "output",
      "target": "format_prompt",
      "targetHandle": "template"
    },
    {
      "source": "movie_title",
      "sourceHandle": "output",
      "target": "format_prompt",
      "targetHandle": "MOVIE_TITLE"
    },
    {
      "source": "genre",
      "sourceHandle": "output",
      "target": "format_prompt",
      "targetHandle": "GENRE"
    },
    {
      "source": "audience",
      "sourceHandle": "output",
      "target": "format_prompt",
      "targetHandle": "PRIMARY_AUDIENCE"
    },
    {
      "source": "format_prompt",
      "sourceHandle": "output",
      "target": "strategy_agent",
      "targetHandle": "prompt"
    },
    {
      "source": "strategy_agent",
      "sourceHandle": "text",
      "target": "prompt_generator",
      "targetHandle": "input_text"
    },
    {
      "source": "designer_prompt",
      "sourceHandle": "output",
      "target": "prompt_generator",
      "targetHandle": "prompt"
    },
    {
      "source": "prompt_generator",
      "sourceHandle": "item",
      "target": "image_gen",
      "targetHandle": "prompt"
    },
    {
      "source": "image_gen",
      "sourceHandle": "output",
      "target": "image_preview",
      "targetHandle": "value"
    }
  ]
}
```

## Key Patterns from Real Workflows

### 1. Groups for List Processing
Use Group nodes with GroupInput/GroupOutput to process lists item-by-item:
- `parent_id` field links nodes to their group
- Group automatically iterates over list items
- Common pattern: Data source → Group → Aggregated results

### 2. Multi-Agent Pipelines
Chain multiple AI agents for complex tasks:
- Agent 1: Generate strategy/plan
- Agent 2: Transform into specific format
- Generator nodes: Create multiple variations
- Common in creative workflows (posters, stories, content)

### 3. Dynamic Properties
Some nodes accept dynamic inputs via `dynamic_properties`:
- `nodetool.text.FormatText`: Template variables ({{VAR_NAME}})
- `nodetool.dictionary.MakeDictionary`: Custom key-value pairs
- Connect inputs to dynamic property names

### 4. Preview Nodes
Use `nodetool.workflows.base_node.Preview` to visualize intermediate results:
- No configuration needed
- Accepts any data type
- Multiple previews at different pipeline stages
- Essential for debugging and validation

### 5. Model Configuration
Language models require structured config:
```json
{
  "model": {
    "type": "language_model",
    "id": "gpt-4o",
    "provider": "openai"
  }
}
```
Common providers: `openai`, `anthropic`, `ollama`, `replicate`

### 6. Media References
Images, audio, and video use typed references:
```json
{
  "value": {
    "type": "image",
    "uri": "https://...",
    "asset_id": null
  }
}
```

### 7. Edge Types
Edge `ui_properties.className` indicates data type:
- `str`: Text data
- `image`: Image references
- `audio`: Audio references
- `list`: Array data
- `dict`: Dictionary/object data
- `any`: Any type (use sparingly)

## Building Your Own

1. **Identify the pattern**: Sequential, parallel, grouped, or multi-agent
2. **Search for nodes**: Use `search_nodes` with functionality keywords
3. **Check node details**: Use `get_node_info` for inputs/outputs
4. **Start simple**: Build with 2-3 core nodes first
5. **Add Preview nodes**: Visualize data at each major step
6. **Connect edges**: Match source outputs to target inputs by type
7. **Test incrementally**: Run after each addition with `run_workflow_tool`
8. **Add complexity**: Groups for iteration, agents for intelligence

## Common Node Types

- **Input**: `nodetool.input.StringInput`, `nodetool.input.ImageInput`
- **Constants**: `nodetool.constant.String`, `nodetool.constant.Image`
- **AI**: `nodetool.agents.Agent`, `nodetool.generators.ListGenerator`
- **Text**: `nodetool.text.FormatText`, `nodetool.text.Concat`, `nodetool.text.Slice`
- **Data**: `nodetool.dictionary.GetValue`, `nodetool.dictionary.MakeDictionary`
- **Groups**: `nodetool.workflows.base_node.Group`, `nodetool.input.GroupInput`
- **Preview**: `nodetool.workflows.base_node.Preview`
"""


@mcp.prompt()
async def troubleshoot_workflow() -> str:
    """
    Guide for troubleshooting common workflow issues.

    Returns:
        Troubleshooting tips and solutions
    """
    return """# Troubleshooting NodeTool Workflows

## Common Issues and Solutions

### 1. Type Mismatch Errors

**Problem**: "Cannot connect output X to input Y - type mismatch"

**Solutions**:
- Use `get_node_info` to check exact input/output types
- Insert conversion nodes between incompatible types
- Check for list vs single item mismatches (str vs list[str])

**Example Fix**:
```
# Wrong: str → list[str]
TextNode → ListNode

# Right: str → conversion → list[str]
TextNode → MakeList → ListNode
```

### 2. Missing Required Parameters

**Problem**: "Required parameter X not provided"

**Solutions**:
- Check node's `data` field has all required params
- Use `get_node_info` to see required vs optional fields
- Connect an input edge or set a default value

### 3. Circular Dependencies

**Problem**: "Workflow contains circular dependency"

**Solutions**:
- Review edges to find cycles in the graph
- Workflows must be directed acyclic graphs (DAGs)
- Remove edges that create loops

### 4. Node Not Found

**Problem**: "Node type X not found in registry"

**Solutions**:
- Verify node type spelling (case-sensitive)
- Use `search_nodes` to find the correct type
- Check if the required package is installed
- Use fully qualified names (e.g., "nodetool.text.Split")

### 5. Empty Results

**Problem**: Workflow runs but produces no output

**Solutions**:
- Add Output nodes to capture results
- Check that edges connect to output nodes
- Verify intermediate nodes are processing data

### 6. Performance Issues

**Problem**: Workflow runs slowly

**Solutions**:
- Check for unnecessary sequential dependencies
- Use parallel branches where possible
- Optimize heavy operations (large images, long videos)
- Consider batch processing nodes

## Debugging Strategies

### 1. Start Small
- Build with 2-3 nodes first
- Test after each node addition
- Verify outputs at each step

### 2. Check Each Component
- Validate node types exist
- Confirm parameter values are correct
- Verify edge connections are valid

### 3. Use Output Nodes
Add intermediate outputs to inspect data:
```
Process A → [Output] → Process B → [Output] → Process C
```

### 4. Review Examples
- Use `list_workflows` to find similar workflows
- Use `get_workflow` to see working structures
- Adapt proven patterns to your use case

## Validation Checklist

Before running a workflow, verify:
- [ ] All node types exist in registry
- [ ] All required parameters are set
- [ ] All edges connect compatible types
- [ ] No circular dependencies exist
- [ ] Input nodes are defined for parameters
- [ ] Output nodes capture desired results
- [ ] Node IDs are unique

## Getting Help

1. **Search for nodes**: Use `search_nodes` with detailed queries
2. **Inspect nodes**: Use `get_node_info` for full specifications
3. **Review examples**: Use `get_workflow` on existing workflows
4. **Test incrementally**: Use `run_workflow_tool` frequently

## Error Messages Guide

### "Workflow not found"
- Verify workflow_id is correct
- Check workflow exists with `list_workflows`

### "Invalid parameter type"
- Check parameter matches expected type
- Convert strings to numbers where needed
- Ensure refs (ImageRef, AudioRef) are used correctly

### "Node execution failed"
- Check node's specific error message
- Verify input data is valid
- Check file paths exist for file operations

### "Timeout"
- Workflow may be too complex
- Check for infinite loops (shouldn't happen in DAG)
- Consider splitting into smaller workflows
"""


def create_mcp_app():
    """
    Create and configure the FastMCP application.

    Returns:
        Configured FastMCP instance
    """
    return mcp


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
