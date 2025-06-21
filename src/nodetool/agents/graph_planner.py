"""GraphPlanner - Creates workflow graphs from objectives using AgentNodes.

The GraphPlanner takes a high-level objective and transforms it into a
workflow graph composed of AgentNodes. This allows for visual workflow
representation and execution within the nodetool workflow system.

## Overview

The GraphPlanner is an AI-powered system that transforms natural language objectives
into executable workflow graphs. It uses a multi-phase approach to analyze requirements,
select appropriate nodes, and design data flow connections.

## Architecture

### 1. Multi-Phase Planning Process

The system operates in three distinct phases:

1. **Analysis Phase**: Interprets the user's objective, creates a high-level plan, and generates a DOT graph visualization
2. **Workflow Design Phase**: Uses the DOT graph as a guide to select specific nodes and define data connections
3. **Graph Creation Phase**: Converts the design into an executable graph structure

### 2. Embedded Edge Format

A key feature of this system is the embedded edge format, where edges are defined
within node properties rather than as separate entities. This eliminates ambiguity
between constant values and data connections.

### 3. Property Value Types

Each property in a node can have one of the following value types:

- **Constants**: String, number, boolean, null
- **Edge Definitions**: `{"type": "edge", "source": "node_id", "sourceHandle": "output_name"}`
- **Asset References**: `{"type": "image|video|audio", "uri": "resource_path"}`

### 4. Node Type Inference

The system includes mappers that automatically infer appropriate Input/Output node types
based on the schema:

- `InputNodeMapper`: Maps TypeMetadata to InputNode subclasses
- `OutputNodeMapper`: Maps TypeMetadata to OutputNode subclasses

### 5. DOT Graph Generation

During the analysis phase, the system generates a DOT graph that serves as a visual blueprint:

- High-level representation of the workflow structure
- Guides the subsequent node selection process
- Helps maintain consistency between planning and implementation
- Can be visualized using Graphviz tools

Example DOT graph:
```dot
digraph workflow {
    input_data [label="Sales Data"];
    aggregate [label="Aggregate by Region"];
    analyze [label="Analyze Trends"];
    report [label="Generate Report"];

    input_data -> aggregate;
    aggregate -> analyze;
    analyze -> report;
}
```

### 6. Type Safety and Validation

The GraphPlanner enforces strict type safety:

- Validates node types exist in the registry
- Checks edge connections for type compatibility
- Ensures Input/Output nodes match schema requirements
- Provides helpful suggestions for type mismatches

## Usage Example

```python
planner = GraphPlanner(
    provider=OpenAIProvider(),
    model="gpt-4o-mini",
    objective="Process sales data and generate a summary report",
    input_schema=[
        GraphInput(name="sales_data", type=TypeMetadata(type="dataframe"),
                   description="Monthly sales data")
    ],
    output_schema=[
        GraphOutput(name="report", type=TypeMetadata(type="string"),
                    description="Summary report")
    ]
)

# Generate the graph
async for update in planner.create_graph(context):
    if isinstance(update, PlanningUpdate):
        print(f"Phase: {update.phase}, Status: {update.status}")

# Access the generated graph
graph = planner.graph
```

## Key Components

- **GraphPlanner**: Main orchestrator class
- **SearchNodesTool**: Helps LLM find available node types
- **Validation Functions**: Ensure graph correctness at each phase
- **Type Compatibility Checker**: Validates connections between nodes
- **Visual Graph Printer**: ASCII representation for debugging

## Benefits of Embedded Edge Format

1. **Eliminates Ambiguity**: Each property can only be either a constant OR an edge
2. **Simplifies Validation**: Properties are self-contained with their connection info
3. **Better LLM Understanding**: More intuitive format for AI to generate
4. **Atomic Operations**: Each node fully describes its inputs

## Workflow Execution

After generation, the graph is converted to the standard APIGraph format with
separate nodes and edges arrays for execution by the WorkflowRunner.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
import traceback
from nodetool.chat.providers.anthropic_provider import AnthropicProvider
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.typecheck import typecheck
from pydantic import BaseModel

from jinja2 import Environment, BaseLoader

from nodetool.agents.tools.help_tools import SearchNodesTool
from nodetool.chat.providers import ChatProvider, OpenAIProvider
from nodetool.metadata.types import (
    Message,
)
from nodetool.packages.registry import Registry
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode, get_node_class
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, PlanningUpdate
from nodetool.types.graph import Graph as APIGraph
from nodetool.workflows.graph import Graph

# Removed AgentConsole import - using logging instead

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_node_type_for_metadata(
    type_metadata: TypeMetadata,
    is_subclass_of: type[BaseNode] | None = None,
) -> str:
    """Find the appropriate InputNode subclass for a given TypeMetadata."""
    from nodetool.packages.registry import Registry

    registry = Registry()
    all_nodes = registry.get_all_installed_nodes()

    for node_meta in all_nodes:
        try:
            node_class = get_node_class(node_meta.node_type)
            if (
                node_class
                and (is_subclass_of is None or issubclass(node_class, is_subclass_of))
                and "ChatInput" not in node_class.get_node_type()
            ):
                # Check the output type of this input node
                outputs = node_class.outputs()
                if outputs and len(outputs) > 0:
                    output_type = outputs[0].type

                    # Check for exact type match
                    if output_type.type == type_metadata.type:
                        return node_class.get_node_type()

                    # Also check for compatible types
                    elif _is_type_compatible(output_type, type_metadata):
                        return node_class.get_node_type()

        except Exception as e:
            logger.debug(f"Could not load node class for {node_meta.node_type}: {e}")
            continue

    raise ValueError(f"No InputNode match found for type: {type_metadata.type}")


def _is_type_compatible(source_type: TypeMetadata, target_type: TypeMetadata) -> bool:
    """Check if source type can be assigned to target type."""
    # Handle any type
    if source_type.type == "any" or target_type.type == "any":
        return True

    # Handle exact matches
    if source_type.type == target_type.type:
        return True

    # Handle numeric conversions
    numeric_types = {"int", "float"}
    if source_type.type in numeric_types and target_type.type in numeric_types:
        return True

    # Handle optional types
    if target_type.optional and not source_type.optional:
        # Can assign non-optional to optional
        return source_type.type == target_type.type

    return False


# Now mappers are always available
MAPPERS_AVAILABLE = True


def print_visual_graph(graph: APIGraph) -> None:
    """Print a visual ASCII representation of the workflow graph."""
    logger.info("\n  Visual Graph Structure:")
    logger.info("  " + "=" * 50)

    # Build adjacency information
    adjacency = {}
    reverse_adjacency = {}
    all_nodes = {node.id: node for node in graph.nodes}

    for edge in graph.edges:
        if edge.source not in adjacency:
            adjacency[edge.source] = []
        adjacency[edge.source].append(
            (edge.target, edge.sourceHandle, edge.targetHandle)
        )

        if edge.target not in reverse_adjacency:
            reverse_adjacency[edge.target] = []
        reverse_adjacency[edge.target].append(
            (edge.source, edge.sourceHandle, edge.targetHandle)
        )

    # Find root nodes (no incoming edges)
    root_nodes = [
        node_id for node_id in all_nodes.keys() if node_id not in reverse_adjacency
    ]

    # If no clear roots, start with first node
    if not root_nodes:
        root_nodes = [list(all_nodes.keys())[0]] if all_nodes else []

    visited = set()

    def print_node_tree(
        node_id: str, depth: int = 0, is_last: bool = True, prefix: str = ""
    ):
        if node_id in visited:
            logger.info(
                f"  {prefix}{'└── ' if is_last else '├── '}[{node_id}] (already shown)"
            )
            return

        visited.add(node_id)
        node = all_nodes.get(node_id)
        node_type = node.type if node else "unknown"

        connector = "└── " if is_last else "├── "
        logger.info(f"  {prefix}{connector}┌─ [{node_id}]")
        logger.info(f"  {prefix}{'    ' if is_last else '│   '}│  Type: {node_type}")

        # Print node properties if available
        if node and hasattr(node, "data") and node.data:
            key_props = []
            for key, value in node.data.items():
                if key in ["objective", "template", "text", "name", "provider"]:
                    val_str = (
                        str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
                    )
                    key_props.append(f"{key}: {val_str}")
            if key_props:
                logger.info(
                    f"  {prefix}{'    ' if is_last else '│   '}│  Props: {', '.join(key_props[:2])}"
                )

        logger.info(f"  {prefix}{'    ' if is_last else '│   '}└─")

        # Print children
        children = adjacency.get(node_id, [])
        for i, (child_id, source_handle, target_handle) in enumerate(children):
            is_last_child = i == len(children) - 1
            child_prefix = prefix + ("    " if is_last else "│   ")

            # Print connection info
            conn_info = f"--[{source_handle}→{target_handle}]-->"
            logger.info(
                f"  {child_prefix}{'└── ' if is_last_child else '├── '}{conn_info}"
            )

            # Print child node
            print_node_tree(
                child_id,
                depth + 1,
                is_last_child,
                child_prefix + ("    " if is_last_child else "│   "),
            )

    # Print each root node tree
    for i, root_id in enumerate(root_nodes):
        is_last_root = i == len(root_nodes) - 1
        print_node_tree(root_id, 0, is_last_root)

    # Print any remaining unvisited nodes
    unvisited = set(all_nodes.keys()) - visited
    if unvisited:
        logger.info("\n  Unconnected nodes:")
        for node_id in unvisited:
            node = all_nodes[node_id]
            logger.info(f"    • [{node_id}] ({node.type})")

    logger.info("  " + "=" * 50)


# Schema for Phase 1: Analysis
WORKFLOW_ANALYSIS_SCHEMA = {
    "type": "object",
    "required": [
        "objective_interpretation",
        "workflow_approach",
        "expected_outcomes",
        "constraints",
        "assumptions",
        "required_namespaces",
        "workflow_graph_dot",
    ],
    "additionalProperties": False,
    "properties": {
        "objective_interpretation": {
            "type": "string",
            "description": "Clear interpretation of what the user wants to achieve",
        },
        "workflow_approach": {
            "type": "string",
            "description": "High-level approach to solve the problem using a graph workflow",
        },
        "expected_outcomes": {
            "type": "array",
            "description": "List of expected outputs or results from the workflow",
            "items": {"type": "string"},
        },
        "constraints": {
            "type": "array",
            "description": "Any constraints or special requirements identified",
            "items": {"type": "string"},
        },
        "assumptions": {
            "type": "array",
            "description": "Assumptions made about the workflow",
            "items": {"type": "string"},
        },
        "required_namespaces": {
            "type": "array",
            "description": "List of required node namespaces for the workflow",
            "items": {"type": "string"},
        },
        "workflow_graph_dot": {
            "type": "string",
            "description": "DOT graph notation representing the planned workflow structure",
        },
    },
}

# Schema for Phase 2: Node Selection and Dataflow
WORKFLOW_DESIGN_SCHEMA = {
    "type": "object",
    "required": ["node_specifications"],
    "additionalProperties": False,
    "properties": {
        "node_specifications": {
            "type": "array",
            "description": "Detailed specifications for processing nodes only (Input/Output nodes are automatically generated)",
            "items": {
                "type": "object",
                "required": [
                    "node_id",
                    "node_type",
                    "purpose",
                    "properties",
                ],
                "additionalProperties": False,
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Unique identifier for the node (e.g., 'input_1', 'agent_1')",
                    },
                    "node_type": {
                        "type": "string",
                        "description": "The exact node type from search_nodes results (e.g., 'nodetool.agents.Agent')",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "What this node does in the workflow and why it's needed",
                    },
                    "properties": {
                        "type": "string",
                        "description": "JSON string of properties for the node",
                    },
                },
            },
        },
    },
}


DEFAULT_GRAPH_PLANNING_SYSTEM_PROMPT = """
# GraphArchitect System Core Directives

## Goal
As GraphArchitect, your primary goal is to transform complex user objectives 
into executable workflow graphs composed of Nodes. You will guide the 
LLM through distinct phases to produce a valid and optimal graph structure.

## Return Format
This system prompt establishes your operational context. For each 
subsequent phase, you will receive specific instructions detailing the 
required output format. Your cumulative output across all phases will be 
a well-structured workflow graph, ultimately generated via the `create_graph` 
tool.

## Core Principles
1. **Graph Structure:** Design workflows as Directed Acyclic Graphs (DAGs) 
   with no cycles.
2. **Data Flow:** Connect nodes via edges that represent data flow
3. **Node Design:** Each node should have a clear, focused purpose
4. **Valid Node Types:** All nodes in the graph **must** correspond to available node types. You **must** use the `search_nodes` tool to discover and verify node types. Do not invent or assume node types.
5. **Type Safety:** Ensure type compatibility throughout the workflow:
   - Only connect compatible types (matching types, numeric conversions, or 'any' type)
   - Use converter nodes when types don't match directly
   - Plan for type conversions during node selection phase

## Understanding Node Metadata
Each node type has specific metadata that defines:
- **properties**: Input fields/parameters the node accepts (these become targetHandles for edges)
- **outputs**: Output slots the node produces (these become sourceHandles for edges)
- **is_dynamic**: Boolean flag indicating if the node supports dynamic properties

Each property and output has a name, type, and description.
- name: The identifier used as targetHandle in edges
- type: The data type it accepts (must be compatible with source output type)
- description: What the property does
- default: Default value if not connected

## Dynamic Properties
Some nodes have `is_dynamic=true` in their metadata, which means:
- **Flexible Configuration**: You can set any property name on these nodes, not just predefined ones
- **Runtime Handling**: The node will dynamically process arbitrary properties during execution
- **Custom Fields**: Create property names that match your specific workflow requirements
- **Beyond Schema**: You're not limited to the properties listed in the node's metadata

Example of using dynamic properties:
```json
{
  "custom_field": "some_value",
  "user_defined_param": {"type": "edge", "source": "input_node", "sourceHandle": "output"},
  "workflow_specific_data": 42
}
```

**Important**: Always include any required properties from the metadata, but feel free to add additional custom properties for dynamic nodes.

## Type Compatibility Rules
- Exact matches are always compatible (string -> string)
- Numeric conversions are allowed (int -> float, float -> int)
- 'any' type is compatible with all types
- Union types are compatible if any member type matches
- Complex types (document, dataframe, image) need converters for primitive types

## Iterator Nodes: Processing Lists Item-by-Item

Iterator nodes are special nodes that process lists by emitting each item individually to downstream nodes. They use a streaming pattern with events to coordinate processing.

### Key Iterator Concepts:

1. **Iterator** (`nodetool.control.Iterator`):
   - Takes a list as input via `input_list` property
   - Emits each item individually through the `output` slot
   - Also provides the current `index` and coordination `event` signals
   - Uses `gen_process()` method for streaming behavior

2. **Output Slots**: Iterator nodes produce three outputs:
   - `output`: The current item from the list
   - `index`: The position (0-based) of the current item
   - `event`: Event signals for coordination (Event type)


### When to Use Iterators:

- **Process each item individually**: When you need to apply operations to each element of a list separately
- **Item-specific transformations**: When each list item requires individual processing
- **Conditional processing**: When you need to apply different logic to different items
- **Parallel processing**: When you want multiple nodes to process the same items
- **Streaming workflows**: When working with large lists that should be processed incrementally

### Important Notes:
- Iterator nodes trigger downstream processing for each item in the list
- Downstream nodes process one item at a time, not the entire list
- Use CollectorNode when you need to gather results back into a list
- Event connections are crucial for proper coordination between iterator and collector
- **Stream to Output Conversion**: When an iterator stream (or any streaming node) connects directly to an Output node of type T, the system automatically collects all streamed values into a list[T]. You don't need a CollectorNode in this case - the Output node handles the collection automatically.

## Streaming Node Behavior and Output Collection

Some nodes, identifiable by `is_streaming=True` in their metadata, are designed to yield multiple outputs sequentially during their execution. This is common for iterators that process list items one-by-one, or nodes that break down data into multiple parts.

### Key Streaming Concepts:

1.  **Multiple Yields**: A node with `is_streaming=True` can produce a sequence of output values on a single output handle from one processing cycle. For example, an `Iterator` processes an input list and yields each item of the list individually through its `output` handle.

2.  **Output Node Collection**: When a streaming node (e.g., an `Iterator` yielding items of type `T`) is connected to an Output node defined in the `output_schema` with type `T` (e.g., `string`), the Output node will automatically collect all the individual streamed items of type `T`. The system handles aggregating these items, typically into a list, when the graph execution completes or the stream ends.
    - **Important**: When designing the graph and defining `output_schema`, if you expect a stream of items of type `T` to be collected by an Output node, you should declare that `GraphOutput` with type `T` (e.g., `string`), NOT `list[T]` (e.g., `list[string]`). The graph execution system manages the collection. This simplifies the graph's output signature.
    - **Avoid `list[T]` for streamed outputs**: Do not define an Output node as `list[T]` if it's meant to collect a stream of `T` items. Use type `T` for the Output node, and the system will provide the collected list.

3.  **Edge Validation for Streaming Outputs**:
    - When connecting a streaming node (source) to an Output node (target), ensure the type compatibility check considers the *individual item type* yielded by the streaming source.
    - For example, if an `Iterator` yields `string` items, it should connect to an Output node defined in the schema as type `string`. The validation should confirm this `string` to `string` compatibility for the individual items, understanding that the Output node will handle the collection.
"""


GRAPH_ANALYSIS_PHASE_TEMPLATE = """
# PHASE 1: OBJECTIVE ANALYSIS

## Goal
Analyze the user's objective and understand what the workflow needs to accomplish.
Focus on breaking down the requirements without getting into specific implementation details.

## Instructions
1. Interpret what the user wants to achieve
2. Identify the high-level approach for solving the problem
3. List expected outcomes from the workflow
4. Note any constraints or special requirements
5. Document assumptions you're making
6. Create a DOT graph representing the planned workflow structure

## DOT Graph Guidelines
Create a simple DOT graph that shows the high-level flow of data through the workflow:
- Use descriptive node labels (not specific node types yet)
- Show the general flow from inputs to outputs
- Include key processing steps
- Use clear, concise labels

Example DOT format:
```dot
digraph workflow {
    input [label="Input Data"];
    process1 [label="Process Step 1"];
    process2 [label="Process Step 2"];
    output [label="Output Result"];
    
    input -> process1;
    process1 -> process2;
    process2 -> output;
}
```

## Context
**User's Objective:**
{{ objective }}
{% if existing_graph_spec -%}
This is an **EDIT** request. You must modify the existing graph below.
The user's objective should be interpreted as an instruction to change this graph.
Your new DOT graph should represent the final, desired state of the graph after the edits.

**Existing Graph Structure:**
```json
{{ existing_graph_spec }}
```
{%- endif %}

**Input Nodes:**
{{ input_nodes }}

**Output Nodes:**
{{ output_nodes }}

Return ONLY the JSON object, no additional text.
"""

WORKFLOW_DESIGN_PROMPT = """
# PHASE 2: WORKFLOW DESIGN (NODE SELECTION & DATAFLOW)

## Goal
Based on the previous analysis, design the PART of the workflow by selecting appropriate intermediate nodes and defining how data flows between them. Use the search_nodes tool to find specific node types and understand their inputs/outputs to create proper connections.

**IMPORTANT: You MUST create ALL nodes including Input and Output nodes. The Input and Output node types are already specified in the context below - use those exact types without searching for them. The system will validate that your Input and Output nodes match the provided schema.**

Review the previous analysis phase conversation above to understand:
- The objective interpretation 
- The planned workflow approach
- The DOT graph representation of the workflow structure
- Any constraints and assumptions identified

Use the DOT graph from the analysis as a guide for your workflow design. You must create ALL nodes shown in the DOT graph including input and output nodes.
{% if existing_graph_spec -%}
This is an **EDIT** request. The user's objective is an instruction to modify the existing graph.
Your primary goal is to produce a NEW, complete `node_specifications` JSON object that represents the final state of the graph AFTER the edits.
You should include ALL nodes (both existing and new) in your final output.

**Existing Graph to Modify:**
```json
{{ existing_graph_spec }}
```
When returning the final `node_specifications`, you may reuse `node_id`s from the existing graph.
Analyze the user's request and the existing graph to determine which nodes to add, remove, or re-wire.
{%- endif %}

## Using `search_nodes` Effectively
When using the `search_nodes` tool to find nodes:
- Provide a `query` with keywords describing the node's function (e.g., "convert", "summarize", "filter data").
- **By default, prefer to omit `input_type` and `output_type` parameters to search broadly across all types.** This allows the tool to find the most relevant nodes regardless of specific type constraints initially.
- **Only specify `input_type` or `output_type` if your initial broad search yields ambiguous or too many irrelevant results, and you need to narrow down the search to a specific data type.** For instance, if a general search for "process data" is too vague, you might then refine it with `input_type="dataframe"` if you are specifically looking for dataframe processing nodes.
- The available types for `input_type` and `output_type` are: "str", "int", "float", "bool", "list", "dict", "tuple", "union", "enum", "any".
- If you are looking for a node that explicitly handles 'any' type or a generic operation, omitting type parameters is the best approach.

## Instructions - Node Selection
1. **Create ALL nodes including Input and Output nodes.** For Input and Output nodes, use the exact node types provided in the context below (do NOT search for them). Only search for intermediate processing nodes.
   
2. **Search for intermediate processing nodes using `search_nodes`**. Apply `input_type` and `output_type` filters whenever the data type is known to get more accurate results.
   - **For dataframe operations**: Search with relevant keywords (e.g., "GroupBy", "Aggregate", "Filter", "Transform", "dataframe"). Many dataframe nodes are in the `nodetool.data` namespace.
   - **For list operations**: Search with `input_type="list"` or `output_type="list"` and relevant keywords.
   - **For text operations**: Search with `input_type="str"` or `output_type="str"` (e.g., "concatenate", "regex", "template").
   - **For agents**: Search "agent". Verify their input/output types by inspecting their metadata from the search results before use.
   
3. Type conversion patterns (use keyword-based searches):
   - dataframe → array: Search "dataframe to array" or "to_numpy"
   - dataframe → string: Search "dataframe to string" or "to_csv"
   - array → dataframe: Search "array to dataframe" or "from_array"
   - list → item: Use iterator node
   - item → list: Use collector node

## Configuration Guidelines
- For nodes (found via `search_nodes`): Check their metadata for required fields and create appropriate property entries.
- **Dynamic Properties**: If a node has `is_dynamic=true` in its metadata, you can set ANY property name on that node, not just the predefined ones. Dynamic nodes will handle arbitrary properties at runtime.
  - For dynamic nodes: You can create custom property names based on your workflow needs
  - Example: `{"custom_field": "value", "another_field": {"type": "edge", "source": "input_1", "sourceHandle": "output"}}`
  - Still include any required properties from the metadata, but feel free to add additional ones
- Edge connections: `{"type": "edge", "source": "source_node_id", "sourceHandle": "output_name"}`
- Encode properties as a JSON string
- Example for constant value: `{"property_name": "property_value"}`
- Example for edge connection: `{"property_name": {"type": "edge", "source": "source_node_id", "sourceHandle": "output_name"}}`

## Important Handle Conventions
- **Most nodes have a single output**: The default output handle is often named "output". Always verify with `search_nodes` if unsure.
- **Input nodes**: Provide data through the `"output"` handle.
- **Output nodes**: Receive data through their `"value"` property.
- **Nodes**: Usually have "output" as their output handle, but **always check metadata from `search_nodes` results for exceptions and exact input property names (targetHandles).**

Example connections:
- From an Input node: `{"type": "edge", "source": "input_id", "sourceHandle": "output"}`
- To an Output node: Connect your final node to the output using a `value` property in your node specifications

## Instructions - Dataflow Design
5. **Connect nodes based on type compatibility:**
   - Exact type matches are always safe
   - Use converter nodes for type mismatches (find them with `search_nodes`)
   - Check node metadata from `search_nodes` for actual input/output types for nodes.

6. **Common connection patterns:**
   - Input → Node → Output
   - Input → Node → Node → Output
   - Input → Node → Node → Node → Output

## Type Compatibility Rules
- ✓ dataframe → dataframe operations → dataframe
- ✓ string → string, int → float, any → any type
- ✗ dataframe → list operations (needs conversion, search for a converter node)
- ✗ LoadCSVFile as input node (use proper Input node type as recommended or found via `search_nodes`)

## Context
**User's Objective:**
{{ objective }}

**Input Nodes:**
{{ input_nodes }}

**Output Nodes:**
{{ output_nodes }}
"""


class GraphInput(BaseModel):
    """Input schema for the graph planner."""

    name: str
    type: TypeMetadata
    description: str


class GraphOutput(BaseModel):
    """Output schema for the graph planner."""

    name: str
    type: TypeMetadata
    description: str


class GraphPlanner:
    """Orchestrates the creation of workflow graphs from high-level objectives.

    The GraphPlanner transforms user objectives into executable workflow graphs
    composed of AgentNodes, InputNodes, and OutputNodes. It uses a multi-phase
    planning approach similar to TaskPlanner but generates graph structures
    instead of task lists.
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: str,
        objective: str,
        inputs: dict[str, Any] = {},
        input_schema: list[GraphInput] = [],
        output_schema: list[GraphOutput] = [],
        existing_graph: Optional[APIGraph] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 20000,
        verbose: bool = True,
    ):
        """Initialize the GraphPlanner.

        Args:
            provider: LLM provider instance
            model: Model identifier to use
            objective: High-level goal to achieve
            inputs: Dictionary of input values to infer types from
            input_schema: List of GraphInput objects defining expected inputs
            output_schema: List of GraphOutput objects defining expected outputs
            existing_graph: Optional existing graph to edit
            system_prompt: Custom system prompt (optional)
            verbose: Enable detailed logging
        """
        self.provider = provider
        self.model = model
        self.objective = objective
        self.inputs = inputs
        self.max_tokens = max_tokens
        self.existing_graph = existing_graph

        # If input_schema is empty but inputs are provided, infer the schema
        if not input_schema and inputs:
            self.input_schema = self._infer_input_schema_from_values(inputs)
            logger.info(
                f"Inferred input schema from provided values: {[inp.model_dump() for inp in self.input_schema]}"
            )
        else:
            self.input_schema = input_schema

        self.output_schema = output_schema
        self.system_prompt = system_prompt or DEFAULT_GRAPH_PLANNING_SYSTEM_PROMPT
        self.verbose = verbose
        self.registry = Registry()

        # Initialize Jinja2 environment
        self.jinja_env = Environment(loader=BaseLoader())

        # Graph storage
        self.graph: Optional[APIGraph] = None

        # Cache for expensive operations
        self._cached_node_metadata: Optional[List] = None
        self._cached_namespaces: Optional[set[str]] = None

        # Get inferred nodes for the context
        self._input_nodes = [
            {
                "node_type": get_node_type_for_metadata(i.type, InputNode),
                "node_id": i.name,
                "purpose": "input",
                "properties": json.dumps({"name": i.name}),
            }
            for i in self.input_schema
        ]
        self._output_nodes = [
            {
                "node_type": get_node_type_for_metadata(o.type, OutputNode),
                "node_id": o.name,
                "purpose": "output",
                "properties": json.dumps(
                    {
                        "name": o.name,
                        "value": {
                            "type": "edge",
                            "source": "source_node_id",
                            "sourceHandle": "output",
                        },
                    }
                ),
            }
            for o in self.output_schema
        ]
        print(self._input_nodes)
        print(self._output_nodes)
        logger.debug(f"GraphPlanner initialized for objective: {objective[:100]}...")

    def _get_node_metadata(self) -> List:
        """Get node metadata with caching."""
        if self._cached_node_metadata is None:
            self._cached_node_metadata = self.registry.get_all_installed_nodes()
        return self._cached_node_metadata

    def _get_namespaces(self) -> set[str]:
        """Get namespaces with caching."""
        if self._cached_namespaces is None:
            node_metadata_list = self._get_node_metadata()
            self._cached_namespaces = {node.namespace for node in node_metadata_list}
        return self._cached_namespaces

    def _convert_graph_to_specifications(
        self, graph: APIGraph
    ) -> List[Dict[str, Any]]:
        """Converts an APIGraph object into the node_specifications format."""
        node_specs = []
        # Create a lookup for edges by their target node ID
        edges_by_target: Dict[str, List[Any]] = {}
        for edge in graph.edges:
            target_id = edge.target
            if target_id not in edges_by_target:
                edges_by_target[target_id] = []
            edges_by_target[target_id].append(edge)

        for node in graph.nodes:
            properties = node.data.copy() if node.data else {}

            # Find incoming edges for the current node and add them to properties
            if node.id in edges_by_target:
                for edge in edges_by_target[node.id]:
                    properties[edge.targetHandle] = {
                        "type": "edge",
                        "source": edge.source,
                        "sourceHandle": edge.sourceHandle,
                    }

            spec = {
                "node_id": node.id,
                "node_type": node.type,
                "purpose": "Existing node in the graph.",
                "properties": json.dumps(properties, indent=2),
            }
            node_specs.append(spec)

        return node_specs

    def _infer_type_metadata_from_value(self, value: Any) -> TypeMetadata:
        """Infer TypeMetadata from a Python value.

        Args:
            value: Python value to infer type from

        Returns:
            TypeMetadata object representing the inferred type
        """
        # Handle None
        if value is None:
            return TypeMetadata(type="none", optional=True)

        # Handle basic types - check bool before int since bool is subclass of int
        if isinstance(value, bool):
            return TypeMetadata(type="bool")
        elif isinstance(value, int):
            return TypeMetadata(type="int")
        elif isinstance(value, float):
            return TypeMetadata(type="float")
        elif isinstance(value, str):
            return TypeMetadata(type="str")
        elif isinstance(value, bytes):
            return TypeMetadata(type="bytes")

        # Handle collections
        elif isinstance(value, list):
            if not value:
                # Empty list, can't infer inner type
                return TypeMetadata(type="list", type_args=[TypeMetadata(type="any")])
            else:
                # Infer from first element (assuming homogeneous list)
                inner_type = self._infer_type_metadata_from_value(value[0])
                return TypeMetadata(type="list", type_args=[inner_type])

        elif isinstance(value, dict):
            # Check if this is an asset reference (has 'type' key with asset type value)
            if "type" in value and isinstance(value.get("type"), str):
                asset_type = value["type"]
                if asset_type in ["image", "video", "audio", "document", "file"]:
                    return TypeMetadata(type=asset_type)

            # Otherwise, it's a regular dict
            if not value:
                return TypeMetadata(
                    type="dict",
                    type_args=[TypeMetadata(type="str"), TypeMetadata(type="any")],
                )
            else:
                # For simplicity, assume string keys and infer value type from first entry
                first_value = next(iter(value.values()))
                value_type = self._infer_type_metadata_from_value(first_value)
                return TypeMetadata(
                    type="dict", type_args=[TypeMetadata(type="str"), value_type]
                )

        elif isinstance(value, tuple):
            # Infer types for each element in the tuple
            type_args = [self._infer_type_metadata_from_value(v) for v in value]
            return TypeMetadata(type="tuple", type_args=type_args)

        # Handle special types
        else:
            # Check for pandas DataFrame
            if hasattr(value, "__class__") and value.__class__.__name__ == "DataFrame":
                return TypeMetadata(type="dataframe")

            # Check for numpy array
            if hasattr(value, "__class__") and value.__class__.__name__ == "ndarray":
                return TypeMetadata(type="tensor")

            # Default to object type for unknown types
            return TypeMetadata(type="object")

    def _infer_input_schema_from_values(
        self, inputs: dict[str, Any]
    ) -> list[GraphInput]:
        """Infer GraphInput schema from a dictionary of input values.

        Args:
            inputs: Dictionary mapping input names to values

        Returns:
            List of GraphInput objects with inferred types
        """
        input_schema = []

        for name, value in inputs.items():
            # Infer the type from the value
            type_metadata = self._infer_type_metadata_from_value(value)

            # Create a descriptive description based on the type
            type_desc = type_metadata.__repr__()
            description = f"Input '{name}' of type {type_desc} (inferred from value)"

            # Create GraphInput
            graph_input = GraphInput(
                name=name, type=type_metadata, description=description
            )

            input_schema.append(graph_input)
            logger.debug(
                f"Inferred input '{name}': type={type_metadata.type}, full_type={type_desc}"
            )

        return input_schema

    def _build_nodes_and_edges_from_specifications(
        self,
        node_specifications: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Build nodes and edges from node specifications.

        Args:
            node_specifications: List of node specification dictionaries

        Returns:
            Tuple of (nodes, edges) lists
        """
        nodes = []
        edges = []

        for spec in node_specifications:
            node_id = spec["node_id"]
            node_type = spec["node_type"]
            node_data = {}

            # Process properties array and extract edges
            properties_string = spec.get("properties", "{}")
            properties = json.loads(properties_string)

            for prop_name, prop_value in properties.items():

                # Check if this is an edge definition
                if isinstance(prop_value, dict) and prop_value.get("type") == "edge":
                    # Create edge
                    edge = {
                        "source": prop_value["source"],
                        "target": node_id,
                        "sourceHandle": prop_value["sourceHandle"],
                        "targetHandle": prop_name,
                    }
                    edges.append(edge)
                else:
                    # Regular property value
                    node_data[prop_name] = prop_value

            # Create node dict in the requested format
            node_dict = {
                "id": node_id,
                "type": node_type,
                "data": node_data,
            }

            nodes.append(node_dict)

        return nodes, edges

    def _get_prompt_context(self, **kwargs) -> Dict[str, Any]:
        """Build context for Jinja2 prompt rendering."""
        context = {
            **kwargs,
            "objective": self.objective,
            "input_nodes": json.dumps(self._input_nodes),
            "output_nodes": json.dumps(self._output_nodes),
            "existing_graph_spec": None,
        }

        if self.existing_graph:
            specs = self._convert_graph_to_specifications(self.existing_graph)
            context["existing_graph_spec"] = json.dumps(specs, indent=2)

        return context

    def _render_prompt(self, template_string: str, **kwargs) -> str:
        """Render a Jinja2 template with context."""
        template = self.jinja_env.from_string(template_string)
        return template.render(self._get_prompt_context(**kwargs))

    async def _run_phase_with_tools(
        self,
        phase_name: str,
        prompt_content: str,
        response_schema: Dict[str, Any],
        schema_name: str,
        tools: List[Any],
        context: ProcessingContext,
        history: List[Message],
        max_iterations: int = 5,
        max_validation_attempts: int = 5,
        validation_fn=None,
    ) -> tuple[List[Message], Dict[str, Any], Optional[PlanningUpdate]]:
        """Generic method for running a phase with looped tool calling until completion with final structured output.

        Args:
            phase_name: Name of the phase for display purposes
            prompt_content: The prompt to send to the LLM
            response_schema: JSON schema for structured output
            schema_name: Name for the schema in structured output
            tools: List of tools available to the LLM
            context: Processing context
            history: Message history
            max_iterations: Maximum tool calling iterations
            max_validation_attempts: Maximum attempts if validation fails
            validation_fn: Optional validation function that returns error message or empty string
        """
        if self.verbose:
            logger.info(
                f"[{phase_name}] Running: Starting {phase_name.lower()} phase..."
            )

        history.append(Message(role="user", content=prompt_content))

        try:
            # Phase 1: Tool usage - let the LLM use tools
            for i in range(max_iterations):
                if self.verbose:
                    logger.info(
                        f"[{phase_name}] Running: LLM interaction (iteration {i + 1}/{max_iterations})..."
                    )

                # print("************************************************")
                # print(prompt_content)
                # print("************************************************")

                response = await self.provider.generate_message(
                    messages=history,
                    model=self.model,
                    tools=tools,
                    max_tokens=self.max_tokens,
                )

                if not response:
                    raise Exception("LLM returned no response.")

                history.append(response)

                if response.tool_calls:
                    if self.verbose:
                        logger.info(
                            f"[{phase_name}] Running: Executing {len(response.tool_calls)} tool call(s)..."
                        )
                    tool_messages_for_history: List[Message] = []
                    for tool_call in response.tool_calls:
                        tool_output_str = ""
                        # Execute the tool call
                        tool_found = False
                        for tool_instance in tools:
                            if tool_call.name == tool_instance.name:
                                tool_found = True
                                try:
                                    params_for_tool = tool_call.args
                                    if not isinstance(params_for_tool, dict):
                                        logger.warning(
                                            f"Tool call arguments for {tool_call.name} is not a dict: {tool_call.args}. Using empty dict."
                                        )
                                        params_for_tool = {}

                                    if self.verbose:
                                        logger.info(
                                            f"[{phase_name}] Running: Processing tool: {tool_call.name} with args: {params_for_tool}"
                                        )
                                    tool_output = await tool_instance.process(
                                        context, params_for_tool
                                    )
                                    tool_output_str = (
                                        json.dumps(tool_output)
                                        if tool_output is not None
                                        else "Tool returned no output."
                                    )
                                    logger.debug(
                                        f"Tool {tool_call.name} output: {tool_output_str[:200]}..."
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error executing tool {tool_call.name}: {e}",
                                        exc_info=True,
                                    )
                                    tool_output_str = f"Error executing tool {tool_call.name}: {str(e)}"
                                    if self.verbose:
                                        logger.error(
                                            f"[{phase_name}] Running: Error in tool {tool_call.name}: {str(e)}"
                                        )
                                break

                        if not tool_found:
                            logger.warning(
                                f"Received unknown tool call: {tool_call.name}"
                            )
                            tool_output_str = (
                                f"Error: Unknown tool {tool_call.name} was called."
                            )

                        tool_messages_for_history.append(
                            Message(
                                role="tool",
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                                content=tool_output_str,
                            )
                        )
                    history.extend(tool_messages_for_history)
                    # Continue to the next iteration to get LLM response after tool execution
                else:
                    # LLM is done with tools, break out to get structured output
                    break

            # Phase 2: Request structured output with validation attempts
            for attempt in range(max_validation_attempts):
                if self.verbose:
                    logger.info(
                        f"[{phase_name}] Running: Generating structured output (attempt {attempt + 1}/{max_validation_attempts})..."
                    )

                # Add instruction to provide structured output
                if attempt == 0:
                    history.append(
                        Message(
                            role="user",
                            content=f"Based on your analysis, provide the {phase_name.lower()} result as a JSON object according to the specified schema. Return ONLY the JSON, no additional text.",
                        )
                    )
                else:
                    # Include validation error from previous attempt
                    history.append(
                        Message(
                            role="user",
                            content=f"The previous output failed validation. Please fix the issues and provide the {phase_name.lower()} result as a JSON object. Return ONLY the JSON, no additional text.",
                        )
                    )

                # Request structured output
                structured_response = await self.provider.generate_message(
                    messages=history,
                    model=self.model,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "schema": response_schema,
                            "strict": True,
                        },
                    },
                    max_tokens=self.max_tokens,
                )

                if not structured_response or not isinstance(
                    structured_response.content, str
                ):
                    raise ValueError(
                        f"Failed to get structured {phase_name.lower()} output"
                    )

                print("************************************************")
                print(structured_response.content)
                print("************************************************")
                try:
                    result = json.loads(structured_response.content)
                except Exception as e:
                    logger.error(f"Error parsing structured output: {e}", exc_info=True)
                    raise ValueError(f"Error parsing structured output: {e}")

                history.append(
                    Message(
                        role="assistant",
                        content=structured_response.content,
                    )
                )

                # Run validation if provided
                if validation_fn:
                    error_message = validation_fn(result)
                    if error_message:
                        logger.warning(
                            f"{phase_name} validation failed: {error_message}"
                        )
                        # Add error to history for next attempt
                        history.append(
                            Message(
                                role="user",
                                content=f"Validation failed: {error_message}",
                            )
                        )
                        continue

                # Success
                if self.verbose:
                    logger.info(f"[{phase_name}] Success: {phase_name} complete")

                logger.debug("************************************************")
                logger.debug(f"{phase_name} Result: {result}")
                logger.debug("************************************************")

                return (
                    history,
                    result,
                    PlanningUpdate(
                        phase=phase_name,
                        status="Success",
                        content=f"{phase_name} complete",
                    ),
                )

            # All validation attempts failed
            raise ValueError(
                f"{phase_name} failed validation after {max_validation_attempts} attempts"
            )

        except Exception as e:
            logger.error(f"{phase_name} phase failed critically: {e}", exc_info=True)
            phase_status = "Failed"
            error_content = f"Error: {str(e)}"
            if self.verbose:
                logger.error(f"[{phase_name}] {phase_status}: {error_content}")

            # Clean history on error
            cleaned_history = []
            for msg in history:
                if msg.role in ["system", "user"]:
                    cleaned_history.append(msg)
                elif msg.role == "assistant" and not msg.tool_calls:
                    cleaned_history.append(msg)

            return (
                cleaned_history,
                {},
                PlanningUpdate(
                    phase=phase_name, status=phase_status, content=error_content
                ),
            )

    async def _run_analysis_phase(
        self, context: ProcessingContext, history: List[Message]
    ) -> tuple[List[Message], Dict[str, Any], Optional[PlanningUpdate]]:
        """Run the analysis phase to understand objectives and design workflow."""
        namespaces = self._get_namespaces()

        analysis_prompt_content = self._render_prompt(
            GRAPH_ANALYSIS_PHASE_TEMPLATE,
            namespaces=list(namespaces),
        )

        return await self._run_phase_with_tools(
            phase_name="Analysis",
            prompt_content=analysis_prompt_content,
            response_schema=WORKFLOW_ANALYSIS_SCHEMA,
            schema_name="WorkflowAnalysis",
            tools=[SearchNodesTool()],
            context=context,
            history=history,
            max_iterations=1,  # No tool iterations needed
            max_validation_attempts=1,  # No validation for analysis phase
            validation_fn=None,
        )

    def _validate_workflow_design(self, result: Dict[str, Any]) -> str:
        """Validate the complete workflow design (nodes + edges)."""
        error_messages = []

        # Validate dataflow analysis using the same logic but with the combined result
        dataflow_errors = self._validate_graph_edge_types(result)
        if dataflow_errors:
            error_messages.append(dataflow_errors)

        # If initial validations pass, create a real Graph and validate edge types
        if not error_messages:
            graph_validation_errors = self._validate_graph_edge_types(result)
            if graph_validation_errors:
                error_messages.append(graph_validation_errors)

        return " ".join(error_messages)

    async def _run_workflow_design_phase(
        self,
        context: ProcessingContext,
        history: List[Message],
    ) -> tuple[List[Message], Dict[str, Any], Optional[PlanningUpdate]]:
        """Run the combined workflow design phase with SearchNodesTool."""
        workflow_design_prompt = self._render_prompt(WORKFLOW_DESIGN_PROMPT)

        return await self._run_phase_with_tools(
            phase_name="Workflow Design",
            prompt_content=workflow_design_prompt,
            response_schema=WORKFLOW_DESIGN_SCHEMA,
            schema_name="WorkflowDesign",
            tools=[
                SearchNodesTool(
                    exclude_namespaces=[
                        "nodetool.agents",
                    ]
                )
            ],
            context=context,
            history=history,
            max_iterations=8,  # Allow more iterations since this is a combined phase
            max_validation_attempts=5,
            validation_fn=self._validate_workflow_design,
        )

    def _is_edge_type_compatible_enhanced(self, source_type, target_type) -> bool:
        """
        Enhanced type compatibility check matching WorkflowRunner implementation.

        This handles special cases like numeric conversions that are allowed in edges.
        """
        # Handle any type
        if source_type.type == "any" or target_type.type == "any":
            return True

        # Handle numeric conversions
        if source_type.type == "float" and target_type.type == "int":
            # Allow float to int conversion (will be truncated at runtime)
            return True
        elif source_type.type == "int" and target_type.type == "float":
            # Allow int to float conversion (lossless)
            return True

        # Handle same types
        if source_type.type == target_type.type:
            # For complex types like lists, check type args recursively
            if (
                source_type.type == "list"
                and len(source_type.type_args) > 0
                and len(target_type.type_args) > 0
            ):
                return self._is_edge_type_compatible_enhanced(
                    source_type.type_args[0], target_type.type_args[0]
                )
            return True

        # Handle ComfyUI types
        if source_type.is_comfy_type() and target_type.is_comfy_type():
            return source_type.type == target_type.type

        # Handle union types
        if source_type.type == "union":
            # Source union can connect if any of its types can connect to target
            return any(
                self._is_edge_type_compatible_enhanced(t, target_type)
                for t in source_type.type_args
            )
        if target_type.type == "union":
            # Target union can accept if source can connect to any of its types
            return any(
                self._is_edge_type_compatible_enhanced(source_type, t)
                for t in target_type.type_args
            )

        # Default: types must match exactly
        return False

    def _validate_graph_edge_types(self, result: Dict[str, Any]) -> str:
        """Create a real Graph object and validate edge types using Graph.validate_edge_types()."""
        try:
            # Enrich node specifications with metadata
            enriched_result = self._enrich_analysis_with_metadata(result)

            # Build nodes and edges using the helper method
            nodes, edges = self._build_nodes_and_edges_from_specifications(
                enriched_result.get("node_specifications", []),
            )
            print("************************************************")
            print(nodes)
            print(edges)
            print("************************************************")

            # Create graph dict
            graph_dict = {"nodes": nodes, "edges": edges}

            # Create Graph object and validate
            graph = Graph.from_dict(graph_dict, skip_errors=False)
            validation_errors = graph.validate_edge_types()

            if validation_errors:
                return "Graph edge type validation errors: " + " ".join(
                    validation_errors
                )

            return self._validate_input_output_nodes(graph)

        except Exception as e:
            logger.error(f"Error validating graph edge types: {e}")
            return f"Failed to validate graph structure: {str(e)}"

    def _validate_input_output_nodes(self, graph: Graph) -> str:
        """Validate InputNode and OutputNode instances against input and output schemas."""
        error_messages = []

        # Find InputNode and OutputNode instances
        input_nodes = [node for node in graph.nodes if isinstance(node, InputNode)]
        output_nodes = [node for node in graph.nodes if isinstance(node, OutputNode)]

        # Check for missing required input nodes
        found_input_names = set()
        for input_node in input_nodes:
            node_name = input_node.name
            node_type = input_node.outputs()[0].type

            # Check if name exists in schema
            for schema_input in self.input_schema:
                if schema_input.name == node_name:
                    found_input_names.add(node_name)
                    if not typecheck(schema_input.type, node_type):
                        error_messages.append(
                            f"InputNode '{node_name}' has type '{node_type}' which cannot be converted from the schema type '{schema_input.type}'."
                        )
                    break

        found_output_names = set()
        for output_node in output_nodes:
            node_name = output_node.name
            node_type = output_node.outputs()[0].type

            for schema_output in self.output_schema:
                if schema_output.name == node_name:
                    found_output_names.add(node_name)
                    if not typecheck(node_type, schema_output.type):
                        error_messages.append(
                            f"OutputNode '{node_name}' has type '{node_type}' which cannot be converted to the schema type '{schema_output.type}'."
                        )
                    break

        # Check for missing input nodes
        missing_input_names = (
            set(schema_input.name for schema_input in self.input_schema)
            - found_input_names
        )
        if missing_input_names:
            error_messages.append(
                f"Missing required InputNodes for: {list(missing_input_names)}. "
                f"You must create InputNodes for all inputs in the schema."
            )

        # Check for missing output nodes
        missing_output_names = (
            set(schema_output.name for schema_output in self.output_schema)
            - found_output_names
        )
        if missing_output_names:
            error_messages.append(
                f"Missing required OutputNodes for: {list(missing_output_names)}. "
                f"You must create OutputNodes for all outputs in the schema."
            )

        return " ".join(error_messages)

    def _enrich_analysis_with_metadata(
        self, analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich the analysis result with actual node metadata from the registry."""
        enriched_result = analysis_result.copy()

        for node_spec in enriched_result.get("node_specifications", []):
            node_type = node_spec.get("node_type")
            if node_type:
                # Get the node class from registry
                node_class = get_node_class(node_type)
                if node_class:
                    # Get node metadata
                    metadata = node_class.get_metadata()

                    # Extract properties and outputs
                    node_spec["metadata_info"] = metadata.model_dump()

                    logger.debug(
                        f"Enriched metadata for {node_type}: {len(metadata.properties)} properties, {len(metadata.outputs)} outputs"
                    )
                else:
                    logger.warning(
                        f"Could not find node class for type: {node_type}"
                    )

        return enriched_result

    def log_graph_summary(self) -> None:
        """Log a summary of the generated graph for debugging."""
        if not self.graph:
            logger.warning("No graph has been generated yet.")
            return

        logger.info("\n" + "=" * 80)
        logger.info("GRAPH PLANNER SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Objective: {self.objective}")
        logger.info(f"Input Schema: {[inp.model_dump() for inp in self.input_schema]}")
        logger.info(
            f"Output Schema: {[out.model_dump() for out in self.output_schema]}"
        )
        logger.info("\nGraph Statistics:")
        logger.info(f"  - Total Nodes: {len(self.graph.nodes)}")
        logger.info(f"  - Total Edges: {len(self.graph.edges)}")

        # Count node types
        node_type_counts = {}
        for node in self.graph.nodes:
            node_type = node.type
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        logger.info("\nNode Type Distribution:")
        for node_type, count in sorted(node_type_counts.items()):
            logger.info(f"  - {node_type}: {count}")

        # Log input/output nodes specifically
        logger.info("\nInput/Output Nodes:")
        for node in self.graph.nodes:
            if "input" in node.type.lower():
                logger.info(f"  - Input: {node.id} ({node.type})")
                if hasattr(node, "data") and "name" in node.data:
                    logger.info(f"    Schema Name: {node.data['name']}")
            elif "output" in node.type.lower():
                logger.info(f"  - Output: {node.id} ({node.type})")
                if hasattr(node, "data") and "name" in node.data:
                    logger.info(f"    Schema Name: {node.data['name']}")

        logger.info("=" * 80 + "\n")

    async def create_graph(
        self, context: ProcessingContext
    ) -> AsyncGenerator[Chunk | PlanningUpdate, None]:
        """Create a workflow graph from the objective.

        Yields PlanningUpdate events during the process.
        """
        logger.info(
            f"Starting graph creation for objective: '{self.objective[:100]}...'"
        )

        # Log start of graph planning
        if self.verbose:
            logger.info("Starting Graph Planner")

        history: List[Message] = [
            Message(role="system", content=self.system_prompt),
        ]

        # Phase 1: Analysis
        current_phase = "Analysis"
        logger.info(f"Starting Phase 1: {current_phase}")
        yield PlanningUpdate(phase=current_phase, status="Starting", content=None)
        history, analysis_result, planning_update = await self._run_analysis_phase(
            context, history
        )
        if planning_update:
            yield planning_update
        if planning_update and planning_update.status == "Failed":
            error_msg = f"Analysis phase failed: {planning_update.content}"
            logger.error(error_msg)
            if self.verbose:
                logger.error(f"[Overall Status] Failed: {error_msg}")
            raise ValueError(error_msg)

        # Pretty print analysis results
        if self.verbose and analysis_result:
            logger.info("\n" + "=" * 60)
            logger.info("WORKFLOW ANALYSIS RESULTS")
            logger.info("=" * 60)
            logger.info("Objective Interpretation:")
            logger.info(f"  {analysis_result.get('objective_interpretation', 'N/A')}")
            logger.info("\nWorkflow Approach:")
            logger.info(f"  {analysis_result.get('workflow_approach', 'N/A')}")
            logger.info("\nExpected Outcomes:")
            for outcome in analysis_result.get("expected_outcomes", []):
                logger.info(f"  • {outcome}")
            logger.info("\nConstraints:")
            for constraint in analysis_result.get("constraints", []):
                logger.info(f"  • {constraint}")
            logger.info("\nAssumptions:")
            for assumption in analysis_result.get("assumptions", []):
                logger.info(f"  • {assumption}")
            logger.info("\nPlanned Workflow Structure (DOT Graph):")
            dot_graph = analysis_result.get("workflow_graph_dot", "N/A")
            if dot_graph != "N/A":
                # Print the DOT graph with indentation
                for line in dot_graph.split("\n"):
                    logger.info(f"  {line}")
            logger.info("=" * 60)

        # Phase 2: Workflow Design (Combined Node Selection & Dataflow)
        current_phase = "Workflow Design"
        logger.info(f"Starting Phase 2: {current_phase}")
        yield PlanningUpdate(phase=current_phase, status="Starting", content=None)
        history, workflow_result, planning_update = (
            await self._run_workflow_design_phase(context, history)
        )
        if planning_update:
            yield planning_update
        if planning_update and planning_update.status == "Failed":
            error_msg = f"Workflow design phase failed: {planning_update.content}"
            logger.error(error_msg)
            if self.verbose:
                logger.error(f"[Overall Status] Failed: {error_msg}")
            raise ValueError(error_msg)

        # Pretty print workflow design results
        if self.verbose and workflow_result:
            logger.info("\n" + "=" * 60)
            logger.info("WORKFLOW DESIGN RESULTS")
            logger.info("=" * 60)

            # Print nodes
            node_specifications = workflow_result.get("node_specifications", [])
            logger.info(f"Nodes ({len(node_specifications)}):")
            for spec in node_specifications:
                logger.info(f"  ┌─ [{spec.get('node_id', 'unknown')}]")
                logger.info(f"  │  Type: {spec.get('node_type', 'unknown')}")
                logger.info(f"  │  Purpose: {spec.get('purpose', 'N/A')}")

                # Print properties
                properties_string = spec.get("properties", "{}")
                properties = json.loads(properties_string)
                if properties:
                    logger.info("  │  Properties:")
                    for prop_name, prop_value in properties.items():
                        if (
                            isinstance(prop_value, dict)
                            and prop_value.get("type") == "edge"
                        ):
                            logger.info(
                                f"  │    - {prop_name}: [edge from {prop_value.get('source')}.{prop_value.get('sourceHandle')}]"
                            )
                        else:
                            logger.info(f"  │    - {prop_name}: {prop_value}")
                logger.info("  └─")

            # Extract and print edges from embedded format
            logger.info("\nEdges (extracted from properties):")
            edge_count = 0
            for spec in node_specifications:
                target_id = spec.get("node_id")
                properties_string = spec.get("properties", "{}")
                properties = json.loads(properties_string)
                for prop_name, prop_value in properties.items():
                    if (
                        isinstance(prop_value, dict)
                        and prop_value.get("type") == "edge"
                    ):
                        source = prop_value.get("source", "unknown")
                        source_handle = prop_value.get("sourceHandle", "unknown")
                        target_handle = prop_name
                        logger.info(
                            f"  • {source}({source_handle}) ──→ {target_id}({target_handle})"
                        )
                        edge_count += 1

            if edge_count == 0:
                logger.info("  (No edges defined)")

            logger.info("=" * 60)

        # Enrich node specifications with actual metadata
        logger.info("Enriching node specifications with metadata from registry...")
        enriched_workflow = self._enrich_analysis_with_metadata(workflow_result)

        yield PlanningUpdate(
            phase="Metadata Enrichment",
            status="Success",
            content=f"Enhanced {len(enriched_workflow.get('node_specifications', []))} nodes with metadata",
        )

        # Create graph directly from the workflow design
        current_phase = "Graph Creation"
        logger.info(f"Starting Phase 3: {current_phase}")
        yield PlanningUpdate(phase=current_phase, status="Starting", content=None)

        try:
            # Build nodes and edges using the helper method
            nodes, edges = self._build_nodes_and_edges_from_specifications(
                enriched_workflow.get("node_specifications", []),
            )

            # Note: Input and Output nodes are now created by the LLM, not auto-generated
            logger.info(
                "Input and Output nodes will be created by the LLM based on schema hints"
            )

            # Create the final graph
            self.graph = APIGraph(
                nodes=nodes,  # type: ignore
                edges=edges,  # type: ignore
            )
            logger.info(
                f"Graph created successfully with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
            )

            # Log the generated graph structure
            logger.info("Generated Graph Structure:")
            logger.info("=" * 80)

            # Log nodes
            logger.info(f"Nodes ({len(self.graph.nodes)}):")
            for node in self.graph.nodes:
                logger.info(f"  - ID: {node.id}")
                logger.info(f"    Type: {node.type}")
                if hasattr(node, "data") and node.data:
                    logger.info(f"    Data: {json.dumps(node.data, indent=6)}")

            # Log edges
            logger.info(f"\nEdges ({len(self.graph.edges)}):")
            for edge in self.graph.edges:
                logger.info(
                    f"  - {edge.source} ({edge.sourceHandle}) -> {edge.target} ({edge.targetHandle})"
                )

            # Log the summary
            self.log_graph_summary()

            yield PlanningUpdate(
                phase=current_phase,
                status="Success",
                content=f"Created graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges",
            )

            if self.verbose:
                logger.info(
                    f"[{current_phase}] Success: Graph created with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
                )

        except Exception as e:
            error_msg = f"Failed to create graph: {str(e)}"
            logger.error(error_msg, exc_info=True)

            yield PlanningUpdate(
                phase=current_phase, status="Failed", content=error_msg
            )

            if self.verbose:
                logger.error(f"[Overall Status] Failed: {error_msg}")

            raise ValueError(error_msg)


async def main():
    """Main function to run a simple GraphPlanner example."""
    # provider = OpenAIProvider()
    # model = "o4-mini"
    provider = AnthropicProvider()
    model = "claude-sonnet-4-20250514"
    objective = "Generate a personalized greeting. The workflow should take a name as input and use a template to create a message like 'Hello, [name]! Welcome to the Nodetool demo.'"

    planner = GraphPlanner(
        provider=provider,
        model=model,
        objective=objective,
        verbose=True,  # Set to False to reduce console output from GraphPlanner
        input_schema=[
            GraphInput(
                name="name",
                type=TypeMetadata(type="str"),
                description="the name of the user",
            ),
        ],
        output_schema=[
            GraphOutput(
                name="greeting",
                type=TypeMetadata(type="str"),
                description="the greeting message",
            ),
        ],
    )

    # Initialize a basic ProcessingContext
    # For this example, many ProcessingContext features are not strictly necessary
    # for GraphPlanner's core graph generation logic, but it expects an instance.
    context = ProcessingContext()

    print(f"Running GraphPlanner with objective: {objective}")
    print("This will involve LLM calls and may take a few moments...")
    try:
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                print(
                    f"Phase: {update.phase}, Status: {update.status}, Content: {str(update.content or '')}"
                )
            elif isinstance(update, Chunk):
                print(f"Received chunk: {update.content}")

        if planner.graph:
            print("\nGenerated Graph:")
            print(
                f"  Title: {getattr(planner.graph, 'title', 'N/A')}"
            )  # Graph title might not be set by default from tool

            # Create visual graph representation
            print_visual_graph(planner.graph)

            print(f"\n  Detailed Nodes ({len(planner.graph.nodes)}):")
            for node in planner.graph.nodes:
                print(f"    {node.id} ({node.type}):")
                if hasattr(node, "data") and node.data:
                    for key, value in node.data.items():
                        print(f"      {key}: {value}")

            print(f"\n  Edges ({len(planner.graph.edges)}):")
            for edge in planner.graph.edges:
                print(
                    f"    - {edge.source} ({edge.sourceHandle}) -> {edge.target} ({edge.targetHandle})"
                )

        else:
            print("\nGraph planning completed, but no graph was generated.")

    except Exception as e:
        print(f"An error occurred during graph planning: {e}")
        traceback.print_exc()
    finally:
        print("\nGraphPlanner example finished.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
