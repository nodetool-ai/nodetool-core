"""GraphPlanner - Creates workflow graphs from objectives using AgentNodes.

The GraphPlanner takes a high-level objective and transforms it into a
workflow graph composed of AgentNodes. This allows for visual workflow
representation and execution within the nodetool workflow system.

## Overview

The GraphPlanner is an AI-powered system that transforms natural language objectives
into executable workflow graphs. It uses a direct approach to analyze requirements,
select appropriate nodes, and design data flow connections.

## Architecture

### 1. Single-Phase Planning Process

The system operates in a single design phase that:
- Interprets the user's objective and creates a workflow design
- Selects specific nodes and defines data connections
- Validates the designed workflow for basic structural integrity

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

### 5. Type Safety and Validation

The GraphPlanner includes type safety features:

- Validates node types through SearchNodesTool registry lookup
- Performs basic type compatibility checking for edge connections
- Ensures Input/Output nodes match schema requirements

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
    ],
    inputs={},  # Optional: Pre-defined input values
    existing_graph=None,  # Optional: Extend existing graph
    system_prompt=None,  # Optional: Custom system prompt
    max_tokens=8192,  # Optional: Token limit for LLM
    verbose=True  # Optional: Enable detailed logging
)

# Generate the graph
async for update in planner.create_graph(context):
    if isinstance(update, PlanningUpdate):
        print(f"Phase: {update.phase}, Status: {update.status}")
    # Note: create_graph yields both Chunk and PlanningUpdate objects

# Access the generated graph
graph = planner.graph
```

## Key Components

- **GraphPlanner**: Main orchestrator class
- **SearchNodesTool**: Helps LLM find available node types (excludes agent namespaces)
- **Validation Functions**: Ensure graph correctness
- **Type Compatibility Checker**: Validates connections between nodes
- **Visual Graph Printer**: Logs visual graph representation for debugging

## Benefits of Embedded Edge Format

1. **Eliminates Ambiguity**: Each property can only be either a constant OR an edge
2. **Simplifies Validation**: Properties are self-contained with their connection info
3. **Better LLM Understanding**: More intuitive format for AI to generate
4. **Atomic Operations**: Each node fully describes its inputs

## Workflow Execution

After generation, the graph is converted to the standard APIGraph format with
separate nodes and edges arrays for execution by the WorkflowRunner.
"""

import json
from nodetool.config.logging_config import get_logger
from typing import Any, AsyncGenerator, Dict, List, Optional, cast
import traceback
from nodetool.providers.anthropic_provider import AnthropicProvider
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.typecheck import typecheck
from pydantic import BaseModel, Field, field_validator, model_validator

from jinja2 import Environment, BaseLoader

from nodetool.agents.tools.help_tools import SearchNodesTool
from nodetool.agents.tools.base import Tool
from nodetool.providers import BaseProvider
from nodetool.metadata.types import (
    Message,
    ToolCall,
)
from nodetool.packages.registry import Registry
from nodetool.workflows.base_node import (
    BaseNode,
    InputNode,
    OutputNode,
    find_node_class_by_name,
    get_node_class,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, PlanningUpdate
from nodetool.types.graph import Graph as APIGraph
from nodetool.workflows.graph import Graph

import logging

# Set up logger for this module
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)


class NodeSpecification(BaseModel):
    """Model for node specification in workflow design."""

    node_id: str = Field(
        description="Unique identifier for the node (e.g., 'input_1', 'agent_1')"
    )
    node_type: str = Field(
        default="",
        description="The exact node type from search_nodes results (e.g., 'nodetool.agents.Agent')",
    )
    purpose: Optional[str] = Field(
        default=None,
        description="What this node does in the workflow and why it's needed",
    )
    properties: str = Field(
        default="{}", description="JSON string of properties for the node"
    )

    @field_validator("properties", mode="before")
    @classmethod
    def validate_properties(cls, v):
        """Convert dict to JSON string if needed, or ensure valid JSON string."""
        if v is None:
            return "{}"
        elif isinstance(v, dict):
            try:
                return json.dumps(v)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize properties dict: {e}")
                return "{}"
        elif isinstance(v, str):
            # Validate that it's proper JSON
            try:
                json.loads(v)  # Just validate, don't modify
                return v
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Invalid JSON string in properties: {e}")
                return "{}"
        else:
            logger.warning(f"Properties must be dict or JSON string, got {type(v)}")
            return "{}"

    @model_validator(mode="after")
    def validate_node_spec(self):
        """Post-validation to ensure node has essential fields."""
        # If node_type is missing but we have a 'type' key in properties, try to extract it
        if not self.node_type:
            try:
                props = json.loads(self.properties)
                if "type" in props:
                    self.node_type = props["type"]
                    logger.debug(
                        f"Extracted node_type '{self.node_type}' from properties for node {self.node_id}"
                    )
                else:
                    raise ValueError(
                        f"No node_type found for node {self.node_id}, properties: {self.properties}"
                    )
            except (json.JSONDecodeError, TypeError):
                raise ValueError(
                    f"Could not parse properties for node {self.node_id}, properties: {self.properties}"
                )

        return self


class WorkflowDesignResult(BaseModel):
    """Result model for workflow design phase."""

    node_specifications: List[NodeSpecification] = Field(
        description="Detailed specifications for all nodes including Input/Output nodes"
    )


# Generate JSON schema from Pydantic model
WORKFLOW_DESIGN_SCHEMA = WorkflowDesignResult.model_json_schema()


def get_node_type_for_metadata(
    type_metadata: TypeMetadata,
    is_subclass_of: type[BaseNode] | None = None,
) -> str:
    """Find the appropriate InputNode subclass for a given TypeMetadata."""
    from nodetool.packages.registry import Registry

    logger.debug(
        f"Finding node type for metadata: {type_metadata.type}, subclass_of: {is_subclass_of}"
    )
    registry = Registry()
    all_nodes = registry.get_all_installed_nodes()
    logger.debug(f"Registry has {len(all_nodes)} nodes to search")

    matches_found = 0
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
                    logger.debug(
                        f"Checking node {node_meta.node_type}: output_type={output_type.type} vs target={type_metadata.type}"
                    )

                    # Check for exact type match
                    if output_type.type == type_metadata.type:
                        logger.debug(
                            f"Found exact type match: {node_class.get_node_type()}"
                        )
                        return node_class.get_node_type()

                    # Also check for compatible types
                    elif _is_type_compatible(output_type, type_metadata):
                        logger.debug(
                            f"Found compatible type match: {node_class.get_node_type()}"
                        )
                        return node_class.get_node_type()

                    matches_found += 1

        except Exception as e:
            logger.debug(f"Could not load node class for {node_meta.node_type}: {e}")
            continue

    logger.debug(
        f"Searched {matches_found} candidate nodes, no match found for type: {type_metadata.type}"
    )
    raise ValueError(f"No InputNode match found for type: {type_metadata.type}")


def _is_type_compatible(source_type: TypeMetadata, target_type: TypeMetadata) -> bool:
    """Check if source type can be assigned to target type."""
    logger.debug(
        f"Checking type compatibility: {source_type.type} -> {target_type.type}"
    )
    # Handle any type
    if source_type.type == "any" or target_type.type == "any":
        logger.debug("Compatible: any type")
        return True

    # Handle exact matches
    if source_type.type == target_type.type:
        logger.debug("Compatible: exact match")
        return True

    # Handle numeric conversions
    numeric_types = {"int", "float"}
    if source_type.type in numeric_types and target_type.type in numeric_types:
        logger.debug("Compatible: numeric conversion")
        return True

    # Handle optional types
    if target_type.optional and not source_type.optional:
        # Can assign non-optional to optional
        result = source_type.type == target_type.type
        logger.debug(f"Compatible optional check: {result}")
        return result

    logger.debug("Not compatible")
    return False


class SubmitDesignResultTool(Tool):
    """Tool for submitting workflow design results."""

    name: str = "submit_design_result"
    description: str = (
        "Submit the final workflow design result. Use this tool when you have completed your workflow design and are ready to provide the structured node specifications."
    )
    input_schema: Dict[str, Any] = WorkflowDesignResult.model_json_schema()

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> WorkflowDesignResult:
        """Process the design result submission."""
        try:
            return WorkflowDesignResult.model_validate(params)
        except Exception as e:
            logger.error(f"Error validating design result: {e}")
            raise ValueError(f"Invalid design result format: {e}")


def print_visual_graph(graph: APIGraph) -> None:
    """Print a visual ASCII representation of the workflow graph."""
    logger.info("\n  Visual Graph Structure:")
    logger.info("  " + "=" * 50)

    # Build adjacency information
    adjacency: Dict[str, List[Any]] = {}
    reverse_adjacency: Dict[str, List[Any]] = {}
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


DEFAULT_GRAPH_PLANNING_SYSTEM_PROMPT = """
# GraphArchitect AI System Core Directives

## Mission
As GraphArchitect AI, you are an intelligent system that transforms natural language objectives into executable workflow graphs. 
Your intelligence lies in automatically understanding what users want to accomplish and creating the appropriate graph structure.

## Core Principles
1. **Graph Structure:** Design workflows as Directed Acyclic Graphs (DAGs) with no cycles
2. **Data Flow:** Connect nodes via edges that represent data flow from inputs through processing to outputs
3. **Node Design:** Each node should have a clear, focused purpose
4. **Valid Node Types:** All nodes **must** correspond to available node types. Always use `search_nodes` to discover and verify node types
5. **Type Safety:** Ensure type compatibility throughout the workflow
6. **User-Centric Design:** Create graphs that solve the user's actual problem, not just technical requirements
7. **Reasoning Privacy:** Think step-by-step internally but do not reveal chain-of-thought. Only provide requested, structured outputs or tool calls.
8. **Determinism & Efficiency:** Minimize tokens. Prefer canonical, compact JSON for node specifications. Avoid markdown in structured outputs.

## Node Metadata Structure
Each node type has specific metadata that defines:
- **properties**: Input fields/parameters the node accepts (these become targetHandles for edges)
- **outputs**: Output slots the node produces (these become sourceHandles for edges)
- **is_dynamic**: Boolean flag indicating if the node supports dynamic properties

## Input and Output Node Mappings
Input nodes: string→StringInput, int→IntegerInput, float→FloatInput, bool→BooleanInput, list[any]→ListInput, image→ImageInput, video→VideoInput, document→DocumentInput, dataframe→DataFrameInput

Output nodes: string→StringOutput, int→IntegerOutput, float→FloatOutput, bool→BooleanOutput, list[any]→ListOutput, image→ImageOutput, video→VideoOutput, document→DocumentOutput, dataframe→DataFrameOutput
"""

WORKFLOW_DESIGN_PROMPT = """
# WORKFLOW DESIGN (NODE SELECTION & DATAFLOW)

## Goal
Design a COMPLETE workflow by selecting appropriate nodes and defining ALL data flow connections between them. Use the search_nodes tool to find specific node types and understand their inputs/outputs to create proper connections.

**CRITICAL: You MUST create ALL nodes including Input and Output nodes AND ALL necessary edge connections between them. Every node that requires input must have its input properties connected via edges. The system will validate that your graph has complete data flow.**

**EDGE CONNECTION REQUIREMENT: After creating all nodes, you MUST ensure that every processing node receives its required inputs through edge connections. Missing edge connections will cause validation failures.**

Conciseness & Output Discipline:
- Keep any explanatory text minimal (<100 tokens) and only when strictly necessary.
- Do not output raw JSON directly. Submit results via the tool call only, with no extra commentary.

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

## Using `search_nodes` Efficiently
**EFFICIENCY PRIORITY:** Minimize the number of search iterations by being strategic and comprehensive:

- **Plan your searches:** Before starting, identify all the different types of processing you need (e.g., data transformation, aggregation, visualization, text generation)
- **Batch similar searches:** If you need multiple data processing nodes, search for them together with broader queries
- **Use specific, descriptive queries:** Instead of generic terms, use specific keywords that target exactly what you need
- **Target the right namespaces:** Most functionality is in `nodetool.data` (dataframes), `nodetool.text` (text processing), `nodetool.code` (custom code), `lib.*` (visualization/specialized tools)

When using the `search_nodes` tool:
- Provide a `query` with keywords describing the node's function (e.g., "convert", "summarize", "filter data").
- **Start with targeted searches using `input_type` and `output_type` when you know the data types** - this reduces irrelevant results and speeds up the process
- **Only use broad searches without type parameters if you're unsure about available node types** 
- The available types for `input_type` and `output_type` are: "str", "int", "float", "bool", "list", "dict", "tuple", "union", "enum", "any"
- **Search for multiple related functionalities in a single query** when possible (e.g., "dataframe group aggregate sum" instead of separate searches)
 - Prefer fewer, more capable nodes over long chains of trivial nodes when functionality overlaps.

## Instructions - Node Selection
1. **Create ALL nodes including Input and Output nodes.** For Input and Output nodes, use the exact node types from the system prompt mappings (do NOT search for them). Only search for intermediate processing nodes.
   - For each item in the Input Schema, create a corresponding Input node with a `name` matching the schema's `name`.
   - For each item in the Output Schema, create a corresponding Output node with a `name` matching the schema's `name`.

2. **Search for intermediate processing nodes using `search_nodes`**. Be strategic with searches - use specific, targeted queries to find the most appropriate nodes. Prefer fewer, more powerful nodes over many simple ones to improve efficiency.
   - **CRITICAL**: When you receive search results, use the EXACT `node_type` value from the results in your node specifications. For example, if the search returns `{"node_type": "lib.browser.WebFetch", ...}`, you MUST use "lib.browser.WebFetch" as the `node_type` in your node specification.
   - **For dataframe operations**: Search with relevant keywords (e.g., "GroupBy", "Aggregate", "Filter", "Transform", "dataframe"). Many dataframe nodes are in the `nodetool.data` namespace.
   - **For list operations**: Search with `input_type="list"` or `output_type="list"` and relevant keywords.
   - **For text operations**: Search with `input_type="str"` or `output_type="str"` (e.g., "concatenate", "regex", "template").
   - **For agents**: Search "agent". Verify their input/output types by inspecting their metadata from the search results before use.

3. **Type conversion patterns** (use keyword-based searches):
   - dataframe → array: Search "dataframe to array" or "to_numpy"
   - dataframe → string: Search "dataframe to string" or "to_csv"
   - array → dataframe: Search "array to dataframe" or "from_array"
   - list → item: Use iterator node
   - item → list: Use collector node

## Configuration Guidelines
- **CRITICAL - Node Specification Format**: When creating node specifications, you MUST use these exact field names:
  - `node_id`: A unique identifier for the node (e.g., "input_1", "fetch_1")
  - `node_type`: The EXACT value from the `node_type` field in search_nodes results (e.g., "lib.browser.WebFetch", NOT just the class name)
  - `purpose`: Brief description of what this node does
  - `properties`: JSON string containing the node's configuration

- **For nodes found via `search_nodes`**:
  - Copy the EXACT `node_type` value from the search results into your node specification
  - Check their metadata for required fields and create appropriate property entries

- **Dynamic Properties**: If a node has `is_dynamic=true` in its metadata, you can set ANY property name on that node, not just the predefined ones. Dynamic nodes will handle arbitrary properties at runtime.
  - For dynamic nodes: You can create custom property names based on your workflow needs
  - Example: `{"custom_field": "value", "another_field": {"type": "edge", "source": "input_1", "sourceHandle": "output"}}`
  - Still include any required properties from the metadata, but feel free to add additional ones

- **Non-dynamic nodes**: Only use properties that exist in the node's metadata. Do not invent properties for non-dynamic nodes.

- **Edge connections**: `{"type": "edge", "source": "source_node_id", "sourceHandle": "output_name"}`

- **Encode properties as a JSON string**
- Example for constant value: `{"property_name": "property_value"}`
- Example for edge connection: `{"property_name": {"type": "edge", "source": "source_node_id", "sourceHandle": "output_name"}}`
- Do not include explanatory prose alongside JSON specifications.

## Important Handle Conventions
- **Most nodes have a single output**: The default output handle is often named "output". Always verify with `search_nodes` if unsure.
- **Input nodes**: Provide data through the `"output"` handle.
- **Output nodes**: Receive data through their `"value"` property.
- **Always check metadata from `search_nodes` results** for exceptions and exact input property names (targetHandles).

Example connections:
- From an Input node: `{"type": "edge", "source": "input_id", "sourceHandle": "output"}`
- To an Output node: Connect your final node to the output using a `value` property in your node specifications

## Example Node Specification (FOLLOW THIS FORMAT EXACTLY)
```json
{
  "node_id": "fetch_1",
  "node_type": "lib.browser.WebFetch",
  "purpose": "Fetch webpage content from URL",
  "properties": "{\"url\": {\"type\": \"edge\", \"source\": \"input_url\", \"sourceHandle\": \"output\"}}"
}
```

Note: The `node_type` field uses the EXACT string from search_nodes results (e.g., "lib.browser.WebFetch"), NOT "type".

## Instructions - Dataflow Design & Edge Creation
4. **MANDATORY: Create ALL required edge connections.** Every processing node must have its input properties connected to appropriate source nodes via edges:
   - Trace data flow from Input nodes through processing nodes to Output nodes
   - Connect EVERY required input property of EVERY processing node
   - Use the node metadata from `search_nodes` to identify which properties need connections
   - Template nodes: ALL template variables (referenced as `{{ variable_name }}`) MUST have corresponding edge connections

5. **Edge Connection Checklist - VERIFY BEFORE SUBMITTING:**
   - ✓ Every Input node connects to at least one processing node
   - ✓ Every processing node has ALL its required input properties connected
   - ✓ Every Output node receives data from a processing node via its "value" property
   - ✓ Template nodes have edge connections for ALL variables used in their templates
   - ✓ Chart/visualization nodes connect their output to encoding/processing nodes if needed
   - ✓ No processing nodes are left as isolated islands without connections

6. **Type Compatibility for Connections:**
   - Exact type matches are always safe (string→string, int→int)
   - Numeric conversions are allowed (int→float, float→int)
   - 'any' type is compatible with all types
   - Union types are compatible if any member type matches
   - Complex types (document, dataframe, image) need converters for primitive types
   - Use converter nodes for type mismatches (find them with `search_nodes`)

7. **Common connection patterns:**
   - Input → Node → Output
   - Input → Node → Node → Output
   - Input → Node → Node → Node → Output

## Special Node Types
- **Iterator Nodes** (`nodetool.control.Iterator`): Process lists item-by-item, emitting each item individually through the `output` slot. Use when you need to apply operations to each element of a list separately.
- **Streaming Nodes** (`is_streaming=True`): Yield multiple outputs sequentially. Output nodes automatically collect streamed items into lists.

## FINAL VALIDATION CHECKLIST
Before submitting your node specifications, verify:

1. **CRITICAL - Correct Field Names:** Every node specification MUST use `node_type` (NOT "type") with the EXACT value from search_nodes results
2. **Complete Data Flow:** Every node that needs input has edge connections defined in its properties
3. **Template Variables:** If using template nodes, ALL variables in the template string have corresponding edge connections
4. **Output Connectivity:** All Output nodes receive data via their "value" property connection
5. **No Orphaned Nodes:** Every node participates in the data flow from inputs to outputs
6. **Proper Node Types:** All intermediate nodes use valid types found via `search_nodes`
7. **Determinism:** No extraneous text; output only the structured node specifications via the tool
8. **Schema Mapping:** There is a one-to-one mapping between schema items and created Input/Output nodes with matching `name` properties
9. **Unique IDs:** All `node_id` values are unique and consistently referenced by edges
10. **Non-Dynamic Props:** Non-dynamic nodes only use properties present in their metadata

**REMEMBER:** The evaluation showed that many workflows fail due to missing edge connections. Your graph must have COMPLETE data flow connectivity to pass validation.

## Context
**User's Objective:**
{{ objective }}

**Input Schema:**
{{ input_schema }}

**Output Schema:**
{{ output_schema }}

You MUST use the submit_design_result tool to submit your workflow design. Do not provide JSON output directly - use the tool to submit your structured node specifications.
Only emit the tool call without extra commentary.
"""


class GraphInput(BaseModel):
    """Input schema for the graph planner."""

    name: str
    type: TypeMetadata
    description: str = ""


class GraphOutput(BaseModel):
    """Output schema for the graph planner."""

    name: str
    type: TypeMetadata
    description: str = ""


class GraphPlanner:
    """Orchestrates the creation of workflow graphs from high-level objectives.

    The GraphPlanner transforms user objectives into executable workflow graphs
    composed of nodes. It uses a single-phase approach to select nodes and
    define data flow connections.
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        objective: str,
        input_schema: list[GraphInput] = [],
        output_schema: list[GraphOutput] = [],
        existing_graph: Optional[APIGraph] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 8192,
        verbose: bool = True,
    ):
        """Initialize the GraphPlanner.

        Args:
            provider: LLM provider instance
            model: Model identifier to use
            objective: High-level goal to achieve
            input_schema: List of GraphInput objects defining expected inputs
            output_schema: List of GraphOutput objects defining expected outputs
            existing_graph: Optional existing graph to edit
            system_prompt: Custom system prompt (optional)
            max_tokens: Token limit for LLM
            verbose: Enable detailed logging
        """
        logger.debug(
            f"GraphPlanner.__init__ called with provider={type(provider).__name__}, model={model}"
        )
        logger.debug(
            f"Objective length: {len(objective)} chars, input_schema: {len(input_schema)} items, output_schema: {len(output_schema)} items"
        )

        self.provider = provider
        self.model = model
        self.objective = objective
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.max_tokens = max_tokens
        self.existing_graph = existing_graph

        logger.debug(f"Using existing graph: {existing_graph is not None}")
        if existing_graph:
            logger.debug(
                f"Existing graph has {len(existing_graph.nodes)} nodes and {len(existing_graph.edges)} edges"
            )

        self.system_prompt = system_prompt or DEFAULT_GRAPH_PLANNING_SYSTEM_PROMPT
        logger.debug(
            f"System prompt length: {len(self.system_prompt)} chars (custom: {system_prompt is not None})"
        )

        self.verbose = verbose
        self.registry = Registry()
        logger.debug(f"Registry initialized")

        # Initialize Jinja2 environment
        self.jinja_env = Environment(loader=BaseLoader())
        logger.debug(f"Jinja2 environment initialized")

        # Graph storage
        self.graph: Optional[APIGraph] = None

        # Cache for expensive operations
        self._cached_node_metadata: Optional[List] = None
        self._cached_namespaces: Optional[set[str]] = None

        logger.debug(f"GraphPlanner initialized for objective: {objective[:100]}...")
        logger.debug(
            f"Input schema details: {[f'{inp.name}:{inp.type.type}' for inp in input_schema]}"
        )
        logger.debug(
            f"Output schema details: {[f'{out.name}:{out.type.type}' for out in output_schema]}"
        )

    def _get_node_metadata(self) -> List:
        """Get node metadata with caching."""
        logger.debug(
            f"_get_node_metadata called, cached: {self._cached_node_metadata is not None}"
        )
        if self._cached_node_metadata is None:
            logger.debug("Loading node metadata from registry...")
            self._cached_node_metadata = self.registry.get_all_installed_nodes()
            logger.debug(
                f"Loaded {len(self._cached_node_metadata)} node metadata entries"
            )
        return self._cached_node_metadata

    def _get_namespaces(self) -> set[str]:
        """Get namespaces with caching."""
        logger.debug(
            f"_get_namespaces called, cached: {self._cached_namespaces is not None}"
        )
        if self._cached_namespaces is None:
            logger.debug("Computing namespaces from node metadata...")
            node_metadata_list = self._get_node_metadata()
            self._cached_namespaces = {node.namespace for node in node_metadata_list}
            logger.debug(
                f"Found {len(self._cached_namespaces)} unique namespaces: {sorted(self._cached_namespaces)}"
            )
        return self._cached_namespaces

    def _convert_graph_to_specifications(self, graph: APIGraph) -> List[Dict[str, Any]]:
        """Converts an APIGraph object into the node_specifications format."""
        logger.debug(
            f"Converting APIGraph to specifications: {len(graph.nodes)} nodes, {len(graph.edges)} edges"
        )
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

            # Include dynamic properties if they exist
            if hasattr(node, "dynamic_properties") and node.dynamic_properties:
                properties.update(node.dynamic_properties)

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

        logger.debug(f"Converted graph to {len(node_specs)} node specifications")
        return node_specs

    def _build_nodes_and_edges_from_specifications(
        self,
        node_specifications: List[NodeSpecification],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Build nodes and edges from node specifications.

        Args:
            node_specifications: List of NodeSpecification models

        Returns:
            Tuple of (nodes, edges) lists
        """
        logger.debug(
            f"Building nodes and edges from {len(node_specifications)} specifications"
        )
        nodes = []
        edges = []

        for spec in node_specifications:
            node_id = spec.node_id
            node_type = spec.node_type
            logger.debug(f"Processing node spec: {node_id} ({node_type})")
            node_data = {}
            dynamic_properties = {}

            # Process properties and extract edges
            properties_string = spec.properties
            logger.debug(
                f"Parsing properties JSON for {node_id}: {len(properties_string)} chars"
            )
            properties = json.loads(properties_string)
            logger.debug(
                f"Parsed {len(properties)} properties for {node_id}: {list(properties.keys())}"
            )

            # Check if this is a dynamic node type
            logger.debug(f"Looking up node class for {node_type}")
            node_class = get_node_class(node_type)
            is_dynamic_node = node_class.is_dynamic() if node_class else False
            logger.debug(
                f"Node {node_id} is_dynamic: {is_dynamic_node}, has_class: {node_class is not None}"
            )
            standard_prop_names = (
                {prop.name for prop in node_class.properties()}
                if is_dynamic_node and node_class
                else set()
            )
            if is_dynamic_node:
                logger.debug(
                    f"Dynamic node {node_id} standard properties: {standard_prop_names}"
                )

            edge_count = 0
            for prop_name, prop_value in properties.items():
                logger.debug(
                    f"Processing property {prop_name} for {node_id}: {type(prop_value)}"
                )

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
                    edge_count += 1
                    logger.debug(
                        f"Created edge for {node_id}.{prop_name} from {prop_value['source']}.{prop_value['sourceHandle']}"
                    )
                else:
                    # For dynamic nodes, check if this property is in the standard schema
                    if is_dynamic_node and prop_name not in standard_prop_names:
                        # This is a dynamic property
                        dynamic_properties[prop_name] = prop_value
                        logger.debug(f"Added dynamic property {prop_name} to {node_id}")
                    else:
                        # This is a standard property
                        node_data[prop_name] = prop_value
                        logger.debug(
                            f"Added standard property {prop_name} to {node_id}"
                        )

            logger.debug(
                f"Node {node_id} processed: {edge_count} edges, {len(node_data)} standard props, {len(dynamic_properties)} dynamic props"
            )

            # Create node dict in the requested format
            node_dict = {
                "id": node_id,
                "type": node_type,
                "data": node_data,
            }

            # Add dynamic_properties only if there are any
            if dynamic_properties:
                node_dict["dynamic_properties"] = dynamic_properties

            nodes.append(node_dict)
            logger.debug(f"Added node {node_id} to nodes list")

        logger.debug(f"Built {len(nodes)} nodes and {len(edges)} edges total")
        return nodes, edges

    def _get_prompt_context(self, **kwargs: Any) -> Dict[str, Any]:
        """Build context for Jinja2 prompt rendering."""
        logger.debug(f"Building prompt context with {len(kwargs)} additional kwargs")
        context = {
            **kwargs,
            "objective": self.objective,
            "input_schema": json.dumps(
                [inp.model_dump() for inp in self.input_schema], indent=2
            ),
            "output_schema": json.dumps(
                [out.model_dump() for out in self.output_schema], indent=2
            ),
            "existing_graph_spec": None,
        }
        logger.debug(
            f"Base context created with objective length {len(self.objective)}"
        )

        if self.existing_graph:
            logger.debug("Converting existing graph to specifications for context")
            specs = self._convert_graph_to_specifications(self.existing_graph)
            context["existing_graph_spec"] = json.dumps(specs, indent=2)
            logger.debug(
                f"Added existing graph spec to context: {len(specs)} specifications"
            )

        logger.debug(f"Final prompt context has {len(context)} keys")
        return context

    def _render_prompt(self, template_string: str, **kwargs: Any) -> str:
        """Render a Jinja2 template with context."""
        logger.debug(
            f"Rendering prompt template: {len(template_string)} chars, {len(kwargs)} extra kwargs"
        )
        template = self.jinja_env.from_string(template_string)
        rendered = template.render(self._get_prompt_context(**kwargs))
        logger.debug(f"Rendered prompt: {len(rendered)} chars")
        return rendered

    async def _run_phase_with_tools(
        self,
        phase_name: str,
        prompt_content: str,
        response_model: type[BaseModel],
        tools: List[Any],
        context: ProcessingContext,
        history: List[Message],
        max_iterations: int = 5,
        max_validation_attempts: int = 5,
        validation_fn: Optional[Any] = None,
    ) -> AsyncGenerator[
        Chunk | ToolCall | tuple[List[Message], BaseModel, Optional[PlanningUpdate]],
        None,
    ]:
        """Generic method for running a phase with single loop tool calling including output tool.

        Args:
            phase_name: Name of the phase for display purposes
            prompt_content: The prompt to send to the LLM
            response_model: Pydantic model class for structured output
            tools: List of tools available to the LLM (output tool will be added)
            context: Processing context
            history: Message history
            max_iterations: Maximum tool calling iterations
            max_validation_attempts: Maximum attempts if validation fails
            validation_fn: Optional validation function that returns error message or empty string
        """
        # Add appropriate output tool based on response model
        output_tool_name = ""
        if response_model == WorkflowDesignResult:
            tools.append(SubmitDesignResultTool())
            output_tool_name = "submit_design_result"

        if self.verbose:
            logger.info(
                f"[{phase_name}] Running: Starting {phase_name.lower()} phase..."
            )
            available_tools = [tool.name for tool in tools]
            logger.info(
                f"[{phase_name}] Running: Available tools: {available_tools} + {output_tool_name} (output)"
            )

        history.append(Message(role="user", content=prompt_content))

        result = None

        try:
            # Single loop with tool calling including output tool
            for attempt in range(max_validation_attempts):
                if self.verbose:
                    if attempt > 0:
                        logger.info(
                            f"[{phase_name}] Running: Retry attempt {attempt + 1}/{max_validation_attempts}..."
                        )
                    else:
                        logger.info(
                            f"[{phase_name}] Running: Starting validation attempt {attempt + 1}/{max_validation_attempts}..."
                        )

                for i in range(max_iterations):
                    if self.verbose:
                        logger.info(
                            f"[{phase_name}] Running: LLM interaction (iteration {i + 1}/{max_iterations})..."
                        )

                    # Use streaming generate_messages and yield all chunks and tool calls
                    response_content = ""
                    tool_calls = []

                    async for chunk in self.provider.generate_messages(  # type: ignore
                        messages=history,
                        model=self.model,
                        tools=tools,
                        max_tokens=self.max_tokens,
                        context_window=8192,
                    ):
                        if isinstance(chunk, Chunk):
                            if chunk.content:
                                response_content += chunk.content
                            # Yield chunk to frontend
                            yield chunk
                        elif isinstance(chunk, ToolCall):
                            tool_calls.append(chunk)
                            # Yield tool call to frontend
                            yield chunk

                    # Create response message from accumulated content and tool calls
                    response = Message(
                        role="assistant",
                        content=response_content if response_content else None,
                        tool_calls=tool_calls if tool_calls else None,
                    )

                    if not response:
                        raise Exception("LLM returned no response.")

                    history.append(response)

                    if response.tool_calls:
                        if self.verbose:
                            logger.info(
                                f"[{phase_name}] Running: Executing {len(response.tool_calls)} tool call(s)..."
                            )
                            tool_names = [tc.name for tc in response.tool_calls]
                            logger.info(
                                f"[{phase_name}] Running: Tool calls: {tool_names}"
                            )
                            logger.info(
                                f"[{phase_name}] Running: Expected output tool: {output_tool_name}"
                            )

                        tool_messages_for_history: List[Message] = []
                        for tool_call in response.tool_calls:
                            logger.debug(
                                f"Processing tool call: {tool_call.name} (id: {tool_call.id})"
                            )
                            tool_output_str = ""
                            # Execute the tool call
                            tool_found = False
                            for tool_instance in tools:
                                if tool_call.name == tool_instance.name:
                                    tool_found = True
                                    logger.debug(
                                        f"Found matching tool instance for {tool_call.name}"
                                    )
                                    try:
                                        params_for_tool = tool_call.args
                                        if not isinstance(params_for_tool, dict):
                                            logger.warning(
                                                f"Tool call arguments for {tool_call.name} is not a dict: {tool_call.args}. Using empty dict."
                                            )
                                            params_for_tool = {}

                                        logger.debug(
                                            f"Tool {tool_call.name} args: {list(params_for_tool.keys()) if isinstance(params_for_tool, dict) else 'not-dict'}"
                                        )
                                        if self.verbose:
                                            logger.info(
                                                f"[{phase_name}] Running: Processing tool: {tool_call.name} with args: {params_for_tool}"
                                            )
                                        logger.debug(
                                            f"Calling tool {tool_call.name}.process()"
                                        )
                                        tool_output = await tool_instance.process(
                                            context, params_for_tool
                                        )
                                        logger.debug(
                                            f"Tool {tool_call.name} returned: {type(tool_output)}"
                                        )

                                        # Check if this is an output tool result
                                        if (
                                            tool_call.name == output_tool_name
                                            and isinstance(tool_output, response_model)
                                        ):
                                            result = tool_output
                                            tool_output_str = (
                                                "Result submitted successfully."
                                            )
                                            logger.debug(
                                                f"Output tool {output_tool_name} successfully returned {response_model.__name__}"
                                            )
                                            if self.verbose:
                                                logger.info(
                                                    f"[{phase_name}] Running: ✅ Output tool {output_tool_name} called successfully! Result captured."
                                                )
                                        else:
                                            tool_output_str = (
                                                json.dumps(tool_output)
                                                if tool_output is not None
                                                else "Tool returned no output."
                                            )
                                            logger.debug(
                                                f"Non-output tool {tool_call.name} returned: {len(tool_output_str)} chars"
                                            )
                                            if tool_call.name == output_tool_name:
                                                logger.warning(
                                                    f"[{phase_name}] Running: ⚠️ Output tool {output_tool_name} was called but result is not the expected type {response_model.__name__}. Got: {type(tool_output)}"
                                                )

                                        logger.debug(
                                            f"Tool {tool_call.name} output: {tool_output_str[:200]}..."
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f"Error executing tool {tool_call.name}: {e}",
                                            exc_info=True,
                                        )
                                        logger.debug(
                                            f"Tool execution failed for {tool_call.name}: {str(e)}"
                                        )
                                        tool_output_str = str(e)
                                        if self.verbose:
                                            logger.error(
                                                f"[{phase_name}] Running: Error in tool {tool_call.name}: {str(e)}"
                                            )
                                    break

                            if not tool_found:
                                logger.warning(
                                    f"Received unknown tool call: {tool_call.name}"
                                )
                                logger.debug(
                                    f"Available tools: {[t.name for t in tools]}"
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
                            logger.debug(
                                f"Added tool message to history for {tool_call.name}: {len(tool_output_str)} chars"
                            )
                        history.extend(tool_messages_for_history)

                        # Check if we got the result from output tool
                        if result is not None:
                            if self.verbose:
                                logger.info(
                                    f"[{phase_name}] Running: ✅ Result obtained from output tool, breaking out of iteration loop"
                                )
                            break
                        else:
                            if self.verbose:
                                logger.info(
                                    f"[{phase_name}] Running: ❌ No result obtained from output tool {output_tool_name}. Continuing to next iteration (iteration {i + 1}/{max_iterations})."
                                )
                            # If this is the last iteration and we still don't have a result, that's a problem
                            if i == max_iterations - 1:
                                if self.verbose:
                                    logger.warning(
                                        f"[{phase_name}] Running: ⚠️ Reached last iteration ({max_iterations}) without calling output tool {output_tool_name}"
                                    )
                        # Continue to the next iteration to get LLM response after tool execution
                    else:
                        # LLM didn't call any tools, this is an error - prompt again for output tool
                        if self.verbose:
                            logger.warning(
                                f"[{phase_name}] Running: ⚠️ LLM didn't call any tools (iteration {i + 1}/{max_iterations}). Prompting for {output_tool_name} tool."
                            )
                        history.append(
                            Message(
                                role="user",
                                content=f"You must call the {output_tool_name} tool to submit your {phase_name.lower()} result. Please use the tool to provide your structured output.",
                            )
                        )
                        continue

                # Check what happened at the end of this validation attempt
                if self.verbose:
                    logger.info(
                        f"[{phase_name}] Running: Completed iteration loop for attempt {attempt + 1}. Result obtained: {result is not None}"
                    )

                # If we got here and have a result, validate it
                if result is not None:
                    if self.verbose:
                        logger.info(
                            f"[{phase_name}] Running: ✅ Result obtained from output tool in attempt {attempt + 1}. Running validation..."
                        )
                    # Run validation if provided
                    if validation_fn:
                        error_message = validation_fn(result)
                        if error_message:
                            if self.verbose:
                                logger.warning(
                                    f"[{phase_name}] Running: ❌ Validation failed for attempt {attempt + 1}: {error_message}"
                                )
                            logger.warning(
                                f"{phase_name} validation failed: {error_message}"
                            )
                            # Add error to history for next attempt
                            history.append(
                                Message(
                                    role="user",
                                    content=f"Validation failed: {error_message}. Please fix the issues and call {output_tool_name} again with corrected data.",
                                )
                            )
                            result = None  # Reset result for retry
                            continue
                        else:
                            if self.verbose:
                                logger.info(
                                    f"[{phase_name}] Running: ✅ Validation passed for attempt {attempt + 1}"
                                )
                    else:
                        if self.verbose:
                            logger.info(
                                f"[{phase_name}] Running: ✅ No validation function provided, accepting result from attempt {attempt + 1}"
                            )

                    # Success
                    if self.verbose:
                        logger.info(f"[{phase_name}] Success: {phase_name} complete")
                        logger.info(
                            f"[{phase_name}] Success: Final result type: {type(result).__name__}"
                        )

                    # Yield the final result as a tuple
                    yield (
                        history,
                        result,
                        PlanningUpdate(
                            phase=phase_name,
                            status="Success",
                            content=f"{phase_name} complete",
                        ),
                    )
                    return
                else:
                    # No result was obtained, add instruction for next attempt
                    if self.verbose:
                        logger.warning(
                            f"[{phase_name}] Running: ❌ No result obtained after {max_iterations} iterations in attempt {attempt + 1}. Adding instruction for next attempt."
                        )
                    history.append(
                        Message(
                            role="user",
                            content=f"You must call the {output_tool_name} tool to submit your {phase_name.lower()} result. Please analyze the requirements and submit your structured output using the tool.",
                        )
                    )

            # All validation attempts failed
            if self.verbose:
                logger.error(
                    f"[{phase_name}] Running: ❌ All {max_validation_attempts} validation attempts exhausted without obtaining valid result"
                )
            raise ValueError(
                f"{phase_name} failed to produce valid result after {max_validation_attempts} attempts"
            )

        except Exception as e:
            logger.error(f"{phase_name} phase failed critically: {e}", exc_info=True)
            raise e

    def _validate_workflow_design(self, result: WorkflowDesignResult) -> str:
        """Validate the complete workflow design (nodes + edges)."""
        logger.debug(
            f"Validating workflow design with {len(result.node_specifications)} node specifications"
        )
        error_messages = []

        # Check for metadata_info properties in node specifications and remove them
        logger.debug("Cleaning node specifications (removing metadata_info)")
        cleaned_specs = []
        for spec in result.node_specifications:
            logger.debug(f"Cleaning spec for node {spec.node_id}")
            try:
                properties = json.loads(spec.properties)
                # Remove metadata_info if it exists
                if "metadata_info" in properties:
                    properties.pop("metadata_info")
                    logger.debug(f"Removed metadata_info from node {spec.node_id}")
                cleaned_properties = json.dumps(properties)
                cleaned_spec = NodeSpecification(
                    node_id=spec.node_id,
                    node_type=spec.node_type,
                    purpose=spec.purpose,
                    properties=cleaned_properties,
                )
                cleaned_specs.append(cleaned_spec)
                logger.debug(f"Successfully cleaned spec for {spec.node_id}")
            except (json.JSONDecodeError, TypeError) as e:
                # If properties are malformed, keep original
                logger.debug(
                    f"Failed to parse properties for {spec.node_id}: {e}, keeping original"
                )
                cleaned_specs.append(spec)

        # Create cleaned result
        logger.debug(
            f"Creating cleaned workflow result with {len(cleaned_specs)} specs"
        )
        cleaned_result = WorkflowDesignResult(node_specifications=cleaned_specs)

        # Validate dataflow analysis using the cleaned result
        logger.debug("Starting dataflow validation")
        dataflow_errors = self._validate_graph_edge_types(cleaned_result)
        if dataflow_errors:
            logger.debug(f"Dataflow validation found errors: {dataflow_errors}")
            error_messages.append(dataflow_errors)
        else:
            logger.debug("Dataflow validation passed")

        # If initial validations pass, create a real Graph and validate edge types
        if not error_messages:
            logger.debug("Running graph validation")
            graph_validation_errors = self._validate_graph_edge_types(cleaned_result)
            if graph_validation_errors:
                logger.debug(
                    f"Graph validation found errors: {graph_validation_errors}"
                )
                error_messages.append(graph_validation_errors)
            else:
                logger.debug("Graph validation passed")

        final_errors = " ".join(error_messages)
        logger.debug(
            f"Workflow design validation complete. Errors: {bool(final_errors)}"
        )
        return final_errors

    async def _run_workflow_design_phase(
        self,
        context: ProcessingContext,
        history: List[Message],
    ) -> AsyncGenerator[
        Chunk
        | ToolCall
        | tuple[List[Message], WorkflowDesignResult, Optional[PlanningUpdate]],
        None,
    ]:
        """Run the workflow design phase with SearchNodesTool."""
        logger.debug("Starting workflow design phase")
        workflow_design_prompt = self._render_prompt(WORKFLOW_DESIGN_PROMPT)
        logger.debug(
            f"Rendered workflow design prompt: {len(workflow_design_prompt)} chars"
        )

        logger.debug("Setting up tools for workflow design phase")
        search_tool = SearchNodesTool(
            exclude_namespaces=[
                "nodetool.agents",
            ]
        )
        logger.debug(
            f"Created SearchNodesTool with excluded namespaces: {search_tool.exclude_namespaces}"
        )

        async for item in self._run_phase_with_tools(
            phase_name="Workflow Design",
            prompt_content=workflow_design_prompt,
            response_model=WorkflowDesignResult,
            tools=[search_tool],
            context=context,
            history=history,
            max_iterations=8,
            max_validation_attempts=5,
            validation_fn=self._validate_workflow_design,
        ):
            if isinstance(item, tuple):
                # Final result tuple
                history, result, update = item
                logger.debug(
                    f"Workflow design phase completed with result type: {type(result)}"
                )
                yield (history, cast(WorkflowDesignResult, result), update)
            else:
                # Stream chunk or tool call
                yield item

    def _validate_graph_edge_types(self, result: WorkflowDesignResult) -> str:
        """Create a real Graph object and validate edge types using Graph.validate_edge_types()."""
        logger.debug(
            f"Starting graph edge type validation for {len(result.node_specifications)} nodes"
        )
        try:
            # Enrich node specifications with metadata
            logger.debug("Enriching node specifications with metadata")
            enriched_result = self._enrich_analysis_with_metadata(result)

            # Build nodes and edges using the helper method
            logger.debug("Building nodes and edges from enriched specifications")
            nodes, edges = self._build_nodes_and_edges_from_specifications(
                enriched_result.node_specifications,
            )
            logger.debug(
                f"Built {len(nodes)} nodes and {len(edges)} edges for validation"
            )

            # Create graph dict
            graph_dict = {"nodes": nodes, "edges": edges}
            logger.debug("Creating Graph object from dict")

            # Create Graph object and validate
            graph = Graph.from_dict(graph_dict, skip_errors=False)
            logger.debug("Graph object created, running edge type validation")
            validation_errors = graph.validate_edge_types()
            logger.debug(
                f"Edge type validation returned {len(validation_errors) if validation_errors else 0} errors"
            )

            if validation_errors:
                error_msg = "Graph edge type validation errors: " + " ".join(
                    validation_errors
                )
                logger.debug(f"Edge validation failed: {error_msg}")
                return error_msg

            logger.debug("Edge type validation passed, checking input/output nodes")
            return self._validate_input_output_nodes(graph)

        except Exception as e:
            logger.error(f"Error validating graph edge types: {e}", exc_info=True)
            return f"Failed to validate graph structure: {str(e)}"

    def _validate_input_output_nodes(self, graph: Graph) -> str:
        """Validate InputNode and OutputNode instances against input and output schemas."""
        logger.debug(f"Validating input/output nodes against schemas")
        logger.debug(f"Graph has {len(graph.nodes)} total nodes")
        error_messages = []

        # Find InputNode and OutputNode instances
        input_nodes = [node for node in graph.nodes if isinstance(node, InputNode)]
        output_nodes = [node for node in graph.nodes if isinstance(node, OutputNode)]
        logger.debug(
            f"Found {len(input_nodes)} InputNodes and {len(output_nodes)} OutputNodes"
        )

        # Check for missing required input nodes
        logger.debug(
            f"Checking {len(input_nodes)} input nodes against {len(self.input_schema)} schema entries"
        )
        found_input_names = set()
        for input_node in input_nodes:
            node_name = input_node.name
            node_type = input_node.outputs()[0].type
            logger.debug(f"Checking InputNode '{node_name}' with type '{node_type}'")

            # Check if name exists in schema
            for schema_input in self.input_schema:
                if schema_input.name == node_name:
                    found_input_names.add(node_name)
                    logger.debug(
                        f"Found matching schema input '{node_name}' with type '{schema_input.type}'"
                    )
                    if not typecheck(schema_input.type, node_type):
                        error_msg = f"InputNode '{node_name}' has type '{node_type}' which cannot be converted from the schema type '{schema_input.type}'."
                        logger.debug(f"Type check failed: {error_msg}")
                        error_messages.append(error_msg)
                    else:
                        logger.debug(f"Type check passed for InputNode '{node_name}'")
                    break

        logger.debug(
            f"Checking {len(output_nodes)} output nodes against {len(self.output_schema)} schema entries"
        )
        found_output_names = set()
        for output_node in output_nodes:
            node_name = output_node.name
            node_type = output_node.outputs()[0].type
            logger.debug(f"Checking OutputNode '{node_name}' with type '{node_type}'")

            for schema_output in self.output_schema:
                if schema_output.name == node_name:
                    found_output_names.add(node_name)
                    logger.debug(
                        f"Found matching schema output '{node_name}' with type '{schema_output.type}'"
                    )
                    if not typecheck(node_type, schema_output.type):
                        error_msg = f"OutputNode '{node_name}' has type '{node_type}' which cannot be converted to the schema type '{schema_output.type}'."
                        logger.debug(f"Type check failed: {error_msg}")
                        error_messages.append(error_msg)
                    else:
                        logger.debug(f"Type check passed for OutputNode '{node_name}'")
                    break

        # Check for missing input nodes
        required_input_names = set(
            schema_input.name for schema_input in self.input_schema
        )
        missing_input_names = required_input_names - found_input_names
        logger.debug(
            f"Required inputs: {required_input_names}, Found: {found_input_names}, Missing: {missing_input_names}"
        )
        if missing_input_names:
            error_msg = f"Missing required InputNodes for: {list(missing_input_names)}. You must create InputNodes for all inputs in the schema."
            logger.debug(f"Missing input validation failed: {error_msg}")
            error_messages.append(error_msg)

        # Check for missing output nodes
        required_output_names = set(
            schema_output.name for schema_output in self.output_schema
        )
        missing_output_names = required_output_names - found_output_names
        logger.debug(
            f"Required outputs: {required_output_names}, Found: {found_output_names}, Missing: {missing_output_names}"
        )
        if missing_output_names:
            error_msg = f"Missing required OutputNodes for: {list(missing_output_names)}. You must create OutputNodes for all outputs in the schema."
            logger.debug(f"Missing output validation failed: {error_msg}")
            error_messages.append(error_msg)

        final_errors = " ".join(error_messages)
        logger.debug(
            f"Input/output node validation complete. Errors: {bool(final_errors)}"
        )
        return final_errors

    def _enrich_analysis_with_metadata(
        self, analysis_result: WorkflowDesignResult
    ) -> WorkflowDesignResult:
        """Enrich the analysis result with actual node metadata from the registry."""
        logger.debug(
            f"Enriching analysis result with {len(analysis_result.node_specifications)} node specifications"
        )
        # Create a copy of the result with enriched node specifications
        enriched_specs = []

        for node_spec in analysis_result.node_specifications:
            node_type = node_spec.node_type
            logger.debug(
                f"Enriching node spec {node_spec.node_id} with type {node_type}"
            )
            if node_type:
                # Get the node class from registry
                if "." in node_type:
                    logger.debug(f"Looking up full node type: {node_type}")
                    node_class = get_node_class(node_type)
                else:
                    logger.debug(f"Looking up node by name: {node_type}")
                    node_class = find_node_class_by_name(node_type)

                if node_class:
                    # Get node metadata
                    logger.debug(f"Getting metadata for {node_type}")
                    metadata = node_class.get_metadata()

                    # Create enriched node specification (keep original properties without metadata_info)
                    enriched_spec = NodeSpecification(
                        node_id=node_spec.node_id,
                        node_type=node_spec.node_type,
                        purpose=node_spec.purpose,
                        properties=node_spec.properties,
                    )
                    enriched_specs.append(enriched_spec)

                    logger.debug(
                        f"Validated metadata for {node_type}: {len(metadata.properties)} properties, {len(metadata.outputs)} outputs"
                    )
                else:
                    logger.warning(f"Could not find node class for type: {node_type}")
                    # Keep original spec if we can't enrich it
                    enriched_specs.append(node_spec)
            else:
                logger.debug(
                    f"No node type specified for {node_spec.node_id}, keeping original"
                )
                # Keep original spec if no node type
                enriched_specs.append(node_spec)

        logger.debug(
            f"Enrichment complete: {len(enriched_specs)} specifications processed"
        )
        return WorkflowDesignResult(node_specifications=enriched_specs)

    def log_graph_summary(self) -> None:
        """Log a summary of the generated graph for debugging."""
        logger.debug("log_graph_summary called")
        if not self.graph:
            logger.warning("No graph has been generated yet.")
            return

        logger.debug(
            f"Logging summary for graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
        )

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
        logger.debug("Computing node type distribution")
        node_type_counts: Dict[str, int] = {}
        for node in self.graph.nodes:
            node_type = node.type
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        logger.debug(f"Found {len(node_type_counts)} unique node types")

        logger.info("\nNode Type Distribution:")
        for node_type, count in sorted(node_type_counts.items()):
            logger.info(f"  - {node_type}: {count}")

        # Log input/output nodes specifically
        logger.info("\nInput/Output Nodes:")
        input_count = output_count = 0
        for node in self.graph.nodes:
            if "input" in node.type.lower():
                input_count += 1
                logger.info(f"  - Input: {node.id} ({node.type})")
                if hasattr(node, "data") and "name" in node.data:
                    logger.info(f"    Schema Name: {node.data['name']}")
            elif "output" in node.type.lower():
                output_count += 1
                logger.info(f"  - Output: {node.id} ({node.type})")
                if hasattr(node, "data") and "name" in node.data:
                    logger.info(f"    Schema Name: {node.data['name']}")

        logger.debug(
            f"Graph summary complete: {input_count} inputs, {output_count} outputs"
        )

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
        logger.debug(f"Full objective: {self.objective}")
        logger.debug(f"Context type: {type(context).__name__}")

        # Log start of graph planning
        if self.verbose:
            logger.info("Starting Graph Planner")

        logger.debug(
            f"Initializing history with system prompt: {len(self.system_prompt)} chars"
        )
        history: List[Message] = [
            Message(role="system", content=self.system_prompt),
        ]
        logger.debug(f"History initialized with {len(history)} messages")

        # Phase 1: Workflow Design
        current_phase = "Workflow Design"
        logger.info(f"Starting Phase 1: {current_phase}")
        logger.debug(f"About to yield PlanningUpdate for {current_phase}")
        yield PlanningUpdate(phase=current_phase, status="Starting", content=None)

        workflow_result = None
        planning_update = None
        logger.debug("Starting workflow design phase iteration")
        async for item in self._run_workflow_design_phase(context, history):
            if isinstance(item, tuple):
                # Final result tuple
                history, workflow_result, planning_update = item
                logger.debug(
                    f"Got final result tuple: history={len(history)} messages, result={workflow_result is not None}, update={planning_update is not None}"
                )
                if planning_update:
                    logger.debug(
                        f"Yielding planning update: {planning_update.phase} - {planning_update.status}"
                    )
                    yield planning_update
            else:
                # Stream chunk or tool call to frontend
                yield item  # type: ignore

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

        # Enrich node specifications with actual metadata
        logger.info("Enriching node specifications with metadata from registry...")
        assert workflow_result is not None
        logger.debug(
            f"Workflow result has {len(workflow_result.node_specifications)} node specifications"
        )
        enriched_workflow = self._enrich_analysis_with_metadata(workflow_result)
        logger.debug(
            f"Enriched workflow has {len(enriched_workflow.node_specifications)} node specifications"
        )

        yield PlanningUpdate(
            phase="Metadata Enrichment",
            status="Success",
            content=f"Enhanced {len(enriched_workflow.node_specifications)} nodes with metadata",
        )

        # Phase 2: Graph Creation
        current_phase = "Graph Creation"
        logger.info(f"Starting Phase 2: {current_phase}")
        logger.debug(f"About to yield PlanningUpdate for {current_phase}")
        yield PlanningUpdate(phase=current_phase, status="Starting", content=None)

        # Build nodes and edges using the helper method
        logger.debug("Building final nodes and edges for graph creation")
        nodes, edges = self._build_nodes_and_edges_from_specifications(
            enriched_workflow.node_specifications,
        )
        logger.debug(f"Final build completed: {len(nodes)} nodes, {len(edges)} edges")

        # Create the final graph
        logger.debug("Creating APIGraph object")
        self.graph = APIGraph(
            nodes=nodes,  # type: ignore
            edges=edges,  # type: ignore
        )
        logger.info(
            f"Graph created successfully with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
        )
        logger.debug(f"Graph object created: {type(self.graph)}")

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

        logger.debug(f"About to yield final PlanningUpdate for {current_phase}")
        yield PlanningUpdate(
            phase=current_phase,
            status="Success",
            content=f"Created graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges",
        )

        if self.verbose:
            logger.info(
                f"[{current_phase}] Success: Graph created with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
            )

        logger.debug("Graph creation completed successfully")
