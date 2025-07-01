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
2. **Design Phase**: Uses the DOT graph as a guide to select specific nodes and define data connections
3. **Validation Phase**: Validates the designed workflow for type compatibility and structural integrity

Graph creation occurs as the final step after successful validation.

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

The GraphPlanner includes type safety features:

- Validates node types through SearchNodesTool registry lookup
- Performs basic type compatibility checking for edge connections
- Ensures Input/Output nodes match schema requirements
- Validation occurs during the dedicated validation phase

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
- **Validation Functions**: Ensure graph correctness at each phase
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

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, cast
import traceback
from nodetool.chat.providers.anthropic_provider import AnthropicProvider
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.typecheck import typecheck
from pydantic import BaseModel, Field
from enum import Enum

from jinja2 import Environment, BaseLoader

from nodetool.agents.tools.help_tools import SearchNodesTool
from nodetool.agents.tools.base import Tool
from nodetool.chat.providers import ChatProvider
from nodetool.metadata.types import (
    Message,
    ToolCall,
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



REVISION_PROMPT_TEMPLATE = """
# Workflow Design Revision Task

The previous workflow design has been validated and requires improvement. You need to revise the design based on the validation feedback below.

## Original Objective
{{ objective }}

## Previous Design
The previous workflow design included:

**Node Count:** {{ node_count }}

**Node Specifications:**
{{ original_nodes }}

## Validation Results

**Validation Status:** {{ validation_status }}

**Objective Fulfillment Score:** {{ obj_fulfillment_score }}/10
**Analysis:** {{ obj_fulfillment_analysis }}

**Design Consistency Score:** {{ design_consistency_score }}/10
**Analysis:** {{ design_consistency_analysis }}

**Flow Correctness Score:** {{ flow_correctness_score }}/10
**Analysis:** {{ flow_correctness_analysis }}

**Recommendations:**
{{ recommendations }}

**Overall Assessment:**
{{ overall_assessment }}

## Revision Instructions

Based on the validation feedback above, please revise the workflow design to address the identified issues:

1. **Address Low Scores:** Focus on areas with scores below 8/10
2. **Follow Recommendations:** Implement the specific recommendations provided
3. **Improve Objective Fulfillment:** Ensure the revised design better addresses the original objective
4. **Enhance Design Consistency:** Make sure the design follows logical flow and best practices
5. **Fix Flow Issues:** Correct any data flow or connection problems identified

## Your Task

Please provide a revised workflow design that addresses the validation feedback. You may:
- Modify existing nodes and their properties
- Add new nodes if needed to improve functionality
- Remove nodes that don't contribute to the objective
- Restructure the data flow to be more logical and efficient
- Use the SearchNodesTool to find better node types if needed

You MUST use the submit_design_result tool to submit your revised workflow design. Do not provide JSON output directly - use the tool to submit your improved node specifications.
"""


# Pydantic models for type safety

class ValidationStatus(str, Enum):
    """Enum for validation status."""
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"


class UsageContext(str, Enum):
    """Enum for usage context."""
    WORKFLOW = "workflow"
    TOOL = "tool"
    HYBRID = "hybrid"


class InferredInput(BaseModel):
    """Model for inferred input schema."""
    name: str = Field(description="Input parameter name")
    type: str = Field(description="Data type (str, int, float, bool, list, dict, image, video, audio, dataframe, document, any)")
    description: str = Field(description="What this input represents")


class InferredOutput(BaseModel):
    """Model for inferred output schema."""
    name: str = Field(description="Output parameter name")
    type: str = Field(description="Data type (str, int, float, bool, list, dict, image, video, audio, dataframe, document, any)")
    description: str = Field(description="What this output represents")


class WorkflowAnalysisResult(BaseModel):
    """Result model for workflow analysis phase."""
    model_config = {"use_enum_values": True}
    
    objective_interpretation: str = Field(description="Clear interpretation of what the user wants to achieve")
    workflow_approach: str = Field(description="High-level approach to solve the problem using a graph workflow")
    expected_outcomes: List[str] = Field(description="List of expected outputs or results from the workflow")
    constraints: List[str] = Field(description="Any constraints or special requirements identified")
    assumptions: List[str] = Field(description="Assumptions made about the workflow")
    required_namespaces: List[str] = Field(description="List of required node namespaces for the workflow")
    workflow_graph_dot: str = Field(description="DOT graph notation representing the planned workflow structure")
    inferred_inputs: List[InferredInput] = Field(description="Inferred input requirements based on the objective")
    inferred_outputs: List[InferredOutput] = Field(description="Inferred output results based on the objective")
    usage_context: UsageContext = Field(description="Whether this graph is meant to be used as a standalone workflow, as a tool for LLMs, or both")


class NodeSpecification(BaseModel):
    """Model for node specification in workflow design."""
    node_id: str = Field(description="Unique identifier for the node (e.g., 'input_1', 'agent_1')")
    node_type: str = Field(description="The exact node type from search_nodes results (e.g., 'nodetool.agents.Agent')")
    purpose: str = Field(description="What this node does in the workflow and why it's needed")
    properties: str = Field(description="JSON string of properties for the node")


class WorkflowDesignResult(BaseModel):
    """Result model for workflow design phase."""
    node_specifications: List[NodeSpecification] = Field(description="Detailed specifications for processing nodes only (Input/Output nodes are automatically generated)")


class ValidationScore(BaseModel):
    """Model for validation score with analysis."""
    score: int = Field(ge=1, le=10, description="Score from 1-10")
    analysis: str = Field(description="Detailed analysis")


class GraphValidationResult(BaseModel):
    """Result model for graph validation phase."""
    model_config = {"use_enum_values": True}
    
    validation_status: ValidationStatus = Field(description="Overall validation status of the generated graph")
    objective_fulfillment: ValidationScore = Field(description="Score and analysis of how well the graph fulfills the original objective")
    design_consistency: ValidationScore = Field(description="Score and analysis of consistency between analysis, design, and final graph")
    flow_correctness: ValidationScore = Field(description="Score and analysis of correctness of data flow and node connections")
    recommendations: List[str] = Field(description="List of recommendations for improvement (if any)")
    overall_assessment: str = Field(description="Overall assessment summary and final verdict on the generated workflow")


# Generate JSON schemas from Pydantic models
WORKFLOW_ANALYSIS_SCHEMA = WorkflowAnalysisResult.model_json_schema()
WORKFLOW_DESIGN_SCHEMA = WorkflowDesignResult.model_json_schema()
GRAPH_VALIDATION_SCHEMA = GraphValidationResult.model_json_schema()


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


class SubmitAnalysisResultTool(Tool):
    """Tool for submitting workflow analysis results."""
    name: str = "submit_analysis_result"
    description: str = "Submit the final analysis result for the objective interpretation phase. Use this tool when you have completed your analysis and are ready to provide the structured result."
    input_schema: Dict[str, Any] = WorkflowAnalysisResult.model_json_schema()

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> WorkflowAnalysisResult:
        """Process the analysis result submission."""
        try:
            return WorkflowAnalysisResult.model_validate(params)
        except Exception as e:
            logger.error(f"Error validating analysis result: {e}")
            raise ValueError(f"Invalid analysis result format: {e}")


class SubmitDesignResultTool(Tool):
    """Tool for submitting workflow design results."""
    name: str = "submit_design_result"
    description: str = "Submit the final workflow design result. Use this tool when you have completed your workflow design and are ready to provide the structured node specifications."
    input_schema: Dict[str, Any] = WorkflowDesignResult.model_json_schema()

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> WorkflowDesignResult:
        """Process the design result submission."""
        try:
            return WorkflowDesignResult.model_validate(params)
        except Exception as e:
            logger.error(f"Error validating design result: {e}")
            raise ValueError(f"Invalid design result format: {e}")


class SubmitValidationResultTool(Tool):
    """Tool for submitting graph validation results."""
    name: str = "submit_validation_result"
    description: str = "Submit the final validation result for the graph. Use this tool when you have completed your validation analysis and are ready to provide the structured validation scores and assessment."
    input_schema: Dict[str, Any] = GraphValidationResult.model_json_schema()

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> GraphValidationResult:
        """Process the validation result submission."""
        try:
            return GraphValidationResult.model_validate(params)
        except Exception as e:
            logger.error(f"Error validating validation result: {e}")
            raise ValueError(f"Invalid validation result format: {e}")


# Now mappers are always available
MAPPERS_AVAILABLE = True


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
Your intelligence lies in automatically understanding what users want to accomplish and creating the appropriate graph structure without requiring manual specification of inputs and outputs.

## Intelligence Principles
1. **Automatic Inference:** You MUST automatically infer required inputs and outputs from the user's objective. Do not require explicit schemas.
2. **Contextual Awareness:** Understand whether the graph should be used as:
   - **Workflow**: Standalone process executed by users
   - **Tool**: Called by LLMs to process specific inputs  
   - **Hybrid**: Designed to work in both contexts
3. **Flexible Adaptation:** Adapt the graph design based on the intended usage context
4. **Smart Defaults:** Make intelligent assumptions about data types, processing steps, and connections

## Core Principles
1. **Graph Structure:** Design workflows as Directed Acyclic Graphs (DAGs) with no cycles
2. **Data Flow:** Connect nodes via edges that represent data flow from inputs through processing to outputs
3. **Node Design:** Each node should have a clear, focused purpose
4. **Valid Node Types:** All nodes **must** correspond to available node types. Always use `search_nodes` to discover and verify node types
5. **Type Safety:** Ensure type compatibility throughout the workflow:
   - Connect compatible types (matching types, numeric conversions, or 'any' type)
   - Use converter nodes when types don't match directly
   - Plan for type conversions during node selection phase
6. **User-Centric Design:** Create graphs that solve the user's actual problem, not just technical requirements

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

## Input and Output Nodes

Input and Output nodes are special nodes that are used to connect the workflow to the outside world.
Use following mapping to create input:
- string -> nodetool.input.StringInput
- int -> nodetool.input.IntegerInput
- float -> nodetool.input.FloatInput
- bool -> nodetool.input.BooleanInput
- list[any] -> nodetool.input.ListInput
- image -> nodetool.input.ImageInput
- video -> nodetool.input.VideoInput
- document -> nodetool.input.DocumentInput
- dataframe -> nodetool.input.DataFrameInput

Use following mapping to create output:
- string -> nodetool.output.StringOutput
- int -> nodetool.output.IntegerOutput
- float -> nodetool.output.FloatOutput
- bool -> nodetool.output.BooleanOutput
- list[any] -> nodetool.output.ListOutput
- image -> nodetool.output.ImageOutput
- video -> nodetool.output.VideoOutput
- document -> nodetool.output.DocumentOutput
- dataframe -> nodetool.output.DataFrameOutput


## Dynamic Properties
Some nodes have `is_dynamic=true` in their metadata, which means:
- **Flexible Configuration**: You can set any property name on these nodes, not just predefined ones
- **Runtime Handling**: The node will dynamically process arbitrary properties during execution
- **Custom Fields**: Create property names that match your specific workflow requirements
- **Beyond Schema**: You're not limited to the properties listed in the node's metadata
```

## Type Compatibility Rules
- Exact matches are always compatible (string -> string)
- Numeric conversions are allowed (int -> float, float -> int)
- 'any' type is compatible with all types
- Union types are compatible if any member type matches
- Complex types (document, dataframe, image) need converters for primitive types

## Iterator Nodes: Processing Lists Item-by-Item

Iterator nodes are special nodes that process lists by emitting each item individually to downstream nodes. 

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
- **Streaming workflows**: When working with large lists that should be processed incrementally

### Important Notes:
- Iterator nodes trigger downstream processing for each item in the list
- Downstream nodes process one item at a time, not the entire list
- Output nodes automatically collect streamed items into a list[T]

## Streaming Node Behavior and Output Collection

Some nodes, identifiable by `is_streaming=True` in their metadata, are designed to yield multiple outputs sequentially during their execution. 
This is common for iterators that process list items one-by-one.

### Key Streaming Concepts:

1.  **Multiple Yields**: A node with `is_streaming=True` can produce a sequence of output values on a single output handle from one processing cycle. 
    For example, an `Iterator` processes an input list and yields each item of the list individually through its `output` handle.

2.  **Output Node Collection**: When a streaming node (e.g., an `Iterator` yielding items of type `T`) is connected to an Output node defined in the `output_schema` with type `T` (e.g., `string`), the Output node will automatically collect all the individual streamed items of type `T`. 
    The system handles aggregating these items, typically into a list, when the graph execution completes or the stream ends.
    - **Important**: When designing the graph and defining `output_schema`, if you expect a stream of items of type `T` to be collected by an Output node, you should declare that `GraphOutput` with type `T` (e.g., `string`), NOT `list[T]` (e.g., `list[string]`). 
    The graph execution system manages the collection. This simplifies the graph's output signature.
    - **Avoid `list[T]` for streamed outputs**: Do not define an Output node as `list[T]` if it's meant to collect a stream of `T` items. 
    Use type `T` for the Output node, and the system will provide the collected list.

3.  **Edge Validation for Streaming Outputs**:
    - When connecting a streaming node (source) to an Output node (target), ensure the type compatibility check considers the *individual item type* yielded by the streaming source.
    - For example, if an `Iterator` yields `string` items, it should connect to an Output node defined in the schema as type `string`. The validation should confirm this `string` to `string` compatibility for the individual items, understanding that the Output node will handle the collection.
"""


GRAPH_ANALYSIS_PHASE_TEMPLATE = """
# PHASE 1: OBJECTIVE ANALYSIS AND SCHEMA INFERENCE

## Goal
Analyze the user's objective to understand what the workflow needs to accomplish.
Intelligently infer the required inputs and outputs from the objective without requiring explicit schemas.

## Instructions
1. Interpret what the user wants to achieve
2. Identify the high-level approach for solving the problem
3. List expected outcomes from the workflow
4. Note any constraints or special requirements
5. Document assumptions you're making
6. **CRITICAL: Infer required inputs and outputs from the objective**
7. Determine usage context (workflow, tool, or hybrid)
8. Create a DOT graph representing the planned workflow structure

## Input/Output Inference Guidelines
Based on the objective, intelligently determine:

### Inputs:
- What data does the workflow need to receive to accomplish the objective?
- Look for nouns that represent data (e.g., "image", "text", "CSV file", "name")
- Look for parameters mentioned (e.g., "interest rate", "principal amount")
- If no explicit inputs are mentioned, consider if the workflow generates data internally
- Name inputs descriptively based on their purpose

### Outputs:
- What results should the workflow produce?
- Look for desired outcomes (e.g., "report", "analysis", "chart", "greeting")
- Consider intermediate results that might be useful
- Name outputs based on what they represent

### Usage Context:
- **workflow**: Standalone process executed by a user
- **tool**: Called by an LLM to process specific inputs
- **hybrid**: Can be used both ways

### Type Mapping:
- Text/strings → "str"
- Numbers → "int" or "float" 
- Yes/no values → "bool"
- Collections → "list" or "dict"
- Media → "image", "video", "audio"
- Tabular data → "dataframe"
- Files → "document"
- Unknown/flexible → "any"

## DOT Graph Guidelines
Create a simple DOT graph that shows the high-level flow:
- Start with inferred inputs
- Show processing steps
- End with inferred outputs
- Use descriptive labels

Example DOT format:
```dot
digraph workflow {
    input_name [label="User Name"];
    generate [label="Generate Greeting"];
    output_greeting [label="Greeting Message"];
    
    input_name -> generate;
    generate -> output_greeting;
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

{% if input_nodes and input_nodes|length > 0 -%}
**Provided Input Schema (use these instead of inferring):**
{{ input_nodes }}
{%- else -%}
**No input schema provided - you MUST infer inputs from the objective**
{%- endif %}

{% if output_nodes and output_nodes|length > 0 -%}
**Provided Output Schema (use these instead of inferring):**
{{ output_nodes }}
{%- else -%}
**No output schema provided - you MUST infer outputs from the objective**
{%- endif %}

You MUST use the submit_analysis_result tool to submit your analysis. Do not provide any JSON output directly - use the tool to submit your structured analysis result.
"""

WORKFLOW_DESIGN_PROMPT = """
# PHASE 2: WORKFLOW DESIGN (NODE SELECTION & DATAFLOW)

## Goal
Based on the previous analysis, design the COMPLETE workflow by selecting appropriate intermediate nodes and defining ALL data flow connections between them. Use the search_nodes tool to find specific node types and understand their inputs/outputs to create proper connections.

**CRITICAL: You MUST create ALL nodes including Input and Output nodes AND ALL necessary edge connections between them. Every node that requires input must have its input properties connected via edges. The system will validate that your graph has complete data flow.**

**EDGE CONNECTION REQUIREMENT: After creating all nodes, you MUST ensure that every processing node receives its required inputs through edge connections. Missing edge connections will cause validation failures.**

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

## Instructions - Node Selection
1. **Create ALL nodes including Input and Output nodes.** For Input and Output nodes, use the exact node types provided in the context below (do NOT search for them). Only search for intermediate processing nodes.
   
2. **Search for intermediate processing nodes using `search_nodes`**. Be strategic with searches - use specific, targeted queries to find the most appropriate nodes. Prefer fewer, more powerful nodes over many simple ones to improve efficiency.
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

## Instructions - Dataflow Design & Edge Creation
5. **MANDATORY: Create ALL required edge connections.** Every processing node must have its input properties connected to appropriate source nodes via edges:
   - Trace data flow from Input nodes through processing nodes to Output nodes
   - Connect EVERY required input property of EVERY processing node
   - Use the node metadata from `search_nodes` to identify which properties need connections
   - Template nodes: ALL template variables (referenced as `{{ variable_name }}`) MUST have corresponding edge connections

6. **Edge Connection Checklist - VERIFY BEFORE SUBMITTING:**
   - ✓ Every Input node connects to at least one processing node
   - ✓ Every processing node has ALL its required input properties connected
   - ✓ Every Output node receives data from a processing node via its "value" property
   - ✓ Template nodes have edge connections for ALL variables used in their templates
   - ✓ Chart/visualization nodes connect their output to encoding/processing nodes if needed
   - ✓ No processing nodes are left as isolated islands without connections

7. **Connect nodes based on type compatibility:**
   - Exact type matches are always safe
   - Use converter nodes for type mismatches (find them with `search_nodes`)
   - Check node metadata from `search_nodes` for actual input/output types for nodes.

8. **Common connection patterns:**
   - Input → Node → Output
   - Input → Node → Node → Output
   - Input → Node → Node → Node → Output

## Type Compatibility Rules
- ✓ dataframe → dataframe operations → dataframe
- ✓ string → string, int → float, any → any type
- ✗ dataframe → list operations (needs conversion, search for a converter node)
- ✗ LoadCSVFile as input node (use proper Input node type as recommended or found via `search_nodes`)

## FINAL VALIDATION CHECKLIST
Before submitting your node specifications, verify:

1. **Complete Data Flow:** Every node that needs input has edge connections defined in its properties
2. **Template Variables:** If using template nodes, ALL variables in the template string have corresponding edge connections
3. **Output Connectivity:** All Output nodes receive data via their "value" property connection
4. **No Orphaned Nodes:** Every node participates in the data flow from inputs to outputs
5. **Proper Node Types:** All intermediate nodes use valid types found via `search_nodes`

**REMEMBER:** The evaluation showed that many workflows fail due to missing edge connections. Your graph must have COMPLETE data flow connectivity to pass validation.

## Context
**User's Objective:**
{{ objective }}

**Input Nodes:**
{{ input_nodes }}

**Output Nodes:**
{{ output_nodes }}

You MUST use the submit_design_result tool to submit your workflow design. Do not provide JSON output directly - use the tool to submit your structured node specifications.
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
        max_tokens: int = 8192,
        verbose: bool = True,
    ):
        """Initialize the GraphPlanner.

        Args:
            provider: LLM provider instance
            model: Model identifier to use
            objective: High-level goal to achieve
            inputs: Dictionary of input values to infer types from
            input_schema: List of GraphInput objects defining expected inputs (optional - will be inferred from objective if empty)
            output_schema: List of GraphOutput objects defining expected outputs (optional - will be inferred from objective if empty)
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

        # Schema handling - prioritize provided schemas, but allow inference
        if not input_schema and inputs:
            # If inputs are provided but no schema, infer from values
            self.input_schema = self._infer_input_schema_from_values(inputs)
            logger.info(
                f"Inferred input schema from provided values: {[inp.model_dump() for inp in self.input_schema]}"
            )
        else:
            # Use provided schema (may be empty - will be inferred during analysis phase)
            self.input_schema = input_schema

        self.output_schema = output_schema
        # Note: Empty schemas will be populated during the analysis phase via LLM inference

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

        # Build initial node representations (may be empty if schemas not provided)
        self._build_initial_node_representations()
        
        logger.debug(f"GraphPlanner initialized for objective: {objective[:100]}...")

    def _build_initial_node_representations(self) -> None:
        """Build initial input/output node representations from current schemas."""
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

            # Include dynamic properties if they exist
            if hasattr(node, 'dynamic_properties') and node.dynamic_properties:
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

    def _process_inferred_schemas(self, analysis_result: WorkflowAnalysisResult) -> None:
        """Process inferred input/output schemas from analysis phase and update instance schemas."""
        # Update input schema if not already provided
        if not self.input_schema and analysis_result.inferred_inputs:
            inferred_inputs = analysis_result.inferred_inputs
            self.input_schema = [
                GraphInput(
                    name=inp.name,
                    type=TypeMetadata(type=inp.type),
                    description=inp.description
                )
                for inp in inferred_inputs
            ]
            logger.info(f"Using inferred input schema: {[inp.name + ':' + inp.type.type for inp in self.input_schema]}")

        # Update output schema if not already provided  
        if not self.output_schema and analysis_result.inferred_outputs:
            inferred_outputs = analysis_result.inferred_outputs
            self.output_schema = [
                GraphOutput(
                    name=out.name,
                    type=TypeMetadata(type=out.type),
                    description=out.description
                )
                for out in inferred_outputs
            ]
            logger.info(f"Using inferred output schema: {[out.name + ':' + out.type.type for out in self.output_schema]}")

        # Rebuild node representations with updated schemas
        self._build_initial_node_representations()

        # Log usage context if available
        if analysis_result.usage_context:
            usage_context = analysis_result.usage_context
            logger.info(f"Graph usage context: {usage_context}")

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
        nodes = []
        edges = []

        for spec in node_specifications:
            node_id = spec.node_id
            node_type = spec.node_type
            node_data = {}
            dynamic_properties = {}

            # Process properties and extract edges
            properties_string = spec.properties
            properties = json.loads(properties_string)

            # Check if this is a dynamic node type
            node_class = get_node_class(node_type)
            is_dynamic_node = node_class.is_dynamic() if node_class else False
            standard_prop_names = {prop.name for prop in node_class.properties()} if is_dynamic_node and node_class else set()

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
                    # For dynamic nodes, check if this property is in the standard schema
                    if is_dynamic_node and prop_name not in standard_prop_names:
                        # This is a dynamic property
                        dynamic_properties[prop_name] = prop_value
                    else:
                        # This is a standard property
                        node_data[prop_name] = prop_value

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

        return nodes, edges

    def _get_prompt_context(self, **kwargs: Any) -> Dict[str, Any]:
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

    def _render_prompt(self, template_string: str, **kwargs: Any) -> str:
        """Render a Jinja2 template with context."""
        template = self.jinja_env.from_string(template_string)
        return template.render(self._get_prompt_context(**kwargs))

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
    ) -> AsyncGenerator[Chunk | ToolCall | tuple[List[Message], BaseModel, Optional[PlanningUpdate]], None]:
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
        if response_model == WorkflowAnalysisResult:
            tools.append(SubmitAnalysisResultTool())
            output_tool_name = "submit_analysis_result"
        elif response_model == WorkflowDesignResult:
            tools.append(SubmitDesignResultTool()) 
            output_tool_name = "submit_design_result"
        elif response_model == GraphValidationResult:
            tools.append(SubmitValidationResultTool())
            output_tool_name = "submit_validation_result"

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
                    
                    async for chunk in self.provider.generate_messages( # type: ignore
                        messages=history,
                        model=self.model,
                        tools=tools,
                        max_tokens=self.max_tokens,
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
                                        
                                        # Check if this is an output tool result
                                        if tool_call.name == output_tool_name and isinstance(tool_output, response_model):
                                            result = tool_output
                                            tool_output_str = "Result submitted successfully."
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
                        logger.info(f"[{phase_name}] Success: Final result type: {type(result).__name__}")

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

    async def _run_analysis_phase(
        self, context: ProcessingContext, history: List[Message]
    ) -> AsyncGenerator[Chunk | ToolCall | tuple[List[Message], WorkflowAnalysisResult, Optional[PlanningUpdate]], None]:
        """Run the analysis phase to understand objectives and design workflow."""
        namespaces = self._get_namespaces()

        analysis_prompt_content = self._render_prompt(
            GRAPH_ANALYSIS_PHASE_TEMPLATE,
            namespaces=list(namespaces),
        )

        async for item in self._run_phase_with_tools(
            phase_name="Analysis",
            prompt_content=analysis_prompt_content,
            response_model=WorkflowAnalysisResult,
            tools=[SearchNodesTool(
                exclude_namespaces=[
                    "nodetool.agents",
                ]
            )],
            context=context,
            history=history,
            max_iterations=5,
            max_validation_attempts=1,
            validation_fn=None,
        ):
            if isinstance(item, tuple):
                # Final result tuple
                history, result, update = item
                yield (history, cast(WorkflowAnalysisResult, result), update)
            else:
                # Stream chunk or tool call
                yield item

    def _validate_workflow_design(self, result: WorkflowDesignResult) -> str:
        """Validate the complete workflow design (nodes + edges)."""
        error_messages = []

        # Check for metadata_info properties in node specifications and remove them
        cleaned_specs = []
        for spec in result.node_specifications:
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
                    properties=cleaned_properties
                )
                cleaned_specs.append(cleaned_spec)
            except (json.JSONDecodeError, TypeError):
                # If properties are malformed, keep original
                cleaned_specs.append(spec)
        
        # Create cleaned result
        cleaned_result = WorkflowDesignResult(node_specifications=cleaned_specs)

        # Validate dataflow analysis using the cleaned result
        dataflow_errors = self._validate_graph_edge_types(cleaned_result)
        if dataflow_errors:
            error_messages.append(dataflow_errors)

        # If initial validations pass, create a real Graph and validate edge types
        if not error_messages:
            graph_validation_errors = self._validate_graph_edge_types(cleaned_result)
            if graph_validation_errors:
                error_messages.append(graph_validation_errors)

        return " ".join(error_messages)

    async def _run_workflow_design_phase(
        self,
        context: ProcessingContext,
        history: List[Message],
    ) -> AsyncGenerator[Chunk | ToolCall | tuple[List[Message], WorkflowDesignResult, Optional[PlanningUpdate]], None]:
        """Run the combined workflow design phase with SearchNodesTool."""
        workflow_design_prompt = self._render_prompt(WORKFLOW_DESIGN_PROMPT)

        async for item in self._run_phase_with_tools(
            phase_name="Workflow Design",
            prompt_content=workflow_design_prompt,
            response_model=WorkflowDesignResult,
            tools=[
                SearchNodesTool(
                    exclude_namespaces=[
                        "nodetool.agents",
                    ]
                )
            ],
            context=context,
            history=history,
            max_iterations=8,
            max_validation_attempts=5,
            validation_fn=self._validate_workflow_design,
        ):
            if isinstance(item, tuple):
                # Final result tuple
                history, result, update = item
                yield (history, cast(WorkflowDesignResult, result), update)
            else:
                # Stream chunk or tool call
                yield item

    async def _run_graph_validation_phase(
        self,
        context: ProcessingContext,
        history: List[Message],
        analysis_result: WorkflowAnalysisResult,
        workflow_result: WorkflowDesignResult,
    ) -> AsyncGenerator[Chunk | ToolCall | tuple[List[Message], GraphValidationResult, Optional[PlanningUpdate]], None]:
        """Run the graph validation phase to review the generated graph against analysis and design."""
        
        # Create a comprehensive prompt for validation
        validation_prompt = self._create_validation_prompt(analysis_result, workflow_result)
        
        async for item in self._run_phase_with_tools(
            phase_name="Graph Validation",
            prompt_content=validation_prompt,
            response_model=GraphValidationResult,
            tools=[
                SearchNodesTool(
                    exclude_namespaces=[
                        "nodetool.agents",
                    ]
                )
            ],
            context=context,
            history=history,
            max_iterations=5,
            max_validation_attempts=3,
            validation_fn=None, 
        ):
            if isinstance(item, tuple):
                # Final result tuple
                history, result, update = item
                yield (history, cast(GraphValidationResult, result), update)
            else:
                # Stream chunk or tool call
                yield item

    async def _run_workflow_design_revision_phase(
        self,
        context: ProcessingContext,
        history: List[Message],
        original_workflow_result: WorkflowDesignResult,
        validation_result: GraphValidationResult,
    ) -> AsyncGenerator[Chunk | ToolCall | tuple[List[Message], WorkflowDesignResult, Optional[PlanningUpdate]], None]:
        """Run a revised workflow design phase using validation feedback to improve the design."""
        
        # Create a revision prompt that includes the original design and validation feedback
        revision_prompt = self._create_design_revision_prompt(original_workflow_result, validation_result)
        
        async for item in self._run_phase_with_tools(
            phase_name="Design Revision",
            prompt_content=revision_prompt,
            response_model=WorkflowDesignResult,
            tools=[
                SearchNodesTool(
                    exclude_namespaces=[
                        "nodetool.agents",
                    ]
                )
            ],
            context=context,
            history=history,
            max_iterations=6,  # Allow fewer iterations for revision
            max_validation_attempts=3,
            validation_fn=self._validate_workflow_design,
        ):
            if isinstance(item, tuple):
                # Final result tuple
                history, result, update = item
                yield (history, cast(WorkflowDesignResult, result), update)
            else:
                # Stream chunk or tool call
                yield item

    def _create_design_revision_prompt(self, original_workflow_result: WorkflowDesignResult, validation_result: GraphValidationResult) -> str:
        """Create a prompt for revising the workflow design based on validation feedback."""
        
        # Extract validation feedback details
        validation_status = validation_result.validation_status
        obj_fulfillment = validation_result.objective_fulfillment
        design_consistency = validation_result.design_consistency
        flow_correctness = validation_result.flow_correctness
        recommendations = validation_result.recommendations
        overall_assessment = validation_result.overall_assessment
        
        # Format original node specifications for reference
        original_nodes = self._format_node_specifications_for_validation(
            original_workflow_result.node_specifications
        )
        
        # Format recommendations
        formatted_recommendations = chr(10).join(['- ' + rec for rec in recommendations]) if recommendations else 'No specific recommendations provided.'
        
        # Render template with variables
        template = self.jinja_env.from_string(REVISION_PROMPT_TEMPLATE)
        revision_prompt = template.render(
            objective=self.objective,
            node_count=len(original_workflow_result.node_specifications),
            original_nodes=original_nodes,
            validation_status=validation_status,
            obj_fulfillment_score=obj_fulfillment.score,
            obj_fulfillment_analysis=obj_fulfillment.analysis,
            design_consistency_score=design_consistency.score,
            design_consistency_analysis=design_consistency.analysis,
            flow_correctness_score=flow_correctness.score,
            flow_correctness_analysis=flow_correctness.analysis,
            recommendations=formatted_recommendations,
            overall_assessment=overall_assessment
        )
        
        return revision_prompt

    def _create_validation_prompt(self, analysis_result: WorkflowAnalysisResult, workflow_result: WorkflowDesignResult) -> str:
        """Create a comprehensive validation prompt that includes analysis, design, and graph details."""
        
        # Get the current graph structure for validation
        graph_summary = self._get_graph_summary_for_validation()
        
        validation_prompt = f"""
# Graph Validation Task

You are a workflow validation expert. Your task is to thoroughly review a generated workflow graph against the original analysis and design to ensure it correctly fulfills the intended objective.

## Original Objective
{self.objective}

## Analysis Results
The initial analysis phase identified the following:

**Objective Interpretation:**
{analysis_result.objective_interpretation}

**Workflow Approach:**
{analysis_result.workflow_approach}

**Expected Outcomes:**
{chr(10).join(['- ' + outcome for outcome in analysis_result.expected_outcomes])}

**Inferred Input Schema:**
{chr(10).join(['- ' + inp.name + ' (' + inp.type + '): ' + inp.description for inp in analysis_result.inferred_inputs])}

**Inferred Output Schema:**
{chr(10).join(['- ' + out.name + ' (' + out.type + '): ' + out.description for out in analysis_result.inferred_outputs])}

**Usage Context:** {analysis_result.usage_context}

**Constraints:**
{chr(10).join(['- ' + constraint for constraint in analysis_result.constraints])}

**Assumptions:**
{chr(10).join(['- ' + assumption for assumption in analysis_result.assumptions])}

## Design Results
The workflow design phase created the following plan:

**Node Count:** {len(workflow_result.node_specifications)}

**Node Specifications:**
{self._format_node_specifications_for_validation(workflow_result.node_specifications)}

## Generated Graph Structure
{graph_summary}

## Validation Requirements

Please perform a comprehensive validation by scoring each aspect from 1-10:

1. **Objective Fulfillment (1-10):** Does the generated graph fully address the original objective?
   - Check if all required functionality is present
   - Verify that the workflow will produce the expected outcomes
   - Assess completeness and correctness

2. **Design Consistency (1-10):** How well does the final graph match the planned design?
   - Compare nodes in the graph vs. planned specifications
   - Check if data flow matches the intended design
   - Verify that all planned components are implemented

3. **Flow Correctness (1-10):** Are the data connections logical and correct?
   - Validate input/output type compatibility
   - Check for missing or incorrect connections
   - Ensure proper data flow from inputs to outputs

4. **Recommendations:** Provide specific, actionable recommendations for improvement (if any)

5. **Overall Assessment:** Provide a comprehensive summary of the workflow's readiness and quality

You MUST use the submit_validation_result tool to submit your validation assessment. Do not provide JSON output directly - use the tool to submit your structured validation results.
"""
        
        return validation_prompt

    def _get_graph_summary_for_validation(self) -> str:
        """Generate a detailed graph summary for validation purposes."""
        if not self.graph:
            return "No graph generated."
        
        summary = "**Graph Overview:**\n"
        summary += f"- Nodes: {len(self.graph.nodes)}\n"
        summary += f"- Edges: {len(self.graph.edges)}\n\n"
        
        summary += "**Nodes:**\n"
        for node in self.graph.nodes:
            summary += f"- {node.id} [{node.type}]\n"
            if hasattr(node, "data") and node.data:
                for key, value in node.data.items():
                    val_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    summary += f"  - {key}: {val_str}\n"
        
        summary += "\n**Edges:**\n"
        for edge in self.graph.edges:
            summary += f"- {edge.source}({edge.sourceHandle}) → {edge.target}({edge.targetHandle})\n"
        
        return summary

    def _format_node_specifications_for_validation(self, node_specs: List[NodeSpecification]) -> str:
        """Format node specifications for the validation prompt."""
        if not node_specs:
            return "No node specifications available."
        
        formatted = ""
        for spec in node_specs:
            node_id = spec.node_id
            node_type = spec.node_type
            purpose = spec.purpose
            
            formatted += f"- **{node_id}** [{node_type}]\n"
            formatted += f"  Purpose: {purpose}\n"
            
            # Format properties
            properties_string = spec.properties
            try:
                properties = json.loads(properties_string)
                if properties:
                    formatted += "  Properties:\n"
                    for prop_name, prop_value in properties.items():
                        if isinstance(prop_value, dict) and prop_value.get("type") == "edge":
                            formatted += f"    - {prop_name}: [edge from {prop_value.get('source')}.{prop_value.get('sourceHandle')}]\n"
                        else:
                            val_str = str(prop_value)[:50] + "..." if len(str(prop_value)) > 50 else str(prop_value)
                            formatted += f"    - {prop_name}: {val_str}\n"
            except (json.JSONDecodeError, TypeError):
                formatted += f"  Properties: {properties_string}\n"
            
            formatted += "\n"
        
        return formatted

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

    def _validate_graph_edge_types(self, result: WorkflowDesignResult) -> str:
        """Create a real Graph object and validate edge types using Graph.validate_edge_types()."""
        try:
            # Enrich node specifications with metadata
            enriched_result = self._enrich_analysis_with_metadata(result)

            # Build nodes and edges using the helper method
            nodes, edges = self._build_nodes_and_edges_from_specifications(
                enriched_result.node_specifications,
            )

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
        self, analysis_result: WorkflowDesignResult
    ) -> WorkflowDesignResult:
        """Enrich the analysis result with actual node metadata from the registry."""
        # Create a copy of the result with enriched node specifications
        enriched_specs = []
        
        for node_spec in analysis_result.node_specifications:
            node_type = node_spec.node_type
            if node_type:
                # Get the node class from registry
                node_class = get_node_class(node_type)
                if node_class:
                    # Get node metadata
                    metadata = node_class.get_metadata()

                    # Create enriched node specification (keep original properties without metadata_info)
                    enriched_spec = NodeSpecification(
                        node_id=node_spec.node_id,
                        node_type=node_spec.node_type,
                        purpose=node_spec.purpose,
                        properties=node_spec.properties
                    )
                    enriched_specs.append(enriched_spec)

                    logger.debug(
                        f"Validated metadata for {node_type}: {len(metadata.properties)} properties, {len(metadata.outputs)} outputs"
                    )
                else:
                    logger.warning(
                        f"Could not find node class for type: {node_type}"
                    )
                    # Keep original spec if we can't enrich it
                    enriched_specs.append(node_spec)
            else:
                # Keep original spec if no node type
                enriched_specs.append(node_spec)

        return WorkflowDesignResult(node_specifications=enriched_specs)

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
        node_type_counts: Dict[str, int] = {}
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
        
        analysis_result = None
        planning_update = None
        async for item in self._run_analysis_phase(context, history):
            if isinstance(item, tuple):
                # Final result tuple
                history, analysis_result, planning_update = item
                if planning_update:
                    yield planning_update
            else:
                # Stream chunk or tool call to frontend
                yield item # type: ignore
        
        if planning_update and planning_update.status == "Failed":
            error_msg = f"Analysis phase failed: {planning_update.content}"
            logger.error(error_msg)
            if self.verbose:
                logger.error(f"[Overall Status] Failed: {error_msg}")
            raise ValueError(error_msg)

        # Process inferred schemas before continuing to design phase
        if analysis_result:
            self._process_inferred_schemas(analysis_result)

        # Pretty print analysis results and yield as chunks
        if self.verbose and analysis_result:
            logger.info("\n" + "=" * 60)
            logger.info("WORKFLOW ANALYSIS RESULTS")
            logger.info("=" * 60)
            
            # Build analysis summary to send to client
            analysis_summary = "\n## 📊 Workflow Analysis Results\n\n"
            
            # Objective Interpretation
            objective_interp = analysis_result.objective_interpretation
            logger.info("Objective Interpretation:")
            logger.info(f"  {objective_interp}")
            analysis_summary += f"**Objective Interpretation:**\n{objective_interp}\n\n"
            
            # Workflow Approach
            workflow_approach = analysis_result.workflow_approach
            logger.info("\nWorkflow Approach:")
            logger.info(f"  {workflow_approach}")
            analysis_summary += f"**Workflow Approach:**\n{workflow_approach}\n\n"
            
            # Expected Outcomes
            logger.info("\nExpected Outcomes:")
            analysis_summary += "**Expected Outcomes:**\n"
            for outcome in analysis_result.expected_outcomes:
                logger.info(f"  • {outcome}")
                analysis_summary += f"- {outcome}\n"
            analysis_summary += "\n"
            
            # Print inferred schemas
            if analysis_result.inferred_inputs:
                logger.info("\nInferred Inputs:")
                analysis_summary += "**Inferred Inputs:**\n"
                for inp in analysis_result.inferred_inputs:
                    logger.info(f"  • {inp.name} ({inp.type}): {inp.description}")
                    analysis_summary += f"- `{inp.name}` ({inp.type}): {inp.description}\n"
                analysis_summary += "\n"
            
            if analysis_result.inferred_outputs:
                logger.info("\nInferred Outputs:")
                analysis_summary += "**Inferred Outputs:**\n"
                for out in analysis_result.inferred_outputs:
                    logger.info(f"  • {out.name} ({out.type}): {out.description}")
                    analysis_summary += f"- `{out.name}` ({out.type}): {out.description}\n"
                analysis_summary += "\n"
            
            usage_context = analysis_result.usage_context
            logger.info(f"\nUsage Context: {usage_context}")
            analysis_summary += f"**Usage Context:** {usage_context}\n\n"
            
            # Constraints and Assumptions
            if analysis_result.constraints:
                logger.info("\nConstraints:")
                analysis_summary += "**Constraints:**\n"
                for constraint in analysis_result.constraints:
                    logger.info(f"  • {constraint}")
                    analysis_summary += f"- {constraint}\n"
                analysis_summary += "\n"
                
            if analysis_result.assumptions:
                logger.info("\nAssumptions:")
                analysis_summary += "**Assumptions:**\n"
                for assumption in analysis_result.assumptions:
                    logger.info(f"  • {assumption}")
                    analysis_summary += f"- {assumption}\n"
                analysis_summary += "\n"
            
            # DOT Graph
            logger.info("\nPlanned Workflow Structure (DOT Graph):")
            dot_graph = analysis_result.workflow_graph_dot
            if dot_graph:
                analysis_summary += "**Planned Workflow Structure (DOT Graph):**\n```dot\n"
                # Print the DOT graph with indentation
                for line in dot_graph.split("\n"):
                    logger.info(f"  {line}")
                analysis_summary += dot_graph + "\n```\n\n"
            logger.info("=" * 60)
            
            # Yield the analysis summary as a chunk
            yield Chunk(content=analysis_summary, done=False)

        # Phase 2: Workflow Design (Combined Node Selection & Dataflow)
        current_phase = "Workflow Design"
        logger.info(f"Starting Phase 2: {current_phase}")
        yield PlanningUpdate(phase=current_phase, status="Starting", content=None)
        
        workflow_result = None
        planning_update = None
        async for item in self._run_workflow_design_phase(context, history):
            if isinstance(item, tuple):
                # Final result tuple
                history, workflow_result, planning_update = item
                if planning_update:
                    yield planning_update
            else:
                # Stream chunk or tool call to frontend
                yield item # type: ignore
        
        if planning_update and planning_update.status == "Failed":
            error_msg = f"Workflow design phase failed: {planning_update.content}"
            logger.error(error_msg)
            if self.verbose:
                logger.error(f"[Overall Status] Failed: {error_msg}")
            raise ValueError(error_msg)

        # Pretty print workflow design results and yield as chunks
        if self.verbose and workflow_result:
            logger.info("\n" + "=" * 60)
            logger.info("WORKFLOW DESIGN RESULTS")
            logger.info("=" * 60)


        # Enrich node specifications with actual metadata
        logger.info("Enriching node specifications with metadata from registry...")
        assert workflow_result is not None
        enriched_workflow = self._enrich_analysis_with_metadata(workflow_result)

        yield PlanningUpdate(
            phase="Metadata Enrichment",
            status="Success",
            content=f"Enhanced {len(enriched_workflow.node_specifications)} nodes with metadata",
        )

        # Graph Creation and Validation Loop - iterate until validation passes or max attempts reached
        max_revision_attempts = 3
        validation_result = None
        
        for revision_attempt in range(max_revision_attempts):
            attempt_suffix = f" (Attempt {revision_attempt + 1}/{max_revision_attempts})" if revision_attempt > 0 else ""
            
            # If this is a revision attempt, use validation feedback to improve the design
            if revision_attempt > 0 and validation_result:
                logger.info(f"Starting revision attempt {revision_attempt + 1} based on validation feedback")
                yield PlanningUpdate(
                    phase="Design Revision",
                    status="Starting", 
                    content=f"Revising design based on validation feedback (attempt {revision_attempt + 1}/{max_revision_attempts})"
                )
                
                # Re-run workflow design phase with validation feedback
                original_workflow_result = workflow_result
                assert original_workflow_result is not None
                workflow_result = None
                planning_update = None
                async for item in self._run_workflow_design_revision_phase(context, history, original_workflow_result, validation_result):
                    if isinstance(item, tuple):
                        # Final result tuple
                        history, workflow_result, planning_update = item
                        if planning_update:
                            yield planning_update
                    else:
                        # Stream chunk or tool call to frontend
                        yield item # type: ignore
                
                if planning_update and planning_update.status == "Failed":
                    error_msg = f"Design revision phase failed: {planning_update.content}"
                    logger.error(error_msg)
                    if self.verbose:
                        logger.error(f"[Overall Status] Failed: {error_msg}")
                    raise ValueError(error_msg)
                
                # Re-enrich with metadata after revision
                assert workflow_result is not None
                enriched_workflow = self._enrich_analysis_with_metadata(workflow_result)

            # Create graph directly from the workflow design
            current_phase = f"Graph Creation{attempt_suffix}"
            logger.info(f"Starting Phase 3: {current_phase}")
            yield PlanningUpdate(phase=current_phase, status="Starting", content=None)

            # Build nodes and edges using the helper method
            nodes, edges = self._build_nodes_and_edges_from_specifications(
                enriched_workflow.node_specifications,
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

            # Phase 4: Graph Validation
            validation_phase = f"Graph Validation{attempt_suffix}"
            logger.info(f"Starting Phase 4: {validation_phase}")
            yield PlanningUpdate(phase=validation_phase, status="Starting", content=None)
            
            validation_result = None
            planning_update = None
            assert analysis_result is not None
            async for item in self._run_graph_validation_phase(context, history, analysis_result, workflow_result):
                if isinstance(item, tuple):
                    # Final result tuple
                    history, validation_result, planning_update = item
                    if planning_update:
                        yield planning_update
                else:
                    # Stream chunk or tool call to frontend
                    yield item # type: ignore
            
            if planning_update and planning_update.status == "Failed":
                error_msg = f"Graph validation phase failed: {planning_update.content}"
                logger.error(error_msg)
                if self.verbose:
                    logger.error(f"[Overall Status] Failed: {error_msg}")
                # Don't raise here - validation failure shouldn't stop graph creation
                validation_result = GraphValidationResult(
                    validation_status=ValidationStatus.FAILED,
                    objective_fulfillment=ValidationScore(score=1, analysis="Validation failed"),
                    design_consistency=ValidationScore(score=1, analysis="Validation failed"), 
                    flow_correctness=ValidationScore(score=1, analysis="Validation failed"),
                    recommendations=["Review validation errors and retry"],
                    overall_assessment="Validation process failed"
                )

            # Pretty print validation results and yield as chunks
            if self.verbose and validation_result:
                logger.info("\n" + "=" * 60)
                logger.info("GRAPH VALIDATION RESULTS")
                logger.info("=" * 60)

                # Build validation summary to send to client
                validation_status = validation_result.validation_status
                logger.info(f"Validation Status: {validation_status}")

                # Objective Fulfillment
                obj_fulfillment = validation_result.objective_fulfillment
                obj_score = obj_fulfillment.score
                obj_analysis = obj_fulfillment.analysis
                logger.info(f"Objective Fulfillment Score: {obj_score}/10")
                logger.info(f"Objective Analysis: {obj_analysis}")

                # Design Consistency
                design_consistency = validation_result.design_consistency
                design_score = design_consistency.score
                design_analysis = design_consistency.analysis
                logger.info(f"Design Consistency Score: {design_score}/10")
                logger.info(f"Design Analysis: {design_analysis}")

                # Flow Correctness
                flow_correctness = validation_result.flow_correctness
                flow_score = flow_correctness.score
                flow_analysis = flow_correctness.analysis
                logger.info(f"Flow Correctness Score: {flow_score}/10")
                logger.info(f"Flow Analysis: {flow_analysis}")

                # Recommendations
                recommendations = validation_result.recommendations
                if recommendations:
                    logger.info("Recommendations:")
                    for rec in recommendations:
                        logger.info(f"  • {rec}")

                # Overall Assessment
                overall_assessment = validation_result.overall_assessment
                logger.info(f"Overall Assessment: {overall_assessment}")

                logger.info("=" * 60)
                
            yield PlanningUpdate(
                phase=validation_phase,
                status="Success",
                content=f"Graph validation completed with status: {validation_result.validation_status if validation_result else 'Unknown'}",
            )

            if self.verbose:
                logger.info(
                    f"[{validation_phase}] Success: Graph validation completed with status: {validation_result.validation_status if validation_result else 'Unknown'}"
                )

            # Check validation result and decide whether to continue or break
            validation_status = validation_result.validation_status if validation_result else ValidationStatus.FAILED
            
            if validation_status == ValidationStatus.PASSED:
                logger.info("Graph validation passed! Graph creation complete.")
                break
            elif validation_status == ValidationStatus.NEEDS_REVISION:
                if revision_attempt < max_revision_attempts - 1:
                    logger.info(f"Graph needs revision. Attempting revision {revision_attempt + 2}/{max_revision_attempts}")
                    continue
                else:
                    logger.warning(f"Graph needs revision but reached maximum attempts ({max_revision_attempts}). Accepting current graph.")
                    break
            else:  # failed or unknown
                if revision_attempt < max_revision_attempts - 1:
                    logger.warning(f"Graph validation failed. Attempting revision {revision_attempt + 2}/{max_revision_attempts}")
                    continue
                else:
                    logger.warning(f"Graph validation failed but reached maximum attempts ({max_revision_attempts}). Accepting current graph.")
                    break



async def main():
    """Main function demonstrating the enhanced GraphPlanner with automatic schema inference."""
    provider = AnthropicProvider()
    model = "claude-sonnet-4-20250514"
    
    # Test objective - NO input/output schemas provided, everything will be inferred automatically
    objective = "Generate a personalized greeting. The workflow should take a name as input and use a template to create a message like 'Hello, [name]! Welcome to the Nodetool demo.'"

    print("🚀 Testing Enhanced GraphPlanner with Automatic Schema Inference")
    print(f"Objective: {objective}")
    print("📋 Note: NO input or output schemas provided - everything will be inferred from the objective!")
    print()

    planner = GraphPlanner(
        provider=provider,
        model=model,
        objective=objective,
        verbose=True,
        # NO input_schema or output_schema provided!
        # The planner will automatically infer these from the objective
    )

    # Initialize a basic ProcessingContext
    # For this example, many ProcessingContext features are not strictly necessary
    # for GraphPlanner's core graph generation logic, but it expects an instance.
    context = ProcessingContext()

    print("⚡ Running Enhanced GraphPlanner...")
    print("This will involve LLM calls and may take a few moments...")
    print()
    
    try:
        async for update in planner.create_graph(context):
            if isinstance(update, PlanningUpdate):
                print(f"📋 {update.phase}: {update.status}")
                if update.content:
                    print(f"   └─ {update.content}")
            elif isinstance(update, Chunk):
                print(f"📨 Chunk: {update.content}")

        if planner.graph:
            print("\n" + "="*60)
            print("🎉 GRAPH GENERATED SUCCESSFULLY!")
            print("="*60)
            
            # Show inferred schemas
            print("\n🔍 AUTOMATICALLY INFERRED SCHEMAS:")
            print(f"📥 Inputs ({len(planner.input_schema)}):")
            for inp in planner.input_schema:
                print(f"   • {inp.name} ({inp.type.type}): {inp.description}")
            
            print(f"\n📤 Outputs ({len(planner.output_schema)}):")
            for out in planner.output_schema:
                print(f"   • {out.name} ({out.type.type}): {out.description}")

            # Create visual graph representation
            print_visual_graph(planner.graph)

            print(f"\n🔧 DETAILED NODES ({len(planner.graph.nodes)}):")
            for node in planner.graph.nodes:
                print(f"   └─ {node.id} [{node.type}]")
                if hasattr(node, "data") and node.data:
                    for key, value in node.data.items():
                        val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                        print(f"      {key}: {val_str}")

            print(f"\n🔗 EDGES ({len(planner.graph.edges)}):")
            for edge in planner.graph.edges:
                print(f"   └─ {edge.source}({edge.sourceHandle}) ──→ {edge.target}({edge.targetHandle})")
                
            print("\n" + "="*60)
            print("✅ SUCCESS: The GraphPlanner automatically inferred the complete workflow!")
            print("🧠 No manual input/output schema specification was required.")
            print("="*60)

        else:
            print("\n❌ Graph planning completed, but no graph was generated.")

    except Exception as e:
        print(f"\n💥 An error occurred during graph planning: {e}")
        traceback.print_exc()
    finally:
        print("\n🏁 GraphPlanner demonstration finished.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
