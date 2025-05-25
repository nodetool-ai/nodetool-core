"""Export NodeTool workflows to Python DSL code.

This module provides functionality to convert a workflow model (which is typically
stored as JSON or loaded from a database) into executable Python code using the
NodeTool DSL (Domain Specific Language).

The primary use case is to allow users to:
1. Design workflows visually or through configuration
2. Export them as Python code for version control, sharing, or direct execution
3. Modify the generated code for advanced use cases

The export process:
1. Analyzes the workflow graph structure (nodes and edges)
2. Generates appropriate imports for each node type
3. Creates variable assignments for each node with their properties
4. Handles connections between nodes by mapping edge connections to tuple references
5. Produces a complete Python script that can recreate and run the workflow

Example:
    A workflow with two nodes (an input and a processor) connected by an edge
    would generate code like:

    ```python
    from nodetool.dsl.input import TextInput
    from nodetool.dsl.llm import ProcessText
    from nodetool.dsl.graph import graph, run_graph

    n0 = TextInput(value="Hello")
    n1 = ProcessText(input=(n0, 'output'), model="gpt-4")

    g = graph(n0, n1)
    result = await run_graph(g)
    print(result)
    ```

The generated code uses the NodeTool DSL which provides a more pythonic way
to define and execute workflows compared to the JSON representation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Sequence

from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.types.graph import Edge


def _edge_map(edges: Sequence[Edge]) -> Dict[tuple[str, str], Edge]:
    mapping: Dict[tuple[str, str], Edge] = {}
    for edge in edges:
        mapping[(edge.target, edge.targetHandle)] = edge
    return mapping


def _dsl_import_path(node_type: str) -> tuple[str, str]:
    """Return the DSL import path and class name for a node type."""
    parts = node_type.split(".")
    module = "nodetool.dsl." + ".".join(parts[:-1])
    return module, parts[-1]


def workflow_to_dsl(workflow: WorkflowModel) -> str:
    """Export a workflow to a Python DSL representation as a function.

    The generated function:
    - Takes input parameters corresponding to input nodes
    - Returns a dictionary mapping output node names to their results
    """
    graph = workflow.get_graph()
    nodes = graph.nodes
    edges = graph.edges

    # Identify input and output nodes
    input_nodes = graph.inputs()
    output_nodes = graph.outputs()

    name_map: Dict[str, str] = {}
    imports: Dict[str, Set[str]] = {}

    for idx, node in enumerate(nodes):
        var = f"n{idx}"
        name_map[node.id] = var
        module, class_name = _dsl_import_path(node.get_node_type())
        imports.setdefault(module, set()).add(class_name)

    edge_lookup = _edge_map(edges)

    lines: List[str] = []

    # Generate imports
    for module, classes in sorted(imports.items()):
        class_list = ", ".join(sorted(classes))
        lines.append(f"from {module} import {class_list}")
    lines.append("from nodetool.dsl.graph import graph, run_graph")
    lines.append("")

    # Generate function signature
    func_name = (
        workflow.name.replace(" ", "_").replace("-", "_").lower()
        if workflow.name
        else "workflow"
    )
    # Sanitize function name to be a valid Python identifier
    func_name = re.sub(r"[^a-zA-Z0-9_]", "_", func_name)
    func_name = re.sub(r"^(\d)", r"_\1", func_name)  # Ensure doesn't start with number

    input_params = []
    for input_node in input_nodes:
        param_name = (
            input_node.name if input_node.name else f"input_{name_map[input_node.id]}"
        )
        input_params.append(param_name)

    param_str = ", ".join(input_params) if input_params else ""
    lines.append(f"async def {func_name}({param_str}):")
    lines.append('    """')
    if workflow.description:
        lines.append(f"    {workflow.description}")
    else:
        lines.append(f"    Generated function from workflow: {workflow.name}")
    lines.append('    """')

    # Generate node definitions (indented for function body)
    for node in nodes:
        var = name_map[node.id]
        _, class_name = _dsl_import_path(node.get_node_type())
        props = []

        # Check if this is an input node
        is_input_node = node in input_nodes

        for name, value in node.node_properties().items():
            edge = edge_lookup.get((node.id, name))
            if edge:
                src_var = name_map[edge.source]
                props.append(f"{name}=({src_var}, '{edge.sourceHandle}')")
            elif is_input_node and name == "value":
                # For input nodes, map the value to the function parameter
                param_name = node.name if node.name else f"input_{var}"
                props.append(f"{name}={param_name}")
            else:
                props.append(f"{name}={repr(value)}")
        prop_str = ", ".join(props)
        lines.append(f"    {var} = {class_name}({prop_str})")
    lines.append("")

    # Generate graph and execution
    node_vars = ", ".join(name_map[n.id] for n in nodes)
    lines.append(f"    g = graph({node_vars})")
    lines.append("    result = await run_graph(g)")
    lines.append("")

    # Generate return statement
    if output_nodes:
        lines.append("    # Return outputs as a dictionary")
        lines.append("    outputs = {}")
        for output_node in output_nodes:
            output_name = (
                output_node.name
                if output_node.name
                else f"output_{name_map[output_node.id]}"
            )
            var = name_map[output_node.id]
            lines.append(f"    outputs['{output_name}'] = result[{var}]")
        lines.append("    return outputs")
    else:
        lines.append("    return result")

    return "\n".join(lines)
