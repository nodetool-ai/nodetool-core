from __future__ import annotations

from typing import Dict, List, Set

from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.types.graph import Edge


def _edge_map(edges: List[Edge]) -> Dict[tuple[str, str], Edge]:
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
    """Export a workflow to a Python DSL representation."""
    graph = workflow.get_graph()
    nodes = graph.nodes
    edges = graph.edges

    name_map: Dict[str, str] = {}
    imports: Dict[str, Set[str]] = {}

    for idx, node in enumerate(nodes):
        var = f"n{idx}"
        name_map[node.id] = var
        module, class_name = _dsl_import_path(node.get_node_type())
        imports.setdefault(module, set()).add(class_name)

    edge_lookup = _edge_map(edges)

    lines: List[str] = []
    for module, classes in sorted(imports.items()):
        class_list = ", ".join(sorted(classes))
        lines.append(f"from {module} import {class_list}")
    lines.append("from nodetool.dsl.graph import graph, run_graph")
    lines.append("")

    for node in nodes:
        var = name_map[node.id]
        _, class_name = _dsl_import_path(node.get_node_type())
        props = []
        for name, value in node.node_properties().items():
            edge = edge_lookup.get((node.id, name))
            if edge:
                src_var = name_map[edge.source]
                props.append(f"{name}=({src_var}, '{edge.sourceHandle}')")
            else:
                props.append(f"{name}={repr(value)}")
        prop_str = ", ".join(props)
        lines.append(f"{var} = {class_name}({prop_str})")
    lines.append("")

    node_vars = ", ".join(name_map[n.id] for n in nodes)
    lines.append(f"g = graph({node_vars})")
    lines.append("result = await run_graph(g)")
    lines.append("print(result)")

    return "\n".join(lines)
