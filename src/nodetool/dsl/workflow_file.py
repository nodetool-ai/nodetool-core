"""
Workflow File Module for NodeTool DSL

This module provides functionality for loading and exporting workflows as Python files.
A workflow file is a Python module that defines:
- Module-level docstring with background and context
- Top-level variables for workflow attributes (name, description, tags, etc.)
- A `graph` variable containing the workflow graph created from DSL nodes

Example workflow file structure:
```python
\"\"\"
This workflow performs sentiment analysis on text input.

Background:
    The workflow uses natural language processing to analyze
    the sentiment of user-provided text.

Context:
    Useful for analyzing customer feedback, social media posts,
    or any text where emotional tone matters.
\"\"\"

from nodetool.dsl.graph import graph
from nodetool.dsl.nodetool.text import TextInput, Sentiment

# Workflow metadata
name = "Sentiment Analysis"
description = "Analyzes the sentiment of input text"
tags = ["nlp", "sentiment", "text"]

# Build the workflow graph
input_node = TextInput(value="Sample text")
sentiment_node = Sentiment(text=input_node.output)

graph = graph(input_node, sentiment_node)
```
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nodetool.types.api_graph import Graph as ApiGraph

from .export import _literal, _sanitize_ident, _snake_case, _split_type, _topo_order


@dataclass
class WorkflowFile:
    """
    Represents a formalized workflow Python file.

    Attributes:
        name: Human-readable name of the workflow
        description: Brief description of what the workflow does
        docstring: Module-level docstring with background and context
        tags: List of tags for categorization
        graph: The workflow graph created from DSL nodes
        settings: Optional workflow settings
        thumbnail: Optional thumbnail path or URL
        path: Original file path if loaded from file
    """

    name: str = ""
    description: str = ""
    docstring: str = ""
    tags: list[str] = field(default_factory=list)
    graph: ApiGraph | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    thumbnail: str | None = None
    thumbnail_url: str | None = None
    path: str | None = None
    tool_name: str | None = None
    run_mode: str | None = None


def load_workflow_file(file_path: str | Path) -> WorkflowFile:
    """
    Load a workflow from a Python file.

    This function imports the Python module and extracts workflow metadata
    from top-level variables and the module docstring.

    Args:
        file_path: Path to the Python workflow file

    Returns:
        WorkflowFile: The loaded workflow with metadata and graph

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a valid workflow file
        ImportError: If the module cannot be imported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {file_path}")

    if not file_path.suffix == ".py":
        raise ValueError(f"Workflow file must be a Python file: {file_path}")

    # Generate a unique module name to avoid conflicts
    module_name = f"_workflow_file_{file_path.stem}_{id(file_path)}"

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load workflow file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Error executing workflow file: {e}") from e

    # Extract workflow metadata
    workflow = WorkflowFile(path=str(file_path))

    # Get module docstring
    if module.__doc__:
        workflow.docstring = module.__doc__

    # Extract top-level variables
    workflow.name = getattr(module, "name", file_path.stem)
    workflow.description = getattr(module, "description", "")
    workflow.tags = getattr(module, "tags", [])
    workflow.settings = getattr(module, "settings", {})
    workflow.thumbnail = getattr(module, "thumbnail", None)
    workflow.thumbnail_url = getattr(module, "thumbnail_url", None)
    workflow.tool_name = getattr(module, "tool_name", None)
    workflow.run_mode = getattr(module, "run_mode", None)

    # Get the graph
    graph = getattr(module, "graph", None)
    if graph is not None:
        if isinstance(graph, ApiGraph):
            workflow.graph = graph
        else:
            raise ValueError(
                f"'graph' variable must be a Graph object (from nodetool.dsl.graph.graph()), "
                f"got {type(graph).__name__}"
            )

    # Clean up the module from sys.modules
    del sys.modules[module_name]

    return workflow


def _extract_workflow_metadata_static(file_path: str | Path) -> dict[str, Any]:
    """
    Statically extract workflow metadata from a Python file without executing it.

    This uses AST parsing to safely extract metadata without running the code.

    Args:
        file_path: Path to the Python workflow file

    Returns:
        Dictionary with extracted metadata (name, description, docstring, tags, etc.)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {file_path}")

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))

    metadata: dict[str, Any] = {
        "name": file_path.stem,
        "description": "",
        "docstring": "",
        "tags": [],
        "path": str(file_path),
    }

    # Extract module docstring
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        metadata["docstring"] = tree.body[0].value.value

    # Extract top-level variable assignments
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    value = _ast_to_value(node.value)
                    if var_name in ("name", "description", "tool_name", "thumbnail", "thumbnail_url", "run_mode"):
                        if isinstance(value, str):
                            metadata[var_name] = value
                    elif var_name == "tags":
                        if isinstance(value, list):
                            metadata["tags"] = value
                    elif var_name == "settings":
                        if isinstance(value, dict):
                            metadata["settings"] = value

    return metadata


def _ast_to_value(node: ast.expr) -> Any:
    """
    Convert an AST node to its Python value for simple literals.

    Handles strings, numbers, lists, dicts, booleans, and None.
    Returns None for complex expressions.
    """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        return [_ast_to_value(elt) for elt in node.elts]
    elif isinstance(node, ast.Dict):
        keys = [_ast_to_value(k) if k else None for k in node.keys]
        values = [_ast_to_value(v) for v in node.values]
        return dict(zip(keys, values, strict=False))
    elif isinstance(node, ast.Tuple):
        return tuple(_ast_to_value(elt) for elt in node.elts)
    elif isinstance(node, ast.Set):
        return {_ast_to_value(elt) for elt in node.elts}
    elif isinstance(node, ast.Name):
        if node.id == "True":
            return True
        elif node.id == "False":
            return False
        elif node.id == "None":
            return None
    return None


def workflow_file_to_py(
    graph: ApiGraph,
    *,
    name: str = "Untitled Workflow",
    description: str = "",
    docstring: str = "",
    tags: list[str] | None = None,
    settings: dict[str, Any] | None = None,
    thumbnail: str | None = None,
    thumbnail_url: str | None = None,
    tool_name: str | None = None,
    run_mode: str | None = None,
) -> str:
    """
    Generate a Python workflow file from a graph.

    This creates a complete, executable Python file that defines a workflow
    with metadata and a graph variable created from DSL nodes.

    Args:
        graph: The API graph to export
        name: Human-readable name of the workflow
        description: Brief description of the workflow
        docstring: Module-level docstring with background and context
        tags: List of tags for categorization
        settings: Optional workflow settings dictionary
        thumbnail: Optional thumbnail path
        thumbnail_url: Optional thumbnail URL
        tool_name: Optional tool name for the workflow
        run_mode: Optional run mode (e.g., "tool", "trigger")

    Returns:
        str: The generated Python source code
    """
    if not isinstance(graph, ApiGraph):
        raise TypeError("workflow_file_to_py accepts only nodetool.types.api_graph.Graph instances")

    tags = tags or []
    settings = settings or {}

    api_nodes = list(graph.nodes)
    api_edges = list(graph.edges)

    incoming: dict[str, dict[str, Any]] = {}
    for edge in api_edges:
        incoming.setdefault(edge.target, {})[edge.targetHandle] = edge

    order = _topo_order(api_nodes, api_edges)
    node_by_id = {node.id: node for node in api_nodes}

    var_names: dict[str, str] = {}
    counters: dict[str, int] = {}
    module_to_classes: dict[str, list[str]] = {}

    def declare_for_node(node: Any) -> None:
        _namespace, cls_name = _split_type(node.type)
        base = _sanitize_ident(_snake_case(cls_name)) or "node"
        counters.setdefault(base, 0)
        counters[base] += 1
        suffix = str(counters[base])
        var = f"{base}_{suffix}"
        var_names[node.id] = var
        module = f"nodetool.dsl.{_namespace}" if _namespace else "nodetool.dsl"
        module_to_classes.setdefault(module, [])
        if cls_name not in module_to_classes[module]:
            module_to_classes[module].append(cls_name)

    for node_id in order:
        declare_for_node(node_by_id[node_id])

    lines: list[str] = []

    # Add module docstring
    if docstring:
        # Format docstring with proper triple quotes
        docstring_lines = docstring.strip().split("\n")
        lines.append('"""')
        for line in docstring_lines:
            lines.append(line)
        lines.append('"""')
        lines.append("")

    # Add imports
    lines.append("from nodetool.dsl.graph import graph")
    lines.append("")

    for module in sorted(module_to_classes.keys()):
        classes = ", ".join(sorted(module_to_classes[module]))
        lines.append(f"from {module} import {classes}")

    if module_to_classes:
        lines.append("")

    # Add workflow metadata as top-level variables
    lines.append("# Workflow metadata")
    lines.append(f"name = {_literal(name)}")
    lines.append(f"description = {_literal(description)}")

    if tags:
        lines.append(f"tags = {_literal(tags)}")

    if tool_name:
        lines.append(f"tool_name = {_literal(tool_name)}")

    if run_mode:
        lines.append(f"run_mode = {_literal(run_mode)}")

    if thumbnail:
        lines.append(f"thumbnail = {_literal(thumbnail)}")

    if thumbnail_url:
        lines.append(f"thumbnail_url = {_literal(thumbnail_url)}")

    if settings:
        lines.append(f"settings = {_literal(settings)}")

    lines.append("")

    # Add node definitions
    lines.append("# Build the workflow graph")

    for node_id in order:
        node = node_by_id[node_id]
        var = var_names[node_id]
        _namespace, cls_name = _split_type(node.type)

        incoming_edges = incoming.get(node_id, {})
        used_keys = set(incoming_edges.keys())
        kwargs: list[str] = []

        sync_mode = node.sync_mode
        if sync_mode and sync_mode != "on_any":
            kwargs.append(f"sync_mode={_literal(sync_mode)}")

        if node.dynamic_outputs:
            from .export import _dynamic_outputs_literal

            dyn = _dynamic_outputs_literal(dict(node.dynamic_outputs))
            kwargs.append(f"dynamic_outputs={dyn}")

        data_attr = node.data
        data = data_attr if isinstance(data_attr, dict) else {}
        for key, value in data.items():
            if key in used_keys:
                continue
            from .export import _map_api_key_to_dsl_arg

            kwargs.append(f"{_sanitize_ident(_map_api_key_to_dsl_arg(key))}={_literal(value)}")

        dyn_props_attr = node.dynamic_properties or {}
        dyn_props = dyn_props_attr if isinstance(dyn_props_attr, dict) else {}
        for key, value in dyn_props.items():
            if key in used_keys:
                continue
            from .export import _map_api_key_to_dsl_arg

            kwargs.append(f"{_sanitize_ident(_map_api_key_to_dsl_arg(key))}={_literal(value)}")

        for handle, edge in sorted(incoming_edges.items()):
            src_var = var_names.get(edge.source)
            if src_var is None:
                continue
            if edge.sourceHandle == "output":
                src_expr = f"{src_var}.output"
            else:
                src_expr = f"{src_var}.out[{_literal(edge.sourceHandle)}]"
            from .export import _map_api_key_to_dsl_arg

            kwargs.append(f"{_sanitize_ident(_map_api_key_to_dsl_arg(handle))}={src_expr}")

        joined = ", ".join(kwargs)
        lines.append(f"{var} = {cls_name}({joined})")

    lines.append("")

    # Add the graph variable
    all_vars = ", ".join(var_names[node_id] for node_id in order)
    lines.append(f"graph = graph({all_vars})")
    lines.append("")

    return "\n".join(lines)


def workflow_to_workflow_file(workflow: WorkflowFile | Any) -> str:
    """
    Convert a WorkflowFile or compatible workflow object to Python source code.

    Args:
        workflow: A WorkflowFile instance or any object with compatible attributes
                  (graph, name, description, docstring, tags, settings, etc.)

    Returns:
        str: The generated Python source code
    """
    if isinstance(workflow, WorkflowFile):
        if workflow.graph is None:
            raise ValueError("WorkflowFile must have a graph")
        return workflow_file_to_py(
            workflow.graph,
            name=workflow.name,
            description=workflow.description,
            docstring=workflow.docstring,
            tags=workflow.tags,
            settings=workflow.settings,
            thumbnail=workflow.thumbnail,
            thumbnail_url=workflow.thumbnail_url,
            tool_name=workflow.tool_name,
            run_mode=workflow.run_mode,
        )
    else:
        # Try to extract compatible attributes from the object
        graph = getattr(workflow, "graph", None)
        if graph is None:
            raise ValueError("Workflow object must have a 'graph' attribute")

        # Convert dict graph to ApiGraph if needed
        if isinstance(graph, dict):
            graph = ApiGraph(
                nodes=graph.get("nodes", []),
                edges=graph.get("edges", []),
            )

        return workflow_file_to_py(
            graph,
            name=getattr(workflow, "name", "Untitled Workflow"),
            description=getattr(workflow, "description", ""),
            docstring=getattr(workflow, "docstring", ""),
            tags=getattr(workflow, "tags", []) or [],
            settings=getattr(workflow, "settings", {}) or {},
            thumbnail=getattr(workflow, "thumbnail", None),
            thumbnail_url=getattr(workflow, "thumbnail_url", None),
            tool_name=getattr(workflow, "tool_name", None),
            run_mode=getattr(workflow, "run_mode", None),
        )


__all__ = [
    "WorkflowFile",
    "load_workflow_file",
    "workflow_file_to_py",
    "workflow_to_workflow_file",
]
