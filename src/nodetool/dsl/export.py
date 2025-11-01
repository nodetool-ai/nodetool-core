from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeAlias

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.types.graph import Edge as ApiEdge, Graph as ApiGraph, Node as ApiNode
from pydantic import BaseModel

DynamicOutputLike: TypeAlias = TypeMetadata | Mapping[str, object]
DynamicOutputsMapping: TypeAlias = Mapping[str, DynamicOutputLike]


def _snake_case(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (not name[i - 1].isupper()):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _sanitize_ident(s: str) -> str:
    s = s.strip()
    if not s:
        return "node"
    ident: list[str] = []
    first = s[0]
    ident.append(first if (first.isalpha() or first == "_") else "_")
    for ch in s[1:]:
        ident.append(ch if (ch.isalnum() or ch == "_") else "_")
    return "".join(ident)


def _topo_order(nodes: Iterable[ApiNode], edges: Iterable[ApiEdge]) -> list[str]:
    """
    Return a topological ordering of node ids. In case of cycles, returns
    a best-effort order where remaining nodes are appended at the end.
    """
    node_ids = [n.id for n in nodes]
    indeg: dict[str, int] = {i: 0 for i in node_ids}
    for edge in edges:
        if edge.source in indeg and edge.target in indeg:
            indeg[edge.target] += 1

    queue: list[str] = [i for i, degree in indeg.items() if degree == 0]
    order: list[str] = []
    queue.sort(key=node_ids.index)
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for edge in edges:
            if edge.source == nid and edge.target in indeg:
                indeg[edge.target] -= 1
                if (
                    indeg[edge.target] == 0
                    and edge.target not in order
                    and edge.target not in queue
                ):
                    queue.append(edge.target)

    if len(order) != len(node_ids):
        remaining = [i for i in node_ids if i not in order]
        order.extend(remaining)
    return order


def _split_type(node_type: str) -> tuple[str, str]:
    """
    Split a node type into (namespace, class_name).
    Example: "apple.notes.CreateNote" -> ("apple.notes", "CreateNote").
    """
    parts = node_type.split(".")
    if len(parts) == 1:
        return "", parts[0]
    return ".".join(parts[:-1]), parts[-1]


def _literal(value: object) -> str:
    """
    Render a Python literal for common JSON-serializable values.
    Fallback to repr() for unknown objects.
    """
    if value is None:
        return "None"
    if isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(_literal(v) for v in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(value, dict):
        items = ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items())
        return "{" + items + "}"
    return repr(value)


def _dynamic_outputs_literal(dynamic_outputs: DynamicOutputsMapping) -> str:
    """
    Represent dynamic outputs as a compact Python literal.

    - Accepts TypeMetadata objects or dicts of the same shape.
    - Emits a minimal dict containing only "type" and, if present, a reduced
      "type_args" list (recursively compacted).
    """

    def compact_tm(meta: object) -> dict[str, object]:
        if isinstance(meta, BaseModel):
            raw = meta.model_dump()
        elif isinstance(meta, Mapping):
            raw = dict(meta)
        else:
            return {"type": str(meta)}

        base: dict[str, object] = {"type": raw.get("type", "any")}
        args = raw.get("type_args") or []
        if isinstance(args, list) and args:
            base["type_args"] = [compact_tm(arg) for arg in args]
        return base

    rendered: list[str] = []
    for name, meta in dynamic_outputs.items():
        rendered.append(f"{_literal(name)}: {_literal(compact_tm(meta))}")
    return "{" + ", ".join(rendered) + "}"


def graph_to_dsl_py(graph: ApiGraph) -> str:
    """
    Generate Python DSL code that reconstructs the provided graph.

    - Accepts an API graph (``nodetool.types.graph.Graph``).
    - Produces Python code using classes under ``nodetool.dsl.<namespace>`` and
      the ``graph(...)`` constructor from ``nodetool.dsl.graph``.

    Returns:
        str: A Python source string.
    """
    if not isinstance(graph, ApiGraph):
        raise TypeError("graph_to_dsl_py accepts only nodetool.types.graph.Graph instances")

    api_nodes = list[ApiNode](graph.nodes)
    api_edges = list[ApiEdge](graph.edges)

    incoming: dict[str, dict[str, ApiEdge]] = {}
    for edge in api_edges:
        incoming.setdefault(edge.target, {})[edge.targetHandle] = edge

    order = _topo_order(api_nodes, api_edges)
    node_by_id = {node.id: node for node in api_nodes}

    var_names: dict[str, str] = {}
    counters: dict[str, int] = {}
    module_to_classes: dict[str, list[str]] = {}

    def declare_for_node(node: ApiNode) -> None:
        namespace, cls_name = _split_type(node.type)
        module = f"nodetool.dsl.{namespace}" if namespace else "nodetool.dsl"
        base = _sanitize_ident(_snake_case(cls_name)) or "node"
        counters.setdefault(base, 0)
        counters[base] += 1
        suffix = str(counters[base])
        var = f"{base}_{suffix}"
        var_names[node.id] = var
        module_to_classes.setdefault(module, [])
        if cls_name not in module_to_classes[module]:
            module_to_classes[module].append(cls_name)

    for node_id in order:
        declare_for_node(node_by_id[node_id])

    lines: list[str] = []
    lines.append("# This file was generated from an API graph using nodetool.dsl.export.graph_to_dsl_py")
    lines.append("from nodetool.dsl.graph import graph\n")

    for module in sorted(module_to_classes.keys()):
        classes = ", ".join(sorted(module_to_classes[module]))
        lines.append(f"from {module} import {classes}")

    if module_to_classes:
        lines.append("")

    for node_id in order:
        node = node_by_id[node_id]
        var = var_names[node_id]
        namespace, cls_name = _split_type(node.type)

        incoming_edges = incoming.get(node_id, {})
        used_keys = set(incoming_edges.keys())
        kwargs: list[str] = []

        sync_mode = node.sync_mode
        if sync_mode and sync_mode != "on_any":
            kwargs.append(f"sync_mode={_literal(sync_mode)}")

        if node.dynamic_outputs:
            dyn = _dynamic_outputs_literal(dict(node.dynamic_outputs))
            kwargs.append(f"dynamic_outputs={dyn}")

        data_attr = node.data
        data = data_attr if isinstance(data_attr, Mapping) else {}
        for key, value in data.items():
            if key in used_keys:
                continue
            kwargs.append(f"{_sanitize_ident(key)}={_literal(value)}")

        dyn_props_attr = node.dynamic_properties or {}
        dyn_props = dyn_props_attr if isinstance(dyn_props_attr, Mapping) else {}
        for key, value in dyn_props.items():
            if key in used_keys:
                continue
            kwargs.append(f"{_sanitize_ident(key)}={_literal(value)}")

        for handle, edge in sorted(incoming_edges.items()):
            src_var = var_names.get(edge.source)
            if src_var is None:
                continue
            if edge.sourceHandle == "output":
                src_expr = f"{src_var}.out"
            else:
                src_expr = f"{src_var}.out[{_literal(edge.sourceHandle)}]"
            kwargs.append(f"{_sanitize_ident(handle)}={src_expr}")

        joined = ", ".join(kwargs)
        module = f"nodetool.dsl.{namespace}" if namespace else "nodetool.dsl"
        lines.append(f"{var} = {cls_name}({joined})")

    lines.append("")
    all_vars = ", ".join(var_names[node_id] for node_id in order)
    lines.append(f"workflow = graph({all_vars})")
    lines.append("")

    return "\n".join(lines)


__all__ = ["graph_to_dsl_py"]
