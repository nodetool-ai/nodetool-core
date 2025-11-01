from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

try:
    # Prefer the API graph model (serialized form)
    from nodetool.types.graph import Graph as ApiGraph, Node as ApiNode, Edge as ApiEdge
except Exception:  # pragma: no cover - import guard for isolated usage
    ApiGraph = Any  # type: ignore
    ApiNode = Any  # type: ignore
    ApiEdge = Any  # type: ignore

try:
    # Also support the in-memory workflow graph (BaseNode instances)
    from nodetool.workflows.graph import Graph as WorkflowGraph
    from nodetool.workflows.base_node import BaseNode
except Exception:  # pragma: no cover - import guard for isolated usage
    WorkflowGraph = Any  # type: ignore
    BaseNode = Any  # type: ignore


def _snake_case(name: str) -> str:
    out: List[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (not name[i - 1].isupper()):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _sanitize_ident(s: str) -> str:
    s = s.strip()
    if not s:
        return "node"
    ident = []
    first = s[0]
    ident.append(first if (first.isalpha() or first == "_") else "_")
    for ch in s[1:]:
        ident.append(ch if (ch.isalnum() or ch == "_") else "_")
    return "".join(ident)


def _to_api_graph(graph: ApiGraph | WorkflowGraph) -> Tuple[List[ApiNode], List[ApiEdge]]:
    """
    Normalize input into API graph components (nodes, edges).
    Accepts both the serialized API graph and the in-memory workflow graph.
    """
    # If already an API graph
    if hasattr(graph, "nodes") and graph.__class__.__module__.endswith("types.graph"):
        return list(graph.nodes), list(graph.edges)

    # If it's a workflow graph (nodes are BaseNode instances)
    if hasattr(graph, "nodes") and hasattr(graph, "edges"):
        api_nodes: List[ApiNode] = []
        for n in graph.nodes:  # type: ignore[attr-defined]
            # BaseNode.to_dict() returns the shape expected by ApiNode
            # Include sync_mode if available (falls back to default in ApiNode otherwise)
            payload: Dict[str, Any] = n.to_dict()  # type: ignore[call-arg]
            api_nodes.append(ApiNode(**payload))
        return api_nodes, list(graph.edges)  # type: ignore[attr-defined]

    raise TypeError("Unsupported graph type for DSL export")


def _topo_order(nodes: Iterable[ApiNode], edges: Iterable[ApiEdge]) -> List[str]:
    """
    Return a topological ordering of node ids. In case of cycles, returns
    a best-effort order where remaining nodes are appended at the end.
    """
    node_ids = [n.id for n in nodes]
    indeg: Dict[str, int] = {i: 0 for i in node_ids}
    for e in edges:
        if e.source in indeg and e.target in indeg:
            indeg[e.target] += 1

    queue: List[str] = [i for i, d in indeg.items() if d == 0]
    order: List[str] = []
    # Deterministic order among zeros: preserve appearance in nodes list
    queue.sort(key=lambda x: node_ids.index(x))
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for e in edges:
            if e.source == nid and e.target in indeg:
                indeg[e.target] -= 1
                if indeg[e.target] == 0 and e.target not in order and e.target not in queue:
                    queue.append(e.target)

    if len(order) != len(node_ids):
        # Append any remaining nodes to preserve completeness
        remaining = [i for i in node_ids if i not in order]
        order.extend(remaining)
    return order


def _split_type(node_type: str) -> Tuple[str, str]:
    """
    Split a node type into (namespace, class_name).
    Example: "apple.notes.CreateNote" -> ("apple.notes", "CreateNote").
    """
    parts = node_type.split(".")
    if len(parts) == 1:
        return "", parts[0]
    return ".".join(parts[:-1]), parts[-1]


def _literal(value: Any) -> str:
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
    # Fallback
    return repr(value)


def _dynamic_outputs_literal(dynamic_outputs: Dict[str, Any]) -> str:
    """
    Represent dynamic outputs as a compact Python literal.

    - Accepts TypeMetadata objects or dicts of the same shape.
    - Emits a minimal dict containing only "type" and, if present, a reduced
      "type_args" list (recursively compacted).
    """

    def compact_tm(meta: Any) -> Dict[str, Any]:
        if hasattr(meta, "model_dump"):
            raw = meta.model_dump()
        elif isinstance(meta, dict):
            raw = dict(meta)
        else:
            try:
                return {"type": str(meta)}
            except Exception:
                return {"type": "any"}

        base: Dict[str, Any] = {"type": raw.get("type", "any")}
        args = raw.get("type_args") or []
        if isinstance(args, list) and args:
            base["type_args"] = [compact_tm(a) for a in args]
        return base

    rendered: List[str] = []
    for name, meta in dynamic_outputs.items():
        rendered.append(f"{_literal(name)}: {_literal(compact_tm(meta))}")
    return "{" + ", ".join(rendered) + "}"


def graph_to_dsl_py(graph: ApiGraph | WorkflowGraph) -> str:
    """
    Generate Python DSL code that reconstructs the provided graph.

    - Accepts either an API graph (``nodetool.types.graph.Graph``) or a
      workflow graph (``nodetool.workflows.graph.Graph``).
    - Produces Python code using classes under ``nodetool.dsl.<namespace>`` and
      the ``graph(...)`` constructor from ``nodetool.dsl.graph``.

    Returns:
        str: A Python source string.
    """
    api_nodes, api_edges = _to_api_graph(graph)

    # Build incoming edge map: target_node_id -> {handle: edge}
    incoming: Dict[str, Dict[str, ApiEdge]] = {}
    for e in api_edges:
        incoming.setdefault(e.target, {})[e.targetHandle] = e

    # Determine module imports and stable variable names
    order = _topo_order(api_nodes, api_edges)
    node_by_id = {n.id: n for n in api_nodes}

    # variable names by node id
    var_names: Dict[str, str] = {}
    counters: Dict[str, int] = {}
    module_to_classes: Dict[str, List[str]] = {}

    def declare_for_node(n: ApiNode) -> Tuple[str, str, str]:
        namespace, cls_name = _split_type(n.type)
        module = f"nodetool.dsl.{namespace}" if namespace else "nodetool.dsl"
        base = _sanitize_ident(_snake_case(cls_name)) or "node"
        counters.setdefault(base, 0)
        counters[base] += 1
        suffix = str(counters[base])
        var = f"{base}_{suffix}"
        var_names[n.id] = var
        module_to_classes.setdefault(module, [])
        if cls_name not in module_to_classes[module]:
            module_to_classes[module].append(cls_name)
        return var, module, cls_name

    for nid in order:
        declare_for_node(node_by_id[nid])

    # Compose source
    lines: List[str] = []
    lines.append("# This file was generated from a workflow graph using nodetool.dsl.export.graph_to_dsl_py")
    lines.append("from nodetool.dsl.graph import graph\n")

    # Grouped imports by module
    for module in sorted(module_to_classes.keys()):
        classes = ", ".join(sorted(module_to_classes[module]))
        lines.append(f"from {module} import {classes}")

    if module_to_classes:
        lines.append("")

    # Node instantiations in topological order
    for nid in order:
        node = node_by_id[nid]
        var = var_names[nid]
        namespace, cls_name = _split_type(node.type)

        # Build kwargs: include unconnected properties and dynamic properties
        used_keys = set((incoming.get(nid) or {}).keys())
        kwargs: List[str] = []

        # sync mode
        try:
            sync_mode = getattr(node, "sync_mode", None)
        except Exception:  # pragma: no cover - defensive
            sync_mode = None
        if sync_mode and sync_mode != "on_any":
            kwargs.append(f"sync_mode={_literal(sync_mode)}")

        # dynamic outputs
        if getattr(node, "dynamic_outputs", None):
            dyn = _dynamic_outputs_literal(dict(node.dynamic_outputs))
            kwargs.append(f"dynamic_outputs={dyn}")

        # regular data properties (skip connected ones)
        data = getattr(node, "data", {}) or {}
        for key, value in data.items():
            if key in used_keys:
                continue
            kwargs.append(f"{_sanitize_ident(key)}={_literal(value)}")

        # dynamic properties (skip connected ones)
        dyn_props = getattr(node, "dynamic_properties", {}) or {}
        for key, value in dyn_props.items():
            if key in used_keys:
                continue
            kwargs.append(f"{_sanitize_ident(key)}={_literal(value)}")

        # Connected inputs from incoming edges
        for handle, e in sorted((incoming.get(nid) or {}).items()):
            src_var = var_names.get(e.source)
            if src_var is None:
                continue
            if e.sourceHandle == "output":
                src_expr = f"{src_var}.out"
            else:
                src_expr = f"{src_var}.out[{_literal(e.sourceHandle)}]"
            kwargs.append(f"{_sanitize_ident(handle)}={src_expr}")

        joined = ", ".join(kwargs)
        module = f"nodetool.dsl.{namespace}" if namespace else "nodetool.dsl"
        lines.append(f"{var} = {cls_name}({joined})")

    lines.append("")
    # Assemble final graph
    all_vars = ", ".join(var_names[nid] for nid in order)
    lines.append(f"workflow = graph({all_vars})")
    lines.append("")

    return "\n".join(lines)


__all__ = ["graph_to_dsl_py"]
