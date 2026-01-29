from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeAlias

from pydantic import BaseModel

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.types.api_graph import Edge as ApiEdge
from nodetool.types.api_graph import Graph as ApiGraph
from nodetool.types.api_graph import Node as ApiNode

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
    indeg: dict[str, int] = dict.fromkeys(node_ids, 0)
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
                if indeg[edge.target] == 0 and edge.target not in order and edge.target not in queue:
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
    if isinstance(value, bool | int | float):
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


def _map_api_key_to_dsl_arg(key: str) -> str:
    """
    Map API graph property/handle names to DSL constructor argument names.

    The API graph schema historically used "content" for some node inputs where
    the Python DSL uses the canonical "instructions" field name.
    """
    return "instructions" if key == "content" else key


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

    - Accepts an API graph (``nodetool.types.api_graph.Graph``).
    - Produces Python code using classes under ``nodetool.dsl.<namespace>`` and
      the ``graph(...)`` constructor from ``nodetool.dsl.graph``.

    Returns:
        str: A Python source string.
    """
    if not isinstance(graph, ApiGraph):
        raise TypeError("graph_to_dsl_py accepts only nodetool.types.api_graph.Graph instances")

    api_nodes = list(graph.nodes)
    api_edges = list(graph.edges)

    incoming: dict[str, dict[str, ApiEdge]] = {}
    for edge in api_edges:
        incoming.setdefault(edge.target, {})[edge.targetHandle] = edge

    order = _topo_order(api_nodes, api_edges)
    node_by_id = {node.id: node for node in api_nodes}

    var_names: dict[str, str] = {}
    counters: dict[str, int] = {}
    module_to_classes: dict[str, list[str]] = {}

    def declare_for_node(node: ApiNode) -> None:
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
            kwargs.append(f"{_sanitize_ident(_map_api_key_to_dsl_arg(key))}={_literal(value)}")

        dyn_props_attr = node.dynamic_properties or {}
        dyn_props = dyn_props_attr if isinstance(dyn_props_attr, Mapping) else {}
        for key, value in dyn_props.items():
            if key in used_keys:
                continue
            kwargs.append(f"{_sanitize_ident(_map_api_key_to_dsl_arg(key))}={_literal(value)}")

        for handle, edge in sorted(incoming_edges.items()):
            src_var = var_names.get(edge.source)
            if src_var is None:
                continue
            if edge.sourceHandle == "output":
                src_expr = f"{src_var}.output"
            else:
                src_expr = f"{src_var}.out[{_literal(edge.sourceHandle)}]"
            kwargs.append(f"{_sanitize_ident(_map_api_key_to_dsl_arg(handle))}={src_expr}")

        joined = ", ".join(kwargs)
        module = f"nodetool.dsl.{namespace}" if namespace else "nodetool.dsl"
        lines.append(f"{var} = {cls_name}({joined})")

    lines.append("")
    all_vars = ", ".join(var_names[node_id] for node_id in order)
    lines.append(f"workflow = graph({all_vars})")
    lines.append("")

    return "\n".join(lines)


def graph_to_gradio_py(
    graph: ApiGraph,
    *,
    app_title: str = "NodeTool Workflow",
    theme: str | None = None,
    description: str | None = None,
    allow_flagging: bool = False,
    queue: bool = True,
) -> str:
    """
    Generate Python source that reconstructs the graph (using the DSL)
    *and* wraps it in a minimal Gradio app capable of running the workflow.

    The generated script includes:
        - The DSL build-out produced by :func:`graph_to_dsl_py`.
        - A runtime shim that exposes heuristic inputs/outputs via Gradio and
          executes the workflow with ``WorkflowRunner`` + ``ProcessingContext``.

    Heuristics:
        - Inputs: zero in-degree nodes, ``*Input`` types, or types containing
          ``.input.``; free (non-wired) properties surface as widgets, preferring
          a ``value`` property when present.
        - Outputs: zero out-degree nodes, ``*Output`` types, or types containing
          ``.output.``; values are read from ``.value`` (or ``.output``).

    Returns:
        str: A standalone Python script. Save it (e.g., to ``app.py``)
             and start the UI with ``python app.py``.
    """
    if not isinstance(graph, ApiGraph):
        raise TypeError("graph_to_gradio_py accepts only nodetool.types.api_graph.Graph instances")

    api_nodes = list(graph.nodes)
    api_edges = list(graph.edges)

    incoming: dict[str, dict[str, ApiEdge]] = {}
    outgoing_count: dict[str, int] = {node.id: 0 for node in api_nodes}
    for edge in api_edges:
        incoming.setdefault(edge.target, {})[edge.targetHandle] = edge
        if edge.source in outgoing_count:
            outgoing_count[edge.source] += 1

    order = _topo_order(api_nodes, api_edges)
    node_by_id = {node.id: node for node in api_nodes}

    var_names: dict[str, str] = {}
    counters: dict[str, int] = {}

    def declare_for_node(node: ApiNode) -> None:
        _namespace, cls_name = _split_type(node.type)
        base = _sanitize_ident(_snake_case(cls_name)) or "node"
        counters.setdefault(base, 0)
        counters[base] += 1
        suffix = str(counters[base])
        var = f"{base}_{suffix}"
        var_names[node.id] = var

    for node_id in order:
        declare_for_node(node_by_id[node_id])

    def is_input_like(node: ApiNode) -> bool:
        _, cls_name = _split_type(node.type)
        return node.id not in incoming or cls_name.endswith("Input") or ".input." in node.type

    def is_output_like(node: ApiNode) -> bool:
        _, cls_name = _split_type(node.type)
        return outgoing_count.get(node.id, 0) == 0 or cls_name.endswith("Output") or ".output." in node.type

    def free_fields(node: ApiNode) -> list[tuple[str, object]]:
        """
        Return non-wired properties for a node (prefer a single 'value' field).
        """
        fields: list[tuple[str, object]] = []
        used = set((incoming.get(node.id) or {}).keys())
        data_attr = node.data if isinstance(node.data, Mapping) else {}
        dyn_attr = node.dynamic_properties if isinstance(node.dynamic_properties, Mapping) else {}

        if "value" in data_attr and "value" not in used:
            return [("value", data_attr.get("value"))]

        for key in sorted(data_attr.keys()):
            if key not in used:
                fields.append((key, data_attr[key]))
        for key in sorted(dyn_attr.keys()):
            if key not in used:
                fields.append((key, dyn_attr[key]))
        return fields

    def guess_component_kind(node_type: str, field_name: str, default: object) -> str:
        """
        Return a canonical widget kind: 'text', 'number', 'checkbox',
        'image', 'audio', 'video', 'file', 'json', 'dataframe'.
        """
        _, cls_name = _split_type(node_type)
        cls_lower = cls_name.lower()
        field_lower = field_name.lower()
        if any(key in cls_lower for key in ["imageinput", "imageoutput", "imageref", "image"]) or field_lower in {
            "image",
            "img",
        }:
            return "image"
        if (
            any(key in cls_lower for key in ["audioinput", "audiooutput", "audioref", "audio"])
            or field_lower == "audio"
        ):
            return "audio"
        if (
            any(key in cls_lower for key in ["videoinput", "videooutput", "videoref", "video"])
            or field_lower == "video"
        ):
            return "video"
        if any(key in cls_lower for key in ["dataframe", "table", "csv"]):
            return "dataframe"
        if any(key in cls_lower for key in ["document", "file", "filepath"]) or field_lower in {
            "path",
            "file",
            "filename",
        }:
            return "file"
        if isinstance(default, bool) or "booleaninput" in cls_lower or "booleanoutput" in cls_lower:
            return "checkbox"
        if (
            isinstance(default, int | float)
            or "integerinput" in cls_lower
            or "floatinput" in cls_lower
            or field_lower in {"number", "count"}
        ):
            return "number"
        if isinstance(default, list | dict):
            return "json"
        return "text"

    input_specs: list[dict[str, object]] = []
    output_specs: list[dict[str, object]] = []

    for node_id in order:
        node = node_by_id[node_id]
        if is_input_like(node):
            for field, default in free_fields(node):
                input_specs.append(
                    {
                        "var": var_names[node_id],
                        "field": field,
                        "label": f"{node.type.split('.')[-1]}:{field}",
                        "kind": guess_component_kind(node.type, field, default),
                        "default": default,
                    }
                )
        if is_output_like(node):
            output_specs.append(
                {
                    "var": var_names[node_id],
                    "label": node.type.split(".")[-1],
                    "kind": guess_component_kind(node.type, "value", None),
                }
            )

    dsl_src = graph_to_dsl_py(graph)

    def input_spec_literal(spec: dict[str, object]) -> str:
        return (
            "InputSpec("
            f"var={_literal(spec['var'])}, "
            f"field={_literal(spec['field'])}, "
            f"label={_literal(spec['label'])}, "
            f"kind={_literal(spec['kind'])}, "
            f"default={_literal(spec.get('default'))}"
            ")"
        )

    def output_spec_literal(spec: dict[str, object]) -> str:
        return (
            f"OutputSpec(var={_literal(spec['var'])}, label={_literal(spec['label'])}, kind={_literal(spec['kind'])})"
        )

    input_specs_py = "[" + ", ".join(input_spec_literal(spec) for spec in input_specs) + "]" if input_specs else "[]"
    output_specs_py = (
        "[" + ", ".join(output_spec_literal(spec) for spec in output_specs) + "]" if output_specs else "[]"
    )

    lines: list[str] = []
    lines.append("# This file was generated from an API graph using nodetool.dsl.export.graph_to_gradio_py")
    lines.append("# It reconstructs the workflow and exposes it as a Gradio app.")
    lines.append("")
    lines.append(dsl_src.rstrip())
    lines.append("")
    lines.append("from nodetool.dsl.gradio_app import GradioAppConfig, InputSpec, OutputSpec, create_gradio_app")
    lines.append("")
    lines.append(f"INPUT_SPECS = {input_specs_py}")
    lines.append(f"OUTPUT_SPECS = {output_specs_py}")
    lines.append("")
    lines.append(
        "APP_CONFIG = GradioAppConfig("
        f"title={_literal(app_title)}, "
        f"theme={_literal(theme)}, "
        f"description={_literal(description)}, "
        f"allow_flagging={_literal(allow_flagging)}, "
        f"queue={_literal(queue)}"
        ")"
    )
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    application = create_gradio_app(")
    lines.append("        workflow=workflow,")
    lines.append("        input_specs=INPUT_SPECS,")
    lines.append("        output_specs=OUTPUT_SPECS,")
    lines.append("        namespace=globals(),")
    lines.append("        config=APP_CONFIG,")
    lines.append("    )")
    lines.append("    application.launch(allow_flagging=APP_CONFIG.allow_flagging)")
    lines.append("")
    return "\n".join(lines)


__all__ = ["graph_to_dsl_py", "graph_to_gradio_py"]
