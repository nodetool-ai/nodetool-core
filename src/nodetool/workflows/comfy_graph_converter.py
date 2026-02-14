"""
Convert a NodeTool API Graph (data-driven Node/Edge models) into a ComfyUI
prompt JSON object, and optionally convert back from a Comfy prompt to a
NodeTool Graph.

This module works entirely with the lightweight ``nodetool.types.api_graph``
data models (``Node``, ``Edge``, ``Graph``) and does **not** depend on the
runtime ``BaseNode`` class hierarchy, making it suitable for backend prompt
submission without needing to instantiate the full node registry.
"""

from __future__ import annotations

import logging
from typing import Any

from nodetool.types.api_graph import Edge, Graph, Node

log = logging.getLogger(__name__)

# Keys in ``Node.data`` that carry internal metadata and should *not* be
# forwarded as ComfyUI input values.
_INTERNAL_DATA_KEYS = frozenset({"_comfy_metadata"})


def _is_comfy_node(node: Node) -> bool:
    """Return True if *node* represents a ComfyUI node."""
    return node.type.startswith("comfy.")


def _class_type(node: Node) -> str:
    """Derive the ComfyUI ``class_type`` from a node's type string."""
    return node.type.removeprefix("comfy.")


def _resolve_output_index(
    source_node: Node,
    source_handle: str,
) -> int:
    """Resolve the numeric output slot index for *source_handle*.

    Resolution order:
    1. If *source_handle* looks like ``output_N``, use ``N``.
    2. Look up the name in ``_comfy_metadata.outputs`` on the source node.
    3. Fallback to ``0``.
    """
    # 1. Index-style handle
    if source_handle.startswith("output_"):
        try:
            return int(source_handle.removeprefix("output_"))
        except ValueError:
            pass

    # 2. Metadata-based lookup
    meta = source_node.data.get("_comfy_metadata", {})
    outputs = meta.get("outputs", [])
    for idx, out in enumerate(outputs):
        if isinstance(out, dict) and out.get("name") == source_handle:
            return idx

    # 3. Fallback
    return 0


def _resolve_input_name(
    target_node: Node,
    target_handle: str,
) -> str:
    """Resolve the semantic input name for *target_handle*.

    Resolution order:
    1. If the handle is *not* in ``input_N`` form, use it as-is (semantic name).
    2. If it matches ``input_N``, look up the name from ``_comfy_metadata.inputs[N]``.
    3. Fallback to the raw handle string.
    """
    if not target_handle.startswith("input_"):
        return target_handle

    try:
        idx = int(target_handle.removeprefix("input_"))
    except ValueError:
        return target_handle

    meta = target_node.data.get("_comfy_metadata", {})
    inputs = meta.get("inputs", [])
    if 0 <= idx < len(inputs):
        entry = inputs[idx]
        if isinstance(entry, dict) and "name" in entry:
            return entry["name"]

    return target_handle


# ---------------------------------------------------------------------------
# Graph ➜ Prompt
# ---------------------------------------------------------------------------


def graph_to_prompt(graph: Graph) -> dict[str, Any]:
    """Convert a NodeTool ``Graph`` to a ComfyUI prompt dict.

    Only nodes whose ``type`` starts with ``comfy.`` are included.  Edges
    are translated to ComfyUI connection tuples
    ``[source_node_id, output_slot_index]``.

    Returns:
        A dict keyed by node id, each value being
        ``{"class_type": str, "inputs": {...}}``.
    """
    nodes_by_id: dict[str, Node] = {n.id: n for n in graph.nodes}
    comfy_nodes = [n for n in graph.nodes if _is_comfy_node(n)]

    prompt: dict[str, Any] = {}
    for node in comfy_nodes:
        # Collect plain input values from data, skipping internal keys
        inputs: dict[str, Any] = {}
        for key, value in (node.data or {}).items():
            if key in _INTERNAL_DATA_KEYS:
                continue
            inputs[key] = value

        prompt[node.id] = {
            "class_type": _class_type(node),
            "inputs": inputs,
        }

    # Apply edge connections (overwrite input values for connected slots)
    for edge in graph.edges:
        if edge.target not in prompt:
            continue  # target not a comfy node

        source_node = nodes_by_id.get(edge.source)
        if source_node is None:
            log.warning(
                "Edge references unknown source node %s; skipping",
                edge.source,
            )
            continue

        target_node = nodes_by_id[edge.target]
        input_name = _resolve_input_name(target_node, edge.targetHandle)
        output_index = _resolve_output_index(source_node, edge.sourceHandle)

        prompt[edge.target]["inputs"][input_name] = [edge.source, output_index]

    log.info(
        "Converted graph to Comfy prompt: %d comfy nodes, %d edges, prompt keys=%s",
        len(comfy_nodes),
        len(graph.edges),
        list(prompt.keys()),
    )
    return prompt


def has_comfy_nodes(graph: Graph) -> bool:
    """Return True if *graph* contains any node with ``type`` starting ``comfy.``."""
    return any(_is_comfy_node(n) for n in graph.nodes)


# ---------------------------------------------------------------------------
# Prompt ➜ Graph  (reverse / import direction)
# ---------------------------------------------------------------------------


def prompt_to_graph(prompt: dict[str, Any]) -> Graph:
    """Convert a raw ComfyUI prompt dict back into a NodeTool ``Graph``.

    This is the reverse of :func:`graph_to_prompt` and is useful for backend
    import endpoints.

    * Node types are prefixed with ``comfy.``.
    * Connection tuples become edges with ``output_N`` / ``input_N`` handles
      (or semantic names when metadata is unavailable).
    * Comfy metadata is preserved under ``_comfy_metadata`` in ``node.data``.
    """
    nodes: list[Node] = []
    edges: list[Edge] = []

    for node_id, node_def in prompt.items():
        class_type = node_def.get("class_type", "Unknown")
        raw_inputs: dict[str, Any] = node_def.get("inputs", {})

        data: dict[str, Any] = {}
        for key, value in raw_inputs.items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                # This is a connection tuple – generate an edge instead
                source_id = str(value[0])
                output_index = value[1]
                edges.append(
                    Edge(
                        source=source_id,
                        sourceHandle=f"output_{output_index}",
                        target=node_id,
                        targetHandle=key,
                    )
                )
            else:
                data[key] = value

        nodes.append(
            Node(
                id=node_id,
                type=f"comfy.{class_type}",
                data=data,
            )
        )

    return Graph(nodes=nodes, edges=edges)
