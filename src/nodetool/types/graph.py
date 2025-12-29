from typing import Any, List

from pydantic import BaseModel, Field

from nodetool.metadata.type_metadata import TypeMetadata


class Node(BaseModel):
    id: str
    parent_id: str | None = None
    type: str = "default"
    data: Any = Field(default_factory=dict)
    ui_properties: Any = Field(default_factory=dict)
    dynamic_properties: dict[str, Any] = Field(default_factory=dict)
    dynamic_outputs: dict[str, TypeMetadata] = Field(default_factory=dict)
    sync_mode: str = "on_any"


class Edge(BaseModel):
    id: str | None = None
    source: str
    sourceHandle: str
    target: str
    targetHandle: str
    ui_properties: dict[str, str] | None = None


class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


def remove_connected_slots(graph: Graph) -> Graph:
    """
    Clears specific slots in the data field of nodes based on connected target handles.

    Args:
        graph (Graph): The graph object containing nodes and edges.

    Returns:
        Graph: The updated graph object with cleared slots.

    """

    # Create a dictionary to store nodes and their connected target handles
    nodes_with_incoming_edges = {}

    # Populate the dictionary
    for edge in graph.edges:
        if edge.target not in nodes_with_incoming_edges:
            nodes_with_incoming_edges[edge.target] = set()
        nodes_with_incoming_edges[edge.target].add(edge.targetHandle)

    # Clear specific slots in the data field of nodes based on connected target handles
    for node in graph.nodes:
        if node.id in nodes_with_incoming_edges:
            connected_handles = nodes_with_incoming_edges[node.id]

            for slot in connected_handles:
                if slot in node.data:
                    del node.data[slot]

    return graph


def get_input_schema(graph: Graph):
    input_schema = {"type": "object", "properties": {}, "required": []}

    for node in graph.nodes:
        if node.type.startswith("nodetool.input."):
            input_type = node.type.split(".")[-1]
            node_schema = {}

            if input_type == "FloatInput":
                node_schema = {
                    "type": "number",
                    "minimum": node.data.get("min", 0),
                    "maximum": node.data.get("max", 100),
                    "default": node.data.get("value", 0),
                }
            elif input_type == "IntegerInput":
                node_schema = {
                    "type": "integer",
                    "minimum": node.data.get("min", 0),
                    "maximum": node.data.get("max", 100),
                    "default": node.data.get("value", 0),
                }
            elif input_type == "StringInput":
                node_schema = {"type": "string", "default": node.data.get("value", "")}
            elif input_type == "BooleanInput":
                node_schema = {
                    "type": "boolean",
                    "default": node.data.get("value", False),
                }
            elif input_type in [
                "ImageInput",
                "VideoInput",
                "AudioInput",
                "DocumentInput",
            ]:
                node_schema = {
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string", "format": "uri"},
                        "type": {
                            "type": "string",
                            "enum": [input_type.lower().replace("input", "")],
                        },
                    },
                    "required": ["uri", "type"],
                }

            if node_schema:
                name = node.data.get("name", node.id)
                node_schema["label"] = node.data.get("label", "")
                input_schema["properties"][name] = node_schema
                input_schema["required"].append(name)

    return input_schema


def get_output_schema(graph: Graph):
    """Generate output schema for graph. Returns empty schema as nodes handle their own outputs."""
    return {"type": "object", "properties": {}, "required": []}
