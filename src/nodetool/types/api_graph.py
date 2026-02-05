from typing import Any

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

    def get_node_type(self) -> str:
        """Return the node type."""
        return self.type

    @property
    def node_id(self) -> str:
        """Alias for id property."""
        return self.id


class Edge(BaseModel):
    id: str | None = None
    source: str
    sourceHandle: str
    target: str
    targetHandle: str
    ui_properties: dict[str, str] | None = None


class Graph(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


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


def get_input_schema(graph: Graph) -> dict[str, Any]:
    input_schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    for node in graph.nodes:
        if node.type.startswith(("nodetool.input.", "nodetool.workflows.test_helper.")):
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


def get_output_schema(graph: Graph) -> dict[str, Any]:
    output_schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    for node in graph.nodes:
        if node.type.startswith("nodetool.output."):
            name = node.data.get("name", node.id)
            output_schema["properties"][name] = {
                "type": "any",
                "label": node.data.get("label", ""),
            }
            output_schema["required"].append(name)

    return output_schema
