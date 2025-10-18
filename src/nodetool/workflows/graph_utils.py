"""
Graph utilities for node and edge operations.
"""

from collections import deque

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.graph import Graph
from nodetool.types.graph import Edge


def find_node(graph: Graph, node_id: str) -> BaseNode:
    """
    Finds a node by its ID.

    Args:
        graph (Graph): The graph to search in.
        node_id (str): The ID of the node to be found.

    Returns:
        BaseNode: The node with the given ID.

    Raises:
        ValueError: If the node with the given ID does not exist.
    """
    node = graph.find_node(node_id)
    if node is None:
        raise ValueError(f"Node with ID {node_id} does not exist")
    return node


def get_node_input_types(graph: Graph, node_id: str) -> dict[str, TypeMetadata | None]:
    """
    Retrieves the input types for a given node, inferred from the output types of the source nodes.

    Args:
        graph (Graph): The graph to analyze.
        node_id (str): The ID of the node.

    Returns:
        dict[str, str]: A dictionary containing the input types for the node, where the keys are the input slot names
        and the values are the types of the corresponding source nodes.
    """

    def output_type(node_id: str, slot: str):
        node = graph.find_node(node_id)
        if node is None:
            return None
        for output in node.outputs():
            if output.name == slot:
                return output.type
        return None

    return {
        edge.targetHandle: output_type(edge.source, edge.sourceHandle)
        for edge in graph.edges
        if edge.target == node_id
    }


def get_downstream_subgraph(
    graph: Graph, node_id: str, source_handle: str, as_subgraph: bool = False
) -> tuple[list[Edge], Graph]:
    """
    Return the entire downstream spanning subgraph starting at that output.

    Args:
        graph (Graph): The graph to analyze.
        node_id (str): The ID of the node to find connections for.
        source_handle (str): The handle of the output to find connections for.
        as_subgraph (bool): Whether to return a subgraph or just the connected nodes.

    Returns:
        tuple[list[Edge], Graph]: A tuple containing the initial edges and the subgraph.
    """
    initial_edges = [
        edge
        for edge in graph.edges
        if edge.source == node_id and edge.sourceHandle == source_handle
    ]

    included_node_ids: set[str] = set()
    included_edges = []

    # Seed the queue with the targets of the initial edges
    queue = deque()
    for e in initial_edges:
        included_edges.append(e)
        included_node_ids.add(e.target)
        queue.append(e.target)

    # Also include the source node of the initial edges
    for e in initial_edges:
        included_node_ids.add(e.source)

    # BFS over outgoing edges to collect downstream nodes and edges
    while queue:
        current = queue.popleft()
        for edge in graph.edges:
            if edge.source == current:
                included_edges.append(edge)
                if edge.target not in included_node_ids:
                    included_node_ids.add(edge.target)
                    queue.append(edge.target)

    # Materialize node instances, skipping any missing nodes gracefully
    included_nodes: list[BaseNode] = []
    for nid in included_node_ids:
        try:
            included_nodes.append(find_node(graph, nid))
        except ValueError:
            # Skip nodes that don't exist in the graph
            pass

    # Filter edges to those whose endpoints are both present in the included set
    filtered_edges = [
        e
        for e in included_edges
        if e.source in included_node_ids and e.target in included_node_ids
    ]

    return initial_edges, Graph(nodes=included_nodes, edges=filtered_edges)
