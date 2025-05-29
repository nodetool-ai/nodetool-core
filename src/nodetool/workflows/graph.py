from typing import Any, List, Sequence, Tuple, Optional
from collections import deque
import logging

from pydantic import BaseModel, Field, ValidationError
from nodetool.types.graph import Edge
from nodetool.workflows.base_node import (
    GroupNode,
    InputNode,
    BaseNode,
    OutputNode,
)


class Graph(BaseModel):
    """
    Represents a graph data structure for workflow management and analysis.

    This class encapsulates the functionality for creating, manipulating, and analyzing
    directed graphs, particularly in the context of workflow systems. It provides methods
    for managing nodes and edges, identifying input and output nodes, generating schemas,
    and performing topological sorting.

    Key features:
    - Node and edge management
    - Input and output node identification
    - JSON schema generation for inputs and outputs
    - Topological sorting of nodes

    The Graph class is designed to support various operations on workflow graphs,
    including dependency analysis, execution order determination, and subgraph handling.
    It is particularly useful for systems that need to represent and process complex,
    interconnected workflows or data pipelines.

    Attributes:
        nodes (list[BaseNode]): A list of nodes in the graph.
        edges (list[Edge]): A list of edges connecting the nodes.

    Methods:
        find_node: Locates a node by its ID.
        from_dict: Creates a Graph instance from a dictionary representation.
        inputs: Returns a list of input nodes.
        outputs: Returns a list of output nodes.
        get_input_schema: Generates a JSON schema for the graph's inputs.
        get_output_schema: Generates a JSON schema for the graph's outputs.
        topological_sort: Performs a topological sort on the graph's nodes.

    The class leverages Pydantic for data validation and serialization, making it
    robust for use in larger systems that require strict type checking and easy
    integration with APIs or databases.
    """

    nodes: Sequence[BaseNode] = Field(default_factory=list)
    edges: Sequence[Edge] = Field(default_factory=list)

    def find_node(self, node_id: str) -> BaseNode | None:
        """
        Find a node by its id.
        """
        for node in self.nodes:
            if node._id == node_id:
                return node
        return None

    def find_edges(self, source: str, source_handle: str) -> List[Edge]:
        """
        Find edges by their source and source_handle.
        """
        return [
            edge
            for edge in self.edges
            if edge.source == source and edge.sourceHandle == source_handle
        ]

    @classmethod
    def from_dict(cls, graph: dict[str, Any]):
        """
        Create a Graph object from a dictionary representation.
        The format is the same as the one used in the frontend.

        Invalid nodes or edges in the input dictionary are skipped, and warnings are logged.
        If all nodes in a workflow are invalid, or if the graph data leads to a Pydantic
        validation error during final Graph instantiation, the resulting graph might be empty
        or incomplete. The method attempts to construct a graph with as much valid data as possible.

        Args:
            graph (dict[str, Any]): The dictionary representing the Graph.
        
        Returns:
            Graph: An instance of the Graph, potentially with fewer nodes/edges than specified
                   in the input if errors were encountered.
        """
        nodes_list = []
        raw_nodes = graph.get("nodes", [])
        for node_data in raw_nodes:
            try:
                node = BaseNode.from_dict(node_data, skip_errors=True)
                nodes_list.append(node)
            except ValueError as e:
                # Log the error and skip this node
                logging.warning(f"Skipping invalid node during Graph.from_dict: {e}. Node data: {node_data}")
                continue
        
        edges_list = []
        raw_edges = graph.get("edges", [])
        for edge_data in raw_edges:
            try:
                # Assuming Edge can be instantiated directly from dict
                # If Edge has its own from_dict or validation, use that
                edges_list.append(Edge(**edge_data))
            except (ValueError, TypeError, ValidationError) as e: # Catching a broader exception for edges if instantiation is complex
                logging.warning(f"Skipping invalid edge during Graph.from_dict: {e}. Edge data: {edge_data}")
                continue

        return cls(
            nodes=nodes_list,
            edges=edges_list,
        )

    def inputs(self) -> List[InputNode]:
        """
        Returns a list of nodes that inherit from InputNode.
        """
        return [node for node in self.nodes if isinstance(node, InputNode)]

    def outputs(self) -> List[OutputNode]:
        """
        Returns a list of nodes that have no outgoing edges.
        """
        return [node for node in self.nodes if isinstance(node, OutputNode)]

    def get_input_schema(self):
        """
        Returns a JSON schema for input nodes of the graph.
        """
        return {
            "type": "object",
            "properties": {node.name: node.get_json_schema() for node in self.inputs()},
        }

    def get_output_schema(self):
        """
        Returns a JSON schema for the output nodes of the graph.
        """
        return {
            "type": "object",
            "properties": {
                node.name: node.get_json_schema() for node in self.outputs()
            },
        }

    def topological_sort(self, parent_id: str | None = None) -> List[List[str]]:
        """
        Perform a topological sort on the graph, grouping nodes by levels.

        This method implements a modified version of Kahn's algorithm for topological sorting.
        It sorts the nodes of the graph into levels, where each level contains nodes
        that can be processed in parallel.

        Args:
            parent_id (str | None, optional): The ID of the parent node to filter results. Defaults to None.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains the node IDs at the same level
                             in the topological order. Nodes in the same list can be processed in parallel.

        Notes:
        - The method does not modify the original graph structure.
        - Nodes are only included in the output if their parent_id matches the given parent_id.
        - If a cycle exists, some nodes may be omitted from the result.
        """
        # child nodes of regular groups (no loops) can be executed like top level nodes
        if parent_id is None:
            group_nodes = {node.id for node in self.nodes if type(node) is GroupNode}
        else:
            group_nodes = set()

        # Filter nodes with given parent_id
        nodes = [
            node
            for node in self.nodes
            if node.parent_id == parent_id or node.parent_id in group_nodes
        ]
        node_ids = {node.id for node in nodes}

        # Filter edges to only include those connected to the filtered nodes
        edges = [
            edge
            for edge in self.edges
            if edge.source in node_ids and edge.target in node_ids
        ]

        indegree: dict[str, int] = {node.id: 0 for node in nodes}

        for edge in edges:
            indegree[edge.target] += 1

        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)

        sorted_nodes = []
        while queue:
            level_nodes = []
            for _ in range(len(queue)):
                n = queue.popleft()
                level_nodes.append(n)
                for edge in edges[:]:  # Iterate over a copy of the list
                    if edge.source == n:
                        edges.remove(edge)
                        indegree[edge.target] -= 1
                        if indegree[edge.target] == 0:
                            queue.append(edge.target)

            if level_nodes:
                sorted_nodes.append(level_nodes)

        if any(indegree[node_id] != 0 for node_id in indegree.keys()):
            print("Graph contains at least one cycle")

        return sorted_nodes
