from typing import Any, List, Sequence
from collections import deque

from nodetool.metadata.typecheck import typecheck
from pydantic import BaseModel, Field
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
    def from_dict(cls, graph: dict[str, Any], skip_errors: bool = True):
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
        # First pass: identify properties that have incoming edges
        properties_with_edges = {}  # {node_id: set(property_names)}
        for edge_data in graph.get("edges", []):
            target_id = edge_data.get("target")
            target_handle = edge_data.get("targetHandle")
            if target_id and target_handle:
                if target_id not in properties_with_edges:
                    properties_with_edges[target_id] = set()
                properties_with_edges[target_id].add(target_handle)

        valid_nodes = []
        valid_node_ids = set()
        
        # Process nodes, collecting valid ones
        for node_data in graph["nodes"]:
            try:
                # Filter out properties that have incoming edges
                node_id = node_data.get("id")
                filtered_node_data = node_data.copy()
                if node_id in properties_with_edges:
                    data = filtered_node_data.get("data", {})
                    connected_properties = properties_with_edges[node_id]
                    filtered_data = {k: v for k, v in data.items() if k not in connected_properties}
                    filtered_node_data["data"] = filtered_data
                
                result = BaseNode.from_dict(filtered_node_data, skip_errors=skip_errors)
                if result is not None and result[0] is not None:
                    valid_nodes.append(result[0])
                    valid_node_ids.add(result[0].id)
            except ValueError as e:
                if not skip_errors:
                    raise
                # If skip_errors is True, skip this node
        
        # Process edges, filtering out invalid ones
        valid_edges = []
        for edge_data in graph["edges"]:
            try:
                # Check if edge has required fields
                if ("sourceHandle" not in edge_data or 
                    "targetHandle" not in edge_data or
                    "source" not in edge_data or 
                    "target" not in edge_data):
                    if skip_errors:
                        continue  # Skip malformed edges
                    else:
                        # Let Pydantic handle the validation error
                        pass
                
                # Check if both source and target nodes exist in valid nodes
                source_id = edge_data.get("source")
                target_id = edge_data.get("target")
                
                if (source_id in valid_node_ids and target_id in valid_node_ids):
                    valid_edges.append(edge_data)
                elif skip_errors:
                    continue  # Skip edges connected to non-existent nodes
                else:
                    # Keep the edge and let downstream validation handle it
                    valid_edges.append(edge_data)
                    
            except Exception:
                if skip_errors:
                    continue
                else:
                    raise
        
        return cls(
            nodes=valid_nodes,
            edges=valid_edges,
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

    def validate_edge_types(self):
        """
        Validate that edge connections have compatible types.

        Returns:
            List[str]: List of validation error messages. Empty list if all edges are valid.
        """
        validation_errors = []

        for edge in self.edges:
            try:
                # Find source and target nodes
                source_node = self.find_node(edge.source)
                target_node = self.find_node(edge.target)

                if not source_node:
                    validation_errors.append(
                        f"Source node '{edge.source}' not found for edge"
                    )
                    continue

                if not target_node:
                    validation_errors.append(
                        f"Target node '{edge.target}' not found for edge"
                    )
                    continue

                # Get node classes to access type metadata
                # Since nodes are already instances, we can use their classes directly
                source_node_class = source_node.__class__
                target_node_class = target_node.__class__

                # Get source output type (find_output is a class method)
                source_output = source_node_class.find_output(edge.sourceHandle)
                if not source_output:
                    validation_errors.append(
                        f"{edge.target}: Output '{edge.sourceHandle}' not found on source node {source_node_class.__name__}"
                    )
                    continue

                # Get target input type (find_property is an instance method)
                target_property = target_node.find_property(edge.targetHandle)
                if not target_property:
                    # Respect dynamic nodes that can accept arbitrary properties
                    if type(target_node).is_dynamic():
                        continue

                    # Align error message format with test expectations ("Property ... not found")
                    validation_errors.append(
                        f"{edge.target}: Property '{edge.targetHandle}' not found on target node {target_node_class.__name__}"
                    )
                    continue

                # Check type compatibility
                source_type = source_output.type
                target_type = target_property.type

                if not typecheck(source_type, target_type):
                    validation_errors.append(
                        f"{edge.target}: Type mismatch for property '{edge.targetHandle}' - "
                        f"{edge.source}.{edge.sourceHandle} outputs {source_type.type} "
                        f"but {edge.target}.{edge.targetHandle} expects {target_type.type}"
                    )

            except Exception as e:
                validation_errors.append(
                    f"Error validating edge {edge.source}->{edge.target}: {str(e)}"
                )

        return validation_errors
