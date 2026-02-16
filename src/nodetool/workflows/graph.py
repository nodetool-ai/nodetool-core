import logging
from collections import defaultdict, deque
from typing import Any, Sequence

from pydantic import BaseModel, Field

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.typecheck import typecheck
from nodetool.types.api_graph import Edge
from nodetool.workflows.base_node import (
    BaseNode,
    GroupNode,
    InputNode,
    OutputNode,
)

log = logging.getLogger(__name__)


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
    _node_index: dict[str, BaseNode] | None = None

    def __init__(self, **data):
        super().__init__(**data)
        # Build node ID index for O(1) lookup
        self._node_index = {node._id: node for node in self.nodes} if self.nodes else {}

    def model_post_init(self, __context: Any):
        # Build node ID index after model initialization
        self._node_index = {node._id: node for node in self.nodes} if self.nodes else {}

    def find_node(self, node_id: str) -> BaseNode | None:
        """
        Find a node by its id using O(1) dictionary lookup.
        """
        return self._node_index.get(node_id) if self._node_index else None

    def find_edges(self, source: str, source_handle: str) -> list[Edge]:
        """
        Find edges by their source and source_handle.
        """
        return [edge for edge in self.edges if edge.source == source and edge.sourceHandle == source_handle]

    @classmethod
    def from_dict(
        cls,
        graph: dict[str, Any],
        skip_errors: bool = True,
        allow_undefined_properties: bool = True,
    ):
        """
        Create a Graph object from a dictionary representation.
        The format is the same as the one used in the frontend.

        Node type resolution: for each node entry, this method delegates to
        `BaseNode.from_dict`, which uses `get_node_class` to resolve the node
        class. Resolution checks the in-memory registry, attempts dynamic
        imports based on the type path, consults installed packages, and finally
        falls back to a class-name match (ignoring an optional "Node" suffix).
        As a result, node types that were previously considered "unregistered"
        may now resolve successfully if the corresponding module/package is
        available.

        Invalid nodes or edges in the input dictionary are skipped when
        `skip_errors=True` (default), and warnings may be logged. If all nodes
        are invalid, or if the graph data leads to a Pydantic validation error
        during final Graph instantiation, the resulting graph might be empty or
        incomplete. The method attempts to construct a graph with as much valid
        data as possible.

        Args:
            graph (dict[str, Any]): The dictionary representing the Graph.
            skip_errors (bool): If True, property assignment errors are collected and returned,
                                not logged directly or raised immediately.
            allow_undefined_properties (bool): If True, properties not defined in the node class are ignored.
                                              Used for backward compatibility to skip deprecated properties.

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
                node_type = node_data.get("type", "<unknown>")
                filtered_node_data = node_data.copy()
                if node_id in properties_with_edges:
                    data = filtered_node_data.get("data", {})
                    connected_properties = properties_with_edges[node_id]
                    filtered_data = {k: v for k, v in data.items() if k not in connected_properties}
                    filtered_node_data["data"] = filtered_data

                result = BaseNode.from_dict(
                    filtered_node_data,
                    skip_errors=skip_errors,
                    allow_undefined_properties=allow_undefined_properties,
                )
                if result is not None and result[0] is not None:
                    valid_nodes.append(result[0])
                    valid_node_ids.add(result[0].id)
                elif skip_errors:
                    log.warning(f"Skipping node {node_id} (type: {node_type}) - failed to instantiate")
            except ValueError as e:
                if not skip_errors:
                    raise ValueError(f"Failed to load node {node_id} (type: {node_type}): {str(e)}") from e
                # If skip_errors is True, log and skip this node
                log.warning(f"Skipping node {node_id} (type: {node_type}) due to error: {str(e)}")
            except Exception as e:
                if not skip_errors:
                    raise RuntimeError(f"Failed to load node {node_id} (type: {node_type}): {str(e)}") from e
                # If skip_errors is True, log and skip this node
                log.error(
                    f"Skipping node {node_id} (type: {node_type}) due to unexpected error: {str(e)}",
                    exc_info=True,
                )

        # Process edges, filtering out invalid ones
        valid_edges = []
        for edge_data in graph["edges"]:
            try:
                # Check if edge has required fields
                if (
                    "sourceHandle" not in edge_data
                    or "targetHandle" not in edge_data
                    or "source" not in edge_data
                    or "target" not in edge_data
                ):
                    if skip_errors:
                        continue  # Skip malformed edges
                    else:
                        # Let Pydantic handle the validation error
                        pass

                # Check if both source and target nodes exist in valid nodes
                source_id = edge_data.get("source")
                target_id = edge_data.get("target")

                if source_id in valid_node_ids and target_id in valid_node_ids:
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

    def inputs(self) -> list[InputNode]:
        """
        Returns a list of nodes that inherit from InputNode.
        """
        return [node for node in self.nodes if isinstance(node, InputNode)]

    def outputs(self) -> list[OutputNode]:
        """
        Returns a list of nodes that are designated as OutputNode instances.
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
            "properties": {node.name: node.get_json_schema() for node in self.outputs()},
        }

    def topological_sort(self, parent_id: str | None = None) -> list[list[str]]:
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
        group_nodes = {node.id for node in self.nodes if type(node) is GroupNode} if parent_id is None else set()

        # Filter nodes with given parent_id
        nodes = [node for node in self.nodes if node.parent_id == parent_id or node.parent_id in group_nodes]
        node_ids = {node.id for node in nodes}

        # Filter edges to only include those connected to the filtered nodes
        edges = [edge for edge in self.edges if edge.source in node_ids and edge.target in node_ids]

        indegree: dict[str, int] = {node.id: 0 for node in nodes}

        outgoing_edges: dict[str, list[Edge]] = {}
        for edge in edges:
            indegree[edge.target] += 1
            if edge.source not in outgoing_edges:
                outgoing_edges[edge.source] = []
            outgoing_edges[edge.source].append(edge)

        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)

        sorted_nodes = []
        while queue:
            level_nodes = []
            for _ in range(len(queue)):
                n = queue.popleft()
                level_nodes.append(n)
                for edge in outgoing_edges.get(n, []):
                    indegree[edge.target] -= 1
                    if indegree[edge.target] == 0:
                        queue.append(edge.target)

            if level_nodes:
                sorted_nodes.append(level_nodes)

        if any(indegree[node_id] != 0 for node_id in indegree):
            log.warning("Graph contains at least one cycle")

        return sorted_nodes

    def get_control_edges(self, target_id: str) -> list[Edge]:
        """Return all control edges targeting the given node."""
        return [
            edge for edge in self.edges
            if edge.target == target_id and edge.edge_type == "control"
        ]

    def get_controller_nodes(self, target_id: str) -> list[BaseNode]:
        """Return all nodes that control the given target node."""
        control_edges = self.get_control_edges(target_id)
        controllers = []
        for edge in control_edges:
            node = self.find_node(edge.source)
            if node:
                controllers.append(node)
        return controllers

    def get_controlled_nodes(self, source_id: str) -> list[str]:
        """Return IDs of all nodes controlled by the given source."""
        return [
            edge.target for edge in self.edges
            if edge.source == source_id and edge.edge_type == "control"
        ]

    def validate_control_edges(self) -> list[str]:
        """
        Validate control edges in the graph.

        Rules:
        - Control edges must originate from Agent-type nodes
        - Control edges must target valid nodes
        - Control edges must use '__control__' as targetHandle
        - Circular control chains are forbidden

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for edge in self.edges:
            if edge.edge_type != "control":
                continue

            # Rule 1: Source must be an Agent node
            # Convention: Agent node types contain "agent" in their type path
            # (e.g., "nodetool.agents.Agent", "nodetool.agents.Classifier")
            source_node = self.find_node(edge.source)
            if not source_node:
                errors.append(f"Control edge {edge.id} has invalid source {edge.source}")
                continue

            if "agent" not in source_node.get_node_type().lower():
                errors.append(
                    f"Control edge {edge.id} source {edge.source} must be an Agent node, "
                    f"got {source_node.get_node_type()}"
                )

            # Rule 2: Target must exist
            target_node = self.find_node(edge.target)
            if not target_node:
                errors.append(f"Control edge {edge.id} has invalid target {edge.target}")
                continue

            # Rule 3: Must use __control__ as targetHandle
            if edge.targetHandle != "__control__":
                errors.append(
                    f"Control edge {edge.id} must use '__control__' as targetHandle, "
                    f"got '{edge.targetHandle}'"
                )

        # Rule 4: Check for circular control dependencies
        circular_errors = self._check_circular_control(self.edges)
        errors.extend(circular_errors)

        return errors

    def _check_circular_control(self, edges: Sequence[Edge]) -> list[str]:
        """
        Check for circular dependencies in control edges.

        Returns:
            List of error messages for circular dependencies
        """
        errors = []

        # Build control adjacency list
        control_graph: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.edge_type == "control":
                control_graph[edge.source].append(edge.target)

        # DFS to detect cycles
        def has_cycle(node: str, visited: set[str], rec_stack: set[str]) -> tuple[bool, list[str]]:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in control_graph.get(node, []):
                if neighbor not in visited:
                    found, path = has_cycle(neighbor, visited, rec_stack)
                    if found:
                        return True, [node, *path]
                elif neighbor in rec_stack:
                    return True, [node, neighbor]

            rec_stack.remove(node)
            return False, []

        visited: set[str] = set()
        for node_id in control_graph:
            if node_id not in visited:
                found, path = has_cycle(node_id, visited, set())
                if found:
                    errors.append(
                        f"Circular control dependency detected: {' -> '.join(path)}"
                    )

        return errors

    def validate_edge_types(self):
        """
        Validate that edge connections have compatible types.

        Multi-edge list input validation:
        - Multiple edges targeting the same property are allowed ONLY if the
          property type is ``list[T]``.
        - When multiple edges target a list property, each source type must be
          compatible with the list element type ``T``.
        - Multiple edges targeting a non-list property result in an error.

        Returns:
            List[str]: List of validation error messages. Empty list if all edges are valid.
        """
        validation_errors = []

        # Group data edges by (target_node_id, targetHandle) to detect multi-edge scenarios
        # Control edges are validated separately via validate_control_edges()
        edges_by_target_handle: dict[tuple[str, str], list[Edge]] = defaultdict(list)
        for edge in self.edges:
            if edge.edge_type == "control":
                continue
            key = (edge.target, edge.targetHandle)
            edges_by_target_handle[key].append(edge)

        # Validate each target handle
        for (target_id, handle), edges in edges_by_target_handle.items():
            target_node = self.find_node(target_id)
            if not target_node:
                validation_errors.append(f"Target node '{target_id}' not found for edge")
                continue

            target_node_class = target_node.__class__

            # Get target input type (find_property is an instance method)
            target_property = target_node.find_property(handle)
            if not target_property:
                # Respect dynamic nodes that can accept arbitrary properties
                if type(target_node).is_dynamic():
                    # Still validate source outputs exist for dynamic nodes
                    for edge in edges:
                        source_node = self.find_node(edge.source)
                        if not source_node:
                            validation_errors.append(f"Source node '{edge.source}' not found for edge")
                            continue
                        source_output = source_node.find_output_instance(edge.sourceHandle)
                        if not source_output:
                            validation_errors.append(
                                f"{edge.target}: Output '{edge.sourceHandle}' not found on source node {source_node.__class__.__name__}"
                            )
                    continue

                # Align error message format with test expectations ("Property ... not found")
                validation_errors.append(
                    f"{target_id}: Property '{handle}' not found on target node {target_node_class.__name__}"
                )
                continue

            target_type = target_property.type

            # Check for multi-edge scenarios
            if len(edges) > 1:
                # Multiple edges to the same property - only allowed for list types
                if not target_type.is_list_type():
                    validation_errors.append(
                        f"{target_id}: Multiple edges target non-list property '{handle}' "
                        f"(type: {target_type.type}). Either change property type to list "
                        f"or use a Collect node."
                    )
                    continue

                # For list properties with multiple edges, validate each source against element type
                element_type = target_type.type_args[0] if target_type.type_args else TypeMetadata(type="any")

                for edge in edges:
                    source_node = self.find_node(edge.source)
                    if not source_node:
                        validation_errors.append(f"Source node '{edge.source}' not found for edge")
                        continue

                    source_output = source_node.find_output_instance(edge.sourceHandle)
                    if not source_output:
                        validation_errors.append(
                            f"{edge.target}: Output '{edge.sourceHandle}' not found on source node {source_node.__class__.__name__}"
                        )
                        continue

                    source_type = source_output.type


                    # For multi-edge list inputs, source type must be compatible with element type
                    # If source is also a list, check if its element type is compatible
                    type_to_check = source_type
                    if source_type.is_list_type() and source_type.type_args:
                        # Source is a list - check its element type compatibility
                        type_to_check = source_type.type_args[0]

                    if not typecheck(type_to_check, element_type):
                        validation_errors.append(
                            f"{target_id}: Edge from {edge.source}.{edge.sourceHandle} "
                            f"has incompatible type '{source_type.type}' for list element type '{element_type.type}' "
                            f"on property '{handle}'"
                        )
            else:
                # Single edge - standard validation
                edge = edges[0]
                source_node = self.find_node(edge.source)
                if not source_node:
                    validation_errors.append(f"Source node '{edge.source}' not found for edge")
                    continue

                source_output = source_node.find_output_instance(edge.sourceHandle)
                if not source_output:
                    validation_errors.append(
                        f"{edge.target}: Output '{edge.sourceHandle}' not found on source node {source_node.__class__.__name__}"
                    )
                    continue

                source_type = source_output.type

                # Check type compatibility
                if not typecheck(source_type, target_type):
                    validation_errors.append(
                        f"{edge.target}: Type mismatch for property '{edge.targetHandle}' - "
                        f"{edge.source}.{edge.sourceHandle} outputs {source_type.type} "
                        f"but {edge.target}.{edge.targetHandle} expects {target_type.type}"
                    )

        # Add control edge validation
        control_errors = self.validate_control_edges()
        validation_errors.extend(control_errors)

        return validation_errors

    def has_streaming_upstream(self, node_id: str) -> bool:
        """Return True if any upstream (direct or transitive) streams outputs.

        A node is considered driven by a stream if any ancestor node implements
        streaming outputs (i.e., overrides `gen_process`). This check is used to
        influence execution/caching behavior of downstream non-streaming nodes.
        """
        visited: set[str] = set()
        queue = deque([node_id])
        reverse_adj: dict[str, list[str]] = {}
        for edge in self.edges:
            reverse_adj.setdefault(edge.target, []).append(edge.source)

        while queue:
            current = queue.popleft()
            for source_id in reverse_adj.get(current, []):
                if source_id in visited:
                    continue
                src = self.find_node(source_id)
                if src is None:
                    continue
                if src.is_streaming_output():
                    return True
                visited.add(source_id)
                queue.append(source_id)
        return False
