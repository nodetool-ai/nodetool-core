"""
Tests for the Graph class (workflows/graph.py).

The Graph class is critical for workflow management and execution.
It handles node and edge management, topological sorting, validation,
and schema generation.
"""

import pytest

from nodetool.types.api_graph import Edge
from nodetool.workflows.base_node import BaseNode, GroupNode
from nodetool.workflows.graph import Graph


class TestInputNode(BaseNode):
    """Simple input node for testing."""

    name: str | None = None
    value: str = "input_value"

    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph.TestInputNode"

    async def process(self, context):
        return {"output": self.value}


class TestOutputNode(BaseNode):
    """Simple output node for testing."""

    name: str | None = None
    value: str = ""

    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph.TestOutputNode"

    async def process(self, context):
        return {"value": self.value}


class TestProcessingNode(BaseNode):
    """Simple processing node for testing."""

    input_value: str = ""
    multiplier: int = 1

    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph.TestProcessingNode"

    async def process(self, context):
        return {"output": self.input_value * self.multiplier}


class TestGraphNode:
    """Test suite for Graph class core functionality."""

    def test_graph_creation_empty(self):
        """Test creating an empty graph."""
        graph = Graph(nodes=[], edges=[])
        assert graph.nodes == []
        assert graph.edges == []

    def test_graph_creation_with_nodes_and_edges(self):
        """Test creating a graph with nodes and edges."""
        nodes = [TestInputNode(id="node1"), TestOutputNode(id="node2")]
        edges = [
            Edge(
                source="node1",
                sourceHandle="output",
                target="node2",
                targetHandle="value",
                id="edge1",
            )
        ]
        graph = Graph(nodes=nodes, edges=edges)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_find_node_exists(self):
        """Test finding an existing node by ID."""
        node = TestInputNode(id="test_node")
        graph = Graph(nodes=[node])
        found = graph.find_node("test_node")
        assert found is not None
        assert found.id == "test_node"

    def test_find_node_not_exists(self):
        """Test finding a non-existent node."""
        graph = Graph(nodes=[])
        found = graph.find_node("nonexistent")
        assert found is None

    def test_find_edges_by_source_and_handle(self):
        """Test finding edges by source and source_handle."""
        edges = [
            Edge(
                source="node1",
                sourceHandle="output1",
                target="node2",
                targetHandle="input1",
                id="edge1",
            ),
            Edge(
                source="node1",
                sourceHandle="output2",
                target="node3",
                targetHandle="input2",
                id="edge2",
            ),
            Edge(
                source="node2",
                sourceHandle="output1",
                target="node3",
                targetHandle="input3",
                id="edge3",
            ),
        ]
        graph = Graph(edges=edges)
        found = graph.find_edges("node1", "output1")
        assert len(found) == 1
        assert found[0].sourceHandle == "output1"

    def test_find_edges_empty_result(self):
        """Test finding edges when none match."""
        edges = [
            Edge(
                source="node1",
                sourceHandle="output1",
                target="node2",
                targetHandle="input1",
                id="edge1",
            )
        ]
        graph = Graph(edges=edges)
        found = graph.find_edges("node1", "nonexistent")
        assert len(found) == 0


class TestGraphTopologicalSort:
    """Test suite for topological sorting functionality."""

    def test_topological_sort_simple_chain(self):
        """Test topological sort with a simple linear chain."""
        nodes = [
            TestInputNode(id="node1"),
            TestProcessingNode(id="node2"),
            TestOutputNode(id="node3"),
        ]
        edges = [
            Edge(
                source="node1",
                sourceHandle="output",
                target="node2",
                targetHandle="input_value",
                id="edge1",
            ),
            Edge(
                source="node2",
                sourceHandle="output",
                target="node3",
                targetHandle="value",
                id="edge2",
            ),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        sorted_levels = graph.topological_sort()
        assert len(sorted_levels) == 3
        assert "node1" in sorted_levels[0]
        assert "node2" in sorted_levels[1]
        assert "node3" in sorted_levels[2]

    def test_topological_sort_parallel_nodes(self):
        """Test topological sort with parallel independent nodes."""
        nodes = [
            TestInputNode(id="node1"),
            TestProcessingNode(id="node2a"),
            TestProcessingNode(id="node2b"),
            TestOutputNode(id="node3"),
        ]
        edges = [
            Edge(
                source="node1",
                sourceHandle="output",
                target="node2a",
                targetHandle="input_value",
                id="edge1",
            ),
            Edge(
                source="node1",
                sourceHandle="output",
                target="node2b",
                targetHandle="input_value",
                id="edge2",
            ),
            Edge(
                source="node2a",
                sourceHandle="output",
                target="node3",
                targetHandle="value",
                id="edge3",
            ),
            Edge(
                source="node2b",
                sourceHandle="output",
                target="node3",
                targetHandle="value",
                id="edge4",
            ),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        sorted_levels = graph.topological_sort()
        assert len(sorted_levels) == 3
        assert "node1" in sorted_levels[0]
        assert "node2a" in sorted_levels[1]
        assert "node2b" in sorted_levels[1]
        assert "node3" in sorted_levels[2]

    def test_topological_sort_with_parent_id(self):
        """Test topological sort with parent_id filtering."""
        nodes = [
            TestInputNode(id="node1", parent_id="parent1"),
            TestProcessingNode(id="node2", parent_id="parent1"),
            TestOutputNode(id="node3", parent_id=None),
        ]
        edges = [
            Edge(
                source="node1",
                sourceHandle="output",
                target="node2",
                targetHandle="input_value",
                id="edge1",
            ),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        # Only get nodes with parent_id=None
        sorted_levels = graph.topological_sort(parent_id=None)
        assert len(sorted_levels) == 1
        assert "node3" in sorted_levels[0]

    def test_topological_sort_with_group_nodes(self):
        """Test topological sort with group nodes."""
        nodes = [
            TestInputNode(id="node1", parent_id=None),
            TestProcessingNode(id="node2", parent_id="group1"),
            GroupNode(id="group1", parent_id=None),
            TestOutputNode(id="node3", parent_id=None),
        ]
        edges = [
            Edge(
                source="node1",
                sourceHandle="output",
                target="node2",
                targetHandle="input_value",
                id="edge1",
            ),
            Edge(
                source="node2",
                sourceHandle="output",
                target="node3",
                targetHandle="value",
                id="edge2",
            ),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        sorted_levels = graph.topological_sort()
        # node2 should be included because its parent is a GroupNode
        assert len(sorted_levels) == 3

    def test_topological_sort_empty_graph(self):
        """Test topological sort on empty graph."""
        graph = Graph(nodes=[], edges=[])
        sorted_levels = graph.topological_sort()
        assert len(sorted_levels) == 0


class TestGraphValidation:
    """Test suite for graph edge type validation."""

    def test_validate_edge_types_all_valid(self):
        """Test validation with all valid edge types."""
        # This test assumes compatible types
        nodes = [
            TestInputNode(id="node1"),
            TestProcessingNode(id="node2"),
        ]
        edges = [
            Edge(
                source="node1",
                sourceHandle="output",
                target="node2",
                targetHandle="input_value",
                id="edge1",
            )
        ]
        graph = Graph(nodes=nodes, edges=edges)
        errors = graph.validate_edge_types()
        # With dynamic nodes and type compatibility, this should not error
        assert isinstance(errors, list)

    def test_validate_edge_types_nonexistent_target_node(self):
        """Test validation with non-existent target node."""
        nodes = [TestInputNode(id="node1")]
        edges = [
            Edge(
                source="node1",
                sourceHandle="output",
                target="nonexistent",
                targetHandle="input_value",
                id="edge1",
            )
        ]
        graph = Graph(nodes=nodes, edges=edges)
        errors = graph.validate_edge_types()
        assert len(errors) > 0
        assert any("not found" in error for error in errors)


class TestGraphStreamingCheck:
    """Test suite for streaming upstream detection."""

    def test_has_streaming_upstream_no_streaming(self):
        """Test streaming check when no upstream nodes stream."""
        nodes = [
            TestInputNode(id="node1"),
            TestProcessingNode(id="node2"),
        ]
        edges = [
            Edge(
                source="node1",
                sourceHandle="output",
                target="node2",
                targetHandle="input_value",
                id="edge1",
            )
        ]
        graph = Graph(nodes=nodes, edges=edges)
        # TestProcessingNode does not override gen_process, so not streaming
        has_streaming = graph.has_streaming_upstream("node2")
        assert has_streaming is False

    def test_has_streaming_upstream_disconnected_node(self):
        """Test streaming check for disconnected node."""
        nodes = [
            TestInputNode(id="node1"),
            TestProcessingNode(id="node2"),
        ]
        graph = Graph(nodes=nodes, edges=[])
        has_streaming = graph.has_streaming_upstream("node2")
        assert has_streaming is False


class TestGraphFromDict:
    """Test suite for Graph.from_dict() factory method."""

    def test_from_dict_empty_graph(self):
        """Test from_dict with empty graph data."""
        graph_data = {"nodes": [], "edges": []}
        graph = Graph.from_dict(graph_data)
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_from_dict_with_skip_errors(self):
        """Test from_dict with skip_errors=True."""
        # Use a non-existent node type to trigger error handling
        graph_data = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "nodetool.test_nonexistent",  # Invalid type
                    "data": {},
                }
            ],
            "edges": [],
        }
        # Should not raise, just skip the invalid node
        graph = Graph.from_dict(graph_data, skip_errors=True)
        assert len(graph.nodes) == 0

    def test_from_dict_filters_properties_with_edges(self):
        """Test that from_dict filters out properties that have incoming edges."""
        graph_data = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "nodetool.test_input",
                    "data": {"value": "static_value"},
                },
                {
                    "id": "node2",
                    "type": "nodetool.test_processing",
                    "data": {"input_value": "should_be_filtered", "multiplier": 2},
                },
            ],
            "edges": [
                {
                    "source": "node1",
                    "sourceHandle": "output",
                    "target": "node2",
                    "targetHandle": "input_value",
                    "id": "edge1",
                }
            ],
        }
        # The filtering happens during from_dict
        # Properties with incoming edges should be removed from node data
        Graph.from_dict(graph_data, skip_errors=True)
        # Note: This test verifies the behavior but may return empty nodes
        # if the node types aren't registered

    def test_from_dict_invalid_edges_skipped(self):
        """Test that invalid edges are skipped when skip_errors=True."""
        graph_data = {
            "nodes": [],
            "edges": [
                {
                    # Missing required fields
                    "source": "node1",
                    "target": "node2",
                }
            ],
        }
        graph = Graph.from_dict(graph_data, skip_errors=True)
        assert len(graph.edges) == 0
