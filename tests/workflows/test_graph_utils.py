"""
Tests for graph utilities (workflows/graph_utils.py).

Tests node lookup, type inference, and subgraph extraction utilities.
"""

import pytest

from nodetool.types.api_graph import Edge
from nodetool.workflows.graph import Graph
from nodetool.workflows.graph_utils import (
    find_node,
    get_downstream_subgraph,
    get_node_input_types,
)
from tests.workflows.test_graph import TestInputNode, TestOutputNode, TestProcessingNode


class TestFindNode:
    """Test find_node utility function."""

    def test_find_existing_node(self):
        """Test finding an existing node in the graph."""
        node = TestInputNode(id="node1")
        graph = Graph(nodes=[node], edges=[])
        found = find_node(graph, "node1")
        assert found is node

    def test_find_nonexistent_node_raises_error(self):
        """Test that finding a nonexistent node raises ValueError."""
        node = TestInputNode(id="node1")
        graph = Graph(nodes=[node], edges=[])
        with pytest.raises(ValueError, match="Node with ID node2 does not exist"):
            find_node(graph, "node2")


class TestGetNodeInputTypes:
    """Test get_node_input_types utility function."""

    def test_get_input_types_for_unconnected_node(self):
        """Test getting input types for a node with no incoming edges."""
        node = TestInputNode(id="node1")
        graph = Graph(nodes=[node], edges=[])
        input_types = get_node_input_types(graph, "node1")

        assert input_types == {}

    def test_get_input_types_with_nonexistent_source_node(self):
        """Test getting input types when source node doesn't exist."""
        edge = Edge(
            id="edge1",
            source="nonexistent",
            sourceHandle="output",
            target="node2",
            targetHandle="input_value",
            type="edge",
        )
        output_node = TestOutputNode(id="node2")
        graph = Graph(nodes=[output_node], edges=[edge])
        input_types = get_node_input_types(graph, "node2")

        # Should return None for missing source node
        assert "input_value" in input_types
        assert input_types["input_value"] is None


class TestGetDownstreamSubgraph:
    """Test get_downstream_subgraph utility function."""

    def test_get_downstream_subgraph_no_downstream(self):
        """Test extracting downstream subgraph when there are no downstream nodes."""
        node = TestInputNode(id="node1")
        graph = Graph(nodes=[node], edges=[])
        initial_edges, _subgraph = get_downstream_subgraph(graph, "node1", "output")

        assert len(initial_edges) == 0
        # Note: empty graph.nodes is expected when no edges connect
