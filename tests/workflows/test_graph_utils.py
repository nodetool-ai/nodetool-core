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
        initial_edges, subgraph = get_downstream_subgraph(graph, "node1", "output")

        assert len(initial_edges) == 0
        assert len(subgraph.nodes) == 0
        assert len(subgraph.edges) == 0

    def test_get_downstream_subgraph_simple_chain(self):
        """Test extracting downstream subgraph for a simple A -> B -> C chain."""
        node_a = TestInputNode(id="A")
        node_b = TestProcessingNode(id="B")
        node_c = TestOutputNode(id="C")
        edge_ab = Edge(id="e1", source="A", sourceHandle="output", target="B", targetHandle="input_value")
        edge_bc = Edge(id="e2", source="B", sourceHandle="output", target="C", targetHandle="value")

        graph = Graph(nodes=[node_a, node_b, node_c], edges=[edge_ab, edge_bc])

        initial_edges, subgraph = get_downstream_subgraph(graph, "A", "output")

        assert initial_edges == [edge_ab]
        assert len(subgraph.nodes) == 3
        assert any(n.id == "A" for n in subgraph.nodes)
        assert any(n.id == "B" for n in subgraph.nodes)
        assert any(n.id == "C" for n in subgraph.nodes)
        assert len(subgraph.edges) == 2
        assert edge_ab in subgraph.edges
        assert edge_bc in subgraph.edges

    def test_get_downstream_subgraph_branching(self):
        """Test extracting downstream subgraph with branching A -> B, A -> C."""
        node_a = TestInputNode(id="A")
        node_b = TestProcessingNode(id="B")
        node_c = TestProcessingNode(id="C")
        # Both edges from same output
        edge_ab = Edge(id="e1", source="A", sourceHandle="output", target="B", targetHandle="input_value")
        edge_ac = Edge(id="e2", source="A", sourceHandle="output", target="C", targetHandle="input_value")

        graph = Graph(nodes=[node_a, node_b, node_c], edges=[edge_ab, edge_ac])

        initial_edges, subgraph = get_downstream_subgraph(graph, "A", "output")

        assert len(initial_edges) == 2
        assert edge_ab in initial_edges
        assert edge_ac in initial_edges
        assert len(subgraph.nodes) == 3
        assert len(subgraph.edges) == 2

    def test_get_downstream_subgraph_cycles(self):
        """Test extracting downstream subgraph with cycles A -> B -> A."""
        node_a = TestInputNode(id="A")
        node_b = TestProcessingNode(id="B")
        edge_ab = Edge(id="e1", source="A", sourceHandle="output", target="B", targetHandle="input_value")
        edge_ba = Edge(id="e2", source="B", sourceHandle="output", target="A", targetHandle="value") # Assuming A has 'value' input for test

        graph = Graph(nodes=[node_a, node_b], edges=[edge_ab, edge_ba])

        initial_edges, subgraph = get_downstream_subgraph(graph, "A", "output")

        assert initial_edges == [edge_ab]
        assert len(subgraph.nodes) == 2
        assert len(subgraph.edges) == 2
        assert edge_ab in subgraph.edges
        assert edge_ba in subgraph.edges

    def test_get_downstream_subgraph_diamond(self):
        """Test extracting downstream subgraph with diamond shape A -> B -> D, A -> C -> D."""
        node_a = TestInputNode(id="A")
        node_b = TestProcessingNode(id="B")
        node_c = TestProcessingNode(id="C")
        node_d = TestProcessingNode(id="D")
        edge_ab = Edge(id="e1", source="A", sourceHandle="output", target="B", targetHandle="input_value")
        edge_ac = Edge(id="e2", source="A", sourceHandle="output", target="C", targetHandle="input_value")
        edge_bd = Edge(id="e3", source="B", sourceHandle="output", target="D", targetHandle="input_value")
        edge_cd = Edge(id="e4", source="C", sourceHandle="output", target="D", targetHandle="multiplier")

        graph = Graph(nodes=[node_a, node_b, node_c, node_d], edges=[edge_ab, edge_ac, edge_bd, edge_cd])

        initial_edges, subgraph = get_downstream_subgraph(graph, "A", "output")

        assert len(initial_edges) == 2
        assert edge_ab in initial_edges
        assert edge_ac in initial_edges
        assert len(subgraph.nodes) == 4
        assert len(subgraph.edges) == 4

    def test_get_downstream_subgraph_handle_filtering(self):
        """Test that only the specified output handle's downstream is included."""
        node_a = TestInputNode(id="A")
        node_b = TestProcessingNode(id="B")
        node_c = TestProcessingNode(id="C")

        # A.output -> B
        edge_ab = Edge(id="e1", source="A", sourceHandle="output", target="B", targetHandle="input_value")
        # A.other_output -> C
        edge_ac = Edge(id="e2", source="A", sourceHandle="other_output", target="C", targetHandle="input_value")

        graph = Graph(nodes=[node_a, node_b, node_c], edges=[edge_ab, edge_ac])

        initial_edges, subgraph = get_downstream_subgraph(graph, "A", "output")

        assert initial_edges == [edge_ab]
        assert any(n.id == "A" for n in subgraph.nodes)
        assert any(n.id == "B" for n in subgraph.nodes)
        assert not any(n.id == "C" for n in subgraph.nodes)
        assert len(subgraph.edges) == 1
        assert edge_ab in subgraph.edges

    def test_get_downstream_subgraph_disconnected(self):
        """Test that unrelated components are not included."""
        node_a = TestInputNode(id="A")
        node_b = TestProcessingNode(id="B")
        node_c = TestInputNode(id="C")
        node_d = TestProcessingNode(id="D")

        edge_ab = Edge(id="e1", source="A", sourceHandle="output", target="B", targetHandle="input_value")
        edge_cd = Edge(id="e2", source="C", sourceHandle="output", target="D", targetHandle="input_value")

        graph = Graph(nodes=[node_a, node_b, node_c, node_d], edges=[edge_ab, edge_cd])

        initial_edges, subgraph = get_downstream_subgraph(graph, "A", "output")

        assert initial_edges == [edge_ab]
        assert len(subgraph.nodes) == 2
        assert any(n.id == "A" for n in subgraph.nodes)
        assert any(n.id == "B" for n in subgraph.nodes)
        assert len(subgraph.edges) == 1
        assert edge_ab in subgraph.edges

    def test_get_downstream_subgraph_missing_nodes(self):
        """Test graceful handling when an edge points to a node not in the graph nodes list."""
        node_a = TestInputNode(id="A")
        # Node B is missing from the nodes list but referenced in edge
        edge_ab = Edge(id="e1", source="A", sourceHandle="output", target="B", targetHandle="input_value")

        graph = Graph(nodes=[node_a], edges=[edge_ab])

        initial_edges, subgraph = get_downstream_subgraph(graph, "A", "output")

        assert initial_edges == [edge_ab]
        # B is in included_node_ids, but find_node(graph, 'B') will fail.
        # So subgraph.nodes should only have A.
        assert len(subgraph.nodes) == 1
        assert subgraph.nodes[0].id == "A"

        # Depending on implementation, edge_ab might still be in subgraph.edges
        # because B is in included_node_ids.
        assert edge_ab in subgraph.edges
