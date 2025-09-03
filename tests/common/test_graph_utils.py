"""
Tests for graph utilities.
"""

import unittest
from unittest.mock import MagicMock, patch

from nodetool.common.graph_utils import (
    find_node,
    get_node_input_types,
    get_downstream_subgraph,
)
from nodetool.types.graph import Edge


class TestGraphUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create mock nodes
        self.node1 = MagicMock()
        self.node1.id = "node1"
        self.node1.outputs.return_value = [MagicMock(name="output1", type="StringType")]

        self.node2 = MagicMock()
        self.node2.id = "node2"
        self.node2.outputs.return_value = [MagicMock(name="output2", type="NumberType")]

        # Create mock graph
        self.graph = MagicMock()
        self.graph.nodes = [self.node1, self.node2]
        self.graph.edges = [
            Edge(
                source="node1",
                sourceHandle="output1",
                target="node2",
                targetHandle="input1",
            )
        ]
        self.graph.find_node.side_effect = lambda node_id: {
            "node1": self.node1,
            "node2": self.node2,
        }.get(node_id)

    def test_find_node_returns_node_when_exists(self):
        """Test that find_node returns the correct node when it exists."""
        result = find_node(self.graph, "node1")
        self.assertEqual(result, self.node1)
        self.graph.find_node.assert_called_once_with("node1")

    def test_find_node_raises_value_error_when_node_not_exists(self):
        """Test that find_node raises ValueError when node doesn't exist."""
        self.graph.find_node.return_value = None
        with self.assertRaises(ValueError) as context:
            find_node(self.graph, "nonexistent")
        self.assertIn("Node with ID nonexistent does not exist", str(context.exception))

    def test_get_node_input_types_returns_correct_types(self):
        """Test that get_node_input_types returns correct input types."""
        # Create a proper mock output object
        mock_output = MagicMock()
        mock_output.name = "output1"
        mock_output.type = "StringType"
        self.node1.outputs.return_value = [mock_output]
        result = get_node_input_types(self.graph, "node2")
        expected = {"input1": "StringType"}
        self.assertEqual(result, expected)

    def test_get_node_input_types_returns_none_for_missing_outputs(self):
        """Test that get_node_input_types handles missing outputs gracefully."""
        # Mock a node with no outputs
        self.node1.outputs.return_value = []
        result = get_node_input_types(self.graph, "node2")
        expected = {"input1": None}
        self.assertEqual(result, expected)

    def test_get_node_input_types_returns_empty_dict_for_no_edges(self):
        """Test that get_node_input_types returns empty dict when no edges target the node."""
        result = get_node_input_types(self.graph, "node1")
        self.assertEqual(result, {})

    @patch("nodetool.common.graph_utils.Graph")
    def test_get_downstream_subgraph_returns_correct_subgraph(self, mock_graph_class):
        """Test that get_downstream_subgraph returns correct subgraph."""
        # Create a more complex graph for testing downstream
        node3 = MagicMock()
        node3.id = "node3"

        complex_graph = MagicMock()
        complex_graph.nodes = [self.node1, self.node2, node3]
        complex_graph.edges = [
            Edge(
                source="node1",
                sourceHandle="output1",
                target="node2",
                targetHandle="input1",
            ),
            Edge(
                source="node2",
                sourceHandle="output2",
                target="node3",
                targetHandle="input2",
            ),
        ]
        complex_graph.find_node.side_effect = lambda node_id: {
            "node1": self.node1,
            "node2": self.node2,
            "node3": node3,
        }.get(node_id)

        # Mock the Graph constructor
        mock_subgraph = MagicMock()
        mock_subgraph.edges = [
            Edge(
                source="node1",
                sourceHandle="output1",
                target="node2",
                targetHandle="input1",
            ),
            Edge(
                source="node2",
                sourceHandle="output2",
                target="node3",
                targetHandle="input2",
            ),
        ]
        mock_subgraph.nodes = [self.node1, self.node2, node3]
        mock_graph_class.return_value = mock_subgraph

        initial_edges, subgraph = get_downstream_subgraph(
            complex_graph, "node1", "output1"
        )

        # Should have 2 edges in the subgraph
        self.assertEqual(len(subgraph.edges), 2)
        # Should have all 3 nodes
        self.assertEqual(len(subgraph.nodes), 3)

    @patch("nodetool.common.graph_utils.Graph")
    def test_get_downstream_subgraph_handles_missing_nodes(self, mock_graph_class):
        """Test that get_downstream_subgraph handles missing nodes gracefully."""
        # Mock the Graph constructor
        mock_subgraph = MagicMock()
        mock_subgraph.edges = [
            Edge(
                source="node1",
                sourceHandle="output1",
                target="node2",
                targetHandle="input1",
            )
        ]
        mock_subgraph.nodes = [self.node1]  # Only node1 since node2 is missing
        mock_graph_class.return_value = mock_subgraph

        # Mock graph.find_node to return None for node2 (simulate missing node)
        self.graph.find_node.side_effect = lambda node_id: {
            "node1": self.node1,
            "node2": None,  # Simulate missing node
        }.get(node_id)

        initial_edges, subgraph = get_downstream_subgraph(
            self.graph, "node1", "output1"
        )

        # Should still work but with fewer nodes
        self.assertEqual(len(subgraph.edges), 1)
        self.assertEqual(len(subgraph.nodes), 1)  # Only node1 should be included


if __name__ == "__main__":
    unittest.main()
