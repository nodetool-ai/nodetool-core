import pytest
from pydantic import ValidationError

from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode, add_node_type
from nodetool.workflows.graph import Graph


class ValidTestNode(BaseNode):
    @classmethod
    def get_node_type(cls):
        return "tests.workflows.ValidTestNode"

class StrictNode(BaseNode):
    int_val: int = 0

    @classmethod
    def get_node_type(cls):
        return "tests.workflows.StrictNode"

class TestGraphFromDictErrors:
    """
    Test suite for error handling in Graph.from_dict.
    Focuses on scenarios where skip_errors=False.
    """

    @classmethod
    def setup_class(cls):
        # Nodes are automatically registered by __init_subclass__, but we can be explicit if needed.
        # Here we rely on the class definitions above having registered them.
        # But to be safe and ensure idempotency if we add explicit registration calls:
        add_node_type(ValidTestNode)
        add_node_type(StrictNode)

    @classmethod
    def teardown_class(cls):
        # Cleanup registered nodes to avoid polluting global state
        if "tests.workflows.ValidTestNode" in NODE_BY_TYPE:
            del NODE_BY_TYPE["tests.workflows.ValidTestNode"]
        if "tests.workflows.StrictNode" in NODE_BY_TYPE:
            del NODE_BY_TYPE["tests.workflows.StrictNode"]

    def test_from_dict_raises_on_invalid_node_type(self):
        """Test that from_dict raises ValueError when a node type is invalid and skip_errors=False."""
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

        with pytest.raises(ValueError, match="Invalid node type"):
            Graph.from_dict(graph_data, skip_errors=False)

    def test_from_dict_raises_on_invalid_property(self):
        """Test that from_dict raises ValueError when a property value is invalid and skip_errors=False."""
        graph_data = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "tests.workflows.StrictNode",
                    "data": {"int_val": "not_an_int"}, # Invalid value for int
                }
            ],
            "edges": [],
        }

        # BaseNode.assign_property returns error message for type mismatch.
        # set_node_properties raises ValueError if skip_errors=False and error message returned.

        with pytest.raises(ValueError, match="Error setting property 'int_val'"):
            Graph.from_dict(graph_data, skip_errors=False)

    def test_from_dict_raises_on_malformed_edge(self):
        """Test that from_dict raises ValidationError when an edge is missing required fields and skip_errors=False."""
        graph_data = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "tests.workflows.ValidTestNode",
                    "data": {},
                },
                {
                    "id": "node2",
                    "type": "tests.workflows.ValidTestNode",
                    "data": {},
                }
            ],
            "edges": [
                {
                    # Missing sourceHandle, targetHandle
                    "source": "node1",
                    "target": "node2",
                    "id": "edge1",
                }
            ],
        }

        # In Graph.from_dict, if skip_errors=False, the malformed edge dict is passed to Graph constructor.
        # Graph constructor validates edges list against List[Edge].
        # Pydantic should raise ValidationError.

        with pytest.raises(ValidationError):
            Graph.from_dict(graph_data, skip_errors=False)

    def test_from_dict_on_edge_with_missing_node(self):
        """
        Test behavior when edge connects to non-existent node with skip_errors=False.
        Current implementation appends the edge and lets downstream handle it.
        So Graph should be created, but contain an edge pointing to nowhere.
        """
        graph_data = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "tests.workflows.ValidTestNode",
                    "data": {},
                }
            ],
            "edges": [
                {
                    "source": "node1",
                    "sourceHandle": "output",
                    "target": "node2", # Non-existent
                    "targetHandle": "input",
                    "id": "edge1",
                }
            ],
        }

        graph = Graph.from_dict(graph_data, skip_errors=False)
        assert len(graph.edges) == 1
        assert graph.edges[0].target == "node2"

        # Verify validation fails
        errors = graph.validate_edge_types()
        assert any("node2" in e and "not found" in e for e in errors)
