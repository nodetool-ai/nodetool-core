"""
Tests for graph validation logic in workflows/graph.py.
"""
from typing import List

import pytest

from nodetool.types.api_graph import Edge
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.graph import Graph

# Define nodes with specific types for testing validation

class StringProducerNode(BaseNode):
    """Node that produces a string output."""
    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph_validation.StringProducerNode"

    async def process(self, context) -> str:
        return "test_string"

class IntProducerNode(BaseNode):
    """Node that produces an int output."""
    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph_validation.IntProducerNode"

    async def process(self, context) -> int:
        return 42

class StringConsumerNode(BaseNode):
    """Node that consumes a string input."""
    input_str: str = ""

    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph_validation.StringConsumerNode"

    async def process(self, context):
        return {"result": self.input_str}

class IntConsumerNode(BaseNode):
    """Node that consumes an int input."""
    input_int: int = 0

    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph_validation.IntConsumerNode"

    async def process(self, context):
        return {"result": self.input_int}

class ListStringConsumerNode(BaseNode):
    """Node that consumes a list of strings."""
    input_list: List[str] = []  # noqa: RUF012

    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_graph_validation.ListStringConsumerNode"

    async def process(self, context):
        return {"result": self.input_list}


class TestGraphEdgeValidation:
    """Test suite for Graph.validate_edge_types()."""

    def test_compatible_types(self):
        """Test validation passes for compatible types (str -> str)."""
        nodes = [
            StringProducerNode(id="producer"),
            StringConsumerNode(id="consumer"),
        ]
        edges = [
            Edge(
                source="producer",
                sourceHandle="output",
                target="consumer",
                targetHandle="input_str",
                id="edge1",
            )
        ]
        graph = Graph(nodes=nodes, edges=edges)
        errors = graph.validate_edge_types()
        assert len(errors) == 0

    def test_incompatible_types(self):
        """Test validation fails for incompatible types (str -> int)."""
        nodes = [
            StringProducerNode(id="producer"),
            IntConsumerNode(id="consumer"),
        ]
        edges = [
            Edge(
                source="producer",
                sourceHandle="output",
                target="consumer",
                targetHandle="input_int",
                id="edge1",
            )
        ]
        graph = Graph(nodes=nodes, edges=edges)
        errors = graph.validate_edge_types()

        assert len(errors) == 1
        # The exact message format depends on type metadata implementation
        # Expected: "... has incompatible type 'str' for property 'input_int' (expected 'int')"
        # Or based on reproduction: "Type mismatch ... outputs str but ... expects int"
        assert "Type mismatch" in errors[0]
        assert "outputs str" in errors[0]
        assert "expects int" in errors[0]

    def test_multi_edge_list_compatible(self):
        """Test validation passes for multiple compatible edges to a list input."""
        nodes = [
            StringProducerNode(id="producer1"),
            StringProducerNode(id="producer2"),
            ListStringConsumerNode(id="consumer"),
        ]
        edges = [
            Edge(
                source="producer1",
                sourceHandle="output",
                target="consumer",
                targetHandle="input_list",
                id="edge1",
            ),
            Edge(
                source="producer2",
                sourceHandle="output",
                target="consumer",
                targetHandle="input_list",
                id="edge2",
            ),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        errors = graph.validate_edge_types()
        assert len(errors) == 0

    def test_multi_edge_list_incompatible(self):
        """Test validation fails if one of the edges is incompatible with list element type."""
        nodes = [
            StringProducerNode(id="producer1"),
            IntProducerNode(id="producer2"),
            ListStringConsumerNode(id="consumer"),
        ]
        edges = [
            Edge(
                source="producer1",
                sourceHandle="output",
                target="consumer",
                targetHandle="input_list",
                id="edge1",
            ),
            Edge(
                source="producer2",
                sourceHandle="output",
                target="consumer",
                targetHandle="input_list",
                id="edge2",
            ),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        errors = graph.validate_edge_types()

        assert len(errors) > 0
        # Based on code: "incompatible type '{source_type.type}' for list element type '{element_type.type}'"
        assert "incompatible type 'int'" in errors[0]
        assert "list element type 'str'" in errors[0]

    def test_multi_edge_non_list_target(self):
        """Test validation fails for multiple edges targeting a non-list property."""
        nodes = [
            StringProducerNode(id="producer1"),
            StringProducerNode(id="producer2"),
            StringConsumerNode(id="consumer"),
        ]
        edges = [
            Edge(
                source="producer1",
                sourceHandle="output",
                target="consumer",
                targetHandle="input_str",
                id="edge1",
            ),
            Edge(
                source="producer2",
                sourceHandle="output",
                target="consumer",
                targetHandle="input_str",
                id="edge2",
            ),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        errors = graph.validate_edge_types()

        assert len(errors) == 1
        assert "Multiple edges target non-list property" in errors[0]
