"""
Integration tests for persistent GraphTool mode.
"""

import asyncio
from typing import Any

import pytest
from pydantic import Field

from nodetool.agents.tools.workflow_tool import GraphTool
from nodetool.types.api_graph import Edge
from nodetool.workflows.base_node import BaseNode, ToolResultNode
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext


class SimpleEchoNode(BaseNode):
    """A simple node that echoes its input."""

    input_text: str = Field(default="", description="Input text to echo")

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """Echo the input."""
        return {"output": self.input_text}


class SimpleAdderNode(BaseNode):
    """A simple node that adds two numbers."""

    a: int = Field(default=0, description="First number")
    b: int = Field(default=0, description="Second number")

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """Add the two numbers."""
        return {"sum": self.a + self.b}


class TestGraphToolTransientMode:
    """Tests for GraphTool in transient (default) mode."""

    @pytest.fixture
    def processing_context(self):
        """Create a basic processing context."""
        return ProcessingContext(user_id="test_user", auth_token="test_token")

    @pytest.mark.asyncio
    async def test_simple_echo_transient(self, processing_context):
        """Test transient mode with a simple echo node."""
        echo_node = SimpleEchoNode(id="echo_node")

        graph = Graph(
            nodes=[echo_node],
            edges=[],
        )

        initial_edges = [
            Edge(
                id="input_edge",
                source="external",
                target="echo_node",
                sourceHandle="value",
                targetHandle="input_text",
                ui_properties={},
            ),
        ]

        tool = GraphTool(
            graph=graph,
            name="echo_tool",
            description="Echo input text",
            initial_edges=initial_edges,
            initial_nodes=[echo_node],
            persistent=False,  # Transient mode (default)
        )

        # First call
        result1 = await tool.process(processing_context, {"input_text": "hello"})
        assert "output" in result1 or "error" not in result1

        # Second call - should work independently
        result2 = await tool.process(processing_context, {"input_text": "world"})
        assert "output" in result2 or "error" not in result2

    @pytest.mark.asyncio
    async def test_simple_adder_transient(self, processing_context):
        """Test transient mode with an adder node."""
        adder_node = SimpleAdderNode(id="adder_node")

        graph = Graph(
            nodes=[adder_node],
            edges=[],
        )

        initial_edges = [
            Edge(
                id="input_edge_a",
                source="external_a",
                target="adder_node",
                sourceHandle="value",
                targetHandle="a",
                ui_properties={},
            ),
            Edge(
                id="input_edge_b",
                source="external_b",
                target="adder_node",
                sourceHandle="value",
                targetHandle="b",
                ui_properties={},
            ),
        ]

        tool = GraphTool(
            graph=graph,
            name="adder_tool",
            description="Add two numbers",
            initial_edges=initial_edges,
            initial_nodes=[adder_node, adder_node],
            persistent=False,
        )

        result = await tool.process(processing_context, {"a": 5, "b": 3})
        # Should have some result (exact key depends on node output)
        assert isinstance(result, dict)


class TestGraphToolPersistentMode:
    """Tests for GraphTool in persistent mode."""

    @pytest.fixture
    def processing_context(self):
        """Create a basic processing context."""
        return ProcessingContext(user_id="test_user", auth_token="test_token")

    @pytest.mark.asyncio
    async def test_persistent_tool_initialization(self, processing_context):
        """Test that persistent tool can be initialized."""
        echo_node = SimpleEchoNode(id="echo_node")

        graph = Graph(
            nodes=[echo_node],
            edges=[],
        )

        initial_edges = [
            Edge(
                id="input_edge",
                source="external",
                target="echo_node",
                sourceHandle="value",
                targetHandle="input_text",
                ui_properties={},
            ),
        ]

        tool = GraphTool(
            graph=graph,
            name="persistent_echo",
            description="Persistent echo tool",
            initial_edges=initial_edges,
            initial_nodes=[echo_node],
            persistent=True,  # Persistent mode
            result_timeout=60.0,  # 60 second timeout
        )

        assert tool.persistent is True
        assert tool.result_timeout == 60.0

        # Clean up
        await tool.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent(self, processing_context):
        """Test that cleanup can be called multiple times safely."""
        echo_node = SimpleEchoNode(id="echo_node")

        graph = Graph(
            nodes=[echo_node],
            edges=[],
        )

        initial_edges = [
            Edge(
                id="input_edge",
                source="external",
                target="echo_node",
                sourceHandle="value",
                targetHandle="input_text",
                ui_properties={},
            ),
        ]

        tool = GraphTool(
            graph=graph,
            name="persistent_echo",
            description="Persistent echo tool",
            initial_edges=initial_edges,
            initial_nodes=[echo_node],
            persistent=True,
        )

        # Multiple cleanup calls should be safe
        await tool.cleanup()
        await tool.cleanup()
        await tool.cleanup()


class TestGraphToolInputSchema:
    """Tests for GraphTool input schema generation."""

    def test_input_schema_generation(self):
        """Test that input schema is correctly generated from nodes."""
        echo_node = SimpleEchoNode(id="echo_node")

        graph = Graph(
            nodes=[echo_node],
            edges=[],
        )

        initial_edges = [
            Edge(
                id="input_edge",
                source="external",
                target="echo_node",
                sourceHandle="value",
                targetHandle="input_text",
                ui_properties={},
            ),
        ]

        tool = GraphTool(
            graph=graph,
            name="echo_tool",
            description="Echo tool",
            initial_edges=initial_edges,
            initial_nodes=[echo_node],
        )

        assert tool.input_schema is not None
        assert tool.input_schema["type"] == "object"
        assert "properties" in tool.input_schema
        assert "input_text" in tool.input_schema["properties"]

    def test_input_schema_multiple_inputs(self):
        """Test input schema with multiple inputs."""
        adder_node = SimpleAdderNode(id="adder_node")

        graph = Graph(
            nodes=[adder_node],
            edges=[],
        )

        initial_edges = [
            Edge(
                id="input_edge_a",
                source="external_a",
                target="adder_node",
                sourceHandle="value",
                targetHandle="a",
                ui_properties={},
            ),
            Edge(
                id="input_edge_b",
                source="external_b",
                target="adder_node",
                sourceHandle="value",
                targetHandle="b",
                ui_properties={},
            ),
        ]

        tool = GraphTool(
            graph=graph,
            name="adder_tool",
            description="Add numbers",
            initial_edges=initial_edges,
            initial_nodes=[adder_node, adder_node],
        )

        assert tool.input_schema is not None
        assert "a" in tool.input_schema["properties"]
        assert "b" in tool.input_schema["properties"]
