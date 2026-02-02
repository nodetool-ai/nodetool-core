"""
Unit tests for SubGraphController.
"""

import asyncio
from typing import Any

import pytest
from pydantic import Field

from nodetool.agents.tools.sub_graph_controller import SubGraphController
from nodetool.types.api_graph import Edge, Graph as ApiGraph, Node
from nodetool.workflows.base_node import BaseNode, ToolResultNode
from nodetool.workflows.processing_context import ProcessingContext


class SimpleEchoNode(BaseNode):
    """A simple node that echoes its input."""

    input_text: str = Field(default="", description="Input text to echo")

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """Echo the input."""
        return {"output": self.input_text}


class StatefulCounterNode(BaseNode):
    """A node that maintains state across invocations."""

    _count: int = 0
    increment: int = Field(default=1, description="Amount to increment")

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """Increment and return count."""
        self._count += self.increment
        return {"count": self._count}


class TestSubGraphController:
    """Tests for SubGraphController lifecycle."""

    @pytest.fixture
    def processing_context(self):
        """Create a basic processing context."""
        return ProcessingContext(user_id="test_user", auth_token="test_token")

    @pytest.fixture
    def simple_api_graph(self):
        """Create a simple graph with one node."""
        return ApiGraph(
            nodes=[
                Node(
                    id="echo_node",
                    type=SimpleEchoNode.get_node_type(),
                    data={"input_text": ""},
                    parent_id=None,
                    ui_properties={},
                    dynamic_properties={},
                    dynamic_outputs={},
                ),
                Node(
                    id="result_node",
                    type=ToolResultNode.get_node_type(),
                    data={},
                    parent_id=None,
                    ui_properties={},
                    dynamic_properties={},
                    dynamic_outputs={},
                ),
            ],
            edges=[
                Edge(
                    id="edge_1",
                    source="echo_node",
                    target="result_node",
                    sourceHandle="output",
                    targetHandle="output",
                    ui_properties={},
                ),
            ],
        )

    @pytest.fixture
    def input_edges(self):
        """Create input edges for injection."""
        return [
            Edge(
                id="input_edge",
                source="external",
                target="echo_node",
                sourceHandle="value",
                targetHandle="input_text",
                ui_properties={},
            ),
        ]

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(
        self, simple_api_graph, input_edges, processing_context
    ):
        """Test that the controller can be started and stopped."""
        controller = SubGraphController(
            api_graph=simple_api_graph,
            input_edges=input_edges,
            parent_context=processing_context,
        )

        # Initially not running
        assert not controller.is_running

        # Start
        await controller.start()
        assert controller.is_running

        # Wait briefly for background task to be up
        await asyncio.sleep(0.1)

        # Stop
        await controller.stop()
        assert not controller.is_running

    @pytest.mark.asyncio
    async def test_cannot_start_twice(
        self, simple_api_graph, input_edges, processing_context
    ):
        """Test that starting twice raises an error."""
        controller = SubGraphController(
            api_graph=simple_api_graph,
            input_edges=input_edges,
            parent_context=processing_context,
        )

        await controller.start()

        with pytest.raises(RuntimeError, match="already started"):
            await controller.start()

        await controller.stop()

    @pytest.mark.asyncio
    async def test_cannot_restart_after_stop(
        self, simple_api_graph, input_edges, processing_context
    ):
        """Test that a stopped controller cannot be restarted."""
        controller = SubGraphController(
            api_graph=simple_api_graph,
            input_edges=input_edges,
            parent_context=processing_context,
        )

        await controller.start()
        await controller.stop()

        with pytest.raises(RuntimeError, match="already stopped"):
            await controller.start()

    @pytest.mark.asyncio
    async def test_context_manager(
        self, simple_api_graph, input_edges, processing_context
    ):
        """Test async context manager usage."""
        controller = SubGraphController(
            api_graph=simple_api_graph,
            input_edges=input_edges,
            parent_context=processing_context,
        )

        async with controller:
            assert controller.is_running

        assert not controller.is_running

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(
        self, simple_api_graph, input_edges, processing_context
    ):
        """Test that calling stop multiple times is safe."""
        controller = SubGraphController(
            api_graph=simple_api_graph,
            input_edges=input_edges,
            parent_context=processing_context,
        )

        await controller.start()

        # Stop multiple times should not raise
        await controller.stop()
        await controller.stop()
        await controller.stop()

        assert not controller.is_running

    @pytest.mark.asyncio
    async def test_inject_before_start_raises(
        self, simple_api_graph, input_edges, processing_context
    ):
        """Test that injecting before start raises an error."""
        controller = SubGraphController(
            api_graph=simple_api_graph,
            input_edges=input_edges,
            parent_context=processing_context,
        )

        with pytest.raises(RuntimeError, match="not started"):
            await controller.inject_and_wait({"input_text": "test"})

    @pytest.mark.asyncio
    async def test_inject_after_stop_raises(
        self, simple_api_graph, input_edges, processing_context
    ):
        """Test that injecting after stop raises an error."""
        controller = SubGraphController(
            api_graph=simple_api_graph,
            input_edges=input_edges,
            parent_context=processing_context,
        )

        await controller.start()
        await controller.stop()

        with pytest.raises(RuntimeError, match="already stopped"):
            await controller.inject_and_wait({"input_text": "test"})
