"""
Tests for the WorkflowTools class.
"""

import pytest


class TestWorkflowToolsFunctions:
    """Test get_tool_functions."""

    def test_get_tool_functions_returns_correct_functions(self):
        """Test that get_tool_functions returns expected functions."""
        from nodetool.tools.workflow_tools import WorkflowTools

        funcs = WorkflowTools.get_tool_functions()
        assert "get_workflow" in funcs
        assert "create_workflow" in funcs
        assert "run_workflow_tool" in funcs
        assert "run_graph" in funcs
        assert "list_workflows" in funcs
        assert "get_example_workflow" in funcs
        assert "validate_workflow" in funcs
        assert "generate_dot_graph" in funcs
        assert "export_workflow_digraph" in funcs

    def test_get_tool_functions_are_callable(self):
        """Test that all returned functions are callable."""
        from nodetool.tools.workflow_tools import WorkflowTools

        funcs = WorkflowTools.get_tool_functions()
        for name, func in funcs.items():
            assert callable(func), f"{name} should be callable"

    def test_get_tool_functions_count(self):
        """Test that exactly 9 functions are returned."""
        from nodetool.tools.workflow_tools import WorkflowTools

        funcs = WorkflowTools.get_tool_functions()
        assert len(funcs) == 9


class TestWorkflowToolsGetWorkflow:
    """Test get_workflow function."""

    @pytest.mark.asyncio
    async def test_get_workflow_not_found(self):
        """Test that get_workflow raises ValueError for unknown workflow."""
        from nodetool.tools.workflow_tools import WorkflowTools

        with pytest.raises(ValueError, match="not found"):
            await WorkflowTools.get_workflow("nonexistent_id", user_id="test_user")


class TestWorkflowToolsGenerateDotGraph:
    """Test generate_dot_graph function."""

    @pytest.mark.asyncio
    async def test_generate_dot_graph_empty(self):
        """Test generating DOT graph from empty graph."""
        from nodetool.tools.workflow_tools import WorkflowTools

        graph = {"nodes": [], "edges": []}
        result = await WorkflowTools.generate_dot_graph(graph, graph_name="test")

        assert "dot" in result
        assert "test" in result["dot"]
        assert result["node_count"] == 0
        assert result["edge_count"] == 0

    @pytest.mark.asyncio
    async def test_generate_dot_graph_with_nodes(self):
        """Test generating DOT graph with nodes."""
        from nodetool.tools.workflow_tools import WorkflowTools

        graph = {
            "nodes": [
                {"id": "node1", "type": "test.Node1", "data": {}},
                {"id": "node2", "type": "test.Node2", "data": {}},
            ],
            "edges": [{"source": "node1", "target": "node2", "sourceHandle": "output", "targetHandle": "input"}],
        }
        result = await WorkflowTools.generate_dot_graph(graph, graph_name="test_workflow")

        assert "dot" in result
        assert result["node_count"] == 2
        assert result["edge_count"] == 1
