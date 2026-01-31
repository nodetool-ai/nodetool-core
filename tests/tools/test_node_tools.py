"""
Tests for the NodeTools class.
"""

import pytest


class TestNodeToolsFunctions:
    """Test get_tool_functions."""

    def test_get_tool_functions_returns_correct_functions(self):
        """Test that get_tool_functions returns expected functions."""
        from nodetool.tools.node_tools import NodeTools

        funcs = NodeTools.get_tool_functions()
        assert "list_nodes" in funcs
        assert "search_nodes" in funcs
        assert "get_node_info" in funcs

    def test_get_tool_functions_are_callable(self):
        """Test that all returned functions are callable."""
        from nodetool.tools.node_tools import NodeTools

        funcs = NodeTools.get_tool_functions()
        for name, func in funcs.items():
            assert callable(func), f"{name} should be callable"

    def test_get_tool_functions_count(self):
        """Test that exactly 3 functions are returned."""
        from nodetool.tools.node_tools import NodeTools

        funcs = NodeTools.get_tool_functions()
        assert len(funcs) == 3


class TestNodeToolsListNodes:
    """Test list_nodes function."""

    @pytest.mark.asyncio
    async def test_list_nodes_returns_list(self):
        """Test that list_nodes returns a list."""
        from nodetool.tools.node_tools import NodeTools

        result = await NodeTools.list_nodes()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_nodes_with_limit(self):
        """Test that list_nodes respects limit."""
        from nodetool.tools.node_tools import NodeTools

        result = await NodeTools.list_nodes(limit=5)
        assert len(result) <= 5


class TestNodeToolsGetNodeInfo:
    """Test get_node_info function."""

    @pytest.mark.asyncio
    async def test_get_node_info_not_found(self):
        """Test that get_node_info raises ValueError for unknown node type."""
        from nodetool.tools.node_tools import NodeTools

        with pytest.raises(ValueError, match="not found"):
            await NodeTools.get_node_info("nonexistent.node.Type")
