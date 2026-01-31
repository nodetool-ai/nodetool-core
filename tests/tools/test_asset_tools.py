"""
Tests for the AssetTools class.
"""

import pytest


class TestAssetToolsFunctions:
    """Test get_tool_functions."""

    def test_get_tool_functions_returns_correct_functions(self):
        """Test that get_tool_functions returns expected functions."""
        from nodetool.tools.asset_tools import AssetTools

        funcs = AssetTools.get_tool_functions()
        assert "list_assets" in funcs
        assert "get_asset" in funcs

    def test_get_tool_functions_are_callable(self):
        """Test that all returned functions are callable."""
        from nodetool.tools.asset_tools import AssetTools

        funcs = AssetTools.get_tool_functions()
        for name, func in funcs.items():
            assert callable(func), f"{name} should be callable"

    def test_get_tool_functions_count(self):
        """Test that exactly 2 functions are returned."""
        from nodetool.tools.asset_tools import AssetTools

        funcs = AssetTools.get_tool_functions()
        assert len(funcs) == 2


class TestAssetToolsListAssets:
    """Test list_assets validation."""

    @pytest.mark.asyncio
    async def test_list_assets_short_query_raises_error(self):
        """Test that search query too short raises ValueError."""
        from nodetool.tools.asset_tools import AssetTools

        with pytest.raises(ValueError, match="at least 2 characters"):
            await AssetTools.list_assets(source="user", query="a", user_id="test")
