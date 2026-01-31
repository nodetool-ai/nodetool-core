"""
Tests for the CollectionTools class.
"""

import pytest


class TestCollectionToolsFunctions:
    """Test get_tool_functions."""

    def test_get_tool_functions_returns_correct_functions(self):
        """Test that get_tool_functions returns expected functions."""
        from nodetool.tools.collection_tools import CollectionTools

        funcs = CollectionTools.get_tool_functions()
        assert "list_collections" in funcs
        assert "get_collection" in funcs
        assert "query_collection" in funcs
        assert "get_documents_from_collection" in funcs

    def test_get_tool_functions_are_callable(self):
        """Test that all returned functions are callable."""
        from nodetool.tools.collection_tools import CollectionTools

        funcs = CollectionTools.get_tool_functions()
        for name, func in funcs.items():
            assert callable(func), f"{name} should be callable"

    def test_get_tool_functions_count(self):
        """Test that exactly 4 functions are returned."""
        from nodetool.tools.collection_tools import CollectionTools

        funcs = CollectionTools.get_tool_functions()
        assert len(funcs) == 4
