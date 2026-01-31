"""
Tests for the ModelTools class.
"""

import pytest


class TestModelToolsFunctions:
    """Test get_tool_functions."""

    def test_get_tool_functions_returns_correct_functions(self):
        """Test that get_tool_functions returns expected functions."""
        from nodetool.tools.model_tools import ModelTools

        funcs = ModelTools.get_tool_functions()
        assert "list_models" in funcs

    def test_get_tool_functions_are_callable(self):
        """Test that all returned functions are callable."""
        from nodetool.tools.model_tools import ModelTools

        funcs = ModelTools.get_tool_functions()
        for name, func in funcs.items():
            assert callable(func), f"{name} should be callable"

    def test_get_tool_functions_count(self):
        """Test that exactly 1 function is returned."""
        from nodetool.tools.model_tools import ModelTools

        funcs = ModelTools.get_tool_functions()
        assert len(funcs) == 1
