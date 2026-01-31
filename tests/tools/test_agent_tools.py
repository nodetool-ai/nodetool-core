"""
Tests for the AgentTools class.
"""

import pytest


class TestAgentToolsFunctions:
    """Test get_tool_functions."""

    def test_get_tool_functions_returns_correct_functions(self):
        """Test that get_tool_functions returns expected functions."""
        from nodetool.tools.agent_tools import AgentTools

        funcs = AgentTools.get_tool_functions()
        assert "run_agent" in funcs
        assert "run_web_research_agent" in funcs
        assert "run_email_agent" in funcs

    def test_get_tool_functions_are_callable(self):
        """Test that all returned functions are callable."""
        from nodetool.tools.agent_tools import AgentTools

        funcs = AgentTools.get_tool_functions()
        for name, func in funcs.items():
            assert callable(func), f"{name} should be callable"

    def test_get_tool_functions_count(self):
        """Test that exactly 3 functions are returned."""
        from nodetool.tools.agent_tools import AgentTools

        funcs = AgentTools.get_tool_functions()
        assert len(funcs) == 3
