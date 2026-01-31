"""
Tests for the JobTools class.
"""

import pytest


class TestJobToolsFunctions:
    """Test get_tool_functions."""

    def test_get_tool_functions_returns_correct_functions(self):
        """Test that get_tool_functions returns expected functions."""
        from nodetool.tools.job_tools import JobTools

        funcs = JobTools.get_tool_functions()
        assert "list_jobs" in funcs
        assert "get_job" in funcs
        assert "get_job_logs" in funcs
        assert "start_background_job" in funcs

    def test_get_tool_functions_are_callable(self):
        """Test that all returned functions are callable."""
        from nodetool.tools.job_tools import JobTools

        funcs = JobTools.get_tool_functions()
        for name, func in funcs.items():
            assert callable(func), f"{name} should be callable"

    def test_get_tool_functions_count(self):
        """Test that exactly 4 functions are returned."""
        from nodetool.tools.job_tools import JobTools

        funcs = JobTools.get_tool_functions()
        assert len(funcs) == 4
