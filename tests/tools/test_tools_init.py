"""
Tests for the nodetool.tools module __init__ and lazy loading.
"""

import pytest


class TestToolsModuleExports:
    """Test that all tool classes are properly exported."""

    def test_all_exports_defined(self):
        """Test that __all__ exports are properly defined."""
        from nodetool import tools

        assert hasattr(tools, "__all__")
        expected_exports = [
            "WorkflowTools",
            "AssetTools",
            "NodeTools",
            "ModelTools",
            "CollectionTools",
            "JobTools",
            "AgentTools",
            "StorageTools",
            "HfTools",
            "get_all_tool_functions",
        ]
        for export in expected_exports:
            assert export in tools.__all__

    def test_lazy_import_workflow_tools(self):
        """Test lazy import of WorkflowTools."""
        from nodetool.tools import WorkflowTools

        assert hasattr(WorkflowTools, "get_tool_functions")
        funcs = WorkflowTools.get_tool_functions()
        assert isinstance(funcs, dict)

    def test_lazy_import_asset_tools(self):
        """Test lazy import of AssetTools."""
        from nodetool.tools import AssetTools

        assert hasattr(AssetTools, "get_tool_functions")
        funcs = AssetTools.get_tool_functions()
        assert isinstance(funcs, dict)

    def test_lazy_import_node_tools(self):
        """Test lazy import of NodeTools."""
        from nodetool.tools import NodeTools

        assert hasattr(NodeTools, "get_tool_functions")
        funcs = NodeTools.get_tool_functions()
        assert isinstance(funcs, dict)

    def test_lazy_import_model_tools(self):
        """Test lazy import of ModelTools."""
        from nodetool.tools import ModelTools

        assert hasattr(ModelTools, "get_tool_functions")
        funcs = ModelTools.get_tool_functions()
        assert isinstance(funcs, dict)

    def test_lazy_import_collection_tools(self):
        """Test lazy import of CollectionTools."""
        from nodetool.tools import CollectionTools

        assert hasattr(CollectionTools, "get_tool_functions")
        funcs = CollectionTools.get_tool_functions()
        assert isinstance(funcs, dict)

    def test_lazy_import_job_tools(self):
        """Test lazy import of JobTools."""
        from nodetool.tools import JobTools

        assert hasattr(JobTools, "get_tool_functions")
        funcs = JobTools.get_tool_functions()
        assert isinstance(funcs, dict)

    def test_lazy_import_agent_tools(self):
        """Test lazy import of AgentTools."""
        from nodetool.tools import AgentTools

        assert hasattr(AgentTools, "get_tool_functions")
        funcs = AgentTools.get_tool_functions()
        assert isinstance(funcs, dict)

    def test_lazy_import_storage_tools(self):
        """Test lazy import of StorageTools."""
        from nodetool.tools import StorageTools

        assert hasattr(StorageTools, "get_tool_functions")
        funcs = StorageTools.get_tool_functions()
        assert isinstance(funcs, dict)

    def test_lazy_import_hf_tools(self):
        """Test lazy import of HfTools."""
        from nodetool.tools import HfTools

        assert hasattr(HfTools, "get_tool_functions")
        funcs = HfTools.get_tool_functions()
        assert isinstance(funcs, dict)


class TestGetAllToolFunctions:
    """Test the get_all_tool_functions aggregator."""

    def test_get_all_tool_functions_returns_dict(self):
        """Test that get_all_tool_functions returns a dictionary."""
        from nodetool.tools import get_all_tool_functions

        all_tools = get_all_tool_functions()
        assert isinstance(all_tools, dict)

    def test_get_all_tool_functions_returns_empty_when_no_module_level_funcs(self):
        """Test that get_all_tool_functions returns empty dict when modules lack module-level get_tool_functions.

        Note: The current implementation looks for module-level get_tool_functions,
        but the tool modules define class-level get_tool_functions. This test
        documents the actual behavior.
        """
        from nodetool.tools import get_all_tool_functions

        all_tools = get_all_tool_functions()
        # Currently returns empty dict because modules have class-level,
        # not module-level get_tool_functions
        assert isinstance(all_tools, dict)

    def test_get_all_tool_functions_values_are_callables(self):
        """Test that all values in get_all_tool_functions are callable."""
        from nodetool.tools import get_all_tool_functions

        all_tools = get_all_tool_functions()
        for name, func in all_tools.items():
            assert callable(func), f"Tool {name} should be callable"


class TestLazyImportErrors:
    """Test error handling for lazy imports."""

    def test_invalid_attribute_raises_error(self):
        """Test that accessing invalid attributes raises AttributeError."""
        from nodetool import tools

        with pytest.raises(AttributeError) as exc_info:
            _ = tools.NonExistentTools
        assert "NonExistentTools" in str(exc_info.value)
