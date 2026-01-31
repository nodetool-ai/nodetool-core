"""
Tests for the HfTools class.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestHfToolsFunctions:
    """Test get_tool_functions."""

    def test_get_tool_functions_returns_correct_functions(self):
        """Test that get_tool_functions returns expected functions."""
        from nodetool.tools.hf_tools import HfTools

        funcs = HfTools.get_tool_functions()
        assert "get_hf_cache_info" in funcs
        assert "inspect_hf_cached_model" in funcs
        assert "query_hf_model_files" in funcs
        assert "search_hf_hub_models" in funcs
        assert "get_hf_model_info" in funcs

    def test_get_tool_functions_are_callable(self):
        """Test that all returned functions are callable."""
        from nodetool.tools.hf_tools import HfTools

        funcs = HfTools.get_tool_functions()
        for name, func in funcs.items():
            assert callable(func), f"{name} should be callable"

    def test_get_tool_functions_count(self):
        """Test that exactly 5 functions are returned."""
        from nodetool.tools.hf_tools import HfTools

        funcs = HfTools.get_tool_functions()
        assert len(funcs) == 5


class TestInspectHfCachedModel:
    """Tests for inspect_hf_cached_model function."""

    @pytest.mark.asyncio
    async def test_model_not_found(self):
        """Test error when model is not found in cache."""
        from nodetool.tools.hf_tools import HfTools

        with patch(
            "nodetool.tools.hf_tools.read_cached_hf_models",
            return_value=[],
        ):
            with pytest.raises(ValueError, match="not found in cache"):
                await HfTools.inspect_hf_cached_model("nonexistent/model")

    @pytest.mark.asyncio
    async def test_model_found(self):
        """Test successful model inspection."""
        from nodetool.tools.hf_tools import HfTools

        # Create a mock model object
        mock_model = MagicMock()
        mock_model.repo_id = "test/model"
        mock_model.name = "Test Model"
        mock_model.type = "model"
        mock_model.path = "/path/to/model"
        mock_model.size_on_disk = 1024 * 1024 * 1024  # 1 GB
        mock_model.downloaded = True

        with patch(
            "nodetool.tools.hf_tools.read_cached_hf_models",
            return_value=[mock_model],
        ):
            result = await HfTools.inspect_hf_cached_model("test/model")

        assert result["repo_id"] == "test/model"
        assert result["name"] == "Test Model"
        assert result["downloaded"] is True
        assert result["size_on_disk_gb"] == 1.0


class TestSearchHfHubModels:
    """Tests for search_hf_hub_models function."""

    def test_function_is_async(self):
        """Test that search_hf_hub_models is an async function."""
        from nodetool.tools.hf_tools import HfTools
        import inspect

        assert inspect.iscoroutinefunction(HfTools.search_hf_hub_models)


class TestGetHfCacheInfo:
    """Tests for get_hf_cache_info function."""

    @pytest.mark.asyncio
    async def test_returns_cache_summary(self):
        """Test that cache info includes expected fields."""
        from nodetool.tools.hf_tools import HfTools

        # Create mock models
        mock_model = MagicMock()
        mock_model.repo_id = "test/model"
        mock_model.type = "model"
        mock_model.size_on_disk = 1024
        mock_model.path = "/path"

        with patch(
            "nodetool.tools.hf_tools.read_cached_hf_models",
            return_value=[mock_model],
        ):
            result = await HfTools.get_hf_cache_info()

        assert "cache_dir" in result
        assert "total_models" in result
        assert "total_size_bytes" in result
        assert "total_size_gb" in result
        assert "models" in result
        assert result["total_models"] == 1
        assert result["total_size_bytes"] == 1024
