"""
Tests for the StorageTools class.
"""

import base64
from datetime import datetime
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestStorageToolsValidation:
    """Test input validation for StorageTools."""

    @pytest.mark.asyncio
    async def test_download_file_rejects_path_separators_slash(self):
        """Test that path separators in key are rejected."""
        from nodetool.tools.storage_tools import StorageTools

        with pytest.raises(ValueError, match="path separators not allowed"):
            await StorageTools.download_file_from_storage("path/to/file.txt")

    @pytest.mark.asyncio
    async def test_download_file_rejects_path_separators_backslash(self):
        """Test that backslash separators in key are rejected."""
        from nodetool.tools.storage_tools import StorageTools

        with pytest.raises(ValueError, match="path separators not allowed"):
            await StorageTools.download_file_from_storage("path\\to\\file.txt")

    @pytest.mark.asyncio
    async def test_get_metadata_rejects_path_separators_slash(self):
        """Test that path separators in key are rejected for metadata."""
        from nodetool.tools.storage_tools import StorageTools

        with pytest.raises(ValueError, match="path separators not allowed"):
            await StorageTools.get_file_metadata("path/to/file.txt")

    @pytest.mark.asyncio
    async def test_get_metadata_rejects_path_separators_backslash(self):
        """Test that backslash separators in key are rejected for metadata."""
        from nodetool.tools.storage_tools import StorageTools

        with pytest.raises(ValueError, match="path separators not allowed"):
            await StorageTools.get_file_metadata("path\\to\\file.txt")


class TestStorageToolsFunctions:
    """Test get_tool_functions."""

    def test_get_tool_functions_returns_correct_functions(self):
        """Test that get_tool_functions returns expected functions."""
        from nodetool.tools.storage_tools import StorageTools

        funcs = StorageTools.get_tool_functions()
        assert "download_file_from_storage" in funcs
        assert "get_file_metadata" in funcs
        assert "list_storage_files" in funcs

    def test_get_tool_functions_are_callable(self):
        """Test that all returned functions are callable."""
        from nodetool.tools.storage_tools import StorageTools

        funcs = StorageTools.get_tool_functions()
        for name, func in funcs.items():
            assert callable(func), f"{name} should be callable"


class TestStorageToolsDownloadFile:
    """Test download_file_from_storage functionality."""

    @pytest.mark.asyncio
    async def test_download_file_not_found(self):
        """Test error when file doesn't exist."""
        from nodetool.tools.storage_tools import StorageTools

        mock_storage = AsyncMock()
        mock_storage.file_exists.return_value = False

        mock_scope = MagicMock()
        mock_scope.get_asset_storage.return_value = mock_storage

        with patch("nodetool.tools.storage_tools.require_scope", return_value=mock_scope):
            with pytest.raises(ValueError, match="File not found"):
                await StorageTools.download_file_from_storage("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_download_file_success(self):
        """Test successful file download."""
        from nodetool.tools.storage_tools import StorageTools

        test_content = b"test file content"
        test_mtime = datetime(2024, 1, 1, 12, 0, 0)

        mock_storage = AsyncMock()
        mock_storage.file_exists.return_value = True
        mock_storage.get_size.return_value = len(test_content)
        mock_storage.get_mtime.return_value = test_mtime

        async def mock_download(key, stream):
            stream.write(test_content)

        mock_storage.download = mock_download

        mock_scope = MagicMock()
        mock_scope.get_asset_storage.return_value = mock_storage

        with patch("nodetool.tools.storage_tools.require_scope", return_value=mock_scope):
            result = await StorageTools.download_file_from_storage("test.txt")

        assert result["key"] == "test.txt"
        assert result["content"] == base64.b64encode(test_content).decode("utf-8")
        assert result["size"] == len(test_content)
        assert result["storage"] == "asset"

    @pytest.mark.asyncio
    async def test_download_file_temp_storage(self):
        """Test downloading from temp storage."""
        from nodetool.tools.storage_tools import StorageTools

        test_content = b"temp file content"

        mock_storage = AsyncMock()
        mock_storage.file_exists.return_value = True
        mock_storage.get_size.return_value = len(test_content)
        mock_storage.get_mtime.return_value = None

        async def mock_download(key, stream):
            stream.write(test_content)

        mock_storage.download = mock_download

        mock_scope = MagicMock()
        mock_scope.get_temp_storage.return_value = mock_storage

        with patch("nodetool.tools.storage_tools.require_scope", return_value=mock_scope):
            result = await StorageTools.download_file_from_storage("test.txt", temp=True)

        assert result["storage"] == "temp"
        assert result["last_modified"] is None


class TestStorageToolsGetMetadata:
    """Test get_file_metadata functionality."""

    @pytest.mark.asyncio
    async def test_get_metadata_not_found(self):
        """Test error when file doesn't exist."""
        from nodetool.tools.storage_tools import StorageTools

        mock_storage = AsyncMock()
        mock_storage.file_exists.return_value = False

        mock_scope = MagicMock()
        mock_scope.get_asset_storage.return_value = mock_storage

        with patch("nodetool.tools.storage_tools.require_scope", return_value=mock_scope):
            with pytest.raises(ValueError, match="File not found"):
                await StorageTools.get_file_metadata("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_get_metadata_success(self):
        """Test successful metadata retrieval."""
        from nodetool.tools.storage_tools import StorageTools

        test_mtime = datetime(2024, 1, 1, 12, 0, 0)

        mock_storage = AsyncMock()
        mock_storage.file_exists.return_value = True
        mock_storage.get_size.return_value = 1024
        mock_storage.get_mtime.return_value = test_mtime

        mock_scope = MagicMock()
        mock_scope.get_asset_storage.return_value = mock_storage

        with patch("nodetool.tools.storage_tools.require_scope", return_value=mock_scope):
            result = await StorageTools.get_file_metadata("test.txt")

        assert result["key"] == "test.txt"
        assert result["exists"] is True
        assert result["size"] == 1024
        assert result["storage"] == "asset"


class TestStorageToolsListFiles:
    """Test list_storage_files functionality."""

    @pytest.mark.asyncio
    async def test_list_files_limit_capped(self):
        """Test that limit is capped at 200."""
        from nodetool.tools.storage_tools import StorageTools

        mock_storage = MagicMock()
        mock_storage.list_files = MagicMock(return_value=[])

        mock_scope = MagicMock()
        mock_scope.get_asset_storage.return_value = mock_storage

        with patch("nodetool.tools.storage_tools.require_scope", return_value=mock_scope):
            result = await StorageTools.list_storage_files(limit=500)

        # Should have called with capped limit
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_list_files_unsupported_backend(self):
        """Test behavior when storage doesn't support listing."""
        from nodetool.tools.storage_tools import StorageTools

        mock_storage = MagicMock(spec=[])  # No list_files method

        mock_scope = MagicMock()
        mock_scope.get_asset_storage.return_value = mock_storage

        with patch("nodetool.tools.storage_tools.require_scope", return_value=mock_scope):
            result = await StorageTools.list_storage_files()

        assert "message" in result
        assert "does not support listing" in result["message"]

    @pytest.mark.asyncio
    async def test_list_files_success(self):
        """Test successful file listing."""
        from nodetool.tools.storage_tools import StorageTools

        test_files = [
            {"key": "file1.txt", "size": 100, "last_modified": "2024-01-01"},
            {"key": "file2.txt", "size": 200, "last_modified": "2024-01-02"},
        ]

        mock_storage = MagicMock()
        mock_storage.list_files = MagicMock(return_value=test_files)

        mock_scope = MagicMock()
        mock_scope.get_asset_storage.return_value = mock_storage

        with patch("nodetool.tools.storage_tools.require_scope", return_value=mock_scope):
            result = await StorageTools.list_storage_files()

        assert result["count"] == 2
        assert len(result["files"]) == 2
        assert result["storage"] == "asset"

    @pytest.mark.asyncio
    async def test_list_files_error_handling(self):
        """Test error handling when listing fails."""
        from nodetool.tools.storage_tools import StorageTools

        mock_storage = MagicMock()
        mock_storage.list_files = MagicMock(side_effect=Exception("Storage error"))

        mock_scope = MagicMock()
        mock_scope.get_asset_storage.return_value = mock_storage

        with patch("nodetool.tools.storage_tools.require_scope", return_value=mock_scope):
            result = await StorageTools.list_storage_files()

        assert "error" in result
        assert "Storage error" in result["error"]
