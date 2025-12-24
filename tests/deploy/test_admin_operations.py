"""Tests for the admin operations module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.deploy.admin_operations import (
    AdminDownloadManager,
    calculate_cache_size,
    convert_cache_info,
    convert_file_info,
    convert_repo_info,
    convert_revision_info,
    delete_hf_model,
    download_hf_model,
    download_ollama_model,
    scan_hf_cache,
    stream_hf_model_download,
    stream_ollama_model_pull,
)


@pytest.fixture
def mock_hf_api():
    """Mock HuggingFace API."""
    api = MagicMock()

    # Mock file objects
    mock_file1 = MagicMock()
    mock_file1.path = "config.json"
    mock_file1.size = 1024

    mock_file2 = MagicMock()
    mock_file2.path = "model.safetensors"
    mock_file2.size = 2048000

    api.list_repo_tree.return_value = [mock_file1, mock_file2]
    return api


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client."""
    client = MagicMock()

    # Mock pull response chunks
    chunk1 = MagicMock()
    chunk1.model_dump.return_value = {"status": "downloading", "progress": 0.5}

    chunk2 = MagicMock()
    chunk2.model_dump.return_value = {"status": "downloading", "progress": 1.0}

    def mock_pull(model_name, stream=False):
        """Mock async generator for pull response."""

        async def async_gen():
            yield chunk1
            yield chunk2

        return async_gen()

    client.pull = mock_pull
    return client


class TestAdminDownloadManager:
    """Test AdminDownloadManager functionality."""

    @pytest.mark.asyncio
    async def test_download_single_file_success(self, mock_hf_api):
        """Test successful single file download."""
        manager = AdminDownloadManager()
        manager.api = mock_hf_api

        with patch("nodetool.deploy.admin_operations.hf_hub_download") as mock_download:
            mock_download.return_value = "/path/to/file.json"

            results = []
            async for chunk in manager.download_with_progress(repo_id="test/repo", file_path="config.json"):
                results.append(chunk)

            # Should have starting, progress, and completed messages
            assert len(results) >= 3
            assert results[0]["status"] == "starting"
            assert results[-1]["status"] == "completed"
            assert results[-1]["local_path"] == "/path/to/file.json"
            mock_download.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_single_file_error(self, mock_hf_api):
        """Test single file download error."""
        manager = AdminDownloadManager()
        manager.api = mock_hf_api

        with patch("nodetool.deploy.admin_operations.hf_hub_download") as mock_download:
            mock_download.side_effect = Exception("Download failed")

            results = []
            async for chunk in manager.download_with_progress(repo_id="test/repo", file_path="config.json"):
                results.append(chunk)

            # Should have starting and error messages
            assert len(results) >= 2
            assert results[0]["status"] == "starting"
            assert results[-1]["status"] == "error"
            assert "Download failed" in results[-1]["error"]

    @pytest.mark.asyncio
    async def test_download_repository_success(self, mock_hf_api):
        """Test successful repository download."""
        manager = AdminDownloadManager()
        manager.api = mock_hf_api

        with (
            patch("nodetool.deploy.admin_operations.hf_hub_download") as mock_download,
            patch("nodetool.deploy.admin_operations.try_to_load_from_cache") as mock_cache,
            patch("nodetool.deploy.admin_operations.filter_repo_paths") as mock_filter,
        ):
            # Mock cache check - no cached files
            mock_cache.return_value = None

            # Mock filter to return both files
            mock_filter.return_value = [
                MagicMock(path="config.json", size=1024),
                MagicMock(path="model.safetensors", size=2048000),
            ]

            mock_download.return_value = "/path/to/file"

            results = []
            async for chunk in manager.download_with_progress(repo_id="test/repo"):
                results.append(chunk)

            # Should have multiple progress updates and final completion
            assert len(results) >= 4
            assert results[0]["status"] == "starting"
            assert results[-1]["status"] == "completed"
            assert results[-1]["downloaded_files"] == 2
            assert mock_download.call_count == 2

    @pytest.mark.asyncio
    async def test_download_repository_all_cached(self, mock_hf_api):
        """Test repository download when all files are cached."""
        manager = AdminDownloadManager()
        manager.api = mock_hf_api

        with (
            patch("nodetool.deploy.admin_operations.try_to_load_from_cache") as mock_cache,
            patch("nodetool.deploy.admin_operations.filter_repo_paths") as mock_filter,
            patch("os.path.exists") as mock_exists,
        ):
            # Mock cache check - all files cached
            mock_cache.return_value = "/cached/path"
            mock_exists.return_value = True

            mock_filter.return_value = [
                MagicMock(path="config.json", size=1024),
                MagicMock(path="model.safetensors", size=2048000),
            ]

            results = []
            async for chunk in manager.download_with_progress(repo_id="test/repo"):
                results.append(chunk)

            # Should complete immediately as all files are cached
            assert len(results) >= 3
            assert results[0]["status"] == "starting"
            assert results[-1]["status"] == "completed"
            assert results[-1]["total_files"] == 0
            assert results[-1]["cached_files"] == 2


class TestConversionFunctions:
    """Test data conversion functions."""

    def test_convert_file_info(self):
        """Test file info conversion."""
        mock_file_info = MagicMock()
        mock_file_info.file_name = "model.bin"
        mock_file_info.size_on_disk = 1024
        mock_file_info.file_path = "/path/to/file"
        mock_file_info.blob_path = "/path/to/blob"

        result = convert_file_info(mock_file_info)

        assert result["file_name"] == "model.bin"
        assert result["size_on_disk"] == 1024
        assert result["file_path"] == "/path/to/file"
        assert result["blob_path"] == "/path/to/blob"

    def test_convert_revision_info(self):
        """Test revision info conversion."""
        mock_file_info = MagicMock()
        mock_file_info.file_name = "model.bin"
        mock_file_info.size_on_disk = 1024
        mock_file_info.file_path = "/path/to/file"
        mock_file_info.blob_path = "/path/to/blob"

        mock_revision_info = MagicMock()
        mock_revision_info.commit_hash = "abc123"
        mock_revision_info.size_on_disk = 2048
        mock_revision_info.snapshot_path = "/path/to/snapshot"
        mock_revision_info.files = [mock_file_info]

        result = convert_revision_info(mock_revision_info)

        assert result["commit_hash"] == "abc123"
        assert result["size_on_disk"] == 2048
        assert result["snapshot_path"] == "/path/to/snapshot"
        assert len(result["files"]) == 1
        assert result["files"][0]["file_name"] == "model.bin"

    def test_convert_repo_info(self):
        """Test repo info conversion."""
        mock_file_info = MagicMock()
        mock_file_info.file_name = "model.bin"
        mock_file_info.size_on_disk = 1024
        mock_file_info.file_path = "/path/to/file"
        mock_file_info.blob_path = "/path/to/blob"

        mock_revision_info = MagicMock()
        mock_revision_info.commit_hash = "abc123"
        mock_revision_info.size_on_disk = 2048
        mock_revision_info.snapshot_path = "/path/to/snapshot"
        mock_revision_info.files = [mock_file_info]

        mock_repo_info = MagicMock()
        mock_repo_info.repo_id = "test/repo"
        mock_repo_info.repo_type = "model"
        mock_repo_info.repo_path = "/path/to/repo"
        mock_repo_info.size_on_disk = 4096
        mock_repo_info.nb_files = 2
        mock_repo_info.revisions = [mock_revision_info]

        result = convert_repo_info(mock_repo_info)

        assert result["repo_id"] == "test/repo"
        assert result["repo_type"] == "model"
        assert result["size_on_disk"] == 4096
        assert result["nb_files"] == 2
        assert len(result["revisions"]) == 1

    def test_convert_cache_info(self):
        """Test cache info conversion."""
        mock_cache_info = MagicMock()
        mock_cache_info.size_on_disk = 8192
        mock_cache_info.repos = []
        mock_cache_info.warnings = ["Warning 1", "Warning 2"]

        result = convert_cache_info(mock_cache_info)

        assert result["size_on_disk"] == 8192
        assert result["repos"] == []
        assert result["warnings"] == ["Warning 1", "Warning 2"]


class TestOllamaOperations:
    """Test Ollama-related operations."""

    @pytest.mark.asyncio
    async def test_stream_ollama_model_pull_success(self, mock_ollama_client):
        """Test successful Ollama model pull."""
        with patch("nodetool.deploy.admin_operations.get_ollama_client") as mock_get_client:
            mock_get_client.return_value = mock_ollama_client

            results = []
            async for chunk in stream_ollama_model_pull("test-model"):
                results.append(chunk)

            # Should have starting, progress updates, and completion
            # Starting + 2 mock chunks + completion = 4 results
            assert len(results) == 4
            assert results[0]["status"] == "starting"
            assert results[0]["model"] == "test-model"
            assert results[1] == {"status": "downloading", "progress": 0.5}
            assert results[2] == {"status": "downloading", "progress": 1.0}
            assert results[-1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_stream_ollama_model_pull_error(self):
        """Test Ollama model pull error."""
        with patch("nodetool.deploy.admin_operations.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.pull.side_effect = Exception("Pull failed")
            mock_get_client.return_value = mock_client

            results = []
            async for chunk in stream_ollama_model_pull("test-model"):
                results.append(chunk)

            # Should have starting and error messages
            assert len(results) >= 2
            assert results[0]["status"] == "starting"
            assert results[-1]["status"] == "error"
            assert "Pull failed" in results[-1]["error"]


class TestHuggingFaceOperations:
    """Test HuggingFace-related operations."""

    @pytest.mark.asyncio
    async def test_stream_hf_model_download_success(self):
        """Test successful HF model download stream."""
        with patch("nodetool.deploy.admin_operations.AdminDownloadManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            async def mock_download(repo_id, cache_dir, file_path, ignore_patterns, allow_patterns):
                yield {"status": "starting", "repo_id": repo_id}
                yield {"status": "completed", "repo_id": repo_id}

            mock_manager.download_with_progress = mock_download

            results = []
            async for chunk in stream_hf_model_download("test/repo"):
                results.append(chunk)

            assert len(results) == 2
            assert results[0]["status"] == "starting"
            assert results[1]["status"] == "completed"


class TestIndividualAdminOperations:
    """Test individual admin operation functions."""

    @pytest.mark.asyncio
    async def test_download_hf_model_streaming(self):
        """Test HF download with streaming."""
        with patch("nodetool.deploy.admin_operations.stream_hf_model_download") as mock_stream:

            async def mock_download(repo_id, cache_dir, file_path, ignore_patterns, allow_patterns):
                yield {"status": "starting", "repo_id": repo_id}
                yield {"status": "completed", "repo_id": repo_id}

            mock_stream.side_effect = mock_download

            results = []
            async for chunk in download_hf_model(repo_id="test/repo", stream=True):
                results.append(chunk)

            assert len(results) == 2
            assert results[0]["status"] == "starting"
            assert results[1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_download_hf_model_non_streaming(self):
        """Test HF download without streaming."""
        with patch("nodetool.deploy.admin_operations.AdminDownloadManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            async def mock_download(*args, **kwargs):
                yield {"status": "starting", "repo_id": "test/repo"}
                yield {"status": "completed", "repo_id": "test/repo"}

            mock_manager.download_with_progress = mock_download

            results = []
            async for chunk in download_hf_model(repo_id="test/repo", stream=False):
                results.append(chunk)

            # Should only get the final result
            assert len(results) == 1
            assert results[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_download_hf_model_missing_repo_id(self):
        """Test HF download with missing repo_id."""
        with pytest.raises(ValueError, match="repo_id is required"):
            async for _ in download_hf_model(repo_id=""):
                pass

    @pytest.mark.asyncio
    async def test_download_ollama_model_streaming(self):
        """Test Ollama download with streaming."""
        with patch("nodetool.deploy.admin_operations.stream_ollama_model_pull") as mock_stream:

            async def mock_pull(model_name):
                yield {"status": "starting", "model": model_name}
                yield {"status": "completed", "model": model_name}

            mock_stream.side_effect = mock_pull

            results = []
            async for chunk in download_ollama_model(model_name="test-model", stream=True):
                results.append(chunk)

            assert len(results) == 2
            assert results[0]["status"] == "starting"
            assert results[1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_download_ollama_model_non_streaming(self):
        """Test Ollama download without streaming."""
        with patch("nodetool.deploy.admin_operations.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client

            results = []
            async for chunk in download_ollama_model(model_name="test-model", stream=False):
                results.append(chunk)

            assert len(results) == 1
            assert results[0]["status"] == "completed"
            mock_client.pull.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    async def test_download_ollama_model_missing_model_name(self):
        """Test Ollama download with missing model_name."""
        with pytest.raises(ValueError, match="model_name is required"):
            async for _ in download_ollama_model(model_name=""):
                pass

    @pytest.mark.asyncio
    async def test_scan_hf_cache_success(self):
        """Test successful cache scan."""
        mock_cache_info = MagicMock()
        mock_cache_info.size_on_disk = 1024
        mock_cache_info.repos = []
        mock_cache_info.warnings = []

        with patch("nodetool.deploy.admin_operations.scan_cache_dir") as mock_scan:
            mock_scan.return_value = mock_cache_info

            results = []
            async for chunk in scan_hf_cache():
                results.append(chunk)

            assert len(results) == 1
            assert results[0]["status"] == "completed"
            assert "cache_info" in results[0]

    @pytest.mark.asyncio
    async def test_scan_hf_cache_error(self):
        """Test cache scan error."""
        with patch("nodetool.deploy.admin_operations.scan_cache_dir") as mock_scan:
            mock_scan.side_effect = Exception("Scan failed")

            results = []
            async for chunk in scan_hf_cache():
                results.append(chunk)

            assert len(results) == 1
            assert results[0]["status"] == "error"
            assert "Scan failed" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_delete_hf_model_success(self):
        """Test successful HF model deletion."""
        with patch(
            "nodetool.deploy.admin_operations.delete_cached_hf_model",
            new_callable=AsyncMock,
        ) as mock_delete:
            results = []
            async for chunk in delete_hf_model(repo_id="test/repo"):
                results.append(chunk)

            assert len(results) == 1
            assert results[0]["status"] == "completed"
            assert results[0]["repo_id"] == "test/repo"
            mock_delete.assert_awaited_once_with("test/repo")

    @pytest.mark.asyncio
    async def test_delete_hf_model_missing_repo_id(self):
        """Test HF model deletion with missing repo_id."""
        with pytest.raises(ValueError, match="repo_id is required"):
            async for _ in delete_hf_model(repo_id=""):
                pass

    @pytest.mark.asyncio
    async def test_calculate_cache_size_success(self):
        """Test successful cache size calculation."""
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.walk") as mock_walk,
            patch("os.path.getsize") as mock_getsize,
        ):
            mock_exists.return_value = True
            mock_walk.return_value = [("/test/cache", [], ["file1.bin", "file2.bin"])]
            mock_getsize.side_effect = [1024, 2048]

            results = []
            async for chunk in calculate_cache_size(cache_dir="/test/cache"):
                results.append(chunk)

            assert len(results) == 1
            assert results[0]["success"] is True
            assert results[0]["total_size_bytes"] == 3072
            assert results[0]["size_gb"] == round(3072 / (1024**3), 2)


if __name__ == "__main__":
    pytest.main([__file__])
