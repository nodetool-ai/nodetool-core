"""Tests for HuggingFace cache download functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from huggingface_hub.errors import GatedRepoError
from huggingface_hub.hf_api import RepoFile

from nodetool.integrations.huggingface.hf_download import (
    DownloadManager,
    DownloadState,
)
from nodetool.integrations.huggingface.hf_cache import (
    filter_repo_paths,
    has_cached_files,
)
from nodetool.integrations.huggingface.huggingface_models import (
    delete_cached_hf_model,
)


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing."""
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


@pytest.fixture
def mock_hf_api():
    """Mock HuggingFace API."""
    api = MagicMock()
    
    # Mock file objects
    mock_file1 = MagicMock(spec=RepoFile)
    mock_file1.path = "config.json"
    mock_file1.size = 1024
    
    mock_file2 = MagicMock(spec=RepoFile)
    mock_file2.path = "model.safetensors"
    mock_file2.size = 2048000
    
    api.list_repo_tree.return_value = [mock_file1, mock_file2]
    return api


class TestDownloadState:
    """Test DownloadState dataclass."""
    
    def test_download_state_initialization(self, mock_websocket):
        """Test that DownloadState initializes correctly."""
        state = DownloadState(repo_id="test/repo", websocket=mock_websocket)
        assert state.repo_id == "test/repo"
        assert state.websocket == mock_websocket
        assert state.status == "idle"
        assert state.downloaded_bytes == 0
        assert state.total_bytes == 0
        assert state.error_message is None
        assert len(state.downloaded_files) == 0


class TestDownloadManager:
    """Test DownloadManager functionality."""
    
    @pytest.mark.asyncio
    async def test_send_update_includes_error_message(self, mock_websocket):
        """Test that send_update includes error message when present."""
        manager = DownloadManager()
        state = DownloadState(repo_id="test/repo", websocket=mock_websocket)
        state.status = "error"
        state.error_message = "Test error message"
        manager.downloads["test/repo"] = state
        
        await manager.send_update("test/repo")
        
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["status"] == "error"
        assert call_args["error"] == "Test error message"
    
    @pytest.mark.asyncio
    async def test_send_update_without_error_message(self, mock_websocket):
        """Test that send_update doesn't include error field when no error."""
        manager = DownloadManager()
        state = DownloadState(repo_id="test/repo", websocket=mock_websocket)
        state.status = "progress"
        manager.downloads["test/repo"] = state
        
        await manager.send_update("test/repo")
        
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["status"] == "progress"
        assert "error" not in call_args
    
    @pytest.mark.asyncio
    async def test_start_download_sets_error_message_on_exception(self, mock_websocket, mock_hf_api):
        """Test that start_download sets error_message and sends it via WebSocket."""
        manager = DownloadManager()
        manager.api = mock_hf_api
        
        # Mock download_huggingface_repo to raise an error
        test_error = GatedRepoError("401 Client Error. Cannot access gated repo")
        
        with (
            patch.object(manager, "download_huggingface_repo", side_effect=test_error),
            patch.object(manager, "run_progress_updates") as mock_progress,
            patch("threading.Thread") as mock_thread,
        ):
            mock_thread.return_value.start = Mock()
            mock_thread.return_value.join = Mock()
            
            with pytest.raises(GatedRepoError):
                await manager.start_download(
                    repo_id="test/repo",
                    path=None,
                    websocket=mock_websocket,
                )
            
            # Verify error message was sent via WebSocket
            assert mock_websocket.send_json.called
            # Check that error status was sent
            calls = mock_websocket.send_json.call_args_list
            error_sent = any(
                call[0][0].get("status") == "error" and "error" in call[0][0]
                for call in calls
            )
            assert error_sent, "Error message should have been sent via WebSocket"
    
    @pytest.mark.asyncio
    async def test_download_huggingface_repo_propagates_exceptions(self, mock_websocket, mock_hf_api):
        """Test that download_huggingface_repo properly propagates exceptions from download tasks."""
        manager = DownloadManager()
        manager.api = mock_hf_api
        
        # Create a mock queue
        from multiprocessing import Manager
        manager_instance = Manager()
        queue = manager_instance.Queue()
        
        state = DownloadState(repo_id="test/repo", websocket=mock_websocket)
        manager.downloads["test/repo"] = state
        
        # Mock try_to_load_from_cache to return None (files not cached)
        with (
            patch("nodetool.integrations.huggingface.hf_download.try_to_load_from_cache", return_value=None),
            patch("nodetool.integrations.huggingface.hf_cache.filter_repo_paths") as mock_filter,
            patch("asyncio.get_running_loop") as mock_loop,
        ):
            # Mock filter to return files
            mock_file = MagicMock(spec=RepoFile)
            mock_file.path = "test.safetensors"
            mock_file.size = 1024
            mock_filter.return_value = [mock_file]
            
            # Mock asyncio.gather to return an exception
            test_error = GatedRepoError("401 Client Error. Cannot access gated repo")
            import asyncio
            
            async def mock_gather(*args, **kwargs):
                return [test_error]
            
            with patch("asyncio.gather", side_effect=mock_gather):
                with pytest.raises(GatedRepoError) as exc_info:
                    await manager.download_huggingface_repo(
                        repo_id="test/repo",
                        path=None,
                        queue=queue,
                    )
                # Should raise the exception
                assert exc_info.value == test_error
    
    @pytest.mark.asyncio
    async def test_download_huggingface_repo_handles_multiple_exceptions(self, mock_websocket, mock_hf_api):
        """Test that download_huggingface_repo raises the first exception found."""
        manager = DownloadManager()
        manager.api = mock_hf_api
        
        from multiprocessing import Manager
        manager_instance = Manager()
        queue = manager_instance.Queue()
        
        state = DownloadState(repo_id="test/repo", websocket=mock_websocket)
        manager.downloads["test/repo"] = state
        
        with (
            patch("nodetool.integrations.huggingface.hf_download.try_to_load_from_cache", return_value=None),
            patch("nodetool.integrations.huggingface.hf_cache.filter_repo_paths") as mock_filter,
        ):
            mock_file = MagicMock(spec=RepoFile)
            mock_file.path = "test.safetensors"
            mock_file.size = 1024
            mock_filter.return_value = [mock_file]
            
            # Create multiple exceptions
            error1 = GatedRepoError("First error")
            error2 = ValueError("Second error")
            
            # Mock asyncio.gather to return exceptions
            import asyncio
            
            async def mock_gather(*args, **kwargs):
                return [error1, error2]
            
            with patch("asyncio.gather", side_effect=mock_gather):
                with pytest.raises(GatedRepoError) as exc_info:
                    await manager.download_huggingface_repo(
                        repo_id="test/repo",
                        path=None,
                        queue=queue,
                    )
                # Should raise the first exception
                assert exc_info.value == error1
    
    @pytest.mark.asyncio
    async def test_endpoint_sends_error_on_exception(self, mock_websocket):
        """Test that huggingface_download_endpoint sends error message on exception."""
        from nodetool.integrations.huggingface.hf_websocket import huggingface_download_endpoint
        
        test_error = GatedRepoError("401 Client Error. Cannot access gated repo")
        
        with (
            patch("nodetool.integrations.huggingface.hf_download.DownloadManager") as mock_manager_class,
        ):
            mock_manager = MagicMock()
            mock_manager.start_download = AsyncMock(side_effect=test_error)
            mock_manager_class.return_value = mock_manager
            
            # Mock websocket methods
            mock_websocket.accept = AsyncMock()
            # Make receive_json raise an exception after first call to break the loop
            call_count = 0
            async def mock_receive_json():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {
                        "command": "start_download",
                        "repo_id": "test/repo",
                        "path": None,
                    }
                else:
                    # Raise an exception to break the while True loop
                    raise ConnectionError("WebSocket closed")
            mock_websocket.receive_json = mock_receive_json
            mock_websocket.close = AsyncMock()
            
            # Run the endpoint - it will catch the exception from start_download
            # and send an error message, then re-raise it
            try:
                await huggingface_download_endpoint(mock_websocket)
            except (ConnectionError, GatedRepoError, Exception):
                pass
            
            # Verify error was sent - should be sent both by start_download and endpoint
            assert mock_websocket.send_json.called
            # Check that at least one call had error status with error field
            calls = mock_websocket.send_json.call_args_list
            error_sent = any(
                call[0][0].get("status") == "error" and "error" in call[0][0]
                for call in calls
            )
            assert error_sent, "Error message should have been sent via WebSocket"


class TestFilterRepoPaths:
    """Test filter_repo_paths function."""
    
    def test_filter_repo_paths_with_allow_patterns(self):
        """Test filtering with allow patterns."""
        file1 = MagicMock(spec=RepoFile)
        file1.path = "config.json"
        file2 = MagicMock(spec=RepoFile)
        file2.path = "model.safetensors"
        
        files = [file1, file2]
        filtered = filter_repo_paths(files, allow_patterns=["*.json"])
        
        assert len(filtered) == 1
        assert filtered[0].path == "config.json"
    
    def test_filter_repo_paths_with_ignore_patterns(self):
        """Test filtering with ignore patterns."""
        file1 = MagicMock(spec=RepoFile)
        file1.path = "config.json"
        file2 = MagicMock(spec=RepoFile)
        file2.path = "model.safetensors"
        
        files = [file1, file2]
        filtered = filter_repo_paths(files, ignore_patterns=["*.json"])
        
        assert len(filtered) == 1
        assert filtered[0].path == "model.safetensors"
    
    def test_filter_repo_paths_no_patterns(self):
        """Test filtering without patterns returns all files."""
        file1 = MagicMock(spec=RepoFile)
        file1.path = "config.json"
        file2 = MagicMock(spec=RepoFile)
        file2.path = "model.safetensors"
        
        files = [file1, file2]
        filtered = filter_repo_paths(files)
        
        assert len(filtered) == 2


class TestHasCachedFiles:
    """Test has_cached_files function."""
    
    @patch("os.path.isdir")
    @patch("os.scandir")
    @patch("os.listdir")
    def test_has_cached_files_returns_true_when_files_exist(self, mock_listdir, mock_scandir, mock_isdir):
        """Test that has_cached_files returns True when cached files exist."""
        mock_isdir.return_value = True
        mock_listdir.return_value = ["revision1"]
        mock_scandir.return_value = [MagicMock()]  # At least one file
        
        result = has_cached_files("test/repo")
        assert result is True
    
    @patch("os.path.isdir")
    def test_has_cached_files_returns_false_when_no_snapshots(self, mock_isdir):
        """Test that has_cached_files returns False when no snapshots exist."""
        mock_isdir.return_value = False
        
        result = has_cached_files("test/repo")
        assert result is False


class TestDeleteCachedHfModel:
    """Tests for delete_cached_hf_model using the fast HF cache."""

    def test_delete_cached_hf_model_success(self):
        """Model is present in fast cache and deleted successfully."""
        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.repo_root",
                return_value="/fake/cache/models--org--repo",
            ) as mock_repo_root,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.os.path.exists",
                return_value=True,
            ) as mock_exists,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.shutil.rmtree"
            ) as mock_rmtree,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.MODEL_INFO_CACHE.delete_pattern"
            ) as mock_delete_pattern,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.invalidate"
            ) as mock_invalidate,
        ):
            result = delete_cached_hf_model("org/repo")

        assert result is True
        mock_repo_root.assert_called_once_with("org/repo", repo_type="model")
        mock_exists.assert_called_once_with("/fake/cache/models--org--repo")
        mock_rmtree.assert_called_once_with("/fake/cache/models--org--repo")
        mock_delete_pattern.assert_called_once_with("cached_hf_*")
        mock_invalidate.assert_called_once_with("org/repo", repo_type="model")

    def test_delete_cached_hf_model_repo_not_found(self):
        """Model is not present in fast cache; nothing is deleted."""
        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.repo_root",
                return_value=None,
            ) as mock_repo_root,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.os.path.exists"
            ) as mock_exists,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.shutil.rmtree"
            ) as mock_rmtree,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.MODEL_INFO_CACHE.delete_pattern"
            ) as mock_delete_pattern,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.invalidate"
            ) as mock_invalidate,
        ):
            result = delete_cached_hf_model("org/repo")

        assert result is False
        mock_repo_root.assert_called_once_with("org/repo", repo_type="model")
        mock_exists.assert_not_called()
        mock_rmtree.assert_not_called()
        mock_delete_pattern.assert_not_called()
        mock_invalidate.assert_not_called()

    def test_delete_cached_hf_model_repo_path_missing(self):
        """Fast cache resolves repo_root, but on-disk path is missing."""
        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.repo_root",
                return_value="/fake/cache/models--org--repo",
            ) as mock_repo_root,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.os.path.exists",
                return_value=False,
            ) as mock_exists,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.shutil.rmtree"
            ) as mock_rmtree,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.MODEL_INFO_CACHE.delete_pattern"
            ) as mock_delete_pattern,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.invalidate"
            ) as mock_invalidate,
        ):
            result = delete_cached_hf_model("org/repo")

        assert result is False
        mock_repo_root.assert_called_once_with("org/repo", repo_type="model")
        mock_exists.assert_called_once_with("/fake/cache/models--org--repo")
        mock_rmtree.assert_not_called()
        mock_delete_pattern.assert_not_called()
        mock_invalidate.assert_not_called()
