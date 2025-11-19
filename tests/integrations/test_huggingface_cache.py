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
    read_cached_hf_models,
    search_cached_hf_models,
    _get_file_size,
    HF_FAST_CACHE,
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

    @pytest.mark.asyncio
    async def test_delete_cached_hf_model_success(self):
        """Model is present in fast cache and deleted successfully."""
        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.repo_root",
                new_callable=AsyncMock,
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
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.model_info_cache.delete_pattern"
            ) as mock_delete_pattern,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.invalidate",
                new_callable=AsyncMock,
            ) as mock_invalidate,
        ):
            result = await delete_cached_hf_model("org/repo")

        assert result is True
        mock_repo_root.assert_awaited_once_with("org/repo", repo_type="model")
        mock_exists.assert_called_once_with("/fake/cache/models--org--repo")
        mock_rmtree.assert_called_once_with("/fake/cache/models--org--repo")
        mock_delete_pattern.assert_called_once_with("cached_hf_*")
        mock_invalidate.assert_awaited_once_with("org/repo", repo_type="model")

    @pytest.mark.asyncio
    async def test_delete_cached_hf_model_repo_not_found(self):
        """Model is not present in fast cache; nothing is deleted."""
        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.repo_root",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_repo_root,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.os.path.exists"
            ) as mock_exists,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.shutil.rmtree"
            ) as mock_rmtree,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.model_info_cache.delete_pattern"
            ) as mock_delete_pattern,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.invalidate",
                new_callable=AsyncMock,
            ) as mock_invalidate,
        ):
            result = await delete_cached_hf_model("org/repo")

        assert result is False
        mock_repo_root.assert_awaited_once_with("org/repo", repo_type="model")
        mock_exists.assert_not_called()
        mock_rmtree.assert_not_called()
        mock_delete_pattern.assert_not_called()
        mock_invalidate.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_cached_hf_model_repo_path_missing(self):
        """Fast cache resolves repo_root, but on-disk path is missing."""
        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.repo_root",
                new_callable=AsyncMock,
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
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.model_info_cache.delete_pattern"
            ) as mock_delete_pattern,
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.invalidate",
                new_callable=AsyncMock,
            ) as mock_invalidate,
        ):
            result = await delete_cached_hf_model("org/repo")

        assert result is False
        mock_repo_root.assert_awaited_once_with("org/repo", repo_type="model")
        mock_exists.assert_called_once_with("/fake/cache/models--org--repo")
        mock_rmtree.assert_not_called()
        mock_delete_pattern.assert_not_called()
        mock_invalidate.assert_not_called()


class TestDiscoverCachedRepos:
    """Tests for HfFastCache.discover_repos method."""

    @pytest.mark.asyncio
    async def test_discover_repos_finds_models(self, tmp_path):
        """Test that discover_repos finds model repos."""
        from nodetool.integrations.huggingface.hf_fast_cache import HfFastCache

        cache_dir = tmp_path
        repo_dir1 = cache_dir / "models--org--repo1"
        repo_dir2 = cache_dir / "models--repo2"
        repo_dir1.mkdir(parents=True)
        repo_dir2.mkdir(parents=True)

        cache = HfFastCache(cache_dir=cache_dir)
        repos = await cache.discover_repos("model")

        assert len(repos) == 2
        repo_ids = [repo_id for repo_id, _ in repos]
        assert "org/repo1" in repo_ids
        assert "repo2" in repo_ids

    @pytest.mark.asyncio
    async def test_discover_repos_filters_by_type(self, tmp_path):
        """Test that discover_repos filters by repo type."""
        from nodetool.integrations.huggingface.hf_fast_cache import HfFastCache

        cache_dir = tmp_path
        model_dir = cache_dir / "models--org--model"
        dataset_dir = cache_dir / "datasets--org--dataset"
        model_dir.mkdir(parents=True)
        dataset_dir.mkdir(parents=True)

        cache = HfFastCache(cache_dir=cache_dir)
        model_repos = await cache.discover_repos("model")
        dataset_repos = await cache.discover_repos("dataset")

        assert len(model_repos) == 1
        assert model_repos[0][0] == "org/model"
        assert len(dataset_repos) == 1
        assert dataset_repos[0][0] == "org/dataset"

    @pytest.mark.asyncio
    async def test_discover_repos_handles_missing_cache_dir(self, tmp_path):
        """Test that discover_repos handles missing cache directory."""
        from nodetool.integrations.huggingface.hf_fast_cache import HfFastCache

        cache_dir = tmp_path / "nonexistent"
        cache = HfFastCache(cache_dir=cache_dir)
        repos = await cache.discover_repos("model")

        assert repos == []


class TestGetFileSize:
    """Tests for _get_file_size function."""

    def test_get_file_size_for_regular_file(self, tmp_path):
        """Test that _get_file_size returns size for regular file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        size = _get_file_size(test_file)
        assert size > 0

    def test_get_file_size_for_symlink(self, tmp_path):
        """Test that _get_file_size handles symlinks correctly."""
        target_file = tmp_path / "target.txt"
        target_file.write_text("target content")
        symlink_file = tmp_path / "link.txt"
        symlink_file.symlink_to(target_file)

        size = _get_file_size(symlink_file)
        assert size > 0

    def test_get_file_size_for_nonexistent_file(self, tmp_path):
        """Test that _get_file_size returns 0 for nonexistent file."""
        nonexistent_file = tmp_path / "nonexistent.txt"

        size = _get_file_size(nonexistent_file)
        assert size == 0


class TestReadCachedHfModels:
    """Tests for read_cached_hf_models function."""

    @pytest.mark.asyncio
    async def test_read_cached_hf_models_with_no_repos(self):
        """Test that read_cached_hf_models returns empty list when no repos found."""
        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.discover_repos",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.model_info_cache.get",
                return_value=None,
            ),
        ):
            result = await read_cached_hf_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_read_cached_hf_models_uses_cache(self):
        """Test that read_cached_hf_models uses cached results when available."""
        mock_cached_models = [MagicMock(repo_id="org/repo")]

        with patch(
            "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.model_info_cache.get",
            return_value=mock_cached_models,
        ):
            result = await read_cached_hf_models()

        assert result == mock_cached_models

    @pytest.mark.asyncio
    async def test_read_cached_hf_models_calculates_size(self, tmp_path):
        """Test that read_cached_hf_models calculates size correctly."""
        cache_dir = tmp_path
        repo_dir = cache_dir / "models--org--repo"
        refs_dir = repo_dir / "refs"
        snapshots_dir = repo_dir / "snapshots"
        commit = "abc123"

        refs_dir.mkdir(parents=True)
        snapshots_dir.mkdir(parents=True)
        snapshot_dir = snapshots_dir / commit
        snapshot_dir.mkdir()
        (refs_dir / "main").write_text(f"{commit}\n", encoding="utf-8")
        (snapshot_dir / "model.bin").write_bytes(b"test" * 100)
        (snapshot_dir / "config.json").write_bytes(b"config")

        mock_model_info = MagicMock()
        mock_model_info.pipeline_tag = "text-generation"
        mock_model_info.tags = []
        mock_model_info.downloads = 1000
        mock_model_info.likes = 100
        mock_model_info.trending_score = 50.0

        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.model_info_cache.get",
                return_value=None,
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.discover_repos",
                new_callable=AsyncMock,
                return_value=[("org/repo", repo_dir)],
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.fetch_model_info",
                new_callable=AsyncMock,
                return_value=mock_model_info,
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.repo_root",
                new_callable=AsyncMock,
                return_value=str(repo_dir),
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.active_snapshot_dir",
                new_callable=AsyncMock,
                return_value=str(snapshot_dir),
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.list_files",
                new_callable=AsyncMock,
                return_value=["model.bin", "config.json"],
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models._get_file_size",
                side_effect=lambda p: p.stat().st_size if p.exists() else 0,
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.model_info_cache.set",
            ),
        ):
            result = await read_cached_hf_models()

        assert len(result) == 1
        assert result[0].repo_id == "org/repo"
        assert result[0].size_on_disk > 0

    @pytest.mark.asyncio
    async def test_read_cached_hf_models_handles_exceptions(self):
        """Test that read_cached_hf_models handles exceptions gracefully."""
        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.model_info_cache.get",
                return_value=None,
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.discover_repos",
                new_callable=AsyncMock,
                side_effect=Exception("Cache error"),
            ),
        ):
            result = await read_cached_hf_models()

        assert result == []


class TestSearchCachedHfModels:
    """Tests for search_cached_hf_models function."""

    @pytest.mark.asyncio
    async def test_search_cached_hf_models_filters_repo_and_files(self, tmp_path):
        snapshot_dir = tmp_path / "models--org--repo" / "snapshots" / "abc"
        snapshot_dir.mkdir(parents=True)

        mock_model_info = MagicMock()
        mock_model_info.pipeline_tag = "text-to-image"
        mock_model_info.tags = ["lora", "diffusers"]
        mock_model_info.author = "org"
        mock_model_info.library_name = "diffusers"

        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.discover_repos",
                new_callable=AsyncMock,
                return_value=[("org/repo", snapshot_dir.parent.parent)],
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.fetch_model_info",
                new_callable=AsyncMock,
                return_value=mock_model_info,
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.repo_root",
                new_callable=AsyncMock,
                return_value=str(snapshot_dir.parent),
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.active_snapshot_dir",
                new_callable=AsyncMock,
                return_value=str(snapshot_dir),
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.list_files",
                new_callable=AsyncMock,
                return_value=[
                    "models/ip_adapter.safetensors",
                    "README.md",
                ],
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models._get_file_size",
                side_effect=[2048, 128],
            ),
        ):
            results = await search_cached_hf_models(
                repo_patterns=["org/*"],
                filename_patterns=["*.safetensors"],
                pipeline_tags=["text-to-image"],
                tags=["lora"],
                authors=["org"],
                library_name="diffusers",
            )

        assert len(results) == 2
        repo_entry = next(model for model in results if model.path is None)
        file_entry = next(model for model in results if model.path is not None)
        assert repo_entry.repo_id == "org/repo"
        assert file_entry.path == "models/ip_adapter.safetensors"
        assert file_entry.size_on_disk == 2048

    @pytest.mark.asyncio
    async def test_search_cached_hf_models_includes_entries_without_metadata(self, tmp_path):
        repo_dir = tmp_path / "models--org--repo"
        repo_dir.mkdir(parents=True)
        snapshot_dir = repo_dir / "snapshots" / "abc"
        snapshot_dir.mkdir(parents=True)

        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.discover_repos",
                new_callable=AsyncMock,
                return_value=[("org/repo", repo_dir)],
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.fetch_model_info",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.repo_root",
                new_callable=AsyncMock,
                return_value=str(repo_dir),
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.active_snapshot_dir",
                new_callable=AsyncMock,
                return_value=str(snapshot_dir),
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.list_files",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models._get_file_size",
                return_value=0,
            ),
        ):
            results = await search_cached_hf_models()

        assert len(results) == 1
        assert results[0].repo_id == "org/repo"

    @pytest.mark.asyncio
    async def test_search_cached_hf_models_requires_metadata_when_filters_used(self, tmp_path):
        repo_dir = tmp_path / "models--org--repo"
        repo_dir.mkdir(parents=True)
        snapshot_dir = repo_dir / "snapshots" / "abc"
        snapshot_dir.mkdir(parents=True)

        with (
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.discover_repos",
                new_callable=AsyncMock,
                return_value=[("org/repo", repo_dir)],
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.fetch_model_info",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.repo_root",
                new_callable=AsyncMock,
                return_value=str(repo_dir),
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.active_snapshot_dir",
                new_callable=AsyncMock,
                return_value=str(snapshot_dir),
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE.list_files",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "nodetool.integrations.huggingface.huggingface_models._get_file_size",
                return_value=0,
            ),
        ):
            results = await search_cached_hf_models(pipeline_tags=["text-generation"])

        assert results == []
