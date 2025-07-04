"""
Tests for the Hugging Face models caching functionality.

This module tests cache operations including:
- Cache read/write operations
- Cache expiration logic
- Concurrent access handling
- Cache corruption recovery
- Platform-specific cache directory resolution
"""

import json
import os
import tempfile
import threading
import time
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from nodetool.common.huggingface_models import (
    get_model_info_cache_directory,
    get_cache_file_path,
    is_cache_valid,
    read_cache_file,
    write_cache_file,
    delete_cache_file,
    cleanup_expired_cache,
    fetch_model_info,
    fetch_model_readme,
    CACHE_VERSION,
)


class TestCacheDirectory:
    """Test cache directory resolution across platforms."""
    
    def test_cache_directory_creation(self, tmp_path):
        """Test that cache directory is created properly."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path / "cache"):
            cache_dir = get_model_info_cache_directory()
            assert cache_dir.exists()
            assert cache_dir.is_dir()
    
    @patch("nodetool.common.huggingface_models.Path.mkdir")
    def test_cache_directory_with_platformdirs(self, mock_mkdir):
        """Test cache directory resolution with platformdirs."""
        with patch("platformdirs.user_cache_dir", return_value="/test/cache/dir"):
            cache_dir = get_model_info_cache_directory()
            assert "model_info_cache" in str(cache_dir)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch("nodetool.common.huggingface_models.Path.mkdir")
    def test_cache_directory_fallback_logic(self, mock_mkdir):
        """Test platform-specific fallback logic without creating directories."""
        # This tests the path logic without actually creating directories
        with patch("platformdirs.user_cache_dir", side_effect=ImportError):
            cache_dir = get_model_info_cache_directory()
            assert "nodetool" in str(cache_dir)
            assert "model_info_cache" in str(cache_dir)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestCacheFileOperations:
    """Test basic cache file operations."""
    
    def test_cache_file_path_generation(self):
        """Test safe filename generation from model IDs."""
        model_id = "meta-llama/Llama-2-7b-hf"
        cache_path = get_cache_file_path(model_id)
        
        # Should create a hashed filename
        assert cache_path.suffix == ".json"
        assert "model_info" in cache_path.name
        # Filename should be a valid MD5 hash
        assert len(cache_path.stem.split("_")[0]) == 32
        
        # Same model ID should produce same path
        cache_path2 = get_cache_file_path(model_id)
        assert cache_path == cache_path2
        
        # Different model IDs should produce different paths
        different_model = "openai/whisper-base"
        different_path = get_cache_file_path(different_model)
        assert cache_path != different_path
    
    def test_write_and_read_cache(self, tmp_path):
        """Test writing and reading cache files."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            model_id = "test-model"
            test_data = {"model": "test", "tags": ["test1", "test2"]}
            
            cache_file = get_cache_file_path(model_id)
            write_cache_file(cache_file, test_data)
            
            # File should exist
            assert cache_file.exists()
            
            # Read back the data
            cached_data = read_cache_file(cache_file)
            assert cached_data == test_data
            
            # Verify cache structure
            with open(cache_file, 'r') as f:
                raw_data = json.load(f)
                assert raw_data["version"] == CACHE_VERSION
                assert "timestamp" in raw_data
                assert raw_data["data"] == test_data
    
    def test_write_cache_with_size_verification(self, tmp_path):
        """Test that cache writing verifies file size."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            model_id = "test-model"
            test_data = {"model": "test", "size": "large" * 100}
            
            cache_file = get_cache_file_path(model_id)
            write_cache_file(cache_file, test_data)
            
            # Verify file was written with correct size
            assert cache_file.exists()
            
            # Manually verify size matches expected
            cache_content = {
                "version": CACHE_VERSION,
                "timestamp": datetime.now().isoformat(),
                "data": test_data
            }
            expected_size = len(json.dumps(cache_content, indent=2, default=str).encode('utf-8'))
            actual_size = cache_file.stat().st_size
            
            # Size should be very close (allowing for timestamp differences)
            assert abs(actual_size - expected_size) < 100
    
    def test_delete_cache_file(self, tmp_path):
        """Test cache file deletion."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            model_id = "test-model"
            test_data = {"model": "test"}
            
            cache_file = get_cache_file_path(model_id)
            write_cache_file(cache_file, test_data)
            assert cache_file.exists()
            
            delete_cache_file(model_id)
            assert not cache_file.exists()
            
            # Deleting non-existent file should not raise
            delete_cache_file("non-existent-model")


class TestCacheExpiration:
    """Test cache expiration logic."""
    
    def test_cache_validity_fresh(self, tmp_path):
        """Test that fresh cache is considered valid."""
        cache_file = tmp_path / "test_cache.json"
        cache_file.write_text("{}")
        
        assert is_cache_valid(cache_file)
    
    def test_cache_validity_expired(self, tmp_path):
        """Test that expired cache is considered invalid."""
        cache_file = tmp_path / "test_cache.json"
        cache_file.write_text("{}")
        
        # Set modification time to 8 days ago
        old_time = time.time() - (8 * 24 * 3600)
        os.utime(cache_file, (old_time, old_time))
        
        assert not is_cache_valid(cache_file)
    
    def test_cache_validity_nonexistent(self, tmp_path):
        """Test that non-existent cache is considered invalid."""
        cache_file = tmp_path / "nonexistent.json"
        assert not is_cache_valid(cache_file)
    
    @patch.dict(os.environ, {"NODETOOL_CACHE_EXPIRY_DAYS": "3"})
    def test_configurable_cache_expiry(self, tmp_path):
        """Test that cache expiry is configurable via environment variable."""
        # Need to reload the module to pick up the env var
        import importlib
        import nodetool.common.huggingface_models
        importlib.reload(nodetool.common.huggingface_models)
        
        cache_file = tmp_path / "test_cache.json"
        cache_file.write_text("{}")
        
        # Set modification time to 4 days ago
        old_time = time.time() - (4 * 24 * 3600)
        os.utime(cache_file, (old_time, old_time))
        
        # Should be invalid with 3-day expiry
        assert not is_cache_valid(cache_file)
    
    def test_cleanup_expired_cache(self, tmp_path):
        """Test cleanup of expired cache files."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            # Create some cache files
            fresh_file = tmp_path / "fresh_cache.json"
            fresh_file.write_text('{"version": "1.0", "data": {}}')
            
            old_file1 = tmp_path / "old_cache1.json"
            old_file1.write_text('{"version": "1.0", "data": {}}')
            
            old_file2 = tmp_path / "old_cache2.json"
            old_file2.write_text('{"version": "1.0", "data": {}}')
            
            # Set old files to 8 days ago
            old_time = time.time() - (8 * 24 * 3600)
            os.utime(old_file1, (old_time, old_time))
            os.utime(old_file2, (old_time, old_time))
            
            # Run cleanup
            removed_count = cleanup_expired_cache()
            
            assert removed_count == 2
            assert fresh_file.exists()
            assert not old_file1.exists()
            assert not old_file2.exists()


class TestCacheCorruption:
    """Test handling of corrupted cache files."""
    
    def test_read_corrupted_json(self, tmp_path):
        """Test reading corrupted JSON cache files."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            model_id = "test-model"
            cache_file = get_cache_file_path(model_id)
            
            # Write corrupted JSON
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text("{corrupted json")
            
            # Should return None for corrupted data
            assert read_cache_file(cache_file) is None
    
    def test_read_wrong_version(self, tmp_path):
        """Test reading cache with wrong version."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            model_id = "test-model"
            cache_file = get_cache_file_path(model_id)
            
            # Write cache with wrong version
            cache_data = {
                "version": "0.1",  # Wrong version
                "timestamp": datetime.now().isoformat(),
                "data": {"test": "data"}
            }
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache_data))
            
            # Should return None for wrong version
            assert read_cache_file(cache_file) is None
    
    def test_read_missing_data_field(self, tmp_path):
        """Test reading cache with missing data field."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            model_id = "test-model"
            cache_file = get_cache_file_path(model_id)
            
            # Write cache without data field
            cache_data = {
                "version": CACHE_VERSION,
                "timestamp": datetime.now().isoformat(),
                # Missing "data" field
            }
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache_data))
            
            # Should return None for missing data
            assert read_cache_file(cache_file) is None


class TestConcurrentAccess:
    """Test concurrent access to cache files."""
    
    def test_concurrent_writes(self, tmp_path):
        """Test multiple threads writing to different cache files."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            num_threads = 10
            results = []
            
            def write_model_cache(model_num):
                model_id = f"test-model-{model_num}"
                test_data = {"model": f"test-{model_num}", "thread": threading.current_thread().name}
                
                cache_file = get_cache_file_path(model_id)
                write_cache_file(cache_file, test_data)
                
                # Verify write
                read_data = read_cache_file(cache_file)
                results.append(read_data == test_data)
            
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=write_model_cache, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # All writes should succeed
            assert all(results)
            assert len(results) == num_threads
    
    def test_concurrent_read_write(self, tmp_path):
        """Test concurrent reads and writes to the same cache file."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            model_id = "shared-model"
            cache_file = get_cache_file_path(model_id)
            
            # Initial write
            initial_data = {"version": 1, "data": "initial"}
            write_cache_file(cache_file, initial_data)
            
            read_results = []
            write_complete = threading.Event()
            
            def reader_thread():
                for _ in range(50):
                    data = read_cache_file(cache_file)
                    if data:
                        read_results.append(data)
                    time.sleep(0.01)
            
            def writer_thread():
                time.sleep(0.1)  # Let readers start
                new_data = {"version": 2, "data": "updated"}
                write_cache_file(cache_file, new_data)
                write_complete.set()
            
            # Start readers and writer
            readers = [threading.Thread(target=reader_thread) for _ in range(3)]
            writer = threading.Thread(target=writer_thread)
            
            for reader in readers:
                reader.start()
            writer.start()
            
            writer.join()
            for reader in readers:
                reader.join()
            
            # Should have read some data
            assert len(read_results) > 0
            
            # Final read should get the updated data
            final_data = read_cache_file(cache_file)
            assert final_data == {"version": 2, "data": "updated"}


class TestAPIIntegration:
    """Test integration with API calls."""
    
    @pytest.mark.asyncio
    async def test_fetch_model_info_cache_hit(self, tmp_path):
        """Test fetching model info with cache hit."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            model_id = "test/model"
            cached_model_data = {
                "_id": "123",
                "id": model_id,
                "modelId": model_id,
                "author": "test",
                "sha": "abc123",
                "lastModified": "2024-01-01T00:00:00Z",
                "private": False,
                "disabled": False,
                "gated": False,
                "tags": ["test"],
                "downloads": 100,
                "likes": 10,
                "createdAt": "2024-01-01T00:00:00Z"
            }
            
            # Pre-populate cache
            cache_file = get_cache_file_path(model_id, "model_info")
            write_cache_file(cache_file, cached_model_data)
            
            # Fetch should use cache
            with patch("httpx.AsyncClient") as mock_client:
                result = await fetch_model_info(model_id)
                
                # Should not make HTTP request
                mock_client.assert_not_called()
                
                # Should return cached data
                assert result is not None
                assert result.id == model_id
                assert result.tags == ["test"]
    
    @pytest.mark.asyncio
    async def test_fetch_model_info_cache_miss(self, tmp_path):
        """Test fetching model info with cache miss."""
        with patch("nodetool.common.huggingface_models.get_model_info_cache_directory", return_value=tmp_path):
            model_id = "test/model"
            api_response_data = {
                "_id": "123",
                "id": model_id,
                "modelId": model_id,
                "author": "test",
                "sha": "abc123",
                "lastModified": "2024-01-01T00:00:00Z",
                "private": False,
                "disabled": False,
                "gated": False,
                "tags": ["new", "from-api"],
                "downloads": 200,
                "likes": 20,
                "createdAt": "2024-01-01T00:00:00Z"
            }
            
            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = api_response_data
            
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client
                
                result = await fetch_model_info(model_id)
                
                # Should make HTTP request
                mock_client.get.assert_called_once_with(f"https://huggingface.co/api/models/{model_id}")
                
                # Should return API data
                assert result is not None
                assert result.id == model_id
                assert result.tags == ["new", "from-api"]
                
                # Should cache the result
                cache_file = get_cache_file_path(model_id, "model_info")
                assert cache_file.exists()
                cached_data = read_cache_file(cache_file)
                assert cached_data == api_response_data
    
    @pytest.mark.asyncio
    async def test_fetch_model_readme_with_hf_cache(self):
        """Test fetching model README using HF hub cache."""
        model_id = "test/model"
        readme_content = "# Test Model\n\nThis is a test model."
        
        with patch("huggingface_hub.try_to_load_from_cache") as mock_try_load:
            with patch("huggingface_hub.hf_hub_download") as mock_download:
                # Simulate cache hit
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                    f.write(readme_content)
                    temp_path = f.name
                
                try:
                    mock_try_load.return_value = temp_path
                    
                    result = await fetch_model_readme(model_id)
                    
                    assert result == readme_content
                    mock_download.assert_not_called()  # Should use cache
                finally:
                    os.unlink(temp_path)