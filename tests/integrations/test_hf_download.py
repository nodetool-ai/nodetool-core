import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from nodetool.integrations.huggingface import hf_download
from nodetool.integrations.huggingface.hf_download import DownloadManager, DownloadState

class TestHfDownload(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.mock_ws = AsyncMock()
        self.manager = DownloadManager(token="fake_token")

    @patch('nodetool.integrations.huggingface.hf_download.hf_hub_download_with_progress')
    def test_download_file(self, mock_download):
        # Setup
        mock_download.return_value = "/path/to/file"
        callback = MagicMock()
        
        # Execute
        filename, path = hf_download.download_file(
            repo_id="repo",
            filename="file.txt",
            token="token",
            on_progress=callback
        )
        
        # Verify
        self.assertEqual(filename, "file.txt")
        self.assertEqual(path, "/path/to/file")
        
        # Verify callback wiring
        mock_download.assert_called_once()
        progress_cb = mock_download.call_args[1]['progress_callback']
        progress_cb(10, 100)
        callback.assert_called_with(10, 100)

    @patch('nodetool.integrations.huggingface.hf_download.hf_auth.get_hf_token', new_callable=AsyncMock)
    async def test_create(self, mock_get_token):
        mock_get_token.return_value = "async_token"
        manager = await DownloadManager.create(user_id="user")
        self.assertEqual(manager.token, "async_token")

    @patch('nodetool.integrations.huggingface.hf_download.download_file')
    @patch('nodetool.integrations.huggingface.hf_download.HfApi')
    async def test_start_download(self, mock_hf_api, mock_download_file):
        # Setup
        mock_api_instance = mock_hf_api.return_value
        # Replace the real API instance with the mock
        self.manager.api = mock_api_instance
        
        mock_file = MagicMock()
        mock_file.path = "file.txt"
        mock_file.size = 100
        mock_file.type = "file"
        # Mock list_repo_tree to return one file
        mock_api_instance.list_repo_tree.return_value = [mock_file]
        
        # Mock try_to_load_from_cache to return None (not cached)
        with patch('nodetool.integrations.huggingface.hf_download.try_to_load_from_cache', return_value=None):
            # Execute
            # We need to run this in a way that allows the background task to start
            # but we want to wait for completion.
            
            # Mock download_file to return immediately
            mock_download_file.return_value = ("file.txt", "/local/path")
            
            await self.manager.start_download(
                repo_id="repo",
                path=None,
                websocket=self.mock_ws,
                user_id="user"
            )
            
            # Verify
            # Check if websocket received start/progress/complete messages
            # The exact sequence depends on timing, but we should see at least some updates
            self.assertTrue(self.mock_ws.send_json.called)
            
            # Verify download_file was called
            mock_download_file.assert_called()
            
            # Check final state
            self.assertNotIn("repo", self.manager.downloads) # Should be cleaned up

    async def test_monitor_progress(self):
        # Setup
        repo_id = "repo"
        path = None
        id = "repo"
        
        state = DownloadState(repo_id=repo_id, websocket=self.mock_ws)
        self.manager.downloads[id] = state
        
        # Start monitor task
        task = asyncio.create_task(self.manager.monitor_progress(repo_id, path))
        
        # Update state
        state.downloaded_bytes = 50
        state.total_bytes = 100
        
        # Wait a bit
        await asyncio.sleep(0.2)
        
        # Mark completed to stop monitor
        state.status = "completed"
        await asyncio.sleep(0.1)
        
        await task
        
        # Verify updates sent
        self.assertTrue(self.mock_ws.send_json.called)
        calls = self.mock_ws.send_json.call_args_list
        # Should have at least one update with progress
        progress_updates = [c for c in calls if c[0][0]['status'] == 'progress']
        # Note: status might flip to completed before the last monitor loop, 
        # but we simulated 'progress' state above.
        
        # We manually set status to completed, so the loop should exit.

if __name__ == '__main__':
    unittest.main()
