import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from nodetool.integrations.huggingface import hf_download
from nodetool.integrations.huggingface.hf_download import DownloadManager, DownloadState, get_download_manager

class TestHfDownload(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Reset singleton for each test
        hf_download._download_managers = {}
        self.mock_ws = AsyncMock()
        self.manager = DownloadManager(token="fake_token")

    @patch('nodetool.integrations.huggingface.hf_download.hf_auth.get_hf_token', new_callable=AsyncMock)
    async def test_get_download_manager_singleton(self, mock_get_token):
        mock_get_token.return_value = "token"
        
        # First call creates instance
        m1 = await get_download_manager("user1")
        self.assertIsInstance(m1, DownloadManager)
        
        # Second call returns same instance
        m2 = await get_download_manager("user1")
        self.assertIs(m1, m2)
        
        # Different user gets different instance
        m3 = await get_download_manager("user2")
        self.assertIsNot(m1, m3)

    def test_websocket_management(self):
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        
        self.manager.add_websocket(ws1)
        self.assertIn(ws1, self.manager.active_websockets)
        
        self.manager.add_websocket(ws2)
        self.assertIn(ws2, self.manager.active_websockets)
        self.assertEqual(len(self.manager.active_websockets), 2)
        
        self.manager.remove_websocket(ws1)
        self.assertNotIn(ws1, self.manager.active_websockets)
        self.assertIn(ws2, self.manager.active_websockets)

    @patch('nodetool.integrations.huggingface.hf_download.async_hf_download')
    @patch('nodetool.integrations.huggingface.hf_download.HfApi')
    async def test_start_download_non_blocking(self, mock_hf_api, mock_async_download):
        # Setup
        mock_api_instance = mock_hf_api.return_value
        self.manager.api = mock_api_instance
        
        mock_file = MagicMock()
        mock_file.path = "file.txt"
        mock_file.size = 100
        mock_file.type = "file"
        mock_api_instance.list_repo_tree.return_value = [mock_file]
        
        # Mock download to take some time
        async def delayed_download(*args, **kwargs):
            await asyncio.sleep(0.1)
            return "/local/path"
        mock_async_download.side_effect = delayed_download
        
        with patch('nodetool.integrations.huggingface.hf_download.try_to_load_from_cache', return_value=None):
            # Add a websocket to receive updates
            self.manager.add_websocket(self.mock_ws)
            
            # Execute start_download
            await self.manager.start_download(
                repo_id="repo",
                path=None,
                user_id="user"
            )
            
            # It should return immediately, but task should be running
            self.assertIn("repo", self.manager.downloads)
            state = self.manager.downloads["repo"]
            self.assertIsNotNone(state.task)
            self.assertFalse(state.task.done())
            
            # Wait for task to complete
            await state.task
            
            # Verify completion
            self.assertEqual(state.status, "completed")
            self.assertTrue(self.mock_ws.send_json.called)

    @patch('nodetool.integrations.huggingface.hf_download.async_hf_download')
    @patch('nodetool.integrations.huggingface.hf_download.HfApi')
    async def test_cancel_download(self, mock_hf_api, mock_async_download):
        # Setup
        mock_api_instance = mock_hf_api.return_value
        self.manager.api = mock_api_instance
        
        mock_file = MagicMock()
        mock_file.path = "file.txt"
        mock_api_instance.list_repo_tree.return_value = [mock_file]
        
        # Mock download that waits for cancellation
        async def cancellable_download(*args, **kwargs):
            cancel_event = kwargs.get('cancel_event')
            while not cancel_event.is_set():
                await asyncio.sleep(0.01)
            raise asyncio.CancelledError()
            
        mock_async_download.side_effect = cancellable_download
        
        with patch('nodetool.integrations.huggingface.hf_download.try_to_load_from_cache', return_value=None):
            self.manager.add_websocket(self.mock_ws)
            
            await self.manager.start_download(repo_id="repo", path=None, user_id="user")
            state = self.manager.downloads["repo"]
            
            # Give it a moment to start
            await asyncio.sleep(0.05)
            
            # Cancel
            await self.manager.cancel_download("repo")
            
            # Wait for task
            try:
                await state.task
            except asyncio.CancelledError:
                pass
            
            # Verify cancelled state
            self.assertTrue(state.cancel.is_set())
            self.assertEqual(state.status, "cancelled")
            
            # Verify update sent
            # We expect multiple calls: initial, progress (maybe), cancelled
            calls = [c[0][0] for c in self.mock_ws.send_json.call_args_list]
            statuses = [c['status'] for c in calls]
            self.assertIn("cancelled", statuses)

    async def test_sync_state(self):
        # Setup state
        state = DownloadState(repo_id="repo1")
        state.status = "progress"
        state.downloaded_bytes = 50
        state.total_bytes = 100
        self.manager.downloads["repo1"] = state
        
        new_ws = AsyncMock()
        
        # Sync
        await self.manager.sync_state(new_ws)
        
        # Verify
        new_ws.send_json.assert_called_once()
        call_args = new_ws.send_json.call_args[0][0]
        self.assertEqual(call_args['repo_id'], "repo1")
        self.assertEqual(call_args['status'], "progress")
        self.assertEqual(call_args['downloaded_bytes'], 50)

if __name__ == '__main__':
    unittest.main()
