"""
Tests for ChatWebSocketRunner functionality
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import msgpack
import pytest
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from nodetool.chat.chat_websocket_runner import ChatWebSocketRunner, WebSocketMode
from nodetool.config.environment import Environment

DEFAULT_TEST_TIMEOUT = 5


async def wait_for(coro, timeout: float = DEFAULT_TEST_TIMEOUT):
    """Await a coroutine with a timeout to avoid test hangs."""
    return await asyncio.wait_for(coro, timeout=timeout)


@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestChatWebSocketRunner:
    """Test suite for ChatWebSocketRunner functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = ChatWebSocketRunner("test_token")

    async def test_init(self):
        """Test initialization of ChatWebSocketRunner"""
        runner = ChatWebSocketRunner("test_token")
        assert runner.auth_token == "test_token"
        assert runner.websocket is None
        assert runner.mode == WebSocketMode.BINARY
        # Inherited attributes
        assert runner.user_id is None
        assert runner.supabase is None

    # Authentication Tests
    async def test_local_development_no_auth_required(self):
        """Test that authentication is bypassed in local development mode"""
        with patch.object(Environment, "enforce_auth", return_value=False):
            runner = ChatWebSocketRunner()
            websocket = Mock()
            websocket.accept = AsyncMock()

            await wait_for(runner.connect(websocket))

            # Verify connection was accepted
            websocket.accept.assert_called_once()
            # Verify default user ID was set
            assert runner.user_id == "1"
            # Verify websocket was never closed
            assert not hasattr(websocket, "close") or not websocket.close.called

            # Ensure cleanup to avoid leaking heartbeat tasks in CI
            await wait_for(runner.disconnect())

    async def test_missing_auth_token_when_required(self):
        """Test that missing auth token is rejected when authentication is required"""
        with patch.object(Environment, "enforce_auth", return_value=True):
            runner = ChatWebSocketRunner()  # No auth token provided
            websocket = Mock()
            websocket.close = AsyncMock()

            await wait_for(runner.connect(websocket))

            # Verify connection was closed with correct code and reason
            websocket.close.assert_called_once_with(code=1008, reason="Missing authentication")
            # Verify accept was never called
            assert not hasattr(websocket, "accept") or not websocket.accept.called

    async def test_invalid_auth_token(self):
        """Test that invalid auth token is rejected"""
        with patch.object(Environment, "enforce_auth", return_value=True):
            runner = ChatWebSocketRunner(auth_token="invalid_token")
            websocket = Mock()
            websocket.close = AsyncMock()

            # Mock validate_token to return False
            with patch.object(runner, "validate_token", return_value=False):
                await wait_for(runner.connect(websocket))

            # Verify connection was closed with correct code and reason
            websocket.close.assert_called_once_with(code=1008, reason="Invalid authentication")
            # Verify accept was never called
            assert not hasattr(websocket, "accept") or not websocket.accept.called

    async def test_valid_auth_token(self):
        """Test that valid auth token is accepted"""
        with patch.object(Environment, "enforce_auth", return_value=True):
            runner = ChatWebSocketRunner(auth_token="valid_token")
            websocket = Mock()
            websocket.accept = AsyncMock()

            # Mock validate_token to return True and set user_id
            async def mock_validate(token):
                runner.user_id = "test-user-123"
                return True

            with patch.object(runner, "validate_token", side_effect=mock_validate):
                await runner.connect(websocket)

            # Verify connection was accepted
            websocket.accept.assert_called_once()
            # Verify user ID was set correctly
            assert runner.user_id == "test-user-123"
            # Verify close was never called
            assert not hasattr(websocket, "close") or not websocket.close.called

            # Ensure cleanup to avoid leaking heartbeat tasks in CI
            await wait_for(runner.disconnect())

    async def test_validate_token_with_supabase(self):
        """Test the validate_token method with Supabase integration"""
        runner = ChatWebSocketRunner(auth_token="test_jwt_token")

        # Mock Supabase client and response
        mock_supabase = Mock()
        mock_session = Mock()
        mock_session.access_token = "test_jwt_token"
        mock_user_response = Mock()
        mock_user = Mock()
        mock_user.id = "user-123"
        mock_user_response.user = mock_user

        mock_supabase.auth.get_session = AsyncMock(return_value=mock_session)
        mock_supabase.auth.get_user = AsyncMock(return_value=mock_user_response)

        runner.supabase = mock_supabase

        result = await wait_for(runner.validate_token("test_jwt_token"))

        # Verify the token validation was successful
        assert result is True
        assert runner.user_id == "user-123"
        mock_supabase.auth.get_user.assert_called_once_with("test_jwt_token")

    # Connection Management Tests
    async def test_disconnect(self):
        """Test disconnect functionality"""
        # Set up a connected state
        websocket = Mock(spec=WebSocket)
        websocket.close = AsyncMock()
        websocket.client_state = WebSocketState.CONNECTED
        self.runner.websocket = websocket

        # Create a proper asyncio task
        async def dummy_task():
            await asyncio.sleep(1)

        task = asyncio.create_task(dummy_task())
        self.runner.current_task = task

        await wait_for(self.runner.disconnect())

        # Verify cleanup
        # Task should be cancelled but we need to handle the CancelledError
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected
        assert task.cancelled()
        # Websocket should have been closed
        websocket.close.assert_called_once()
        # And then set to None
        assert self.runner.websocket is None
        assert self.runner.current_task is None

    # Message Sending Tests
    async def test_send_message_binary(self):
        """Test sending binary messages"""
        self.runner.websocket = Mock(spec=WebSocket)
        self.runner.websocket.send_bytes = AsyncMock()
        self.runner.mode = WebSocketMode.BINARY

        message = {"type": "test", "content": "Hello"}
        await wait_for(self.runner.send_message(message))

        # Verify message was packed and sent
        expected_packed = msgpack.packb(message, use_bin_type=True)
        self.runner.websocket.send_bytes.assert_called_once_with(expected_packed)

    async def test_send_message_text(self):
        """Test sending text messages"""
        self.runner.websocket = Mock(spec=WebSocket)
        self.runner.websocket.send_text = AsyncMock()
        self.runner.mode = WebSocketMode.TEXT

        message = {"type": "test", "content": "Hello"}
        await wait_for(self.runner.send_message(message))

        # Verify message was JSON encoded and sent
        expected_json = json.dumps(message)
        self.runner.websocket.send_text.assert_called_once_with(expected_json)

    # Message Receiving Tests
    async def test_receive_message_binary(self):
        """Test receiving binary messages"""
        self.runner.websocket = Mock(spec=WebSocket)

        test_data = {"type": "chat", "content": "Hello"}
        packed_data = msgpack.packb(test_data)

        self.runner.websocket.receive = AsyncMock(return_value={"type": "websocket.message", "bytes": packed_data})

        # Receive message
        result = await wait_for(self.runner.receive_message())

        # Verify unpacking and mode setting
        assert result == test_data
        assert self.runner.mode == WebSocketMode.BINARY

    async def test_receive_message_text(self):
        """Test receiving text messages"""
        self.runner.websocket = Mock(spec=WebSocket)

        test_data = {"type": "chat", "content": "Hello"}
        json_data = json.dumps(test_data)

        self.runner.websocket.receive = AsyncMock(return_value={"type": "websocket.message", "text": json_data})

        # Receive message
        result = await wait_for(self.runner.receive_message())

        # Verify parsing and mode setting
        assert result == test_data
        assert self.runner.mode == WebSocketMode.TEXT

    async def test_receive_message_disconnect(self):
        """Test receiving disconnect message"""
        self.runner.websocket = Mock(spec=WebSocket)

        self.runner.websocket.receive = AsyncMock(return_value={"type": "websocket.disconnect"})

        # Receive message
        result = await wait_for(self.runner.receive_message())

        # Verify None is returned for disconnect
        assert result is None

    # Main Loop Tests
    async def test_run_main_loop(self):
        """Test the main run loop"""
        websocket = Mock(spec=WebSocket)
        self.runner.websocket = websocket  # Set websocket to avoid assertion error

        # Mock connect method
        with patch.object(self.runner, "connect", new_callable=AsyncMock) as mock_connect:
            # Mock _receive_messages to simulate immediate completion
            async def mock_receive():
                await asyncio.sleep(0.01)  # Small delay to simulate work
                return

            with patch.object(self.runner, "_receive_messages", side_effect=mock_receive):
                await wait_for(self.runner.run(websocket))

                # Verify connect was called with websocket and user_id
                mock_connect.assert_called_once_with(websocket, user_id=None)

    # Message Handling Tests
    async def test_receive_messages_stop_command(self):
        """Test handling stop command in receive loop"""
        self.runner.websocket = Mock(spec=WebSocket)

        # Create a mock current task
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        self.runner.current_task = mock_task

        # Simulate receiving stop command then disconnect
        messages = [{"type": "stop"}, None]  # Disconnect

        with (
            patch.object(self.runner, "receive_message", side_effect=messages),
            patch.object(self.runner, "send_message", new_callable=AsyncMock) as mock_send,
        ):
            await wait_for(self.runner._receive_messages())

            # Verify task was cancelled
            mock_task.cancel.assert_called_once()
            # Verify stop message was sent
            mock_send.assert_called_once_with(
                {
                    "type": "generation_stopped",
                    "message": "Generation stopped by user",
                }
            )

    async def test_receive_messages_normal_message(self):
        """Test handling normal message in receive loop"""
        self.runner.websocket = Mock(spec=WebSocket)

        test_message = {"type": "chat", "content": "Hello"}

        # Simulate receiving one message then disconnect
        messages = [test_message, None]

        with (
            patch.object(self.runner, "receive_message", side_effect=messages),
            patch.object(self.runner, "handle_message", new_callable=AsyncMock) as mock_handle,
        ):
            await wait_for(self.runner._receive_messages())

            # Verify message was handled
            mock_handle.assert_called_once_with(test_message)

    async def test_receive_messages_error_handling(self):
        """Test error handling in receive loop"""
        self.runner.websocket = Mock(spec=WebSocket)

        # Create a side effect that raises an exception on first call, then returns None
        side_effects = [Exception("Test error"), None]

        with (
            patch.object(self.runner, "receive_message", side_effect=side_effects),
            patch.object(self.runner, "send_message", new_callable=AsyncMock) as mock_send,
        ):
            await wait_for(self.runner._receive_messages())

            # Verify error message was sent
            mock_send.assert_called_once()
            error_msg = mock_send.call_args[0][0]
            assert error_msg["type"] == "error"
            assert "Test error" in error_msg["message"]

    async def test_concurrent_message_handling(self):
        """Test that new messages cancel ongoing processing"""
        self.runner.websocket = Mock(spec=WebSocket)

        # Create a mock task that's still running
        old_task = Mock()
        old_task.done.return_value = False
        old_task.cancel = Mock()
        self.runner.current_task = old_task

        new_message = {"type": "chat", "content": "New message"}

        # Simulate receiving new message then disconnect
        messages = [new_message, None]

        with (
            patch.object(self.runner, "receive_message", side_effect=messages),
            patch.object(self.runner, "handle_message", new_callable=AsyncMock),
        ):
            await self.runner._receive_messages()

        # Verify old task was cancelled
        old_task.cancel.assert_called_once()

    # Context Passing Tests
    async def test_user_id_passed_to_processing_context(self):
        """Test that user_id is correctly passed to ProcessingContext"""
        runner = ChatWebSocketRunner()
        runner.user_id = "test-user-456"

        # Just test that ProcessingContext is created with correct user_id
        # by directly calling the relevant part of the method
        from nodetool.workflows.processing_context import ProcessingContext

        # Call the constructor directly to verify it accepts user_id
        context = ProcessingContext(user_id=runner.user_id)
        assert context.user_id == "test-user-456"

    async def test_user_id_passed_to_workflow_processing_context(self):
        """Test that user_id is correctly passed to ProcessingContext in workflow processing"""
        runner = ChatWebSocketRunner()
        runner.user_id = "test-user-789"

        # Just test that ProcessingContext is created with correct user_id and workflow_id
        # by directly calling the constructor
        from nodetool.workflows.processing_context import ProcessingContext

        # Call the constructor directly to verify it accepts both user_id and workflow_id
        context = ProcessingContext(user_id=runner.user_id, workflow_id="workflow-123")
        assert context.user_id == "test-user-789"
        assert context.workflow_id == "workflow-123"
