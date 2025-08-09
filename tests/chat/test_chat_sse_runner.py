"""
Tests for ChatSSERunner functionality
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from nodetool.chat.chat_sse_runner import ChatSSERunner
from nodetool.common.environment import Environment
from nodetool.metadata.types import MessageTextContent, MessageImageContent


@pytest.mark.asyncio
class TestChatSSERunner:
    """Test suite for ChatSSERunner functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = ChatSSERunner("test_token")

    async def test_init(self):
        """Test initialization of ChatSSERunner"""
        runner = ChatSSERunner("test_token")
        assert runner.auth_token == "test_token"
        assert isinstance(runner.message_queue, asyncio.Queue)
        assert runner.is_connected is False
        # Inherited attributes
        assert runner.user_id is None
        assert runner.supabase is None
        assert runner.all_tools == []

    async def test_connect_local_development(self):
        """Test connection in local development mode"""
        with patch.object(Environment, "use_remote_auth", return_value=False):
            await self.runner.connect(user_id="custom_user")

            # Verify connection state
            assert self.runner.is_connected is True
            assert self.runner.user_id == "custom_user"

    async def test_connect_default_user_id(self):
        """Test connection with default user ID in local mode"""
        with patch.object(Environment, "use_remote_auth", return_value=False):
            await self.runner.connect()

            # Verify default user ID
            assert self.runner.user_id == "1"

    async def test_connect_with_valid_auth(self):
        """Test connection with valid authentication"""
        with patch.object(Environment, "use_remote_auth", return_value=True):
            with patch.object(
                self.runner, "validate_token", return_value=True
            ) as mock_validate:
                await self.runner.connect()

                # Verify token was validated
                mock_validate.assert_called_once_with("test_token")
                assert self.runner.is_connected is True

    async def test_connect_missing_auth(self):
        """Test connection with missing authentication"""
        runner = ChatSSERunner()  # No auth token

        with patch.object(Environment, "use_remote_auth", return_value=True):
            with pytest.raises(ValueError, match="Missing authentication token"):
                await runner.connect()

    async def test_connect_invalid_auth(self):
        """Test connection with invalid authentication"""
        with patch.object(Environment, "use_remote_auth", return_value=True):
            with patch.object(self.runner, "validate_token", return_value=False):
                with pytest.raises(ValueError, match="Invalid authentication token"):
                    await self.runner.connect()

    async def test_disconnect(self):
        """Test disconnect functionality"""
        self.runner.is_connected = True

        # Create a proper asyncio task
        async def dummy_task():
            await asyncio.sleep(1)

        task = asyncio.create_task(dummy_task())
        self.runner.current_task = task

        await self.runner.disconnect()

        # Verify cleanup
        assert task.cancelled()
        assert self.runner.is_connected is False
        # Verify None was put in queue to signal end
        assert self.runner.message_queue.qsize() == 1
        assert await self.runner.message_queue.get() is None

    async def test_send_message(self):
        """Test sending messages to queue"""
        self.runner.is_connected = True

        message = {"type": "content", "content": "Hello"}
        await self.runner.send_message(message)

        # Verify message was queued
        assert self.runner.message_queue.qsize() == 1
        queued_message = await self.runner.message_queue.get()
        assert queued_message == message

    async def test_send_message_when_disconnected(self):
        """Test sending messages when disconnected"""
        self.runner.is_connected = False

        message = {"type": "content", "content": "Hello"}
        await self.runner.send_message(message)

        # Verify message was not queued
        assert self.runner.message_queue.empty()

    async def test_receive_message(self):
        """Test that receive_message returns None (SSE is one-way)"""
        result = await self.runner.receive_message()
        assert result is None

    def test_convert_openai_content_text(self):
        """Test converting OpenAI text content to internal format"""
        # Test simple string content
        result = self.runner._convert_openai_content("Hello world")
        assert len(result) == 1
        assert isinstance(result[0], MessageTextContent)
        assert result[0].text == "Hello world"

    def test_convert_openai_content_multimodal(self):
        """Test converting OpenAI multimodal content to internal format"""
        # Test multi-modal content
        content = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]
        result = self.runner._convert_openai_content(content)
        assert len(result) == 2
        assert isinstance(result[0], MessageTextContent)
        assert result[0].text == "What's in this image?"
        assert isinstance(result[1], MessageImageContent)
        assert result[1].image.uri == "https://example.com/image.jpg"

    def test_convert_internal_to_openai_chunk(self):
        """Test converting internal messages to OpenAI chunks"""
        from nodetool.workflows.types import Chunk
        
        # Test content chunk
        internal_chunk = Chunk(content="Hello world")
        chunk = self.runner._convert_internal_to_openai_chunk(internal_chunk, "gpt-4")
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == "Hello world"
        assert chunk.choices[0].finish_reason is None

        # Test another chunk
        done_chunk = Chunk(content="Final response")
        chunk = self.runner._convert_internal_to_openai_chunk(done_chunk, "gpt-4")
        assert chunk.choices[0].delta.content == "Final response"

    async def test_stream_response_success(self):
        """Test successful streaming of OpenAI format responses"""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
            "stream": True,
        }

        # Mock handle_message to add messages to queue
        async def mock_handle_message(data):
            await self.runner.send_message({"type": "chunk", "content": "Response 1"})
            await self.runner.send_message(
                {"type": "chunk", "content": "Response 2"}
            )
            await self.runner.message_queue.put(None)  # End of stream

        self.runner.is_connected = True
        with patch.object(
            self.runner, "handle_message", side_effect=mock_handle_message
        ):
            with patch.object(self.runner, "disconnect", new_callable=AsyncMock):
                # Collect streamed events
                events = []
                async for event in self.runner.stream_response(request_data):
                    events.append(event)

                # Verify OpenAI format events
                assert len(events) == 3  # 2 content chunks + [DONE]
                assert "chat.completion.chunk" in events[0]
                assert "Response 1" in events[0]
                assert "chat.completion.chunk" in events[1]
                assert "Response 2" in events[1]
                assert events[2] == "data: [DONE]\n\n"

    async def test_stream_response_error(self):
        """Test error handling in stream_response"""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
        }

        # Mock handle_message to raise error
        async def mock_handle_message(data):
            raise Exception("Processing error")

        with patch.object(
            self.runner, "handle_message", side_effect=mock_handle_message
        ):
            with patch.object(self.runner, "disconnect", new_callable=AsyncMock):
                # Collect streamed events
                events = []
                async for event in self.runner.stream_response(request_data):
                    events.append(event)

                # Verify error event was sent in OpenAI format
                assert len(events) == 2  # Error chunk + [DONE]
                assert "chat.completion.chunk" in events[0]
                assert "Error: Processing error" in events[0]
                assert events[1] == "data: [DONE]\n\n"

    async def test_stream_response_cancellation(self):
        """Test cancellation handling in stream_response"""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
        }

        # Mock handle_message to be cancellable
        async def mock_handle_message(data):
            await asyncio.sleep(10)  # Long running task

        with patch.object(
            self.runner, "handle_message", side_effect=mock_handle_message
        ):
            with patch.object(self.runner, "disconnect", new_callable=AsyncMock):
                # Create a task that we'll cancel
                async def run_stream():
                    events = []
                    async for event in self.runner.stream_response(request_data):
                        events.append(event)
                    return events

                task = asyncio.create_task(run_stream())
                await asyncio.sleep(0.1)  # Let it start
                task.cancel()

                try:
                    events = await task
                except asyncio.CancelledError:
                    # This is expected
                    pass

    async def test_process_single_request_success(self):
        """Test processing a single OpenAI request with authentication"""
        request_data = {
            "auth_token": "request_token",
            "user_id": "user_123",
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
        }

        # Mock connect and stream_response
        with patch.object(
            self.runner, "connect", new_callable=AsyncMock
        ) as mock_connect:
            # Mock stream_response to yield OpenAI format events
            async def mock_stream():
                yield 'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Response"}}]}\n\n'
                yield "data: [DONE]\n\n"

            with patch.object(
                self.runner, "stream_response", return_value=mock_stream()
            ):
                with patch.object(self.runner, "disconnect", new_callable=AsyncMock):
                    # Process request
                    events = []
                    async for event in self.runner.process_single_request(request_data):
                        events.append(event)

                    # Verify auth token was updated
                    assert self.runner.auth_token == "request_token"

                    # Verify connect was called with user_id
                    mock_connect.assert_called_once_with(user_id="user_123")

                    # Verify OpenAI format events were yielded
                    assert len(events) == 2
                    assert "chat.completion.chunk" in events[0]
                    assert events[1] == "data: [DONE]\n\n"

    async def test_process_single_request_error(self):
        """Test error handling in process_single_request"""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
        }

        # Mock connect to raise error
        with patch.object(
            self.runner, "connect", side_effect=Exception("Connection failed")
        ):
            with patch.object(self.runner, "disconnect", new_callable=AsyncMock):
                # Process request
                events = []
                async for event in self.runner.process_single_request(request_data):
                    events.append(event)

                # Verify error event was sent (simple JSON format for connection errors)
                assert len(events) == 1
                assert "Connection failed" in events[0]

    async def test_message_queue_timeout_handling(self):
        """Test that stream_response handles queue timeouts properly"""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
        }

        # Mock handle_message to not add any messages
        async def mock_handle_message(data):
            # Don't add any messages to the queue
            pass

        self.runner.is_connected = True
        with patch.object(
            self.runner, "handle_message", side_effect=mock_handle_message
        ):
            with patch.object(self.runner, "disconnect", new_callable=AsyncMock):
                # Collect streamed events
                events = []
                async for event in self.runner.stream_response(request_data):
                    events.append(event)

                # Should complete with just [DONE] message
                assert len(events) == 1
                assert events[0] == "data: [DONE]\n\n"
