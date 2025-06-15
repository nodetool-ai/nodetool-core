"""Test error handling in ChatWebSocketRunner, specifically for connection errors."""

import pytest
import json
import msgpack
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from nodetool.common.chat_websocket_runner import ChatWebSocketRunner
from nodetool.metadata.types import Message


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.sent_messages = []
        self.closed = False
        self.close_code = None
        self.close_reason = None
        
    async def accept(self):
        pass
        
    async def close(self, code=1000, reason=""):
        self.closed = True
        self.close_code = code
        self.close_reason = reason
        
    async def send_bytes(self, data):
        unpacked = msgpack.unpackb(data)
        self.sent_messages.append(unpacked)
        
    async def send_text(self, data):
        parsed = json.loads(data)
        self.sent_messages.append(parsed)
        
    async def receive(self):
        # Simulate receiving a message that will trigger a connection error
        return {
            "type": "websocket.receive",
            "bytes": msgpack.packb({
                "role": "user",
                "content": "Hello, can you help me?",
                "model": "gpt-4"
            })
        }


@pytest.mark.asyncio
async def test_httpx_connect_error_handling():
    """Test that httpx.ConnectError is properly caught and sent to the client."""
    
    runner = ChatWebSocketRunner()
    websocket = MockWebSocket()
    
    # Connect without auth (in test mode)
    await runner.connect(websocket)
    
    # Mock the provider to raise httpx.ConnectError
    with patch('nodetool.common.chat_websocket_runner.provider_from_model') as mock_provider_from_model:
        mock_provider = AsyncMock()
        mock_provider_from_model.return_value = mock_provider
        
        # Make generate_messages raise httpx.ConnectError
        async def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
            # This is to make it an async generator
            yield  # pragma: no cover
        
        mock_provider.generate_messages = raise_connect_error
        
        # Process a message (this will trigger the error)
        runner.chat_history = [Message(role="user", content="Hello", model="gpt-4")]
        response = await runner.process_messages()
        
        # Check that an error message was sent to the client
        error_messages = [msg for msg in websocket.sent_messages if msg.get("type") == "error"]
        assert len(error_messages) == 1
        
        error_msg = error_messages[0]
        assert error_msg["type"] == "error"
        assert "Unable to resolve hostname" in error_msg["message"]
        assert error_msg["error_type"] == "connection_error"
        
        # Check that the response message contains the error
        assert response.role == "assistant"
        assert "connection error" in response.content.lower()


@pytest.mark.asyncio
async def test_httpx_connect_error_in_help_messages():
    """Test that httpx.ConnectError is properly caught in help message processing."""
    
    runner = ChatWebSocketRunner()
    websocket = MockWebSocket()
    
    # Connect without auth (in test mode)
    await runner.connect(websocket)
    
    # Mock the provider to raise httpx.ConnectError
    with patch('nodetool.common.chat_websocket_runner.provider_from_model') as mock_provider_from_model:
        mock_provider = AsyncMock()
        mock_provider_from_model.return_value = mock_provider
        
        # Mock create_help_answer to raise httpx.ConnectError
        with patch('nodetool.common.chat_websocket_runner.create_help_answer') as mock_help:
            async def raise_connect_error(*args, **kwargs):
                raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
                yield  # pragma: no cover
            
            mock_help.return_value = raise_connect_error()
            
            # Process a help message
            runner.chat_history = [Message(role="user", content="Help me", model="help:gpt-4")]
            response = await runner._process_help_messages("gpt-4")
            
            # Check that an error message was sent to the client
            error_messages = [msg for msg in websocket.sent_messages if msg.get("type") == "error"]
            assert len(error_messages) == 1
            
            error_msg = error_messages[0]
            assert error_msg["type"] == "error"
            assert "Unable to resolve hostname" in error_msg["message"]
            assert error_msg["error_type"] == "connection_error"
            
            # Check that the response message contains the error
            assert response.role == "assistant"
            assert "connection error while processing the help request" in response.content.lower()


@pytest.mark.asyncio  
async def test_other_httpx_connect_errors():
    """Test handling of other types of httpx.ConnectError."""
    
    runner = ChatWebSocketRunner()
    websocket = MockWebSocket()
    
    # Connect without auth (in test mode)
    await runner.connect(websocket)
    
    # Mock the provider to raise a different httpx.ConnectError
    with patch('nodetool.common.chat_websocket_runner.provider_from_model') as mock_provider_from_model:
        mock_provider = AsyncMock()
        mock_provider_from_model.return_value = mock_provider
        
        # Make generate_messages raise a different httpx.ConnectError
        async def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")
            yield  # pragma: no cover
        
        mock_provider.generate_messages = raise_connect_error
        
        # Process a message
        runner.chat_history = [Message(role="user", content="Hello", model="gpt-4")]
        response = await runner.process_messages()
        
        # Check that an error message was sent
        error_messages = [msg for msg in websocket.sent_messages if msg.get("type") == "error"]
        assert len(error_messages) == 1
        
        error_msg = error_messages[0]
        assert error_msg["type"] == "error"
        assert "Connection error: Connection refused" in error_msg["message"]
        assert error_msg["error_type"] == "connection_error"