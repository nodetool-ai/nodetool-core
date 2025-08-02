"""
Integration test demonstrating SSE usage with FastAPI
"""

import pytest
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock

from nodetool.common.chat_sse_runner import ChatSSERunner
from nodetool.common.environment import Environment


# Create a test FastAPI app with SSE endpoint
app = FastAPI()


@app.post("/chat/sse")
async def chat_sse_endpoint(request: Request):
    """
    Example SSE endpoint for chat
    """
    # Get request data
    data = await request.json()
    
    # Extract auth token from headers
    auth_header = request.headers.get("Authorization", "")
    auth_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None
    
    # Create SSE runner
    runner = ChatSSERunner(auth_token)
    
    # Create streaming response
    return StreamingResponse(
        runner.process_single_request(data),
        media_type="text/event-stream"
    )


@pytest.mark.asyncio
class TestChatSSEIntegration:
    """Integration tests for SSE chat functionality"""

    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)

    @patch.object(Environment, 'use_remote_auth', return_value=False)
    def test_sse_chat_request(self, mock_auth):
        """Test making an SSE chat request"""
        request_data = {
            "thread_id": "thread_123",
            "role": "user",
            "content": "Hello, world!",
            "model": "gpt-4",
            "provider": "openai"
        }
        
        # Mock the runner's handle_message to simulate a response
        async def mock_handle_message(data):
            # Simulate some processing
            await asyncio.sleep(0.01)
            
        with patch('nodetool.common.chat_sse_runner.ChatSSERunner.handle_message', side_effect=mock_handle_message):
            with patch('nodetool.common.chat_sse_runner.ChatSSERunner._initialize_tools'):
                # Make the request
                response = self.client.post(
                    "/chat/sse",
                    json=request_data,
                    headers={"Authorization": "Bearer test_token"}
                )
                
                # Verify response
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_sse_event_parsing(self):
        """Test parsing SSE events from response"""
        # Helper function to parse SSE events
        def parse_sse_events(text: str):
            events = []
            current_event = {}
            
            for line in text.strip().split('\n'):
                if line.startswith('event:'):
                    current_event['event'] = line[6:].strip()
                elif line.startswith('data:'):
                    current_event['data'] = line[5:].strip()
                elif line == '' and current_event:
                    events.append(current_event)
                    current_event = {}
                    
            if current_event:
                events.append(current_event)
                
            return events
        
        # Test parsing
        sse_text = """event: message
data: {"type": "content", "content": "Hello"}

data: {"type": "content", "content": "World"}

event: error
data: {"type": "error", "message": "Test error"}

"""
        events = parse_sse_events(sse_text)
        
        assert len(events) == 3
        assert events[0] == {'event': 'message', 'data': '{"type": "content", "content": "Hello"}'}
        assert events[1] == {'data': '{"type": "content", "content": "World"}'}
        assert events[2] == {'event': 'error', 'data': '{"type": "error", "message": "Test error"}'}


# Example of how to use SSE runner directly
@pytest.mark.asyncio
async def test_direct_sse_usage():
    """Test using SSE runner directly without FastAPI"""
    
    with patch.object(Environment, 'use_remote_auth', return_value=False):
        runner = ChatSSERunner()
        
        # Mock handle_message to generate some events
        async def mock_handle(data):
            await runner.send_message({"type": "start", "message": "Processing"})
            await asyncio.sleep(0.01)
            await runner.send_message({"type": "content", "content": "Hello from SSE!"})
            await asyncio.sleep(0.01)
            await runner.send_message({"type": "end", "message": "Complete"})
            # Signal completion
            await runner.message_queue.put(None)
        
        with patch.object(runner, 'handle_message', side_effect=mock_handle):
            with patch.object(runner, '_initialize_tools'):
                # Process request
                request = {
                    "thread_id": "test_thread",
                    "content": "Test message",
                    "model": "gpt-4",
                    "provider": "openai"
                }
                
                # Collect events
                events = []
                async for event in runner.process_single_request(request):
                    events.append(event)
                
                # Verify events
                assert len(events) == 3
                assert '"type": "start"' in events[0]
                assert '"type": "content"' in events[1]
                assert '"type": "end"' in events[2]


# Example client code for consuming SSE
class SSEClient:
    """Example client for consuming SSE streams"""
    
    @staticmethod
    async def consume_sse_stream(url: str, data: dict, headers: dict | None = None):
        """
        Example of how a client would consume SSE events
        
        In a real implementation, you would use aiohttp or httpx
        """
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data:'):
                        yield line[5:].strip()
                    elif line.startswith('event:'):
                        # Handle special events
                        pass


# Performance test for SSE streaming
@pytest.mark.asyncio
async def test_sse_streaming_performance():
    """Test SSE streaming performance with multiple messages"""
    
    with patch.object(Environment, 'use_remote_auth', return_value=False):
        runner = ChatSSERunner()
        
        # Track timing
        start_time = asyncio.get_event_loop().time()
        message_count = 100
        
        # Mock handle_message to generate many events quickly
        async def mock_handle(data):
            for i in range(message_count):
                await runner.send_message({
                    "type": "content", 
                    "content": f"Message {i}",
                    "index": i
                })
            await runner.message_queue.put(None)
        
        with patch.object(runner, 'handle_message', side_effect=mock_handle):
            with patch.object(runner, '_initialize_tools'):
                # Process request
                request = {"content": "Test"}
                
                # Collect events
                events = []
                async for event in runner.process_single_request(request):
                    events.append(event)
                
                # Calculate performance
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                
                # Verify all messages were streamed
                assert len(events) == message_count
                
                # Should be able to stream 100 messages in under 1 second
                assert duration < 1.0
                
                # Verify message ordering
                for i, event in enumerate(events):
                    assert f'"index": {i}' in event