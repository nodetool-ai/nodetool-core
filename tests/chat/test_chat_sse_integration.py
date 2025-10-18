"""
Integration test demonstrating OpenAI-compatible SSE usage with FastAPI
"""

import pytest
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from unittest.mock import patch

from nodetool.chat.chat_sse_runner import ChatSSERunner
from nodetool.config.environment import Environment


# Create a test FastAPI app with SSE endpoint
app = FastAPI()


@app.post("/v1/chat/completions")
async def openai_chat_completions_endpoint(request: Request):
    """
    OpenAI-compatible chat completions endpoint with SSE
    """
    # Get request data
    data = await request.json()

    # Extract auth token from headers
    auth_header = request.headers.get("Authorization", "")
    auth_token = (
        auth_header.replace("Bearer ", "")
        if auth_header.startswith("Bearer ")
        else None
    )

    # Create SSE runner
    runner = ChatSSERunner(auth_token)

    # Create streaming response
    return StreamingResponse(
        runner.process_single_request(data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@pytest.mark.asyncio
class TestOpenAIChatSSEIntegration:
    """Integration tests for OpenAI-compatible SSE chat functionality"""

    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)

    def teardown_method(self):
        """Ensure the TestClient is closed to avoid event loop/resource leaks"""
        try:
            self.client.close()
        except Exception:
            pass

    @patch.object(Environment, "use_remote_auth", return_value=False)
    def test_openai_chat_completions_request(self, mock_auth):
        """Test making an OpenAI-compatible chat completions request"""
        request_data = {
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "model": "gpt-4o-mini",
            "stream": True,
            "max_tokens": 150,
        }

        # Mock the runner's handle_message to simulate a response
        async def mock_handle_message(data):
            # Simulate some processing
            await asyncio.sleep(0.01)

        with patch(
            "nodetool.chat.chat_sse_runner.ChatSSERunner.handle_message",
            side_effect=mock_handle_message,
        ):
            # Make the request
            response = self.client.post(
                "/v1/chat/completions",
                json=request_data,
                headers={"Authorization": "Bearer test_token"},
            )

            # Verify response
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

    def test_openai_sse_event_parsing(self):
        """Test parsing OpenAI-compatible SSE events from response"""

        # Helper function to parse SSE events
        def parse_sse_events(text: str):
            events = []
            current_event = {}

            for line in text.strip().split("\n"):
                if line.startswith("event:"):
                    current_event["event"] = line[6:].strip()
                elif line.startswith("data:"):
                    current_event["data"] = line[5:].strip()
                elif line == "" and current_event:
                    events.append(current_event)
                    current_event = {}

            if current_event:
                events.append(current_event)

            return events

        # Test parsing OpenAI format
        sse_text = """data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Hello"}}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " World"}}]}

data: [DONE]

"""
        events = parse_sse_events(sse_text)

        assert len(events) == 3
        assert "chat.completion.chunk" in events[0]["data"]
        assert "Hello" in events[0]["data"]
        assert " World" in events[1]["data"]
        assert events[2]["data"] == "[DONE]"


# Example of how to use SSE runner directly
@pytest.mark.asyncio
async def test_direct_openai_sse_usage():
    """Test using OpenAI-compatible SSE runner directly without FastAPI"""

    with patch.object(Environment, "use_remote_auth", return_value=False):
        runner = ChatSSERunner()

        # Mock handle_message to generate some events
        async def mock_handle(data):
            await runner.send_message({"type": "chunk", "content": "Hello"})
            await asyncio.sleep(0.01)
            await runner.send_message({"type": "chunk", "content": " from SSE!"})
            # Signal completion
            await runner.message_queue.put(None)

        with patch.object(runner, "handle_message", side_effect=mock_handle):
            # Process OpenAI-compatible request
            request = {
                "messages": [{"role": "user", "content": "Test message"}],
                "model": "gpt-4o-mini",
                "stream": True,
            }

            # Collect events
            events = []
            async for event in runner.process_single_request(request):
                events.append(event)

            # Verify OpenAI-compatible events
            assert len(events) == 3  # 2 content chunks + [DONE]
            assert "chat.completion.chunk" in events[0]
            assert "Hello" in events[0]
            assert " from SSE!" in events[1]
            assert events[2] == "data: [DONE]\n\n"


# Example client code for consuming OpenAI-compatible SSE
class OpenAISSEClient:
    """Example client for consuming OpenAI-compatible SSE streams"""

    @staticmethod
    async def consume_openai_sse_stream(
        url: str, data: dict, headers: dict | None = None
    ):
        """
        Example of how a client would consume OpenAI-compatible SSE events

        In a real implementation, you would use aiohttp or httpx
        """
        import aiohttp
        import json

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data:"):
                        data_content = line[5:].strip()
                        if data_content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_content)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            pass


# Performance test for SSE streaming
@pytest.mark.asyncio
async def test_openai_sse_streaming_performance():
    """Test OpenAI-compatible SSE streaming performance with multiple messages"""

    with patch.object(Environment, "use_remote_auth", return_value=False):
        runner = ChatSSERunner()

        # Track timing
        start_time = asyncio.get_event_loop().time()
        message_count = 50  # Reduced count since OpenAI chunks are larger

        # Mock handle_message to generate many events quickly
        async def mock_handle(data):
            for i in range(message_count):
                await runner.send_message(
                    {
                        "type": "chunk",
                        "content": f"Message {i}",
                        "index": i,
                    }
                )
            await runner.message_queue.put(None)

        with patch.object(runner, "handle_message", side_effect=mock_handle):
            # Process OpenAI-compatible request
            request = {
                "messages": [{"role": "user", "content": "Performance test"}],
                "model": "gpt-4o-mini",
            }

            # Collect events
            events = []
            async for event in runner.process_single_request(request):
                events.append(event)

            # Calculate performance
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            # Verify all messages were streamed plus [DONE]
            assert len(events) == message_count + 1

            # Should be able to stream messages quickly (allow more time for CI/slower machines)
            assert duration < 10.0

            # Verify OpenAI format and message ordering
            for i in range(message_count):
                assert "chat.completion.chunk" in events[i]
                assert f"Message {i}" in events[i]

            # Verify [DONE] termination
            assert events[-1] == "data: [DONE]\n\n"
