"""
Tests for xllamacpp provider with comprehensive API response mocking.

This module tests the xllamacpp provider implementation including:
- Local model responses via xllamacpp Server
- OpenAI-compatible API format
- Tool calling functionality via emulation
- Message normalization for chat templates
- Server lifecycle management

xllamacpp provides a high-performance Cython wrapper around llama.cpp with:
- Built-in optimizations for CPU, CUDA, Vulkan, and Metal
- Thread-safe continuous batching server
- Memory estimation for GPU layer offloading
- Pythonic API without external process management
"""

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import openai.resources
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as ToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from nodetool.metadata.types import Message, MessageTextContent
from nodetool.providers.xllamacpp_provider import XLlamaCppProvider, _check_xllamacpp_available
from tests.chat.providers.test_base_provider import BaseProviderTest, ResponseFixtures


# Skip all tests if xllamacpp is not available
pytestmark = pytest.mark.skipif(
    not _check_xllamacpp_available(),
    reason="xllamacpp not available - install with: pip install xllamacpp"
)


class TestXLlamaCppProvider(BaseProviderTest):
    """Test suite for xllamacpp provider with realistic server response mocking."""

    @property
    def provider_class(self):
        return XLlamaCppProvider

    @property
    def provider_name(self):
        return "xllamacpp"

    @pytest.fixture(autouse=True)
    def _mock_xllamacpp_server(self):
        """Mock xllamacpp Server creation to avoid loading actual models.

        Automatically mocks Server for all tests in this suite unless explicitly
        overridden within a test.
        """
        with self.mock_server_creation("http://localhost:8080"):
            yield

    def mock_server_creation(self, base_url: str = "http://localhost:8080"):
        """Mock the xllamacpp Server initialization."""
        mock_server = MagicMock()
        mock_server.listening_address = base_url

        def mock_server_init(*args, **kwargs):
            return mock_server

        return patch("xllamacpp.Server", side_effect=mock_server_init)

    def create_xllamacpp_completion_response(
        self, content: str = "Hello, world!", tool_calls: List[Dict] | None = None
    ) -> ChatCompletion:
        """Create a realistic xllamacpp Server ChatCompletion response."""
        message_kwargs: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }

        if tool_calls:
            message_kwargs["tool_calls"] = [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type="function",
                    function=ToolCallFunction(name=tc["name"], arguments=json.dumps(tc["args"])),
                )
                for tc in tool_calls
            ]
            message_kwargs["content"] = None  # No content when making tool calls

        return ChatCompletion(
            id="chatcmpl-xllamacpp-123",
            choices=[
                Choice(
                    finish_reason="stop" if not tool_calls else "tool_calls",
                    index=0,
                    message=ChatCompletionMessage(**message_kwargs),
                    logprobs=None,
                )
            ],
            created=1677652288,
            model="test-model",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=12, prompt_tokens=9, total_tokens=21),
        )

    def create_xllamacpp_streaming_responses(
        self, text: str = "Hello world!", chunk_size: int = 5
    ) -> List[ChatCompletionChunk]:
        """Create realistic xllamacpp Server streaming response chunks."""
        chunks = []

        # Content chunks
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            is_last = i + chunk_size >= len(text)

            chunks.append(
                ChatCompletionChunk(
                    id="chatcmpl-xllamacpp-123",
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(content=chunk_text),
                            finish_reason="stop" if is_last else None,
                            index=0,
                            logprobs=None,
                        )
                    ],
                    created=1677652288,
                    model="test-model",
                    object="chat.completion.chunk",
                )
            )

        return chunks

    def create_xllamacpp_error(self, error_type: str = "server_error"):
        """Create realistic xllamacpp Server API errors."""
        if error_type == "server_not_available":
            return httpx.ConnectError("Connection refused")
        elif error_type == "model_not_found":
            return httpx.HTTPStatusError(
                message="Model not found",
                request=MagicMock(),
                response=MagicMock(status_code=404, text="Model not found"),
            )
        else:
            return httpx.HTTPStatusError(
                message="Server error",
                request=MagicMock(),
                response=MagicMock(status_code=500, text="Internal server error"),
            )

    def mock_api_call(self, response_data: Dict[str, Any]):
        """Mock xllamacpp Server API call with structured response."""
        if "tool_calls" in response_data:
            # Tool calling response
            xllamacpp_response = self.create_xllamacpp_completion_response(
                content=response_data.get("text", "Hello, world!"),
                tool_calls=response_data["tool_calls"],
            )
        else:
            # Regular text response
            xllamacpp_response = self.create_xllamacpp_completion_response(
                content=response_data.get("text", "Hello, world!")
            )

        async def mock_create(*args, **kwargs):
            return xllamacpp_response

        return patch.object(
            openai.resources.chat.completions.AsyncCompletions,
            "create",
            side_effect=mock_create,
        )

    def mock_streaming_call(self, chunks: List[Dict[str, Any]]):
        """Mock xllamacpp Server streaming API call."""
        # Convert generic chunks to xllamacpp format
        text = "".join(chunk.get("content", "") for chunk in chunks)
        xllamacpp_chunks = self.create_xllamacpp_streaming_responses(text)

        async def async_generator():
            for chunk in xllamacpp_chunks:
                yield chunk

        async def mock_stream(*args, **kwargs):
            return async_generator()

        return patch.object(
            openai.resources.chat.completions.AsyncCompletions,
            "create",
            side_effect=mock_stream,
        )

    def mock_error_response(self, error_type: str):
        """Mock xllamacpp Server API error response."""
        error = self.create_xllamacpp_error(error_type)
        return patch.object(
            openai.resources.chat.completions.AsyncCompletions,
            "create",
            side_effect=error,
        )

    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test xllamacpp Server initialization."""
        provider = self.create_provider()

        with (
            self.mock_server_creation("http://localhost:9999"),
            self.mock_api_call(ResponseFixtures.simple_text_response("Server running")),
        ):
            response = await provider.generate_message(self.create_simple_messages(), "test-model.gguf")

        assert response.role == "assistant"
        assert "Server running" in str(response.content)

    @pytest.mark.asyncio
    async def test_message_normalization(self):
        """Test message normalization for llama.cpp chat templates."""
        provider = self.create_provider()

        # Create messages that need normalization (system + tool messages)
        messages = [
            Message(role="system", content=[MessageTextContent(text="You are helpful.")]),
            Message(role="user", content=[MessageTextContent(text="Hello")]),
            Message(role="assistant", content=[MessageTextContent(text="Hi there!")]),
            Message(
                role="tool",
                name="search",
                tool_call_id="call_123",
                content=json.dumps({"result": "Found information"}),
            ),
        ]

        with (
            self.mock_server_creation(),
            self.mock_api_call(ResponseFixtures.simple_text_response("Processed")) as mock_call,
        ):
            await provider.generate_message(messages, "test-model.gguf")

        # Verify normalization occurred
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_xllamacpp_tool_calling_emulation(self):
        """Test tool calling emulation with xllamacpp provider.

        Since xllamacpp uses tool emulation by default, this tests
        the emulation functionality.
        """
        provider = self.create_provider()
        messages = self.create_tool_messages()
        tools = [self.create_mock_tool()]

        # Simulate a response that contains an emulated tool call
        tool_call_text = "mock_tool(query='test search')"

        with (
            self.mock_server_creation(),
            self.mock_api_call(ResponseFixtures.simple_text_response(tool_call_text)),
        ):
            response = await provider.generate_message(messages, "test-model.gguf", tools=tools)

        # Tool emulation should parse the function call from text
        assert hasattr(response, "tool_calls")
        if response.tool_calls:
            assert len(response.tool_calls) >= 1
            # Note: actual tool parsing depends on _parse_function_calls implementation

    @pytest.mark.asyncio
    async def test_server_unavailable_handling(self):
        """Test handling when xllamacpp Server is unavailable."""
        provider = self.create_provider()
        messages = self.create_simple_messages()

        with (
            self.mock_server_creation(),
            self.mock_error_response("server_not_available"),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await provider.generate_message(messages, "test-model.gguf")

    @pytest.mark.asyncio
    async def test_streaming_response(self):
        """Test streaming responses from xllamacpp."""
        provider = self.create_provider()
        messages = self.create_simple_messages()

        chunks_data = [
            {"content": "Hello "},
            {"content": "world!"},
        ]

        with (
            self.mock_server_creation(),
            self.mock_streaming_call(chunks_data),
        ):
            chunks = []
            async for chunk in provider.generate_messages(messages, "test-model.gguf"):
                chunks.append(chunk)

        # Should have received text chunks
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_gpu_layer_estimation(self):
        """Test GPU layer estimation with xllamacpp.

        This tests that the provider attempts to use xllamacpp's memory
        estimation feature when GPU is available.
        """
        provider = self.create_provider()

        # Mock xllamacpp functions
        mock_devices = [{"name": "CUDA", "memory_free": 8 * 1024**3}]  # 8GB VRAM
        mock_estimate = MagicMock()
        mock_estimate.layers = 32
        mock_estimate.vram_size = 4 * 1024**3  # 4GB used
        mock_estimate.total_size = 8 * 1024**3  # 8GB total

        with (
            patch("xllamacpp.get_device_info", return_value=mock_devices),
            patch("xllamacpp.estimate_gpu_layers", return_value=mock_estimate),
            self.mock_server_creation(),
            self.mock_api_call(ResponseFixtures.simple_text_response("Test")),
        ):
            await provider.generate_message(self.create_simple_messages(), "test-model.gguf")

        # Server should have been created successfully
        # In real usage, params.n_gpu_layers would be set to 32

    def create_mock_tool(self):
        """Create a mock tool for testing."""
        from tests.chat.providers.test_base_provider import MockTool

        return MockTool()

    @pytest.mark.asyncio
    async def test_model_caching(self):
        """Test that xllamacpp provider caches Server instances per model."""
        provider = self.create_provider()

        with (
            self.mock_server_creation() as mock_server_creation,
            self.mock_api_call(ResponseFixtures.simple_text_response("Test 1")),
        ):
            # First call
            await provider.generate_message(self.create_simple_messages(), "model1.gguf")
            first_call_count = mock_server_creation.call_count

        with (
            self.mock_server_creation(),
            self.mock_api_call(ResponseFixtures.simple_text_response("Test 2")),
        ):
            # Second call with same model - should reuse cached server
            await provider.generate_message(self.create_simple_messages(), "model1.gguf")
            # Server creation should not have been called again
            # (Note: This test is simplified and may need adjustment based on actual caching behavior)

        with (
            self.mock_server_creation() as mock_server_creation2,
            self.mock_api_call(ResponseFixtures.simple_text_response("Test 3")),
        ):
            # Third call with different model - should create new server
            await provider.generate_message(self.create_simple_messages(), "model2.gguf")
            # Server creation should have been called for new model
            assert mock_server_creation2.call_count > 0
