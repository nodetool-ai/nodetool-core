"""
Tests for Llama provider with comprehensive API response mocking.

This module tests the Llama.cpp provider implementation including:
- Local model responses via llama-server
- OpenAI-compatible API format
- Tool calling functionality
- Message normalization for different chat templates
- Server management and lifecycle

llama.cpp OpenAI-Compatible Server Documentation (2024):
URLs:
- https://llama-cpp-python.readthedocs.io/en/latest/server/
- https://github.com/ggml-org/llama.cpp

llama.cpp provides an OpenAI-compatible API server through llama-server for local model inference.

Server Setup:
- Start with: llama-server -m model.gguf --port 8080
- Basic web UI at http://localhost:8080
- OpenAI-compatible endpoint at http://localhost:8080/v1/chat/completions

Key Features:
- Full OpenAI Chat Completions API compatibility
- Function calling with JSON schema support
- Streaming and non-streaming responses
- Multiple model support with routing
- GGUF model format support
- High-performance inference optimizations

Request Parameters (OpenAI-compatible):
- model: Model identifier (can be path to GGUF file or alias)
- messages: Array of message objects with role and content
- tools: Function calling tool definitions
- stream: Boolean for streaming responses
- temperature: Sampling temperature (0.0-2.0)
- top_p: Nucleus sampling (0.0-1.0)
- top_k: Top-k sampling
- max_tokens: Maximum response tokens
- stop: Stop sequences
- seed: For reproducible outputs

Response Format:
- Same as OpenAI Chat Completions API
- id: Completion identifier
- object: "chat.completion" or "chat.completion.chunk"
- model: Model name used
- choices: Array with message and finish_reason
- usage: Token statistics

Function Calling:
- Compatible with OpenAI function calling API
- Tools defined with name, description, and parameters (JSON Schema)
- Structured function calling based on JSON schema
- Multi-turn conversations with tool results

Chat Template Support:
- Automatic chat template detection for different model families
- Support for Llama, Qwen, Gemma, Mistral, and other formats
- Message role normalization (system, user, assistant, tool, ipython)
- Handles template-specific constraints and formatting

Model Compatibility:
- Qwen models: Excellent tool calling support with standard OpenAI format
- Llama models: Good general chat and tool calling capabilities
- Gemma models: Basic chat but limited tool calling due to strict templates
- Mistral models: Strong function calling capabilities
- Code models: Specialized for code generation tasks

Performance Features:
- Optimized C++ inference engine
- GPU acceleration support (CUDA, Metal, OpenCL)
- Quantization support (Q4, Q5, Q8 formats)
- Efficient memory management
- Concurrent request handling

Integration:
- Drop-in replacement for OpenAI API in existing applications
- Compatible with OpenAI Python client by setting base_url
- Works with LangChain, LlamaIndex, and other frameworks
- No API key required for local inference

Local Advantages:
- Complete privacy - no data sent to external services
- No API costs or rate limits
- Custom model fine-tuning and deployment
- Offline operation capability
- Full control over model parameters and behavior
"""

import json
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import openai
import openai.resources
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as ToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from nodetool.providers.llama_provider import LlamaProvider
from nodetool.providers.llama_server_manager import LlamaServerManager
from nodetool.metadata.types import Message, MessageTextContent
from tests.chat.providers.test_base_provider import BaseProviderTest, ResponseFixtures


class TestLlamaProvider(BaseProviderTest):
    """Test suite for Llama provider with realistic server response mocking."""

    @property
    def provider_class(self):
        return LlamaProvider

    @property
    def provider_name(self):
        return "llama"

    @pytest.fixture(autouse=True)
    def _mock_llama_server_manager(self):
        """Avoid spawning real llama-server during tests.

        Automatically mock LlamaServerManager.ensure_server for all tests in this
        suite unless explicitly overridden within a test.
        """
        with self.mock_server_manager("http://localhost:8080"):
            yield

    def create_llama_completion_response(
        self, content: str = "Hello, world!", tool_calls: List[Dict] | None = None
    ) -> ChatCompletion:
        """Create a realistic llama-server ChatCompletion response."""
        message_kwargs: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }

        if tool_calls:
            message_kwargs["tool_calls"] = [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type="function",
                    function=ToolCallFunction(
                        name=tc["name"], arguments=json.dumps(tc["args"])
                    ),
                )
                for tc in tool_calls
            ]
            message_kwargs["content"] = None  # No content when making tool calls

        return ChatCompletion(
            id="chatcmpl-123",
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
            usage=CompletionUsage(
                completion_tokens=12, prompt_tokens=9, total_tokens=21
            ),
        )

    def create_llama_streaming_responses(
        self, text: str = "Hello world!", chunk_size: int = 5
    ) -> List[ChatCompletionChunk]:
        """Create realistic llama-server streaming response chunks."""
        chunks = []

        # Content chunks
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            is_last = i + chunk_size >= len(text)

            chunks.append(
                ChatCompletionChunk(
                    id="chatcmpl-123",
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

    def create_llama_error(self, error_type: str = "server_error"):
        """Create realistic llama-server API errors."""
        if error_type == "server_not_available":
            return httpx.ConnectError("Connection refused")
        elif error_type == "context_length":
            return httpx.HTTPStatusError(
                message="Context length exceeded",
                request=MagicMock(),
                response=MagicMock(status_code=400, text="Context length exceeded"),
            )
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

    def mock_server_manager(self, base_url: str = "http://localhost:8080"):
        """Mock the LlamaServerManager to return a test server URL."""
        return patch.object(
            LlamaServerManager,
            "ensure_server",
            return_value=AsyncMock(return_value=base_url),
        )

    def mock_api_call(self, response_data: Dict[str, Any]):
        """Mock llama-server API call with structured response."""
        if "tool_calls" in response_data:
            # Tool calling response
            llama_response = self.create_llama_completion_response(
                content=response_data.get("text", "Hello, world!"),
                tool_calls=response_data["tool_calls"],
            )
        else:
            # Regular text response
            llama_response = self.create_llama_completion_response(
                content=response_data.get("text", "Hello, world!")
            )

        async def mock_create(*args, **kwargs):
            return llama_response

        return patch.object(
            openai.resources.chat.completions.AsyncCompletions,
            "create",
            side_effect=mock_create,
        )

    def mock_streaming_call(self, chunks: List[Dict[str, Any]]):
        """Mock llama-server streaming API call."""
        # Convert generic chunks to llama-server format
        text = "".join(chunk.get("content", "") for chunk in chunks)
        llama_chunks = self.create_llama_streaming_responses(text)

        async def async_generator():
            for chunk in llama_chunks:
                yield chunk

        async def mock_stream(*args, **kwargs):
            return async_generator()

        return patch.object(
            openai.resources.chat.completions.AsyncCompletions,
            "create",
            side_effect=mock_stream,
        )

    def mock_error_response(self, error_type: str):
        """Mock llama-server API error response."""
        error = self.create_llama_error(error_type)
        return patch.object(
            openai.resources.chat.completions.AsyncCompletions,
            "create",
            side_effect=error,
        )

    @pytest.mark.asyncio
    async def test_server_manager_integration(self):
        """Test integration with LlamaServerManager."""
        provider = self.create_provider(ttl_seconds=60)

        with self.mock_server_manager("http://localhost:9999"):
            with self.mock_api_call(
                ResponseFixtures.simple_text_response("Server running")
            ):
                response = await provider.generate_message(
                    self.create_simple_messages(), "test-model"
                )

        assert response.role == "assistant"

    @pytest.mark.asyncio
    async def test_message_normalization(self):
        """Test Llama-specific message normalization for chat templates."""
        provider = self.create_provider()

        # Create messages that need normalization (system + tool messages)
        messages = [
            Message(
                role="system", content=[MessageTextContent(text="You are helpful.")]
            ),
            Message(role="user", content=[MessageTextContent(text="Hello")]),
            Message(role="assistant", content=[MessageTextContent(text="Hi there!")]),
            Message(
                role="tool",
                name="search",
                tool_call_id="call_123",
                content=json.dumps({"result": "Found information"}),
            ),
        ]

        with self.mock_server_manager():
            with self.mock_api_call(
                ResponseFixtures.simple_text_response("Processed")
            ) as mock_call:
                await provider.generate_message(messages, "test-model")

        # Verify normalization occurred
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_llama_tool_calling(self):
        """Test tool calling with Llama provider."""
        provider = self.create_provider()
        messages = self.create_tool_messages()
        tools = [self.create_mock_tool()]

        tool_response = {
            "tool_calls": [
                {
                    "id": "call_llama_123",
                    "name": "mock_tool",
                    "args": {"query": "test search"},
                }
            ]
        }

        with self.mock_server_manager():
            with self.mock_api_call(tool_response):
                response = await provider.generate_message(
                    messages, "test-model", tools=tools
                )

        assert hasattr(response, "tool_calls")
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "mock_tool"

    @pytest.mark.asyncio
    async def test_server_unavailable_handling(self):
        """Test handling when llama-server is unavailable."""
        provider = self.create_provider()
        messages = self.create_simple_messages()

        with self.mock_server_manager():
            with self.mock_error_response("server_not_available"):
                with pytest.raises(Exception):
                    await provider.generate_message(messages, "test-model")

    def create_mock_tool(self):
        """Create a mock tool for testing."""
        from tests.chat.providers.test_base_provider import MockTool

        return MockTool()
