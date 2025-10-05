"""
Base test class for chat providers with comprehensive test scenarios.

This module provides:
- Base test class with common test scenarios
- Mock response management for different providers
- Test fixtures for various conversation patterns
- Best practices for API response testing

API Documentation References (2024):
- OpenAI Chat Completions: https://platform.openai.com/docs/api-reference/chat/create
- Anthropic Messages API: https://docs.anthropic.com/en/api/messages
- Ollama OpenAI-compatible: https://github.com/ollama/ollama/blob/main/docs/openai.md
- Google Gemini API: https://ai.google.dev/gemini-api/docs
- HuggingFace Text Generation Inference: https://huggingface.co/docs/text-generation-inference/messages_api
- llama.cpp OpenAI Server: https://llama-cpp-python.readthedocs.io/en/latest/server/
"""

import json
import pytest
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.agents.tools.base import Tool
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import Message, MessageTextContent, ToolCall, Provider
from nodetool.workflows.types import Chunk
from nodetool.workflows.processing_context import ProcessingContext


class MockTool(Tool):
    """Mock tool for testing tool calling functionality."""

    def __init__(self, name: str = "mock_tool", description: str = "A mock tool"):
        self.name = name
        self.description = description
        self.input_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer", "default": 1},
            },
            "required": ["query"],
        }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        """Process the tool call with mock response."""
        return {
            "result": f"Mock result for: {params.get('query', 'unknown')}",
            "count": params.get("count", 1),
        }


class ResponseFixtures:
    """
    Centralized fixtures for API responses across different providers.

    This class manages mock responses for different scenarios:
    - Simple text responses
    - Streaming responses
    - Tool calling responses
    - Error responses
    """

    @staticmethod
    def simple_text_response(text: str = "Hello, world!") -> Dict[str, Any]:
        """Standard text response fixture."""
        return {
            "text": text,
            "role": "assistant",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @staticmethod
    def streaming_response_chunks(
        text: str = "Hello world!", chunk_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate streaming response chunks."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            is_last = i + chunk_size >= len(text)
            chunks.append({"content": chunk_text, "done": is_last, "delta": True})
        return chunks

    @staticmethod
    def tool_call_response(
        tool_name: str = "search", args: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Tool call response fixture."""
        if args is None:
            args = {"query": "test query"}

        return {
            "tool_calls": [{"id": "call_123", "name": tool_name, "args": args}],
            "role": "assistant",
            "content": None,
        }

    @staticmethod
    def tool_result_response(
        result: str = "Tool execution completed",
    ) -> Dict[str, Any]:
        """Response after tool execution."""
        return {
            "text": result,
            "role": "assistant",
            "usage": {"prompt_tokens": 25, "completion_tokens": 8, "total_tokens": 33},
        }

    @staticmethod
    def error_response(
        error_type: str = "rate_limit", message: str = "Rate limit exceeded"
    ) -> Dict[str, Any]:
        """Error response fixtures."""
        error_templates = {
            "rate_limit": {
                "error": {
                    "type": "rate_limit_error",
                    "message": message,
                    "code": "rate_limit_exceeded",
                }
            },
            "context_length": {
                "error": {
                    "type": "invalid_request_error",
                    "message": "This model's maximum context length is 4096 tokens",
                    "code": "context_length_exceeded",
                }
            },
            "invalid_api_key": {
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key provided",
                    "code": "invalid_api_key",
                }
            },
            "timeout": {
                "error": {
                    "type": "timeout_error",
                    "message": "Request timed out",
                    "code": "timeout",
                }
            },
        }
        return error_templates.get(error_type, error_templates["rate_limit"])


class BaseProviderTest(ABC):
    """
    Base test class for chat providers.

    This class defines comprehensive test scenarios that all providers should pass:
    - Basic text generation
    - Streaming responses
    - Tool calling
    - Error handling
    - Context length management
    - Usage tracking
    """

    @property
    @abstractmethod
    def provider_class(self) -> Type[BaseProvider]:
        """Return the provider class to test."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for identification."""
        pass

    def create_provider(self, **kwargs) -> BaseProvider:
        """Create a provider instance with optional configuration."""
        return self.provider_class(**kwargs)

    def create_simple_messages(self, content: str = "Hello") -> List[Message]:
        """Create simple test messages."""
        return [Message(role="user", content=[MessageTextContent(text=content)])]

    def create_conversation_messages(self) -> List[Message]:
        """Create a multi-turn conversation."""
        return [
            Message(
                role="user",
                content=[MessageTextContent(text="What's the weather like?")],
            ),
            Message(
                role="assistant",
                content=[
                    MessageTextContent(
                        text="I don't have access to current weather data."
                    )
                ],
            ),
            Message(
                role="user",
                content=[MessageTextContent(text="Can you suggest what to wear?")],
            ),
        ]

    def create_tool_messages(self) -> List[Message]:
        """Create messages that should trigger tool calls."""
        return [
            Message(
                role="user",
                content=[
                    MessageTextContent(
                        text="Search for information about Python testing"
                    )
                ],
            )
        ]

    @pytest.mark.asyncio
    async def test_basic_text_generation(self):
        """Test basic text generation functionality."""
        provider = self.create_provider()
        messages = self.create_simple_messages("Say hello")

        with self.mock_api_call(ResponseFixtures.simple_text_response("Hello!")):
            response = await provider.generate_message(messages, "test-model")

        assert response.role == "assistant"
        assert response.content is not None
        # Provider-specific content format validation should be in subclasses

    @pytest.mark.asyncio
    async def test_streaming_generation(self):
        """Test streaming response functionality."""
        provider = self.create_provider()
        messages = self.create_simple_messages("Count to 5")

        chunks = ResponseFixtures.streaming_response_chunks("1 2 3 4 5", chunk_size=2)

        with self.mock_streaming_call(chunks):
            collected_chunks = []
            async for chunk in provider.generate_messages(messages, "test-model"):
                collected_chunks.append(chunk)

        assert len(collected_chunks) > 0
        # Verify streaming behavior
        assert any(isinstance(chunk, Chunk) for chunk in collected_chunks)

        # Last chunk should be marked as done
        if collected_chunks:
            last_chunk = next(
                (c for c in reversed(collected_chunks) if isinstance(c, Chunk)), None
            )
            if last_chunk:
                assert last_chunk.done

    @pytest.mark.asyncio
    async def test_tool_calling(self):
        """Test tool calling functionality."""
        provider = self.create_provider()
        messages = self.create_tool_messages()
        tools = [MockTool()]

        tool_response = ResponseFixtures.tool_call_response(
            "mock_tool", {"query": "test"}
        )

        with self.mock_api_call(tool_response):
            response = await provider.generate_message(
                messages, "test-model", tools=tools
            )

        if hasattr(response, "tool_calls") and response.tool_calls:
            assert len(response.tool_calls) > 0
            assert response.tool_calls[0].name == "mock_tool"

    @pytest.mark.asyncio
    async def test_tool_result_continuation(self):
        """Test conversation continuation after tool execution."""
        provider = self.create_provider()
        messages = self.create_tool_messages()
        tools = [MockTool()]

        # First response with tool call
        tool_response = ResponseFixtures.tool_call_response(
            "mock_tool", {"query": "test"}
        )

        with self.mock_api_call(tool_response):
            response = await provider.generate_message(
                messages, "test-model", tools=tools
            )

        if hasattr(response, "tool_calls") and response.tool_calls:
            # Add tool result to conversation
            tool_call = response.tool_calls[0]
            tool = next((t for t in tools if t.name == tool_call.name), None)
            assert tool is not None

            context = ProcessingContext()
            result = await tool.process(context, tool_call.args or {})

            # Add tool result message
            messages.append(
                Message(
                    role="tool",
                    name=tool.name,
                    tool_call_id=tool_call.id,
                    content=json.dumps(result),
                )
            )

            # Test continuation
            continuation_response = ResponseFixtures.tool_result_response(
                "Task completed"
            )

            with self.mock_api_call(continuation_response):
                final_response = await provider.generate_message(
                    messages, "test-model", tools=tools
                )

            assert final_response.role == "assistant"

    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        """Test token usage tracking."""
        provider = self.create_provider()
        messages = self.create_simple_messages()

        initial_usage = provider.usage

        with self.mock_api_call(ResponseFixtures.simple_text_response()):
            await provider.generate_message(messages, "test-model")

        final_usage = provider.usage

        # Usage should have been updated (exact behavior depends on provider)
        assert isinstance(final_usage, dict)
        assert "total_tokens" in final_usage

    @pytest.mark.asyncio
    async def test_context_length_error_handling(self):
        """Test handling of context length errors."""
        provider = self.create_provider()

        # Create very long messages that would exceed context length
        long_content = "A" * 10000  # Very long message
        messages = self.create_simple_messages(long_content)

        with self.mock_error_response("context_length"):
            with pytest.raises(Exception) as exc_info:
                await provider.generate_message(messages, "test-model")

            # Provider should recognize this as a context length error
            assert provider.is_context_length_error(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self):
        """Test handling of rate limit errors."""
        provider = self.create_provider()
        messages = self.create_simple_messages()

        with self.mock_error_response("rate_limit"):
            with pytest.raises(Exception):
                await provider.generate_message(messages, "test-model")

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self):
        """Test handling of authentication errors."""
        provider = self.create_provider()
        messages = self.create_simple_messages()

        with self.mock_error_response("invalid_api_key"):
            with pytest.raises(Exception):
                await provider.generate_message(messages, "test-model")

    @pytest.mark.asyncio
    async def test_model_parameter_passing(self):
        """Test that model parameters are correctly passed."""
        provider = self.create_provider()
        messages = self.create_simple_messages()

        with self.mock_api_call(ResponseFixtures.simple_text_response()) as mock_call:
            await provider.generate_message(
                messages, "test-model", max_tokens=100, temperature=0.7
            )

        # Verify model parameters were passed (implementation specific)
        mock_call.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import asyncio

        provider = self.create_provider()
        messages = self.create_simple_messages()

        with self.mock_api_call(ResponseFixtures.simple_text_response()):
            # Create multiple concurrent requests
            tasks = [
                provider.generate_message(messages, f"model-{i}") for i in range(3)
            ]

            responses = await asyncio.gather(*tasks)

            assert len(responses) == 3
            for response in responses:
                assert response.role == "assistant"

    @pytest.mark.asyncio
    async def test_empty_message_handling(self):
        """Test handling of edge cases like empty messages."""
        provider = self.create_provider()

        # Test empty message list
        with pytest.raises((ValueError, Exception)):
            await provider.generate_message([], "test-model")

    def test_context_length_detection(self):
        """Test context length detection for different models."""
        provider = self.create_provider()

        # Test with common model names
        context_length = provider.get_context_length("test-model")
        assert isinstance(context_length, int)
        assert context_length > 0

    def test_container_environment(self):
        """Test container environment variable configuration."""
        provider = self.create_provider()

        env_vars = provider.get_container_env()
        assert isinstance(env_vars, dict)
        # All values should be strings
        for key, value in env_vars.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    # Abstract methods that subclasses must implement for provider-specific mocking

    @abstractmethod
    def mock_api_call(self, response_data: Dict[str, Any]) -> MagicMock:
        """Mock a single API call with the given response data."""
        pass

    @abstractmethod
    def mock_streaming_call(self, chunks: List[Dict[str, Any]]) -> MagicMock:
        """Mock a streaming API call with the given chunks."""
        pass

    @abstractmethod
    def mock_error_response(self, error_type: str) -> MagicMock:
        """Mock an error response of the given type."""
        pass


# Utility functions for test data management


def create_test_conversation(turns: int = 3) -> List[Message]:
    """Create a multi-turn conversation for testing."""
    messages = []
    for i in range(turns):
        if i % 2 == 0:
            messages.append(
                Message(
                    role="user",
                    content=[MessageTextContent(text=f"User message {i//2 + 1}")],
                )
            )
        else:
            messages.append(
                Message(
                    role="assistant",
                    content=[MessageTextContent(text=f"Assistant response {i//2 + 1}")],
                )
            )
    return messages


def create_tool_calling_scenario() -> Dict[str, Any]:
    """Create a complete tool calling scenario."""
    return {
        "initial_message": Message(
            role="user",
            content=[MessageTextContent(text="Calculate the square root of 16")],
        ),
        "tool_call": ToolCall(
            id="call_123", name="calculator", args={"operation": "sqrt", "value": 16}
        ),
        "tool_result": {"result": 4.0, "operation": "sqrt"},
        "final_response": "The square root of 16 is 4.0",
    }
