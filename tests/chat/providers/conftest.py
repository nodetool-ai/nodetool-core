"""
Pytest configuration and fixtures for chat provider tests.

This module provides:
- Common fixtures for all provider tests
- Test configuration and markers
- Shared utilities and helpers
- Mock management and cleanup
"""

import asyncio
import os
from typing import Any, ClassVar, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import Message, MessageTextContent, ToolCall
from nodetool.workflows.processing_context import ProcessingContext


# Mock tiktoken to avoid network calls during tests
@pytest.fixture(scope="session", autouse=True)
def mock_tiktoken():
    """Mock tiktoken to prevent network calls for encoding downloads."""
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # Mock token IDs
    mock_encoding.decode.return_value = "mocked text"

    with (
        patch("tiktoken.get_encoding", return_value=mock_encoding),
        patch("tiktoken.encoding_for_model", return_value=mock_encoding),
    ):
        yield mock_encoding


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring external services",
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_api_key: mark test as requiring API key configuration")
    config.addinivalue_line("markers", "requires_server: mark test as requiring local server")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names and requirements."""
    for item in items:
        # Mark integration tests
        if "integration" in item.name or "real_api" in item.name:
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if any(keyword in item.name for keyword in ["comprehensive", "stress", "performance"]):
            item.add_marker(pytest.mark.slow)

        # Mark tests requiring API keys
        if any(provider in item.name for provider in ["openai", "anthropic", "gemini"]):
            item.add_marker(pytest.mark.requires_api_key)

        # Mark tests requiring servers
        if any(provider in item.name for provider in ["llama", "ollama"]):
            item.add_marker(pytest.mark.requires_server)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def simple_messages() -> List[Message]:
    """Fixture providing simple test messages."""
    return [Message(role="user", content=[MessageTextContent(text="Hello, how are you?")])]


@pytest.fixture
def conversation_messages() -> List[Message]:
    """Fixture providing a multi-turn conversation."""
    return [
        Message(role="user", content=[MessageTextContent(text="What's the weather like?")]),
        Message(
            role="assistant",
            instructions=[MessageTextContent(text="I don't have access to current weather data.")],
        ),
        Message(role="user", content=[MessageTextContent(text="What should I wear then?")]),
    ]


@pytest.fixture
def tool_messages() -> List[Message]:
    """Fixture providing messages that should trigger tool calls."""
    return [
        Message(
            role="user",
            instructions=[MessageTextContent(text="Search for information about machine learning")],
        )
    ]


@pytest.fixture
def mock_tool():
    """Fixture providing a mock tool for testing."""

    class TestTool(Tool):
        name = "test_search"
        description = "Search for information"
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }

        async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
            return {
                "results": [
                    {
                        "title": f"Result for: {params.get('query', 'unknown')}",
                        "score": 0.9,
                    }
                ],
                "total": 1,
            }

    return TestTool()


@pytest.fixture
def calculator_tool():
    """Fixture providing a calculator tool for testing."""

    class CalculatorTool(Tool):
        name = "calculator"
        description = "Perform mathematical calculations"
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                },
                "precision": {"type": "integer", "default": 2},
            },
            "required": ["expression"],
        }

        async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
            expression = params.get("expression", "")
            try:
                # Simple evaluation for testing (in real implementation, use safe evaluation)
                result = eval(expression)  # Only for testing!
                return {
                    "result": result,
                    "expression": expression,
                    "precision": params.get("precision", 2),
                }
            except Exception as e:
                return {"error": str(e), "expression": expression}

    return CalculatorTool()


@pytest.fixture
def processing_context():
    """Fixture providing a processing context."""
    return ProcessingContext()


@pytest.fixture
def sample_tool_call() -> ToolCall:
    """Fixture providing a sample tool call."""
    return ToolCall(id="call_test_123", name="test_search", args={"query": "test query", "limit": 3})


@pytest.fixture(scope="session")
def api_key_config():
    """Fixture providing API key configuration from environment."""
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "huggingface": os.getenv("HF_TOKEN"),
    }


@pytest.fixture
def mock_http_client():
    """Fixture providing a mock HTTP client."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_openai_client():
    """Fixture providing a mock OpenAI client."""
    with patch("openai.AsyncOpenAI") as mock_client:
        yield mock_client


@pytest.fixture
def mock_anthropic_client():
    """Fixture providing a mock Anthropic client."""
    with patch("anthropic.AsyncAnthropic") as mock_client:
        yield mock_client


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Automatically cleanup environment after each test."""
    yield
    # Clean up any environment variables that might have been set during testing
    test_env_vars = [
        "TEST_OPENAI_API_KEY",
        "TEST_ANTHROPIC_API_KEY",
        "TEST_GEMINI_API_KEY",
    ]
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def temp_model_cache():
    """Fixture providing a temporary model cache directory."""
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="test_model_cache_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Test data fixtures


@pytest.fixture
def sample_responses():
    """Fixture providing sample responses for different scenarios."""
    return {
        "simple_text": {
            "content": "Hello! How can I help you today?",
            "role": "assistant",
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        },
        "with_tool_call": {
            "content": None,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "name": "search",
                    "args": {"query": "machine learning"},
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 0, "total_tokens": 15},
        },
        "tool_result": {
            "content": "I found some information about machine learning. It's a subset of artificial intelligence focused on algorithms that improve through experience.",
            "role": "assistant",
            "usage": {"prompt_tokens": 25, "completion_tokens": 22, "total_tokens": 47},
        },
        "error": {
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded",
                "code": "rate_limit_exceeded",
            }
        },
    }


@pytest.fixture
def streaming_chunks():
    """Fixture providing streaming response chunks."""
    return [
        {"content": "Hello", "done": False},
        {"content": " there", "done": False},
        {"content": "! How", "done": False},
        {"content": " can I", "done": False},
        {"content": " help?", "done": True},
    ]


# Performance testing fixtures


@pytest.fixture
def performance_metrics():
    """Fixture for collecting performance metrics during tests."""
    metrics = {
        "start_time": None,
        "end_time": None,
        "memory_usage": [],
        "response_times": [],
    }

    import time

    import psutil

    def start_monitoring():
        metrics["start_time"] = time.time()
        metrics["memory_usage"].append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB

    def stop_monitoring():
        metrics["end_time"] = time.time()
        metrics["memory_usage"].append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB

    def record_response_time(duration: float):
        metrics["response_times"].append(duration)

    metrics["start"] = start_monitoring
    metrics["stop"] = stop_monitoring
    metrics["record"] = record_response_time

    return metrics


# Skip conditions


def pytest_runtest_setup(item):
    """Skip tests based on environment conditions."""
    # Skip integration tests if no API keys are available
    if item.get_closest_marker("requires_api_key"):
        api_keys_available = any(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GEMINI_API_KEY"),
                os.getenv("HF_TOKEN"),
            ]
        )
        if not api_keys_available:
            pytest.skip("No API keys configured for integration testing")

    # Skip server tests if servers are not available
    if item.get_closest_marker("requires_server"):
        # Could add server availability checks here
        pass
