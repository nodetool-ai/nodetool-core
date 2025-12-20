"""
Tests for Claude Agent SDK message processor.

This module tests the ClaudeAgentMessageProcessor which uses Anthropic's
native tool_runner (Claude Agent SDK) for agent mode processing.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from nodetool.messaging.claude_agent_message_processor import (
    ClaudeAgentMessageProcessor,
    _convert_message_to_anthropic,
    _create_anthropic_tool_wrapper,
    _extract_system_message,
)
from nodetool.metadata.types import Message as ApiMessage
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext


@pytest.fixture
def test_thread_id():
    """Create a test thread ID."""
    return str(uuid4())


@pytest.fixture
def test_user_id():
    """Create a test user ID."""
    return "test_user_1"


@pytest.fixture
def processing_context(test_user_id):
    """Create a processing context for testing."""
    return ProcessingContext(user_id=test_user_id)


@pytest.fixture
def mock_anthropic_provider():
    """Create a mock AnthropicProvider."""
    from nodetool.providers.anthropic_provider import AnthropicProvider

    # Create a real provider with mocked client
    with patch.object(AnthropicProvider, "__init__", lambda self, secrets: None):
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.api_key = "test-api-key"
        provider.client = MagicMock()
        provider.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        provider.cost = 0.0
        return provider


@pytest.fixture
def test_message(test_thread_id):
    """Create a test message for agent processing."""
    return ApiMessage(
        thread_id=test_thread_id,
        role="user",
        content="Test objective for agent",
        provider=Provider.Anthropic,
        model="claude-3-5-sonnet-20241022",
        agent_mode=True,
        use_claude_agent_sdk=True,
    )


class TestMessageConversion:
    """Tests for message conversion functions."""

    def test_convert_user_message_string(self):
        """Test converting a user message with string content."""
        message = ApiMessage(
            role="user",
            content="Hello, Claude!",
            provider=Provider.Anthropic,
        )
        result = _convert_message_to_anthropic(message)
        assert result == {"role": "user", "content": "Hello, Claude!"}

    def test_convert_assistant_message_string(self):
        """Test converting an assistant message with string content."""
        message = ApiMessage(
            role="assistant",
            content="Hello! How can I help?",
            provider=Provider.Anthropic,
        )
        result = _convert_message_to_anthropic(message)
        assert result == {"role": "assistant", "content": "Hello! How can I help?"}

    def test_convert_system_message_returns_none(self):
        """Test that system messages return None (handled separately)."""
        message = ApiMessage(
            role="system",
            content="You are a helpful assistant.",
            provider=Provider.Anthropic,
        )
        result = _convert_message_to_anthropic(message)
        assert result is None

    def test_convert_tool_message(self):
        """Test converting a tool result message."""
        message = ApiMessage(
            role="tool",
            content="Tool result",
            tool_call_id="tool_123",
            provider=Provider.Anthropic,
        )
        result = _convert_message_to_anthropic(message)
        assert result is not None
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "tool_123"


class TestExtractSystemMessage:
    """Tests for system message extraction."""

    def test_extract_system_message_string(self):
        """Test extracting a system message with string content."""
        messages = [
            ApiMessage(
                role="system",
                content="You are a helpful assistant.",
                provider=Provider.Anthropic,
            ),
            ApiMessage(
                role="user",
                content="Hello",
                provider=Provider.Anthropic,
            ),
        ]
        result = _extract_system_message(messages)
        assert result == "You are a helpful assistant."

    def test_extract_system_message_default(self):
        """Test default system message when none provided."""
        messages = [
            ApiMessage(
                role="user",
                content="Hello",
                provider=Provider.Anthropic,
            ),
        ]
        result = _extract_system_message(messages)
        assert result == "You are a helpful assistant."


class TestToolWrapper:
    """Tests for tool wrapper creation."""

    @pytest.mark.asyncio
    async def test_create_tool_wrapper(self, processing_context):
        """Test creating a tool wrapper for a nodetool Tool."""
        from nodetool.agents.tools.base import Tool

        # Create a mock tool
        class MockTool(Tool):
            name = "mock_tool"
            description = "A mock tool for testing"
            input_schema = {
                "type": "object",
                "properties": {"param1": {"type": "string", "description": "A test parameter"}},
                "required": ["param1"],
            }

            async def process(self, context, params):
                return f"Result: {params.get('param1')}"

        tool = MockTool()
        wrapped = _create_anthropic_tool_wrapper(tool, processing_context)

        assert wrapped.name == "mock_tool"
        assert wrapped.description == "A mock tool for testing"
        assert wrapped.input_schema == tool.input_schema


class TestClaudeAgentMessageProcessor:
    """Tests for the ClaudeAgentMessageProcessor class."""

    def test_init_requires_anthropic_provider(self):
        """Test that initialization requires an AnthropicProvider."""
        from nodetool.providers.base import BaseProvider

        mock_provider = MagicMock(spec=BaseProvider)
        with pytest.raises(ValueError, match="requires an AnthropicProvider"):
            ClaudeAgentMessageProcessor(mock_provider)

    def test_init_with_anthropic_provider(self, mock_anthropic_provider):
        """Test initialization with AnthropicProvider succeeds."""
        processor = ClaudeAgentMessageProcessor(mock_anthropic_provider)
        assert processor.provider == mock_anthropic_provider

    @pytest.mark.asyncio
    async def test_process_sends_planning_update(self, mock_anthropic_provider, test_message, processing_context):
        """Test that process sends planning update messages."""
        processor = ClaudeAgentMessageProcessor(mock_anthropic_provider)
        chat_history = [test_message]

        # Mock the tool runner - create an empty async generator
        async def mock_runner_iter():
            return
            yield  # pragma: no cover - makes this an async generator

        mock_runner = MagicMock()
        mock_runner.__aiter__ = lambda self: mock_runner_iter()

        # Mock until_done to return a final message
        mock_final_message = MagicMock()
        mock_final_message.content = [MagicMock(text="Final response")]
        mock_runner.until_done = AsyncMock(return_value=mock_final_message)

        mock_anthropic_provider.client.beta.messages.tool_runner = MagicMock(return_value=mock_runner)

        # Collect sent messages
        sent_messages = []
        original_send = processor.send_message

        async def capture_send(msg):
            sent_messages.append(msg)
            await original_send(msg)

        processor.send_message = capture_send

        await processor.process(
            chat_history=chat_history,
            processing_context=processing_context,
        )

        # Verify planning updates were sent
        planning_updates = [
            msg
            for msg in sent_messages
            if msg.get("type") == "message" and msg.get("execution_event_type") == "planning_update"
        ]
        assert len(planning_updates) >= 1

    @pytest.mark.asyncio
    async def test_process_non_anthropic_provider_raises(self, test_message, processing_context):
        """Test that process fails when provider is not Anthropic."""
        # Change the message to use a different provider
        test_message.provider = Provider.OpenAI

        from nodetool.providers.anthropic_provider import AnthropicProvider

        with patch.object(AnthropicProvider, "__init__", lambda self, secrets: None):
            provider = AnthropicProvider.__new__(AnthropicProvider)
            provider.client = MagicMock()

        processor = ClaudeAgentMessageProcessor(provider)

        with pytest.raises(AssertionError, match="requires Anthropic provider"):
            await processor.process(
                chat_history=[test_message],
                processing_context=processing_context,
            )


class TestUseClaudeAgentSdkFlag:
    """Tests for the use_claude_agent_sdk flag in Message type."""

    def test_message_has_flag(self):
        """Test that Message type has use_claude_agent_sdk field."""
        message = ApiMessage(
            role="user",
            content="Test",
            provider=Provider.Anthropic,
            model="claude-3-5-sonnet-20241022",
            agent_mode=True,
            use_claude_agent_sdk=True,
        )
        assert message.use_claude_agent_sdk is True

    def test_message_flag_default_is_none(self):
        """Test that use_claude_agent_sdk defaults to None."""
        message = ApiMessage(
            role="user",
            content="Test",
            provider=Provider.Anthropic,
        )
        assert message.use_claude_agent_sdk is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
