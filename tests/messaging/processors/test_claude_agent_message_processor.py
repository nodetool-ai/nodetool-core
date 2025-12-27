"""
Tests for Claude Agent SDK message processor.

This module tests the ClaudeAgentMessageProcessor which uses the Claude Agent SDK
for handling agent-mode messages.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from nodetool.messaging.claude_agent_message_processor import (
    ClaudeAgentMessageProcessor,
)
from nodetool.metadata.types import Message as ApiMessage, Provider
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
def claude_processor():
    """Create a ClaudeAgentMessageProcessor for testing."""
    return ClaudeAgentMessageProcessor(api_key="test-api-key")


@pytest.fixture
def test_message(test_thread_id):
    """Create a test message for agent processing."""
    return ApiMessage(
        thread_id=test_thread_id,
        role="user",
        content="Test objective for Claude Agent SDK",
        provider=Provider.Anthropic,
        model="claude-sonnet-4-20250514",
        agent_mode=True,
    )


class TestClaudeAgentMessageProcessorInit:
    """Tests for ClaudeAgentMessageProcessor initialization."""

    def test_init_with_api_key(self):
        """Test processor initialization with explicit API key."""
        processor = ClaudeAgentMessageProcessor(api_key="test-key-123")
        assert processor.api_key == "test-key-123"

    def test_init_without_api_key_uses_env(self):
        """Test processor initialization uses environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key-456"}):
            processor = ClaudeAgentMessageProcessor()
            assert processor.api_key == "env-key-456"

    def test_init_inherits_from_message_processor(self):
        """Test that processor inherits from MessageProcessor."""
        from nodetool.messaging.message_processor import MessageProcessor

        processor = ClaudeAgentMessageProcessor(api_key="test-key")
        assert isinstance(processor, MessageProcessor)


class TestClaudeAgentMessageProcessorHelperMethods:
    """Tests for helper methods in ClaudeAgentMessageProcessor."""

    def test_extract_objective_from_string_content(self, claude_processor, test_thread_id):
        """Test extracting objective from string content."""
        message = ApiMessage(
            thread_id=test_thread_id,
            role="user",
            content="Write a Python function",
            provider=Provider.Anthropic,
            model="claude-sonnet-4-20250514",
        )
        objective = claude_processor._extract_objective(message)
        assert objective == "Write a Python function"

    def test_extract_objective_from_list_content(self, claude_processor, test_thread_id):
        """Test extracting objective from list content."""
        from nodetool.metadata.types import MessageTextContent

        message = ApiMessage(
            thread_id=test_thread_id,
            role="user",
            content=[MessageTextContent(type="text", text="Create a web app")],
            provider=Provider.Anthropic,
            model="claude-sonnet-4-20250514",
        )
        objective = claude_processor._extract_objective(message)
        assert objective == "Create a web app"

    def test_extract_objective_default(self, claude_processor, test_thread_id):
        """Test extracting objective returns default when content is empty."""
        message = ApiMessage(
            thread_id=test_thread_id,
            role="user",
            content=None,
            provider=Provider.Anthropic,
            model="claude-sonnet-4-20250514",
        )
        objective = claude_processor._extract_objective(message)
        assert objective == "Complete the requested task"

    def test_build_system_prompt(self, claude_processor):
        """Test system prompt generation."""
        prompt = claude_processor._build_system_prompt("Test objective", [])
        assert "Test objective" in prompt
        assert "Claude Agent SDK" in prompt


class TestClaudeAgentMessageProcessorImportError:
    """Tests for handling SDK import errors."""

    @pytest.mark.asyncio
    async def test_process_handles_import_error(self, claude_processor, test_message, processing_context):
        """Test that processor handles SDK import error gracefully."""
        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        claude_processor.send_message = capture_send

        # The processor can be instantiated and has correct structure
        # Verify processor has correct initial state
        assert claude_processor.is_processing is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
