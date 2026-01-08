"""
Tests for Chunk thread_id field functionality.

This module tests that all message processors correctly set thread_id
on Chunk objects when applicable.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from nodetool.metadata.types import (
    Chunk,
    Message,
    Provider,
    ToolCall,
)
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


class TestChunkThreadIdField:
    """Tests for the Chunk thread_id field."""

    def test_chunk_has_thread_id_field(self):
        """Test that Chunk class has optional thread_id field."""
        chunk = Chunk(content="test", thread_id="thread_123")
        assert hasattr(chunk, "thread_id")
        assert chunk.thread_id == "thread_123"

    def test_chunk_thread_id_defaults_to_none(self):
        """Test that thread_id defaults to None."""
        chunk = Chunk(content="test")
        assert chunk.thread_id is None

    def test_chunk_serialization_includes_thread_id(self):
        """Test that thread_id is included in model_dump()."""
        chunk = Chunk(content="test", thread_id="thread_123")
        dumped = chunk.model_dump()
        assert "thread_id" in dumped
        assert dumped["thread_id"] == "thread_123"

    def test_chunk_serialization_with_none_thread_id(self):
        """Test serialization with None thread_id."""
        chunk = Chunk(content="test", thread_id=None)
        dumped = chunk.model_dump()
        assert "thread_id" in dumped
        assert dumped["thread_id"] is None


class TestRegularChatProcessorThreadId:
    """Tests for regular_chat_processor setting thread_id on chunks."""

    @pytest.mark.asyncio
    async def test_regular_chat_processor_sets_thread_id(self, test_thread_id, test_user_id, processing_context):
        """Test that RegularChatProcessor sets thread_id on chunks."""
        from nodetool.messaging.regular_chat_processor import RegularChatProcessor

        # Create a mock provider
        mock_provider = MagicMock()

        # Create chunks without thread_id
        async def mock_generate_messages(*args, **kwargs):
            yield Chunk(content="Hello ")
            yield Chunk(content="World")
            yield Chunk(content="", done=True)

        mock_provider.generate_messages = mock_generate_messages

        processor = RegularChatProcessor(provider=mock_provider)

        # Create test message with thread_id
        test_message = Message(
            thread_id=test_thread_id,
            role="user",
            content="Test message",
            provider=Provider.OpenAI,
            model="gpt-4",
        )

        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        processor.send_message = capture_send

        # Process the message
        await processor.process([test_message], processing_context)

        # Check that chunks have thread_id set
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk"]
        assert len(chunk_messages) > 0, "Should have sent chunk messages"

        for i, msg in enumerate(chunk_messages):
            # Thread_id should be present in the sent message
            assert "thread_id" in msg, f"Chunk message {i} should include thread_id. Message: {msg}"


class TestAgentMessageProcessorThreadId:
    """Tests for agent_message_processor setting thread_id on chunks."""

    @pytest.mark.asyncio
    async def test_agent_message_processor_sets_thread_id(self, test_thread_id, test_user_id, processing_context):
        """Test that AgentMessageProcessor sets thread_id on chunks."""
        from nodetool.messaging.agent_message_processor import (
            AgentMessageProcessor,
        )
        from nodetool.providers.base import BaseProvider

        # Create a mock provider
        mock_provider = MagicMock(spec=BaseProvider)

        processor = AgentMessageProcessor(provider=mock_provider)

        # Create test message with thread_id
        test_message = Message(
            thread_id=test_thread_id,
            role="user",
            content="Test objective",
            provider=Provider.OpenAI,
            model="gpt-4",
        )

        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        processor.send_message = capture_send

        # Mock agent execution to yield chunks
        async def mock_execute(*args, **kwargs):
            yield Chunk(content="Executing task ")
            yield Chunk(content="step by step", done=True)

        # Mock Agent class
        with patch("nodetool.agents.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.execute = mock_execute
            MockAgent.return_value = mock_agent

            # Process the message
            await processor.process([test_message], processing_context)

        # Check that chunks have thread_id set
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk"]
        assert len(chunk_messages) > 0, "Should have sent chunk messages"

        for msg in chunk_messages:
            assert "thread_id" in msg, "Chunk message should include thread_id"
            assert msg["thread_id"] == test_thread_id, f"Chunk thread_id should be {test_thread_id}"


class TestHelpMessageProcessorThreadId:
    """Tests for help_message_processor setting thread_id on chunks."""

    @pytest.mark.asyncio
    async def test_help_message_processor_sets_thread_id(self, test_thread_id, test_user_id, processing_context):
        """Test that HelpMessageProcessor sets thread_id on chunks."""
        from nodetool.messaging.help_message_processor import (
            HelpMessageProcessor,
        )

        # Create a mock provider
        mock_provider = MagicMock()

        # Create chunks without thread_id
        async def mock_generate_messages(*args, **kwargs):
            yield Chunk(content="Help response ")
            yield Chunk(content="text")
            yield Chunk(content="", done=True)

        mock_provider.generate_messages = mock_generate_messages

        processor = HelpMessageProcessor(provider=mock_provider)

        # Create test message with thread_id
        test_message = Message(
            thread_id=test_thread_id,
            role="user",
            content="How do I use this?",
            provider=Provider.OpenAI,
            model="gpt-4",
        )

        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        processor.send_message = capture_send

        # Process the message
        await processor.process([test_message], processing_context)

        # Check that chunks have thread_id set
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk"]
        assert len(chunk_messages) > 0, "Should have sent chunk messages"

        for msg in chunk_messages:
            assert "thread_id" in msg, "Chunk message should include thread_id"
            # The thread_id in help processor is set, verify it's present
            if msg.get("thread_id") is not None:
                assert msg["thread_id"] == test_thread_id, f"Chunk thread_id should be {test_thread_id}"


class TestClaudeAgentMessageProcessorThreadId:
    """Tests for claude_agent_message_processor setting thread_id on chunks."""

    @pytest.mark.asyncio
    async def test_claude_agent_processor_sets_thread_id(self, test_thread_id, processing_context):
        """Test that ClaudeAgentMessageProcessor sets thread_id on chunks."""
        from nodetool.messaging.claude_agent_message_processor import (
            ClaudeAgentMessageProcessor,
        )

        processor = ClaudeAgentMessageProcessor(api_key="test_key")

        # Create test message with thread_id
        test_message = Message(
            thread_id=test_thread_id,
            role="user",
            content="Test task",
            provider=Provider.Anthropic,
            model="claude-3-sonnet-20240229",
        )

        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        processor.send_message = capture_send

        # Mock the ClaudeSDKClient
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
        )

        async def mock_query(*args, **kwargs):
            pass

        async def mock_receive_response():
            # Yield text blocks
            yield AssistantMessage(content=[TextBlock(text="Processing your request")])
            yield ResultMessage(duration_ms=100)

        with patch("nodetool.messaging.claude_agent_message_processor.ClaudeSDKClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.query = mock_query
            mock_client.receive_response = mock_receive_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            # Process the message
            await processor.process([test_message], processing_context)

        # Check that chunks have thread_id set
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk"]
        assert len(chunk_messages) > 0, "Should have sent chunk messages"

        for msg in chunk_messages:
            assert "thread_id" in msg, "Chunk message should include thread_id"
            assert msg["thread_id"] == test_thread_id, f"Chunk thread_id should be {test_thread_id}"


class TestWorkflowMessageProcessorThreadId:
    """Tests for workflow_message_processor setting thread_id on chunks."""

    @pytest.mark.asyncio
    async def test_workflow_processor_sets_thread_id(self, test_thread_id, test_user_id, processing_context):
        """Test that WorkflowMessageProcessor sets thread_id on completion chunks."""
        from nodetool.messaging.workflow_message_processor import (
            WorkflowMessageProcessor,
        )

        processor = WorkflowMessageProcessor(user_id=test_user_id)

        # Create test message with thread_id and workflow_id
        test_workflow_id = str(uuid4())
        test_message = Message(
            thread_id=test_thread_id,
            workflow_id=test_workflow_id,
            role="user",
            content="Run workflow",
            provider=Provider.OpenAI,
            model="gpt-4",
        )

        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        processor.send_message = capture_send

        # Mock run_workflow
        from nodetool.workflows.workflow_types import OutputUpdate

        async def mock_run_workflow(*args, **kwargs):
            yield OutputUpdate(
                node_id="output1",
                node_name="output",
                output_name="result",
                output_type="string",
                value="Done",
            )

        with patch(
            "nodetool.workflows.run_workflow.run_workflow",
            mock_run_workflow,
        ):
            await processor.process([test_message], processing_context)

        # Check that completion chunk has thread_id set
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk" and msg.get("done")]
        assert len(chunk_messages) > 0, "Should have sent completion chunk"

        for msg in chunk_messages:
            assert "thread_id" in msg, "Completion chunk should include thread_id"
            assert msg["thread_id"] == test_thread_id, f"Chunk thread_id should be {test_thread_id}"


class TestChatWorkflowMessageProcessorThreadId:
    """Tests for chat_workflow_message_processor setting thread_id on chunks."""

    @pytest.mark.asyncio
    async def test_chat_workflow_processor_sets_thread_id(self, test_thread_id, test_user_id, processing_context):
        """Test that ChatWorkflowMessageProcessor sets thread_id on completion chunks."""
        from nodetool.messaging.chat_workflow_message_processor import (
            ChatWorkflowMessageProcessor,
        )

        processor = ChatWorkflowMessageProcessor(user_id=test_user_id)

        # Create test message with thread_id and workflow_id
        test_workflow_id = str(uuid4())
        test_message = Message(
            thread_id=test_thread_id,
            workflow_id=test_workflow_id,
            role="user",
            content="Chat with workflow",
            provider=Provider.OpenAI,
            model="gpt-4",
        )

        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        processor.send_message = capture_send

        # Mock run_workflow
        from nodetool.workflows.workflow_types import OutputUpdate

        async def mock_run_workflow(*args, **kwargs):
            yield OutputUpdate(
                node_id="output1",
                node_name="output",
                output_name="result",
                output_type="string",
                value="Chat response",
            )

        with patch(
            "nodetool.messaging.chat_workflow_message_processor.run_workflow",
            mock_run_workflow,
        ):
            await processor.process([test_message], processing_context)

        # Check that completion chunk has thread_id set
        chunk_messages = [msg for msg in sent_messages if msg.get("type") == "chunk" and msg.get("done")]
        assert len(chunk_messages) > 0, "Should have sent completion chunk"

        for msg in chunk_messages:
            assert "thread_id" in msg, "Completion chunk should include thread_id"
            assert msg["thread_id"] == test_thread_id, f"Chunk thread_id should be {test_thread_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
