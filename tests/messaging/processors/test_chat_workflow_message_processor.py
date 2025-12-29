"""
Tests for Chat Workflow Message Processor.

This module tests the ChatWorkflowMessageProcessor which handles workflows
with run_mode="chat", processing messages and running workflows with ChatInput nodes.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from nodetool.messaging.chat_workflow_message_processor import (
    ChatWorkflowMessageProcessor,
)
from nodetool.metadata.types import (
    ImageRef,
    Message as ApiMessage,
    MessageTextContent,
    Provider,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import OutputUpdate


@pytest.fixture
def test_thread_id():
    """Create a test thread ID."""
    return str(uuid4())


@pytest.fixture
def test_user_id():
    """Create a test user ID."""
    return "test_user_1"


@pytest.fixture
def test_workflow_id():
    """Create a test workflow ID."""
    return str(uuid4())


@pytest.fixture
def processing_context(test_user_id):
    """Create a processing context for testing."""
    return ProcessingContext(user_id=test_user_id)


@pytest.fixture
def chat_processor(test_user_id):
    """Create a ChatWorkflowMessageProcessor for testing."""
    return ChatWorkflowMessageProcessor(user_id=test_user_id)


@pytest.fixture
def test_message(test_thread_id, test_workflow_id):
    """Create a test message for chat workflow processing."""
    return ApiMessage(
        thread_id=test_thread_id,
        workflow_id=test_workflow_id,
        role="user",
        content="Hello, how are you?",
        provider=Provider.OpenAI,
        model="gpt-4",
    )


@pytest.fixture
def chat_history(test_thread_id, test_workflow_id):
    """Create a chat history for testing."""
    return [
        ApiMessage(
            thread_id=test_thread_id,
            workflow_id=test_workflow_id,
            role="user",
            content="Hello",
            provider=Provider.OpenAI,
            model="gpt-4",
        ),
        ApiMessage(
            thread_id=test_thread_id,
            workflow_id=test_workflow_id,
            role="assistant",
            content="Hi there!",
            provider=Provider.OpenAI,
            model="gpt-4",
        ),
        ApiMessage(
            thread_id=test_thread_id,
            workflow_id=test_workflow_id,
            role="user",
            content="How are you?",
            provider=Provider.OpenAI,
            model="gpt-4",
        ),
    ]


class TestChatWorkflowMessageProcessorInit:
    """Tests for ChatWorkflowMessageProcessor initialization."""

    def test_init_with_user_id(self, test_user_id):
        """Test processor initialization with user ID."""
        processor = ChatWorkflowMessageProcessor(user_id=test_user_id)
        assert processor.user_id == test_user_id

    def test_init_inherits_from_message_processor(self, test_user_id):
        """Test that processor inherits from MessageProcessor."""
        from nodetool.messaging.message_processor import MessageProcessor

        processor = ChatWorkflowMessageProcessor(user_id=test_user_id)
        assert isinstance(processor, MessageProcessor)


class TestChatWorkflowMessageProcessorHelperMethods:
    """Tests for helper methods in ChatWorkflowMessageProcessor."""

    def test_extract_text_content_from_string(self, chat_processor):
        """Test extracting text content from string message."""
        message = ApiMessage(
            role="user",
            content="Test content",
            provider=Provider.OpenAI,
            model="gpt-4",
        )
        result = chat_processor._extract_text_content(message)
        assert result == "Test content"

    def test_extract_text_content_from_list(self, chat_processor):
        """Test extracting text content from list of content items."""
        message = ApiMessage(
            role="user",
            content=[
                MessageTextContent(type="text", text="First part"),
                MessageTextContent(type="text", text="Second part"),
            ],
            provider=Provider.OpenAI,
            model="gpt-4",
        )
        result = chat_processor._extract_text_content(message)
        assert result == "First part Second part"

    def test_extract_text_content_empty(self, chat_processor):
        """Test extracting text content from empty message."""
        message = ApiMessage(
            role="user",
            content=None,
            provider=Provider.OpenAI,
            model="gpt-4",
        )
        result = chat_processor._extract_text_content(message)
        assert result == ""

    def test_prepare_chat_input_params(self, chat_processor, chat_history):
        """Test preparation of ChatInput parameters from chat history."""
        params = chat_processor._prepare_chat_input_params(chat_history)
        
        # Should have both parameter names
        assert "chat_input" in params
        assert "messages" in params
        
        # Both should contain the same data
        assert params["chat_input"] == params["messages"]
        
        # Should have correct number of messages
        assert len(params["messages"]) == 3
        
        # Check message structure
        for msg_data in params["messages"]:
            assert "role" in msg_data
            assert "content" in msg_data
            assert "created_at" in msg_data

    def test_create_response_message_with_text(self, chat_processor, test_message):
        """Test creating response message from text workflow result."""
        result = {"output": "This is the response"}
        response = chat_processor._create_response_message(result, test_message)
        
        assert response.role == "assistant"
        assert response.thread_id == test_message.thread_id
        assert response.workflow_id == test_message.workflow_id
        assert response.workflow_assistant is True
        assert len(response.content) == 1
        assert isinstance(response.content[0], MessageTextContent)
        assert response.content[0].text == "This is the response"

    def test_create_response_message_with_list(self, chat_processor, test_message):
        """Test creating response message from list workflow result."""
        result = {"output": ["item1", "item2", "item3"]}
        response = chat_processor._create_response_message(result, test_message)
        
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert isinstance(response.content[0], MessageTextContent)
        assert response.content[0].text == "item1 item2 item3"

    def test_create_response_message_with_image(self, chat_processor, test_message):
        """Test creating response message from image workflow result."""
        from nodetool.metadata.types import MessageImageContent
        
        result = {
            "image_output": {
                "type": "image",
                "uri": "file:///tmp/test.png",
            }
        }
        response = chat_processor._create_response_message(result, test_message)
        
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert isinstance(response.content[0], MessageImageContent)
        assert response.content[0].image.uri == "file:///tmp/test.png"

    def test_create_response_message_empty_result(self, chat_processor, test_message):
        """Test creating response message from empty workflow result."""
        result = {}
        response = chat_processor._create_response_message(result, test_message)
        
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert isinstance(response.content[0], MessageTextContent)
        assert response.content[0].text == "Workflow completed successfully."


class TestChatWorkflowMessageProcessorProcess:
    """Tests for the main process method."""

    @pytest.mark.asyncio
    async def test_process_requires_workflow_id(self, chat_processor, processing_context):
        """Test that process requires a workflow_id."""
        message = ApiMessage(
            role="user",
            content="Test",
            workflow_id=None,  # No workflow ID
            provider=Provider.OpenAI,
            model="gpt-4",
        )
        
        with pytest.raises(AssertionError, match="Workflow ID is required"):
            await chat_processor.process([message], processing_context)

    @pytest.mark.asyncio
    async def test_process_sends_updates(
        self, chat_processor, chat_history, processing_context, test_workflow_id
    ):
        """Test that process sends workflow updates and completion messages."""
        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        chat_processor.send_message = capture_send

        # Mock run_workflow to yield some updates
        mock_updates = [
            OutputUpdate(
                node_id="output1",
                node_name="output",
                output_name="result",
                output_type="string",
                value="Test response",
            ),
        ]

        async def mock_run_workflow(*args, **kwargs):
            for update in mock_updates:
                yield update

        with patch(
            "nodetool.workflows.run_workflow.run_workflow",
            mock_run_workflow,
        ):
            await chat_processor.process(chat_history, processing_context)

        # Check that messages were sent
        assert len(sent_messages) > 0
        
        # Should have: update message(s), completion chunk, final response message
        assert any(msg.get("type") == "chunk" and msg.get("done") is True for msg in sent_messages)
        assert any(msg.get("type") == "message" for msg in sent_messages)
        
        # Check that processing is marked as complete
        assert chat_processor.is_processing is False

    @pytest.mark.asyncio
    async def test_process_handles_errors(
        self, chat_processor, chat_history, processing_context, test_workflow_id
    ):
        """Test that process handles errors gracefully."""
        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        chat_processor.send_message = capture_send

        # Mock run_workflow to raise an error
        async def mock_run_workflow(*args, **kwargs):
            raise ValueError("Test error")
            yield  # Make it a generator

        with patch(
            "nodetool.workflows.run_workflow.run_workflow",
            mock_run_workflow,
        ):
            with pytest.raises(ValueError, match="Test error"):
                await chat_processor.process(chat_history, processing_context)

        # Should have sent error and completion messages
        assert any(msg.get("type") == "error" for msg in sent_messages)
        assert any(msg.get("type") == "chunk" and msg.get("done") is True for msg in sent_messages)
        
        # Processing should be marked as complete even on error
        assert chat_processor.is_processing is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
