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
    MessageImageContent,
    MessageTextContent,
    Provider,
)
from nodetool.metadata.types import (
    Message as ApiMessage,
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
        """Test preparation of workflow params from chat history."""
        params = chat_processor._prepare_workflow_params(chat_history, chat_history[-1])

        # New interface: full message and history
        assert "message" in params
        assert "messages" in params

        # Legacy support: old interface still preserved
        assert "chat_input" in params

        # Check message structure - message is a Message object
        assert params["message"].role == chat_history[-1].role
        assert len(params["messages"]) == 3

    def test_create_response_message_with_text(self, chat_processor, test_message):
        """Test creating response message from text workflow result."""
        result = {"output": "This is the response"}
        response = chat_processor._create_response_message(result, test_message)

        assert response.role == "assistant"
        assert response.thread_id == test_message.thread_id
        assert response.workflow_id == test_message.workflow_id
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

    def test_prepare_workflow_params_includes_full_message(self, chat_processor, chat_history):
        """Test that _prepare_workflow_params passes full message object."""
        params = chat_processor._prepare_workflow_params(chat_history, chat_history[-1])

        assert "message" in params
        assert "messages" in params
        assert "chat_input" in params

        # Full message should be a Message object with all fields
        full_message = params["message"]
        assert full_message.role == chat_history[-1].role
        assert full_message.content is not None
        assert hasattr(full_message, "created_at")

        # Legacy messages list should preserve all data (this is chat_input dict format)
        assert len(params["messages"]) == 3
        for msg in params["messages"]:
            assert "role" in msg
            assert "content" in msg
            assert "created_at" in msg


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

        with pytest.raises(ValueError, match="Workflow ID is required"):
            await chat_processor.process([message], processing_context)

    @pytest.mark.asyncio
    async def test_process_sends_updates(self, chat_processor, chat_history, processing_context, test_workflow_id):
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
    async def test_process_handles_errors(self, chat_processor, chat_history, processing_context, test_workflow_id):
        """Test that process handles errors gracefully."""
        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        chat_processor.send_message = capture_send

        async def mock_run_workflow(*args, **kwargs):
            raise ValueError("Test error")
            yield

        with patch(
            "nodetool.messaging.chat_workflow_message_processor.run_workflow",
            mock_run_workflow,
        ), pytest.raises(ValueError, match="Test error"):
            await chat_processor.process(chat_history, processing_context)

        assert any(msg.get("type") == "error" for msg in sent_messages)
        assert any(msg.get("type") == "chunk" and msg.get("done") is True for msg in sent_messages)

        assert chat_processor.is_processing is False


class TestChatWorkflowMessageProcessorIssues:
    """Tests to expose issues found during code review."""

    @pytest.mark.asyncio
    async def test_process_returns_none_not_message(self, chat_processor, chat_history, processing_context):
        """
        Test that process() returns None instead of Message (LSP violation).
        The base class signature says it should return Message.
        """
        import inspect

        from nodetool.messaging.message_processor import MessageProcessor

        sig = inspect.signature(MessageProcessor.process)
        return_annotation = sig.return_annotation

        chat_processor.send_message = AsyncMock()

        async def mock_run_workflow(*args, **kwargs):
            yield OutputUpdate(
                node_id="output1",
                node_name="output",
                output_name="result",
                output_type="string",
                value="Test response",
            )

        with patch("nodetool.messaging.chat_workflow_message_processor.run_workflow", mock_run_workflow):
            result = await chat_processor.process(chat_history, processing_context)

        assert result is None, (
            f"process() returns {type(result).__name__} but base class signature declares {return_annotation}. "
            "This is an LSP violation."
        )

    def test_user_id_is_passed_to_processing_context(self, test_user_id):
        """
        Test that user_id is passed to ProcessingContext.
        This ensures the user_id is actually used.
        """
        import inspect

        from nodetool.messaging.chat_workflow_message_processor import ChatWorkflowMessageProcessor

        source = inspect.getsource(ChatWorkflowMessageProcessor)
        uses_user_id = "self.user_id" in source and "processing_context.user_id" in source

        assert uses_user_id, "user_id should be passed to ProcessingContext to ensure it's used"

    def test_create_response_message_with_invalid_image_format(self, chat_processor, test_message):
        """
        Test that create_response_message handles invalid image format gracefully.
        Previously this would crash. Now it should silently ignore invalid fields.
        """
        result = {
            "image_output": {
                "type": "image",
                "uri": "file:///tmp/test.png",
                "invalid_field": "should be ignored",
            }
        }

        response = chat_processor._create_response_message(result, test_message)
        assert len(response.content) == 1
        assert isinstance(response.content[0], MessageImageContent)
        assert response.content[0].image.uri == "file:///tmp/test.png"

    def test_extract_text_content_with_image_only(self, chat_processor):
        """
        Test that extracting text from image-only messages returns empty string.
        This loses context from non-text content.
        """
        from nodetool.metadata.types import ImageRef, MessageImageContent

        message = ApiMessage(
            role="user",
            content=[MessageImageContent(image=ImageRef(uri="file:///tmp/test.png"))],
            provider=Provider.OpenAI,
            model="gpt-4",
        )

        result = chat_processor._extract_text_content(message)
        assert result == "", (
            "Image-only messages return empty string, losing context. "
            "Consider preserving image references in chat history."
        )

    @pytest.mark.asyncio
    async def test_process_uses_value_error_not_assert(self, chat_processor, processing_context):
        """
        Test that process uses ValueError instead of AssertionError for validation.
        AssertionError should not be used for user-facing validation.
        """
        message = ApiMessage(
            role="user",
            content="Test",
            workflow_id=None,
            provider=Provider.OpenAI,
            model="gpt-4",
        )

        with pytest.raises(ValueError, match="Workflow ID is required"):
            await chat_processor.process([message], processing_context)

    def test_create_response_message_with_nested_list(self, chat_processor, test_message):
        """
        Test that create_response_message with nested lists produces poor output.
        This is a design issue - nested structures are stringified with brackets.
        """
        result = {"output": [["nested"], ["list"]], "other": {"key": "value"}}
        response = chat_processor._create_response_message(result, test_message)

        actual_text = " ".join(c.text for c in response.content if isinstance(c, MessageTextContent))
        assert "['nested']" in actual_text, "Nested lists should be stringified"
        assert "{'key':" in actual_text or '{"key":' in actual_text, "Dicts should be stringified"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
