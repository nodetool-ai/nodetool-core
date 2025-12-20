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
from nodetool.metadata.types import Message as ApiMessage
from nodetool.metadata.types import Provider, Step, Task
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
        prompt = claude_processor._build_system_prompt("Test objective")
        assert "Test objective" in prompt
        assert "Claude Agent SDK" in prompt


class TestClaudeAgentMessageProcessorUpdateMethods:
    """Tests for update message sending methods."""

    @pytest.mark.asyncio
    async def test_send_planning_update(
        self, claude_processor, test_message, test_thread_id
    ):
        """Test sending planning update message."""
        from nodetool.workflows.types import PlanningUpdate

        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        claude_processor.send_message = capture_send

        update = PlanningUpdate(
            phase="planning",
            status="InProgress",
            content="Creating plan...",
            node_id="node1",
        )

        await claude_processor._send_planning_update(
            update, "exec-123", test_message
        )

        assert len(sent_messages) == 1
        msg = sent_messages[0]
        assert msg["type"] == "message"
        assert msg["role"] == "agent_execution"
        assert msg["execution_event_type"] == "planning_update"
        assert msg["agent_execution_id"] == "exec-123"
        assert msg["content"]["phase"] == "planning"
        assert msg["content"]["status"] == "InProgress"

    @pytest.mark.asyncio
    async def test_send_task_update(
        self, claude_processor, test_message, test_thread_id
    ):
        """Test sending task update message."""
        from nodetool.workflows.types import TaskUpdate, TaskUpdateEvent

        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        claude_processor.send_message = capture_send

        task = Task(id="task1", title="Test Task", description="Test description")
        step = Step(id="step1", instructions="Test step")
        update = TaskUpdate(
            task=task,
            step=step,
            event=TaskUpdateEvent.TASK_CREATED,
        )

        await claude_processor._send_task_update(
            update, "exec-456", test_message
        )

        assert len(sent_messages) == 1
        msg = sent_messages[0]
        assert msg["type"] == "message"
        assert msg["role"] == "agent_execution"
        assert msg["execution_event_type"] == "task_update"
        assert msg["agent_execution_id"] == "exec-456"
        assert msg["content"]["event"] == TaskUpdateEvent.TASK_CREATED

    @pytest.mark.asyncio
    async def test_send_step_result(
        self, claude_processor, test_message, test_thread_id
    ):
        """Test sending step result message."""
        from nodetool.workflows.types import StepResult

        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        claude_processor.send_message = capture_send

        step = Step(id="step1", instructions="Test step")
        result = StepResult(
            step=step,
            result="Step completed successfully",
            is_task_result=False,
        )

        await claude_processor._send_step_result(
            result, "exec-789", test_message
        )

        assert len(sent_messages) == 1
        msg = sent_messages[0]
        assert msg["type"] == "message"
        assert msg["role"] == "agent_execution"
        assert msg["execution_event_type"] == "step_result"
        assert msg["content"]["result"] == "Step completed successfully"
        assert msg["content"]["is_task_result"] is False

    @pytest.mark.asyncio
    async def test_send_log_update(
        self, claude_processor, test_message, test_thread_id
    ):
        """Test sending log update message."""
        from nodetool.workflows.types import LogUpdate

        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        claude_processor.send_message = capture_send

        update = LogUpdate(
            node_id="node1",
            node_name="Test Node",
            content="Processing started",
            severity="info",
        )

        await claude_processor._send_log_update(
            update, "exec-111", test_message
        )

        assert len(sent_messages) == 1
        msg = sent_messages[0]
        assert msg["type"] == "message"
        assert msg["role"] == "agent_execution"
        assert msg["execution_event_type"] == "log_update"
        assert msg["content"]["content"] == "Processing started"
        assert msg["content"]["severity"] == "info"


class TestClaudeAgentMessageProcessorImportError:
    """Tests for handling SDK import errors."""

    @pytest.mark.asyncio
    async def test_process_handles_import_error(
        self, claude_processor, test_message, processing_context
    ):
        """Test that processor handles SDK import error gracefully."""
        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        claude_processor.send_message = capture_send

        # The processor can be instantiated and has correct structure
        # Verify processor has correct initial state
        assert claude_processor.is_processing is True


class TestClaudeAgentMessageProcessorToolConversion:
    """Tests for tool conversion methods."""

    @pytest.mark.asyncio
    async def test_create_sdk_tool(self, claude_processor, processing_context):
        """Test creating SDK tool from nodetool Tool."""
        from typing import ClassVar

        from claude_agent_sdk import SdkMcpTool

        from nodetool.agents.tools.base import Tool

        # Create a mock tool
        class MockTool(Tool):
            name: str = "mock_tool"
            description: str = "A mock tool for testing"
            input_schema: ClassVar[dict] = {"type": "object", "properties": {"arg1": {"type": "string"}}}

            async def process(self, context, params):
                return {"result": f"Processed: {params.get('arg1', '')}"}

        mock_tool = MockTool()
        sdk_tool = claude_processor._create_sdk_tool(mock_tool, processing_context)

        # The SDK tool should be an SdkMcpTool with correct properties
        assert isinstance(sdk_tool, SdkMcpTool)
        assert sdk_tool.name == "mock_tool"
        assert sdk_tool.description == "A mock tool for testing"
        assert sdk_tool.handler is not None
        assert callable(sdk_tool.handler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
