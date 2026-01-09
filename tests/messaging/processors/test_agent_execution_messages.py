"""
Tests for agent execution message storage in the database.

This module tests that agent execution events (planning_update, task_update, step_result)
are properly stored as messages with role="agent_execution" and that they are filtered out
when loading chat history for LLM processing.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from nodetool.chat.base_chat_runner import BaseChatRunner
from nodetool.messaging.agent_message_processor import AgentMessageProcessor
from nodetool.metadata.types import Message as ApiMessage
from nodetool.metadata.types import Provider, Step, Task
from nodetool.models.message import Message as DBMessage
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import (
    Chunk,
    PlanningUpdate,
    StepResult,
    TaskUpdate,
    TaskUpdateEvent,
)


# Concrete implementation of BaseChatRunner for testing
class TestChatRunner(BaseChatRunner):
    """Test implementation of BaseChatRunner for testing purposes."""

    async def connect(self, **kwargs):
        pass

    async def disconnect(self):
        pass

    async def send_message(self, message: dict):
        pass

    async def receive_message(self):
        return None


@pytest.fixture
async def test_thread_id():
    """Create a test thread ID."""
    return str(uuid4())


@pytest.fixture
async def test_user_id():
    """Create a test user ID."""
    return "test_user_1"


@pytest.fixture
async def processing_context(test_user_id):
    """Create a processing context for testing."""
    return ProcessingContext(user_id=test_user_id)


@pytest.fixture
async def agent_processor():
    """Create an AgentMessageProcessor for testing."""
    from nodetool.providers.base import BaseProvider

    # Create a mock provider
    mock_provider = MagicMock(spec=BaseProvider)
    return AgentMessageProcessor(provider=mock_provider)


@pytest.fixture
async def test_message(test_thread_id):
    """Create a test message for agent processing."""
    return ApiMessage(
        thread_id=test_thread_id,
        role="user",
        instructions="Test objective for agent",
        provider=Provider.OpenAI,
        model="gpt-4o",
        agent_mode=True,
    )


class TestAgentExecutionMessageStorage:
    """Tests for storing agent execution events as database messages."""

    @pytest.mark.asyncio
    async def test_task_update_saved_as_message(
        self,
        agent_processor,
        test_message,
        processing_context,
        test_thread_id,
        test_user_id,
    ):
        """Test that TaskUpdate events are saved as agent_execution messages."""
        chat_history = [test_message]

        # Create a mock agent that yields a TaskUpdate event
        task = Task(id="task1", title="Test Task", description="Test Description")
        step = Step(id="step1", instructions="Test Subtask")
        task_update = TaskUpdate(event=TaskUpdateEvent.STEP_COMPLETED, task=task, step=step)

        # Mock the agent execution to yield the task update
        with patch("nodetool.agents.agent.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.results = "Test result"

            # Create an async generator that yields the task update and a completion chunk
            async def mock_execute(context):
                yield task_update
                yield Chunk(content="", done=True)

            mock_agent_instance.execute = mock_execute
            MockAgent.return_value = mock_agent_instance

            # Collect sent messages instead of mocking send_message
            sent_messages = []
            original_send = agent_processor.send_message

            async def capture_send(msg):
                sent_messages.append(msg)
                await original_send(msg)

            agent_processor.send_message = capture_send

            # Execute the agent processor
            await agent_processor.process(
                chat_history=chat_history,
                processing_context=processing_context,
            )

            # Verify that a task_update message was sent
            task_update_messages = [
                msg
                for msg in sent_messages
                if msg.get("type") == "message" and msg.get("execution_event_type") == "task_update"
            ]

            assert len(task_update_messages) > 0, "TaskUpdate should be saved as message"

            message = task_update_messages[0]
            assert message["thread_id"] == test_thread_id
            assert message["role"] == "agent_execution"
            assert message["execution_event_type"] == "task_update"
            assert "agent_execution_id" in message

            content = message["content"]
            assert content["type"] == "task_update"
            assert content["event"] == TaskUpdateEvent.STEP_COMPLETED
            assert content["task"] is not None
            assert content["step"] is not None

    @pytest.mark.asyncio
    async def test_planning_update_saved_as_message(
        self,
        agent_processor,
        test_message,
        processing_context,
        test_thread_id,
        test_user_id,
    ):
        """Test that PlanningUpdate events are saved as agent_execution messages."""
        chat_history = [test_message]

        # Create a mock planning update - use "Success" status to trigger message sending
        planning_update = PlanningUpdate(
            phase="planning",
            status="Success",
            instructions="Creating task plan",
            node_id="node1",
        )

        with patch("nodetool.agents.agent.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.results = "Test result"

            async def mock_execute(context):
                yield planning_update
                yield Chunk(content="", done=True)

            mock_agent_instance.execute = mock_execute
            MockAgent.return_value = mock_agent_instance

            # Collect sent messages
            sent_messages = []
            original_send = agent_processor.send_message

            async def capture_send(msg):
                sent_messages.append(msg)
                await original_send(msg)

            agent_processor.send_message = capture_send

            await agent_processor.process(
                chat_history=chat_history,
                processing_context=processing_context,
            )

            planning_messages = [
                msg
                for msg in sent_messages
                if msg.get("type") == "message" and msg.get("execution_event_type") == "planning_update"
            ]

            assert len(planning_messages) > 0, "PlanningUpdate should be saved as message"

            message = planning_messages[0]
            assert message["thread_id"] == test_thread_id
            assert message["role"] == "agent_execution"
            assert message["execution_event_type"] == "planning_update"

            content = message["content"]
            assert content["type"] == "planning_update"
            assert content["phase"] == "planning"
            assert content["status"] == "Success"

    @pytest.mark.asyncio
    async def test_step_result_saved_as_message(
        self,
        agent_processor,
        test_message,
        processing_context,
        test_thread_id,
        test_user_id,
    ):
        """Test that StepResult events are saved as agent_execution messages."""
        chat_history = [test_message]

        # Create a mock step result
        step = Step(id="step1", instructions="Test Subtask")
        step_result = StepResult(
            step=step,
            result="Subtask completed successfully",
            is_task_result=False,
        )

        with patch("nodetool.agents.agent.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.results = "Test result"

            async def mock_execute(context):
                yield step_result
                yield Chunk(content="", done=True)

            mock_agent_instance.execute = mock_execute
            MockAgent.return_value = mock_agent_instance

            # Collect sent messages
            sent_messages = []
            original_send = agent_processor.send_message

            async def capture_send(msg):
                sent_messages.append(msg)
                await original_send(msg)

            agent_processor.send_message = capture_send

            await agent_processor.process(
                chat_history=chat_history,
                processing_context=processing_context,
            )

            step_messages = [
                msg
                for msg in sent_messages
                if msg.get("type") == "message" and msg.get("execution_event_type") == "step_result"
            ]

            assert len(step_messages) > 0, "StepResult should be saved as message"

            message = step_messages[0]
            assert message["thread_id"] == test_thread_id
            assert message["role"] == "agent_execution"
            assert message["execution_event_type"] == "step_result"

            content = message["content"]
            assert content["type"] == "step_result"
            assert content["result"] == "Subtask completed successfully"

    @pytest.mark.asyncio
    async def test_all_events_share_same_execution_id(self, agent_processor, test_message, processing_context):
        """Test that all events from the same execution share the same agent_execution_id."""
        chat_history = [test_message]

        # Create multiple events - use statuses that trigger message sending
        task = Task(id="task1", title="Test Task")
        step = Step(id="step1", instructions="Test Subtask")
        task_update = TaskUpdate(event=TaskUpdateEvent.STEP_COMPLETED, task=task, step=step)
        planning_update = PlanningUpdate(phase="planning", status="Success", instructions="Planning", node_id="node1")

        with patch("nodetool.agents.agent.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.results = "Test result"

            async def mock_execute(context):
                yield planning_update
                yield task_update
                yield Chunk(content="", done=True)

            mock_agent_instance.execute = mock_execute
            MockAgent.return_value = mock_agent_instance

            # Collect sent messages
            sent_messages = []
            original_send = agent_processor.send_message

            async def capture_send(msg):
                sent_messages.append(msg)
                await original_send(msg)

            agent_processor.send_message = capture_send

            await agent_processor.process(
                chat_history=chat_history,
                processing_context=processing_context,
            )

            # Get all execution message calls
            execution_messages = [
                msg for msg in sent_messages if msg.get("type") == "message" and msg.get("role") == "agent_execution"
            ]

            assert len(execution_messages) >= 2, "Should have multiple execution events"

            # Extract execution IDs
            execution_ids = [msg["agent_execution_id"] for msg in execution_messages]

            # All execution IDs should be the same
            assert len(set(execution_ids)) == 1, "All events should share the same execution ID"


class TestAgentExecutionMessageFiltering:
    """Tests for filtering agent_execution messages from chat history."""

    @pytest.mark.asyncio
    async def test_agent_execution_messages_filtered_from_history(self, test_thread_id, test_user_id):
        """Test that agent_execution messages are filtered out when loading chat history."""
        # Create a mix of message types
        messages_to_create = [
            {
                "thread_id": test_thread_id,
                "user_id": test_user_id,
                "role": "user",
                "content": "Hello",
            },
            {
                "thread_id": test_thread_id,
                "user_id": test_user_id,
                "role": "agent_execution",
                "execution_event_type": "planning_update",
                "content": json.dumps({"type": "planning_update", "phase": "planning"}),
                "agent_execution_id": str(uuid4()),
            },
            {
                "thread_id": test_thread_id,
                "user_id": test_user_id,
                "role": "assistant",
                "content": "Hello back",
            },
            {
                "thread_id": test_thread_id,
                "user_id": test_user_id,
                "role": "agent_execution",
                "execution_event_type": "task_update",
                "content": json.dumps({"type": "task_update", "event": "started"}),
                "agent_execution_id": str(uuid4()),
            },
        ]

        # Mock the paginate method to return our test messages
        # Note: DBMessage.create is async but we're just creating instances for testing
        mock_db_messages = []
        for msg in messages_to_create:
            # Create instances directly for testing (not using async create)
            message = DBMessage(id=str(uuid4()), created_at=None, **msg)
            mock_db_messages.append(message)

        with patch.object(DBMessage, "paginate", new_callable=AsyncMock) as mock_paginate:
            mock_paginate.return_value = (mock_db_messages, None)

            # Create a test chat runner
            chat_runner = TestChatRunner()
            chat_runner.user_id = test_user_id

            # Get chat history
            history = await chat_runner.get_chat_history_from_db(test_thread_id)

            # Verify that agent_execution messages are filtered out
            assert len(history) == 2, "Should only return user and assistant messages"

            roles = [msg.role for msg in history]
            assert "user" in roles
            assert "assistant" in roles
            assert "agent_execution" not in roles

    @pytest.mark.asyncio
    async def test_only_regular_messages_sent_to_llm(self, test_thread_id, test_user_id):
        """Test that only user, assistant, system, and tool messages are sent to LLM."""
        messages_to_create = [
            {
                "thread_id": test_thread_id,
                "user_id": test_user_id,
                "role": "system",
                "content": "You are a helpful assistant",
            },
            {
                "thread_id": test_thread_id,
                "user_id": test_user_id,
                "role": "user",
                "content": "What is 2+2?",
            },
            {
                "thread_id": test_thread_id,
                "user_id": test_user_id,
                "role": "agent_execution",
                "execution_event_type": "planning_update",
                "content": json.dumps({"type": "planning_update"}),
                "agent_execution_id": str(uuid4()),
            },
            {
                "thread_id": test_thread_id,
                "user_id": test_user_id,
                "role": "agent_execution",
                "execution_event_type": "task_update",
                "content": json.dumps({"type": "task_update"}),
                "agent_execution_id": str(uuid4()),
            },
            {
                "thread_id": test_thread_id,
                "user_id": test_user_id,
                "role": "assistant",
                "content": "2+2 equals 4",
            },
            {
                "thread_id": test_thread_id,
                "user_id": test_user_id,
                "role": "tool",
                "name": "calculator",
                "content": "4",
            },
        ]

        # Create message instances for testing
        mock_db_messages = []
        for msg in messages_to_create:
            message = DBMessage(id=str(uuid4()), created_at=None, **msg)
            mock_db_messages.append(message)

        with patch.object(DBMessage, "paginate", new_callable=AsyncMock) as mock_paginate:
            mock_paginate.return_value = (mock_db_messages, None)

            chat_runner = TestChatRunner()
            chat_runner.user_id = test_user_id

            history = await chat_runner.get_chat_history_from_db(test_thread_id)

            # Should have 4 messages (system, user, assistant, tool)
            assert len(history) == 4

            roles = [msg.role for msg in history]
            assert "system" in roles
            assert "user" in roles
            assert "assistant" in roles
            assert "tool" in roles
            assert "agent_execution" not in roles

    @pytest.mark.asyncio
    async def test_error_in_saving_execution_message_doesnt_fail_execution(
        self, agent_processor, test_message, processing_context
    ):
        """Test that errors in saving execution messages don't stop agent execution."""
        chat_history = [test_message]

        task_update = TaskUpdate(
            event=TaskUpdateEvent.STEP_STARTED,
            task=Task(id="task1", title="Test Task"),
            step=Step(id="step1", instructions="Test Subtask"),
        )

        with patch("nodetool.agents.agent.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.results = "Test result"

            async def mock_execute(context):
                yield task_update
                yield Chunk(content="", done=True)

            mock_agent_instance.execute = mock_execute
            MockAgent.return_value = mock_agent_instance

            agent_processor.send_message = AsyncMock()

            # Mock DBMessage.create to raise an exception
            with patch.object(DBMessage, "create", side_effect=Exception("DB Error")):
                # This should not raise an exception
                await agent_processor.process(
                    chat_history=chat_history,
                    processing_context=processing_context,
                )

                # Verify that send_message was still called (WebSocket sending works)
                assert agent_processor.send_message.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
