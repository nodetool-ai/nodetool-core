"""
Tests for agent execution message storage in the database.

This module tests that agent execution events (planning_update, task_update, subtask_result)
are properly stored as messages with role="agent_execution" and that they are filtered out
when loading chat history for LLM processing.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nodetool.models.message import Message as DBMessage
from nodetool.metadata.types import Message as ApiMessage, Provider
from nodetool.messaging.processors.agent import AgentMessageProcessor
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import TaskUpdate, PlanningUpdate, SubTaskResult, Chunk, TaskUpdateEvent
from nodetool.metadata.types import Task, SubTask
from nodetool.chat.base_chat_runner import BaseChatRunner


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
        content="Test objective for agent",
        provider=Provider.OpenAI,
        model="gpt-4o",
        agent_mode=True,
    )


class TestAgentExecutionMessageStorage:
    """Tests for storing agent execution events as database messages."""

    @pytest.mark.asyncio
    async def test_task_update_saved_as_message(
        self, agent_processor, test_message, processing_context, test_thread_id, test_user_id
    ):
        """Test that TaskUpdate events are saved as agent_execution messages."""
        chat_history = [test_message]

        # Create a mock agent that yields a TaskUpdate event
        task = Task(id="task1", title="Test Task", description="Test Description")
        subtask = SubTask(id="subtask1", content="Test Subtask")
        task_update = TaskUpdate(event=TaskUpdateEvent.SUBTASK_STARTED, task=task, subtask=subtask)

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

            # Mock the send_message method to avoid actual websocket sending
            agent_processor.send_message = AsyncMock()

            # Mock DBMessage.create to track calls
            with patch.object(DBMessage, "create", new_callable=AsyncMock) as mock_create:
                # Execute the agent processor
                await agent_processor.process(
                    chat_history=chat_history,
                    processing_context=processing_context,
                )

                # Verify that DBMessage.create was called with correct parameters
                task_update_calls = [
                    call for call in mock_create.call_args_list
                    if call[1].get("execution_event_type") == "task_update"
                ]

                assert len(task_update_calls) > 0, "TaskUpdate should be saved as message"

                call_kwargs = task_update_calls[0][1]
                assert call_kwargs["thread_id"] == test_thread_id
                assert call_kwargs["user_id"] == test_user_id
                assert call_kwargs["role"] == "agent_execution"
                assert call_kwargs["execution_event_type"] == "task_update"
                assert "agent_execution_id" in call_kwargs

                content = call_kwargs["content"]
                assert content["type"] == "task_update"
                assert content["event"] == TaskUpdateEvent.SUBTASK_STARTED
                assert content["task"] is not None
                assert content["subtask"] is not None

    @pytest.mark.asyncio
    async def test_planning_update_saved_as_message(
        self, agent_processor, test_message, processing_context, test_thread_id, test_user_id
    ):
        """Test that PlanningUpdate events are saved as agent_execution messages."""
        chat_history = [test_message]

        # Create a mock planning update
        planning_update = PlanningUpdate(
            phase="planning",
            status="in_progress",
            content="Creating task plan",
            node_id="node1"
        )

        with patch("nodetool.agents.agent.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.results = "Test result"

            async def mock_execute(context):
                yield planning_update
                yield Chunk(content="", done=True)

            mock_agent_instance.execute = mock_execute
            MockAgent.return_value = mock_agent_instance

            agent_processor.send_message = AsyncMock()

            with patch.object(DBMessage, "create", new_callable=AsyncMock) as mock_create:
                await agent_processor.process(
                    chat_history=chat_history,
                    processing_context=processing_context,
                )

                planning_calls = [
                    call for call in mock_create.call_args_list
                    if call[1].get("execution_event_type") == "planning_update"
                ]

                assert len(planning_calls) > 0, "PlanningUpdate should be saved as message"

                call_kwargs = planning_calls[0][1]
                assert call_kwargs["thread_id"] == test_thread_id
                assert call_kwargs["user_id"] == test_user_id
                assert call_kwargs["role"] == "agent_execution"
                assert call_kwargs["execution_event_type"] == "planning_update"

                content = call_kwargs["content"]
                assert content["type"] == "planning_update"
                assert content["phase"] == "planning"
                assert content["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_subtask_result_saved_as_message(
        self, agent_processor, test_message, processing_context, test_thread_id, test_user_id
    ):
        """Test that SubTaskResult events are saved as agent_execution messages."""
        chat_history = [test_message]

        # Create a mock subtask result
        subtask = SubTask(id="subtask1", content="Test Subtask")
        subtask_result = SubTaskResult(
            subtask=subtask,
            result="Subtask completed successfully",
            is_task_result=False
        )

        with patch("nodetool.agents.agent.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.results = "Test result"

            async def mock_execute(context):
                yield subtask_result
                yield Chunk(content="", done=True)

            mock_agent_instance.execute = mock_execute
            MockAgent.return_value = mock_agent_instance

            agent_processor.send_message = AsyncMock()

            with patch.object(DBMessage, "create", new_callable=AsyncMock) as mock_create:
                await agent_processor.process(
                    chat_history=chat_history,
                    processing_context=processing_context,
                )

                subtask_calls = [
                    call for call in mock_create.call_args_list
                    if call[1].get("execution_event_type") == "subtask_result"
                ]

                assert len(subtask_calls) > 0, "SubTaskResult should be saved as message"

                call_kwargs = subtask_calls[0][1]
                assert call_kwargs["thread_id"] == test_thread_id
                assert call_kwargs["user_id"] == test_user_id
                assert call_kwargs["role"] == "agent_execution"
                assert call_kwargs["execution_event_type"] == "subtask_result"

                content = call_kwargs["content"]
                assert content["type"] == "subtask_result"
                assert content["result"] == "Subtask completed successfully"

    @pytest.mark.asyncio
    async def test_all_events_share_same_execution_id(
        self, agent_processor, test_message, processing_context
    ):
        """Test that all events from the same execution share the same agent_execution_id."""
        chat_history = [test_message]

        # Create multiple events
        task = Task(id="task1", title="Test Task")
        subtask = SubTask(id="subtask1", content="Test Subtask")
        task_update = TaskUpdate(event=TaskUpdateEvent.SUBTASK_STARTED, task=task, subtask=subtask)
        planning_update = PlanningUpdate(
            phase="planning",
            status="in_progress",
            content="Planning",
            node_id="node1"
        )

        with patch("nodetool.agents.agent.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.results = "Test result"

            async def mock_execute(context):
                yield planning_update
                yield task_update
                yield Chunk(content="", done=True)

            mock_agent_instance.execute = mock_execute
            MockAgent.return_value = mock_agent_instance

            agent_processor.send_message = AsyncMock()

            with patch.object(DBMessage, "create", new_callable=AsyncMock) as mock_create:
                await agent_processor.process(
                    chat_history=chat_history,
                    processing_context=processing_context,
                )

                # Get all execution message calls
                execution_calls = [
                    call for call in mock_create.call_args_list
                    if call[1].get("role") == "agent_execution"
                ]

                assert len(execution_calls) >= 2, "Should have multiple execution events"

                # Extract execution IDs
                execution_ids = [
                    call[1]["agent_execution_id"]
                    for call in execution_calls
                ]

                # All execution IDs should be the same
                assert len(set(execution_ids)) == 1, "All events should share the same execution ID"


class TestAgentExecutionMessageFiltering:
    """Tests for filtering agent_execution messages from chat history."""

    @pytest.mark.asyncio
    async def test_agent_execution_messages_filtered_from_history(
        self, test_thread_id, test_user_id
    ):
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
    async def test_only_regular_messages_sent_to_llm(
        self, test_thread_id, test_user_id
    ):
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
            event=TaskUpdateEvent.SUBTASK_STARTED,
            task=Task(id="task1", title="Test Task"),
            subtask=SubTask(id="subtask1", content="Test Subtask")
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
