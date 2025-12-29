"""
Test fixtures and utilities for chat runner tests
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import Message as ApiMessage
from nodetool.models.message import Message as DBMessage
from nodetool.models.thread import Thread


class MockTool(Tool):
    """Mock tool for testing"""

    def __init__(self, name: str = "mock_tool"):
        self.name = name
        self.description = f"Mock tool {name}"
        self.call_count = 0
        self.last_args = None

    async def execute(self, **kwargs):
        self.call_count += 1
        self.last_args = kwargs
        return {"result": f"Mock result from {self.name}"}


class MockMessageProcessor:
    """Mock message processor for testing"""

    def __init__(self):
        self.messages = []
        self.is_processing = False
        self._message_queue = asyncio.Queue()

    def has_messages(self):
        return not self._message_queue.empty() or self.is_processing

    async def get_message(self):
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=0.1)
        except TimeoutError:
            return None

    async def process(self, chat_history, processing_context, tools, **kwargs):
        self.is_processing = True
        # Simulate processing
        await self._message_queue.put({"type": "start", "message": "Processing started"})
        await asyncio.sleep(0.01)
        await self._message_queue.put({"type": "content", "content": "Test response"})
        await asyncio.sleep(0.01)
        await self._message_queue.put({"type": "message", "role": "assistant", "content": "Final response"})
        self.is_processing = False


def create_mock_db_message(
    id: str = "msg_123",
    thread_id: str = "thread_123",
    user_id: str = "user_123",
    role: str = "user",
    content: str = "Test message",
    **kwargs,
) -> Mock:
    """Create a mock database message"""
    message = Mock(spec=DBMessage)
    message.id = id
    message.thread_id = thread_id
    message.user_id = user_id
    message.role = role
    message.content = content
    message.workflow_id = kwargs.get("workflow_id")
    message.graph = kwargs.get("graph")
    message.tools = kwargs.get("tools", [])
    message.tool_call_id = kwargs.get("tool_call_id")
    message.name = kwargs.get("name")
    message.tool_calls = kwargs.get("tool_calls", [])
    message.collections = kwargs.get("collections", [])
    message.input_files = kwargs.get("input_files", [])
    message.output_files = kwargs.get("output_files", [])
    message.created_at = Mock(isoformat=Mock(return_value="2024-01-01T00:00:00"))
    message.provider = kwargs.get("provider", "openai")
    message.model = kwargs.get("model", "gpt-4")
    message.agent_mode = kwargs.get("agent_mode", False)
    message.help_mode = kwargs.get("help_mode", False)
    return message


def create_api_message(
    role: str = "user",
    content: str = "Test message",
    thread_id: str = "thread_123",
    **kwargs,
) -> ApiMessage:
    """Create an API message for testing"""
    return ApiMessage(
        role=role,
        instructions=content,
        thread_id=thread_id,
        id=kwargs.get("id"),
        workflow_id=kwargs.get("workflow_id"),
        graph=kwargs.get("graph"),
        tools=kwargs.get("tools", []),
        tool_call_id=kwargs.get("tool_call_id"),
        name=kwargs.get("name"),
        tool_calls=kwargs.get("tool_calls", []),
        collections=kwargs.get("collections", []),
        input_files=kwargs.get("input_files", []),
        output_files=kwargs.get("output_files", []),
        created_at=kwargs.get("created_at"),
        provider=kwargs.get("provider", "openai"),
        model=kwargs.get("model", "gpt-4"),
        agent_mode=kwargs.get("agent_mode", False),
        workflow_assistant=kwargs.get("workflow_assistant", False),
        help_mode=kwargs.get("help_mode", False),
    )


def create_mock_thread(id: str = "thread_123", user_id: str = "user_123") -> Mock:
    """Create a mock thread"""
    thread = Mock(spec=Thread)
    thread.id = id
    thread.user_id = user_id
    return thread


class MockWebSocket:
    """Mock WebSocket for testing"""

    def __init__(self):
        self.accepted = False
        self.closed = False
        self.close_code = None
        self.close_reason = None
        self.sent_messages = []
        self.receive_queue = asyncio.Queue()

    async def accept(self):
        self.accepted = True

    async def close(self, code: int = 1000, reason: str = ""):
        self.closed = True
        self.close_code = code
        self.close_reason = reason

    async def send_bytes(self, data: bytes):
        self.sent_messages.append(("bytes", data))

    async def send_text(self, data: str):
        self.sent_messages.append(("text", data))

    async def receive(self):
        return await self.receive_queue.get()

    def add_message(self, message: dict):
        """Add a message to be received"""
        self.receive_queue.put_nowait(message)


class ChatHistoryBuilder:
    """Helper class to build chat histories for testing"""

    def __init__(self):
        self.messages: List[ApiMessage] = []

    def add_user_message(self, content: str, **kwargs) -> "ChatHistoryBuilder":
        self.messages.append(create_api_message(role="user", content=content, **kwargs))
        return self

    def add_assistant_message(self, content: str, **kwargs) -> "ChatHistoryBuilder":
        self.messages.append(create_api_message(role="assistant", content=content, **kwargs))
        return self

    def add_tool_call(self, tool_name: str, args: dict, call_id: str = "call_123") -> "ChatHistoryBuilder":
        self.messages.append(
            create_api_message(
                role="assistant",
                instructions="",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": args},
                    }
                ],
            )
        )
        return self

    def add_tool_result(self, result: str, call_id: str = "call_123") -> "ChatHistoryBuilder":
        self.messages.append(create_api_message(role="tool", instructions=result, tool_call_id=call_id))
        return self

    def build(self) -> List[ApiMessage]:
        return self.messages


@pytest.fixture
def mock_websocket():
    """Fixture for mock WebSocket"""
    return MockWebSocket()


@pytest.fixture
def mock_tool():
    """Fixture for mock tool"""
    return MockTool()


@pytest.fixture
def chat_history_builder():
    """Fixture for chat history builder"""
    return ChatHistoryBuilder()


@pytest.fixture
def mock_message_processor():
    """Fixture for mock message processor"""
    return MockMessageProcessor()


@pytest.fixture
def mock_supabase_client():
    """Fixture for mock Supabase client"""
    client = Mock()
    client.auth = Mock()
    client.auth.get_session = AsyncMock()
    client.auth.get_user = AsyncMock()
    return client


# Helper functions for common test scenarios


async def simulate_chat_interaction(runner, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simulate a chat interaction and collect responses

    Args:
        runner: The chat runner instance
        messages: List of message dictionaries to send

    Returns:
        List of response messages
    """
    responses = []

    for message in messages:
        # Handle the message
        await runner.handle_message(message)

        # Collect any sent messages
        if hasattr(runner, "sent_messages"):
            responses.extend(runner.sent_messages)
            runner.sent_messages.clear()

    return responses


def assert_message_contains(message: dict, expected_type: str, expected_content: Optional[str] = None):
    """Assert that a message has expected type and optionally content"""
    assert message.get("type") == expected_type, f"Expected type {expected_type}, got {message.get('type')}"

    if expected_content is not None:
        content = message.get("content") or message.get("message", "")
        assert expected_content in content, f"Expected '{expected_content}' in '{content}'"
