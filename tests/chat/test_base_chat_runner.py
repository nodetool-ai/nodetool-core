"""
Tests for BaseChatRunner functionality
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from nodetool.chat.base_chat_runner import BaseChatRunner
from nodetool.metadata.types import Message as ApiMessage
from nodetool.models.message import Message as DBMessage
from nodetool.models.thread import Thread
from nodetool.types.api_graph import Graph


class TestChatRunner(BaseChatRunner):
    """Concrete implementation of BaseChatRunner for testing"""

    def __init__(self, auth_token: str | None = None):
        super().__init__(auth_token)
        self.sent_messages = []
        self.connection_state = False

    async def connect(self, **kwargs):
        self.connection_state = True

    async def disconnect(self):
        self.connection_state = False

    async def send_message(self, message: dict):
        self.sent_messages.append(message)

    async def receive_message(self):
        return None

    async def handle_message(self, message_data: dict):
        """Handle a message by converting to API format and delegating to implementation"""
        # Ensure thread exists
        thread_id = await self.ensure_thread_exists(message_data.get("thread_id"))
        message_data["thread_id"] = thread_id

        # Save message to database
        db_message = await self._save_message_to_db_async(message_data)

        # Convert to API message and get full chat history
        self._db_message_to_metadata_message(db_message)
        messages = await self.get_chat_history_from_db(thread_id)

        # Call implementation
        await self.handle_message_impl(messages)


@pytest.mark.asyncio
class TestBaseChatRunner:
    """Test suite for BaseChatRunner functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = TestChatRunner()

    async def test_init(self):
        """Test initialization of BaseChatRunner"""
        runner = TestChatRunner("test_token")
        assert runner.auth_token == "test_token"
        assert runner.user_id is None
        assert runner.supabase is None
        assert runner.current_task is None

    async def test_db_message_to_metadata_message(self):
        """Test conversion from database message to metadata message"""
        # Create a mock DB message
        db_message = Mock(spec=DBMessage)
        db_message.id = "test_id"
        db_message.workflow_id = "workflow_123"
        db_message.graph = {"nodes": [], "edges": []}
        db_message.thread_id = "thread_123"
        db_message.tools = ["tool1", "tool2"]
        db_message.tool_call_id = "tool_call_123"
        db_message.role = "user"
        db_message.name = "test_user"
        db_message.content = "Test content"
        db_message.tool_calls = []
        db_message.collections = []
        db_message.input_files = []
        db_message.output_files = []
        db_message.created_at = Mock(isoformat=Mock(return_value="2024-01-01T00:00:00"))
        db_message.provider = "openai"
        db_message.model = "gpt-4"
        db_message.agent_mode = True
        db_message.help_mode = False
        db_message.agent_execution_id = None
        db_message.execution_event_type = None
        db_message.workflow_target = None

        # Convert to metadata message
        api_message = self.runner._db_message_to_metadata_message(db_message)

        # Verify conversion
        assert api_message.id == "test_id"
        assert api_message.workflow_id == "workflow_123"
        assert isinstance(api_message.graph, Graph)
        assert api_message.thread_id == "thread_123"
        assert api_message.tools == ["tool1", "tool2"]
        assert api_message.role == "user"
        assert api_message.content == "Test content"
        assert api_message.provider == "openai"
        assert api_message.model == "gpt-4"
        assert api_message.agent_mode is True

    async def test_save_message_to_db_async(self):
        """Test asynchronous message saving to database"""
        self.runner.user_id = "user_123"

        message_data = {
            "thread_id": "thread_123",
            "role": "user",
            "content": "Test message",
            "model": "gpt-4",
            "provider": "openai",
        }

        # Mock DBMessage.create
        with patch.object(DBMessage, "create") as mock_create:
            mock_db_message = Mock()
            mock_db_message.id = "msg_123"
            mock_create.return_value = mock_db_message

            # Save message
            result = await self.runner._save_message_to_db_async(message_data)

            # Verify result
            assert result == mock_db_message
            mock_create.assert_called_once()

    async def test_get_chat_history_from_db(self):
        """Test fetching chat history from database"""
        self.runner.user_id = "user_123"

        # Mock database messages
        mock_messages = [
            Mock(spec=DBMessage, id=f"msg_{i}", instructions=f"Message {i}", role="user") for i in range(3)
        ]

        # Set attributes for proper conversion
        for msg in mock_messages:
            msg.workflow_id = None
            msg.graph = None
            msg.thread_id = "thread_123"
            msg.tools = []
            msg.tool_call_id = None
            msg.name = None
            msg.tool_calls = []
            msg.collections = []
            msg.input_files = []
            msg.output_files = []
            msg.created_at = None
            msg.provider = "openai"
            msg.model = "gpt-4"
            msg.agent_mode = False
            msg.help_mode = False
            msg.agent_execution_id = None
            msg.execution_event_type = None
            msg.workflow_target = None

        # Mock DBMessage.paginate
        with patch.object(DBMessage, "paginate", return_value=(mock_messages, None)):
            # Get chat history
            history = await self.runner.get_chat_history_from_db("thread_123")

            # Verify results
            assert len(history) == 3
            assert all(isinstance(msg, ApiMessage) for msg in history)
            assert history[0].content == "Message 0"
            assert history[1].content == "Message 1"
            assert history[2].content == "Message 2"

    async def test_ensure_thread_exists_create_new(self):
        """Test thread creation when no thread_id is provided"""
        self.runner.user_id = "user_123"

        # Mock Thread.create
        with patch.object(Thread, "create") as mock_create:
            mock_thread = Mock()
            mock_thread.id = "new_thread_123"
            mock_create.return_value = mock_thread

            # Ensure thread exists
            thread_id = await self.runner.ensure_thread_exists(None)

            # Verify new thread was created
            assert thread_id == "new_thread_123"
            mock_create.assert_called_once_with(user_id="user_123")

    async def test_ensure_thread_exists_verify_existing(self):
        """Test thread verification for existing thread"""
        self.runner.user_id = "user_123"

        # Mock Thread.find
        with patch.object(Thread, "find") as mock_find:
            mock_thread = Mock()
            mock_thread.id = "existing_thread_123"
            mock_find.return_value = mock_thread

            # Ensure thread exists
            thread_id = await self.runner.ensure_thread_exists("existing_thread_123")

            # Verify existing thread was found
            assert thread_id == "existing_thread_123"
            mock_find.assert_called_once_with(user_id="user_123", id="existing_thread_123")

    async def test_ensure_thread_exists_create_with_client_id(self):
        """Test thread creation with client-provided ID when thread doesn't exist"""
        self.runner.user_id = "user_123"

        # Mock Thread.find to return None (thread doesn't exist)
        with patch.object(Thread, "find", return_value=None), patch.object(Thread, "create") as mock_create:
            mock_thread = Mock()
            mock_thread.id = "client_thread_456"
            mock_create.return_value = mock_thread

            # Ensure thread exists with client-provided ID
            thread_id = await self.runner.ensure_thread_exists("client_thread_456")

            # Verify new thread was created with the client-provided ID
            assert thread_id == "client_thread_456"
            mock_create.assert_called_once_with(user_id="user_123", id="client_thread_456")

    async def test_validate_token_success(self):
        """Test successful token validation"""
        # Mock supabase client
        mock_supabase = Mock()
        mock_session = Mock()
        mock_session.access_token = "valid_token"
        mock_supabase.auth.get_session = AsyncMock(return_value=mock_session)

        mock_user_response = Mock()
        mock_user_response.user = Mock(id="user_123")
        mock_supabase.auth.get_user = AsyncMock(return_value=mock_user_response)

        self.runner.supabase = mock_supabase

        # Validate token
        is_valid = await self.runner.validate_token("valid_token")

        # Verify validation
        assert is_valid is True
        assert self.runner.user_id == "user_123"

    async def test_handle_message_regular_chat(self):
        """Test handling a regular chat message"""
        self.runner.user_id = "user_123"

        message_data = {
            "thread_id": "thread_123",
            "role": "user",
            "content": "Test message",
            "model": "gpt-4",
            "provider": "openai",
        }

        # Mock dependencies
        with (
            patch.object(self.runner, "ensure_thread_exists", return_value="thread_123"),
            patch.object(self.runner, "_save_message_to_db_async") as mock_save,
            patch.object(self.runner, "get_chat_history_from_db") as mock_history,
            patch.object(self.runner, "handle_message_impl") as mock_handle_impl,
        ):
            # Create properly configured mock DB message
            mock_db_message = Mock(spec=DBMessage)
            mock_db_message.id = "msg_123"
            mock_db_message.workflow_id = None
            mock_db_message.graph = None
            mock_db_message.thread_id = "thread_123"
            mock_db_message.tools = []
            mock_db_message.tool_call_id = None
            mock_db_message.role = "user"
            mock_db_message.name = None
            mock_db_message.content = "Test message"
            mock_db_message.tool_calls = []
            mock_db_message.collections = []
            mock_db_message.input_files = []
            mock_db_message.output_files = []
            mock_db_message.created_at = None
            mock_db_message.provider = "openai"
            mock_db_message.model = "gpt-4"
            mock_db_message.agent_mode = False
            mock_db_message.help_mode = False
            mock_db_message.agent_execution_id = None
            mock_db_message.execution_event_type = None
            mock_db_message.workflow_target = None

            mock_save.return_value = mock_db_message
            mock_history.return_value = []

            # Handle message
            await self.runner.handle_message(message_data)

            # Verify processing
            mock_save.assert_called_once()
            mock_handle_impl.assert_called_once()

    async def test_handle_message_agent_mode(self):
        """Test handling a message in agent mode"""
        self.runner.user_id = "user_123"

        message_data = {
            "thread_id": "thread_123",
            "role": "user",
            "content": "Test message",
            "model": "gpt-4",
            "provider": "openai",
            "agent_mode": True,
        }

        # Mock dependencies
        with (
            patch.object(self.runner, "ensure_thread_exists", return_value="thread_123"),
            patch.object(self.runner, "_save_message_to_db_async") as mock_save,
            patch.object(self.runner, "get_chat_history_from_db") as mock_history,
            patch.object(self.runner, "handle_message_impl") as mock_handle_impl,
        ):
            # Create properly configured mock DB message
            mock_db_message = Mock(spec=DBMessage)
            mock_db_message.id = "msg_123"
            mock_db_message.workflow_id = None
            mock_db_message.graph = None
            mock_db_message.thread_id = "thread_123"
            mock_db_message.tools = []
            mock_db_message.tool_call_id = None
            mock_db_message.role = "user"
            mock_db_message.name = None
            mock_db_message.content = "Test message"
            mock_db_message.tool_calls = []
            mock_db_message.collections = []
            mock_db_message.input_files = []
            mock_db_message.output_files = []
            mock_db_message.created_at = None
            mock_db_message.provider = "openai"
            mock_db_message.model = "gpt-4"
            mock_db_message.agent_mode = True
            mock_db_message.help_mode = False
            mock_db_message.agent_execution_id = None
            mock_db_message.execution_event_type = None
            mock_db_message.workflow_target = None

            mock_save.return_value = mock_db_message
            mock_history.return_value = []

            # Handle message
            await self.runner.handle_message(message_data)

            # Verify processing
            mock_save.assert_called_once()
            mock_handle_impl.assert_called_once()

    async def test_handle_message_workflow(self):
        """Test handling a workflow message"""
        self.runner.user_id = "user_123"

        message_data = {
            "thread_id": "thread_123",
            "role": "user",
            "content": "Test message",
            "workflow_id": "workflow_123",
        }

        # Mock dependencies
        with (
            patch.object(self.runner, "ensure_thread_exists", return_value="thread_123"),
            patch.object(self.runner, "_save_message_to_db_async") as mock_save,
            patch.object(self.runner, "get_chat_history_from_db") as mock_history,
            patch.object(self.runner, "handle_message_impl") as mock_handle_impl,
        ):
            # Create properly configured mock DB message
            mock_db_message = Mock(spec=DBMessage)
            mock_db_message.id = "msg_123"
            mock_db_message.workflow_id = "workflow_123"
            mock_db_message.graph = None
            mock_db_message.thread_id = "thread_123"
            mock_db_message.tools = []
            mock_db_message.tool_call_id = None
            mock_db_message.role = "user"
            mock_db_message.name = None
            mock_db_message.content = "Test message"
            mock_db_message.tool_calls = []
            mock_db_message.collections = []
            mock_db_message.input_files = []
            mock_db_message.output_files = []
            mock_db_message.created_at = None
            mock_db_message.provider = "openai"
            mock_db_message.model = "gpt-4"
            mock_db_message.agent_mode = False
            mock_db_message.help_mode = False
            mock_db_message.agent_execution_id = None
            mock_db_message.execution_event_type = None
            mock_db_message.workflow_target = None

            mock_save.return_value = mock_db_message
            mock_history.return_value = []

            # Handle message
            await self.runner.handle_message(message_data)

            # Verify processing
            mock_save.assert_called_once()
            mock_handle_impl.assert_called_once()
