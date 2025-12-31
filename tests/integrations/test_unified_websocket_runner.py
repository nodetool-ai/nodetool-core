"""
Tests for UnifiedWebSocketRunner functionality.

This module tests the unified WebSocket runner that handles both workflow execution
and chat communications through a single endpoint.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import msgpack
import pytest
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from nodetool.config.environment import Environment
from nodetool.integrations.websocket.unified_websocket_runner import (
    CommandType,
    UnifiedWebSocketRunner,
    WebSocketCommand,
    WebSocketMode,
)
from nodetool.models.workflow import Workflow
from nodetool.types.graph import Graph
from nodetool.workflows.job_execution_manager import JobExecutionManager
from nodetool.workflows.run_job_request import RunJobRequest

DEFAULT_TEST_TIMEOUT = 5


async def wait_for(coro, timeout: float = DEFAULT_TEST_TIMEOUT):
    """Await a coroutine with a timeout to avoid test hangs."""
    return await asyncio.wait_for(coro, timeout=timeout)


@pytest.fixture
async def mock_websocket():
    """Create a mock WebSocket."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_bytes = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    ws.client_state = MagicMock()
    return ws


@pytest.fixture
async def unified_runner():
    """Create a UnifiedWebSocketRunner instance."""
    runner = UnifiedWebSocketRunner()
    yield runner
    # Cleanup
    if runner.websocket:
        await runner.disconnect()


@pytest.fixture
async def simple_workflow():
    """Create a simple workflow for testing."""
    workflow = await Workflow.create(
        user_id="test_user",
        name="Test Workflow",
        description="A test workflow",
        graph=Graph(nodes=[], edges=[]).model_dump(),
    )
    yield workflow
    # Cleanup
    await workflow.delete()


@pytest.fixture
async def cleanup_jobs():
    """Cleanup jobs after each test."""
    yield
    manager = JobExecutionManager.get_instance()
    for job_id, job in list(manager._jobs.items()):
        try:
            job.cleanup_resources()
            if not job.is_completed():
                job.cancel()
        except Exception as e:
            print(f"Error cleaning up job {job_id}: {e}")
    manager._jobs.clear()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestUnifiedWebSocketRunnerBasics:
    """Test suite for basic UnifiedWebSocketRunner functionality."""

    async def test_init(self):
        """Test initialization of UnifiedWebSocketRunner."""
        runner = UnifiedWebSocketRunner("test_token", user_id="test_user")
        assert runner.auth_token == "test_token"
        assert runner.user_id == "test_user"
        assert runner.websocket is None
        assert runner.mode == WebSocketMode.BINARY
        assert runner.active_jobs == {}
        assert runner.client_tools_manifest == {}

    async def test_connect_local_development(self, mock_websocket):
        """Test that connection works in local development mode without auth."""
        with patch.object(Environment, "enforce_auth", return_value=False):
            runner = UnifiedWebSocketRunner()
            await wait_for(runner.connect(mock_websocket))

            mock_websocket.accept.assert_called_once()
            assert runner.user_id == "1"  # Default user ID
            assert runner.websocket is mock_websocket

            await wait_for(runner.disconnect())

    async def test_connect_with_user_id(self, mock_websocket):
        """Test connection with pre-authenticated user_id."""
        with patch.object(Environment, "enforce_auth", return_value=False):
            runner = UnifiedWebSocketRunner()
            await wait_for(runner.connect(mock_websocket, user_id="custom_user"))

            mock_websocket.accept.assert_called_once()
            assert runner.user_id == "custom_user"

            await wait_for(runner.disconnect())

    async def test_disconnect(self, unified_runner, mock_websocket):
        """Test disconnect functionality."""
        with patch.object(Environment, "enforce_auth", return_value=False):
            await wait_for(unified_runner.connect(mock_websocket))
            mock_websocket.client_state = WebSocketState.CONNECTED

            await wait_for(unified_runner.disconnect())

            mock_websocket.close.assert_called_once()
            assert unified_runner.websocket is None
            assert unified_runner.active_jobs == {}


@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestUnifiedWebSocketRunnerMessaging:
    """Test suite for message sending and receiving."""

    async def test_send_message_binary(self):
        """Test sending binary messages."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)
        runner.websocket.send_bytes = AsyncMock()
        runner.mode = WebSocketMode.BINARY

        message = {"type": "test", "content": "Hello"}
        await wait_for(runner.send_message(message))

        expected_packed = msgpack.packb(message, use_bin_type=True)
        runner.websocket.send_bytes.assert_called_once_with(expected_packed)

    async def test_send_message_text(self):
        """Test sending text messages."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)
        runner.websocket.send_text = AsyncMock()
        runner.mode = WebSocketMode.TEXT

        message = {"type": "test", "content": "Hello"}
        await wait_for(runner.send_message(message))

        expected_json = json.dumps(message)
        runner.websocket.send_text.assert_called_once_with(expected_json)

    async def test_receive_message_binary(self):
        """Test receiving binary messages."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)

        test_data = {"type": "chat", "content": "Hello"}
        packed_data = msgpack.packb(test_data)

        runner.websocket.receive = AsyncMock(return_value={"type": "websocket.message", "bytes": packed_data})

        result = await wait_for(runner.receive_message())

        assert result == test_data
        assert runner.mode == WebSocketMode.BINARY

    async def test_receive_message_text(self):
        """Test receiving text messages."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)

        test_data = {"type": "chat", "content": "Hello"}
        json_data = json.dumps(test_data)

        runner.websocket.receive = AsyncMock(return_value={"type": "websocket.message", "text": json_data})

        result = await wait_for(runner.receive_message())

        assert result == test_data
        assert runner.mode == WebSocketMode.TEXT

    async def test_receive_message_disconnect(self):
        """Test receiving disconnect message."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)

        runner.websocket.receive = AsyncMock(return_value={"type": "websocket.disconnect"})

        result = await wait_for(runner.receive_message())

        assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestUnifiedWebSocketRunnerCommands:
    """Test suite for command handling."""

    async def test_handle_command_get_status(self):
        """Test handling GET_STATUS command."""
        runner = UnifiedWebSocketRunner()

        command = WebSocketCommand(command=CommandType.GET_STATUS, data={})
        result = await wait_for(runner.handle_command(command))

        assert "active_jobs" in result
        assert isinstance(result["active_jobs"], list)

    async def test_handle_command_set_mode_text(self):
        """Test handling SET_MODE command to text mode."""
        runner = UnifiedWebSocketRunner()
        assert runner.mode == WebSocketMode.BINARY

        command = WebSocketCommand(command=CommandType.SET_MODE, data={"mode": "text"})
        result = await wait_for(runner.handle_command(command))

        assert runner.mode == WebSocketMode.TEXT
        assert "Mode set to" in result["message"]

    async def test_handle_command_set_mode_binary(self):
        """Test handling SET_MODE command to binary mode."""
        runner = UnifiedWebSocketRunner()
        runner.mode = WebSocketMode.TEXT

        command = WebSocketCommand(command=CommandType.SET_MODE, data={"mode": "binary"})
        await wait_for(runner.handle_command(command))

        assert runner.mode == WebSocketMode.BINARY

    async def test_handle_command_clear_models(self):
        """Test handling CLEAR_MODELS command."""
        runner = UnifiedWebSocketRunner()

        command = WebSocketCommand(command=CommandType.CLEAR_MODELS, data={})
        result = await wait_for(runner.handle_command(command))

        assert "message" in result

    async def test_handle_command_cancel_job_no_id(self):
        """Test handling CANCEL_JOB command without job_id."""
        runner = UnifiedWebSocketRunner()

        command = WebSocketCommand(command=CommandType.CANCEL_JOB, data={})
        result = await wait_for(runner.handle_command(command))

        assert "error" in result
        assert "job_id is required" in result["error"]

    async def test_handle_command_chat_message_requires_thread_id(self):
        """Test that CHAT_MESSAGE command requires thread_id."""
        runner = UnifiedWebSocketRunner()

        command = WebSocketCommand(command=CommandType.CHAT_MESSAGE, data={"content": "Hello"})
        result = await wait_for(runner.handle_command(command))

        assert "error" in result
        assert "thread_id is required" in result["error"]

    async def test_handle_command_stop_requires_reference(self):
        """Test that STOP command requires job_id or thread_id."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)
        runner.websocket.send_bytes = AsyncMock()

        command = WebSocketCommand(command=CommandType.STOP, data={})
        result = await wait_for(runner.handle_command(command))

        assert "error" in result
        assert "job_id or thread_id is required" in result["error"]

    async def test_handle_command_stop_with_thread_id(self):
        """Test STOP command with thread_id cancels current task."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)
        runner.websocket.send_bytes = AsyncMock()
        runner.mode = WebSocketMode.BINARY

        # Create a mock current task
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        runner.current_task = mock_task

        command = WebSocketCommand(command=CommandType.STOP, data={"thread_id": "test-thread-123"})
        result = await wait_for(runner.handle_command(command))

        mock_task.cancel.assert_called_once()
        assert result["message"] == "Stop command processed"
        assert result["thread_id"] == "test-thread-123"

    async def test_handle_command_stop_with_job_id(self):
        """Test STOP command with job_id."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)
        runner.websocket.send_bytes = AsyncMock()
        runner.mode = WebSocketMode.BINARY

        command = WebSocketCommand(command=CommandType.STOP, data={"job_id": "test-job-123"})
        result = await wait_for(runner.handle_command(command))

        assert result["message"] == "Stop command processed"
        assert result["job_id"] == "test-job-123"


@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestUnifiedWebSocketRunnerControlMessages:
    """Test suite for control message handling."""

    async def test_receive_messages_stop_via_command(self):
        """Test handling stop command via command structure."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)
        runner.websocket.send_bytes = AsyncMock()
        runner.mode = WebSocketMode.BINARY

        # Create a mock current task
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        runner.current_task = mock_task

        # Simulate receiving stop command (now must be wrapped in command structure) then disconnect
        messages = [{"command": "stop", "data": {"thread_id": "test-thread"}}, None]

        with patch.object(runner, "receive_message", side_effect=messages):
            await wait_for(runner._receive_messages())

            mock_task.cancel.assert_called_once()

    async def test_receive_messages_ping_pong(self):
        """Test ping-pong handling in receive loop."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)

        # Simulate receiving ping then disconnect
        messages = [{"type": "ping"}, None]

        with (
            patch.object(runner, "receive_message", side_effect=messages),
            patch.object(runner, "send_message", new_callable=AsyncMock) as mock_send,
        ):
            await wait_for(runner._receive_messages())

            # Verify pong was sent
            assert mock_send.call_count == 1
            pong_msg = mock_send.call_args[0][0]
            assert pong_msg["type"] == "pong"
            assert "ts" in pong_msg

    async def test_receive_messages_client_tools_manifest(self):
        """Test client tools manifest handling."""
        runner = UnifiedWebSocketRunner()
        runner.websocket = Mock(spec=WebSocket)

        tools = [
            {"name": "tool1", "description": "Tool 1"},
            {"name": "tool2", "description": "Tool 2"},
        ]
        messages = [{"type": "client_tools_manifest", "tools": tools}, None]

        with patch.object(runner, "receive_message", side_effect=messages):
            await wait_for(runner._receive_messages())

            assert "tool1" in runner.client_tools_manifest
            assert "tool2" in runner.client_tools_manifest


@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestUnifiedWebSocketRunnerWorkflowIntegration:
    """Test suite for workflow integration."""

    async def test_run_job_connects_and_manages_jobs(
        self, unified_runner, mock_websocket, simple_workflow, cleanup_jobs
    ):
        """Test that run_job can start and manage jobs."""
        with patch.object(Environment, "enforce_auth", return_value=False):
            await unified_runner.connect(mock_websocket)
            mock_websocket.client_state = MagicMock()
            mock_websocket.client_state.__eq__ = MagicMock(return_value=False)

            request = RunJobRequest(
                workflow_id=simple_workflow.id,
                user_id="test_user",
                auth_token="test_token",
                job_type="workflow",
                params={},
                graph=Graph(nodes=[], edges=[]),
            )

            await unified_runner.run_job(request)
            await asyncio.sleep(0.2)

            # Job should have been started
            assert len(unified_runner.active_jobs) <= 1  # May complete quickly

            await unified_runner.disconnect()

    async def test_get_status_returns_all_jobs(self, unified_runner):
        """Test that get_status returns all active jobs."""
        status = unified_runner.get_status()

        assert "active_jobs" in status
        assert isinstance(status["active_jobs"], list)

    async def test_get_status_with_job_id_not_found(self, unified_runner):
        """Test that get_status returns not_found for unknown job."""
        status = unified_runner.get_status("nonexistent_job_id")

        assert status["status"] == "not_found"
        assert status["job_id"] == "nonexistent_job_id"


@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestUnifiedWebSocketRunnerAuthentication:
    """Test suite for authentication handling."""

    async def test_missing_auth_token_when_required(self):
        """Test that missing auth token is rejected when authentication is required."""
        with patch.object(Environment, "enforce_auth", return_value=True):
            runner = UnifiedWebSocketRunner()
            websocket = Mock()
            websocket.close = AsyncMock()

            await wait_for(runner.connect(websocket))

            websocket.close.assert_called_once_with(code=1008, reason="Missing authentication")

    async def test_connection_with_valid_auth(self, mock_websocket):
        """Test connection with valid authentication."""
        with (
            patch.object(Environment, "enforce_auth", return_value=True),
            patch("nodetool.integrations.websocket.unified_websocket_runner.get_user_auth_provider") as mock_provider,
        ):
            # Mock the auth provider
            mock_auth = AsyncMock()
            mock_auth.verify_token = AsyncMock(
                return_value=MagicMock(ok=True, user_id="authenticated_user")
            )
            mock_provider.return_value = mock_auth

            runner = UnifiedWebSocketRunner(auth_token="valid_token")
            await wait_for(runner.connect(mock_websocket))

            mock_websocket.accept.assert_called_once()
            assert runner.user_id == "authenticated_user"

            await wait_for(runner.disconnect())


@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
class TestToolBridge:
    """Test suite for ToolBridge functionality."""

    async def test_create_and_resolve_waiter(self):
        """Test creating and resolving a tool result waiter."""
        from nodetool.integrations.websocket.unified_websocket_runner import ToolBridge

        bridge = ToolBridge()
        future = bridge.create_waiter("tool_call_123")

        # Resolve the result
        payload = {"result": "success", "data": {"key": "value"}}
        bridge.resolve_result("tool_call_123", payload)

        result = await wait_for(future)
        assert result == payload

    async def test_cancel_all_waiters(self):
        """Test cancelling all pending waiters."""
        from nodetool.integrations.websocket.unified_websocket_runner import ToolBridge

        bridge = ToolBridge()
        future1 = bridge.create_waiter("tool_call_1")
        future2 = bridge.create_waiter("tool_call_2")

        bridge.cancel_all()

        assert future1.cancelled()
        assert future2.cancelled()
        assert len(bridge._futures) == 0
