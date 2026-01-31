"""Tests for gateway client."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.gateway.client import GatewayClient
from nodetool.gateway.protocol import (
    CommandRequest,
    NodeRegistration,
    WorkflowRequest,
)


@pytest.fixture
def gateway_client():
    """Create a gateway client for testing."""
    return GatewayClient(
        gateway_url="ws://localhost:8080",
        node_id="test-node",
        user_id="test-user",
    )


def test_client_initialization(gateway_client):
    """Test client initialization."""
    assert gateway_client.gateway_url == "ws://localhost:8080"
    assert gateway_client.node_id == "test-node"
    assert gateway_client.user_id == "test-user"
    assert gateway_client.connected is False
    assert gateway_client.running is False


def test_client_with_auth_token():
    """Test client initialization with auth token."""
    client = GatewayClient(
        gateway_url="ws://localhost:8080",
        auth_token="test-token",
    )
    assert client.auth_token == "test-token"


def test_client_auto_generates_node_id():
    """Test that node_id is auto-generated if not provided."""
    client = GatewayClient(gateway_url="ws://localhost:8080")
    assert client.node_id is not None
    assert client.node_id.startswith("node-")


@pytest.mark.asyncio
async def test_send_registration(gateway_client):
    """Test sending node registration message."""
    # Mock websocket
    gateway_client.websocket = AsyncMock()
    gateway_client.connected = True

    await gateway_client._send_registration()

    # Verify registration message was sent
    assert gateway_client.websocket.send.called
    call_args = gateway_client.websocket.send.call_args[0][0]
    data = json.loads(call_args)

    assert data["type"] == "node_registration"
    assert data["node_id"] == "test-node"
    assert "capabilities" in data


@pytest.mark.asyncio
async def test_handle_workflow_request(gateway_client):
    """Test handling workflow request."""
    # Mock websocket and workflow execution
    gateway_client.websocket = AsyncMock()
    gateway_client.connected = True

    request_data = {
        "type": "workflow_request",
        "request_id": "req-123",
        "workflow_id": "wf-abc",
        "params": {},
        "user_id": "test-user",
        "timestamp": "2024-01-30T12:00:00Z",
    }

    # Mock the workflow execution
    with patch("nodetool.gateway.client.WorkflowTools") as mock_tools:
        mock_tools.run_workflow_tool = AsyncMock(
            return_value={"job_id": "job-123", "status": "completed"}
        )

        await gateway_client._handle_workflow_request(request_data)

        # Verify ACK was sent
        assert gateway_client.websocket.send.called


@pytest.mark.asyncio
async def test_handle_command_request(gateway_client):
    """Test handling command request."""
    # Mock websocket
    gateway_client.websocket = AsyncMock()
    gateway_client.connected = True

    request_data = {
        "type": "command_request",
        "request_id": "req-789",
        "command": "list_workflows",
        "args": {"limit": 10},
        "timestamp": "2024-01-30T12:00:00Z",
    }

    await gateway_client._handle_command_request(request_data)

    # Verify ACK was sent
    assert gateway_client.websocket.send.called


@pytest.mark.asyncio
async def test_dispatch_command_list_workflows(gateway_client):
    """Test dispatching list_workflows command."""
    with patch("nodetool.gateway.client.WorkflowTools") as mock_tools:
        mock_tools.list_workflows = AsyncMock(
            return_value={"workflows": []}
        )

        result = await gateway_client._dispatch_command(
            "list_workflows",
            {"limit": 10}
        )

        assert "workflows" in result
        mock_tools.list_workflows.assert_called_once_with(limit=10)


@pytest.mark.asyncio
async def test_dispatch_command_list_jobs(gateway_client):
    """Test dispatching list_jobs command."""
    with patch("nodetool.gateway.client.JobTools") as mock_tools:
        mock_tools.list_jobs = AsyncMock(
            return_value={"jobs": []}
        )

        result = await gateway_client._dispatch_command(
            "list_jobs",
            {"limit": 20}
        )

        assert "jobs" in result
        mock_tools.list_jobs.assert_called_once_with(limit=20)


@pytest.mark.asyncio
async def test_dispatch_unknown_command(gateway_client):
    """Test dispatching unknown command raises error."""
    with pytest.raises(ValueError, match="Unknown command"):
        await gateway_client._dispatch_command(
            "unknown_command",
            {}
        )


@pytest.mark.asyncio
async def test_disconnect(gateway_client):
    """Test client disconnect."""
    # Mock websocket and tasks
    gateway_client.websocket = AsyncMock()
    gateway_client.connected = True
    gateway_client.running = True

    # Create mock tasks
    gateway_client._receive_task = asyncio.create_task(asyncio.sleep(10))
    gateway_client._heartbeat_task = asyncio.create_task(asyncio.sleep(10))

    await gateway_client.disconnect()

    assert gateway_client.connected is False
    assert gateway_client.running is False
    assert gateway_client.websocket.close.called


def test_message_handlers_registered(gateway_client):
    """Test that message handlers are registered."""
    assert "workflow_request" in gateway_client._handlers
    assert "command_request" in gateway_client._handlers
    assert "ack" in gateway_client._handlers
    assert "error" in gateway_client._handlers


@pytest.mark.asyncio
async def test_execute_workflow_with_workflow_id(gateway_client):
    """Test executing workflow by ID."""
    request = WorkflowRequest(
        request_id="req-123",
        workflow_id="wf-abc",
        params={"input": "test"},
    )

    gateway_client.websocket = AsyncMock()
    gateway_client.connected = True

    with patch("nodetool.gateway.client.WorkflowTools") as mock_tools:
        mock_tools.run_workflow_tool = AsyncMock(
            return_value={
                "job_id": "job-123",
                "status": "completed",
                "result": {}
            }
        )

        await gateway_client._execute_workflow(request)

        # Verify workflow was executed
        mock_tools.run_workflow_tool.assert_called_once()

        # Verify response was sent
        assert gateway_client.websocket.send.called


@pytest.mark.asyncio
async def test_execute_workflow_with_graph(gateway_client):
    """Test executing workflow with inline graph."""
    graph = {
        "nodes": [{"id": "node-1", "type": "input"}],
        "edges": [],
    }

    request = WorkflowRequest(
        request_id="req-123",
        graph=graph,
        params={},
    )

    gateway_client.websocket = AsyncMock()
    gateway_client.connected = True

    with patch("nodetool.gateway.client.WorkflowTools") as mock_tools:
        mock_tools.run_graph = AsyncMock(
            return_value={"job_id": "job-123", "status": "completed"}
        )

        await gateway_client._execute_workflow(request)

        # Verify workflow was executed with graph
        mock_tools.run_graph.assert_called_once()


@pytest.mark.asyncio
async def test_execute_workflow_error(gateway_client):
    """Test workflow execution error handling."""
    request = WorkflowRequest(
        request_id="req-123",
        workflow_id="wf-abc",
        params={},
    )

    gateway_client.websocket = AsyncMock()
    gateway_client.connected = True

    with patch("nodetool.gateway.client.WorkflowTools") as mock_tools:
        mock_tools.run_workflow_tool = AsyncMock(
            side_effect=Exception("Execution failed")
        )

        await gateway_client._execute_workflow(request)

        # Verify error response was sent
        assert gateway_client.websocket.send.called
        call_args = gateway_client.websocket.send.call_args[0][0]
        data = json.loads(call_args)

        assert data["type"] == "workflow_response"
        assert data["status"] == "failed"
        assert "Execution failed" in data["error"]


@pytest.mark.asyncio
async def test_execute_command_success(gateway_client):
    """Test command execution success."""
    request = CommandRequest(
        request_id="req-789",
        command="list_nodes",
        args={},
    )

    gateway_client.websocket = AsyncMock()
    gateway_client.connected = True

    with patch("nodetool.gateway.client.NodeTools") as mock_tools:
        mock_tools.list_nodes = AsyncMock(
            return_value=[{"id": "node-1"}]
        )

        await gateway_client._execute_command(request)

        # Verify success response was sent
        assert gateway_client.websocket.send.called
        call_args = gateway_client.websocket.send.call_args[0][0]
        data = json.loads(call_args)

        assert data["type"] == "command_response"
        assert data["status"] == "success"


@pytest.mark.asyncio
async def test_execute_command_error(gateway_client):
    """Test command execution error handling."""
    request = CommandRequest(
        request_id="req-789",
        command="invalid_command",
        args={},
    )

    gateway_client.websocket = AsyncMock()
    gateway_client.connected = True

    await gateway_client._execute_command(request)

    # Verify error response was sent
    assert gateway_client.websocket.send.called
    call_args = gateway_client.websocket.send.call_args[0][0]
    data = json.loads(call_args)

    assert data["type"] == "command_response"
    assert data["status"] == "error"
    assert "Unknown command" in data["error"]
