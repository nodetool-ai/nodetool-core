"""Tests for OpenClaw schemas."""

from datetime import datetime

import pytest

from nodetool.integrations.openclaw.schemas import (
    GatewayMessage,
    NodeCapability,
    NodeRegistration,
    NodeRegistrationResponse,
    NodeStatus,
    NodeStatusResponse,
    TaskExecutionRequest,
    TaskExecutionResponse,
    TaskStatus,
)


def test_node_capability_creation():
    """Test creating a NodeCapability."""
    capability = NodeCapability(
        name="test_capability",
        description="A test capability",
        input_schema={"type": "object", "properties": {"param": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
    )

    assert capability.name == "test_capability"
    assert capability.description == "A test capability"
    assert "param" in capability.input_schema["properties"]
    assert "result" in capability.output_schema["properties"]


def test_node_registration():
    """Test creating a NodeRegistration."""
    capabilities = [
        NodeCapability(
            name="test_cap",
            description="Test capability",
        )
    ]

    registration = NodeRegistration(
        node_id="test-node-1",
        node_name="Test Node",
        node_version="1.0.0",
        capabilities=capabilities,
        endpoint="http://localhost:7777/openclaw",
        metadata={"key": "value"},
    )

    assert registration.node_id == "test-node-1"
    assert registration.node_name == "Test Node"
    assert len(registration.capabilities) == 1
    assert registration.endpoint == "http://localhost:7777/openclaw"


def test_node_registration_response():
    """Test creating a NodeRegistrationResponse."""
    response = NodeRegistrationResponse(
        success=True,
        node_id="test-node-1",
        token="test-token-123",
        message="Registration successful",
    )

    assert response.success is True
    assert response.node_id == "test-node-1"
    assert response.token == "test-token-123"
    assert response.message == "Registration successful"


def test_task_execution_request():
    """Test creating a TaskExecutionRequest."""
    request = TaskExecutionRequest(
        task_id="task-123",
        capability_name="workflow_execution",
        parameters={"workflow_id": "wf-456", "params": {"input": "test"}},
        callback_url="http://gateway.example.com/callback",
        metadata={"priority": "high"},
    )

    assert request.task_id == "task-123"
    assert request.capability_name == "workflow_execution"
    assert request.parameters["workflow_id"] == "wf-456"
    assert request.callback_url == "http://gateway.example.com/callback"


def test_task_execution_response():
    """Test creating a TaskExecutionResponse."""
    response = TaskExecutionResponse(
        task_id="task-123",
        status=TaskStatus.RUNNING,
        message="Task is executing",
    )

    assert response.task_id == "task-123"
    assert response.status == TaskStatus.RUNNING
    assert response.message == "Task is executing"
    assert response.result is None


def test_task_execution_response_with_result():
    """Test creating a TaskExecutionResponse with result."""
    result = {"output": "success", "data": [1, 2, 3]}

    response = TaskExecutionResponse(
        task_id="task-123",
        status=TaskStatus.COMPLETED,
        message="Task completed successfully",
        result=result,
    )

    assert response.task_id == "task-123"
    assert response.status == TaskStatus.COMPLETED
    assert response.result["output"] == "success"


def test_node_status_response():
    """Test creating a NodeStatusResponse."""
    response = NodeStatusResponse(
        node_id="test-node-1",
        status=NodeStatus.ONLINE,
        uptime_seconds=3600.5,
        active_tasks=2,
        total_tasks_completed=100,
        total_tasks_failed=5,
        system_info={"cpu_percent": 45.2, "memory_percent": 60.1},
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
    )

    assert response.node_id == "test-node-1"
    assert response.status == NodeStatus.ONLINE
    assert response.uptime_seconds == 3600.5
    assert response.active_tasks == 2
    assert response.total_tasks_completed == 100
    assert response.total_tasks_failed == 5
    assert response.system_info["cpu_percent"] == 45.2


def test_gateway_message():
    """Test creating a GatewayMessage."""
    message = GatewayMessage(
        message_id="msg-123",
        message_type="task_result",
        source_node_id="node-1",
        target_node_id="node-2",
        payload={"result": "success", "data": {"key": "value"}},
    )

    assert message.message_id == "msg-123"
    assert message.message_type == "task_result"
    assert message.source_node_id == "node-1"
    assert message.target_node_id == "node-2"
    assert message.payload["result"] == "success"


def test_task_status_enum():
    """Test TaskStatus enum values."""
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.RUNNING == "running"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"


def test_node_status_enum():
    """Test NodeStatus enum values."""
    assert NodeStatus.ONLINE == "online"
    assert NodeStatus.OFFLINE == "offline"
    assert NodeStatus.BUSY == "busy"
    assert NodeStatus.ERROR == "error"
