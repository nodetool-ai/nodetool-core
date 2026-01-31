"""Tests for gateway protocol message types."""

from datetime import datetime

import pytest

from nodetool.gateway.protocol import (
    AckMessage,
    CommandRequest,
    CommandResponse,
    ErrorMessage,
    NodeHeartbeat,
    NodeRegistration,
    WorkflowRequest,
    WorkflowResponse,
    WorkflowUpdate,
)


def test_node_registration():
    """Test NodeRegistration message creation and serialization."""
    msg = NodeRegistration(
        node_id="test-node",
        capabilities={"workflow_execution": True},
        metadata={"version": "1.0"},
    )

    assert msg.type == "node_registration"
    assert msg.node_id == "test-node"
    assert msg.capabilities["workflow_execution"] is True
    assert msg.metadata["version"] == "1.0"

    # Test JSON serialization
    json_str = msg.model_dump_json()
    assert "node_registration" in json_str
    assert "test-node" in json_str


def test_heartbeat():
    """Test NodeHeartbeat message."""
    msg = NodeHeartbeat(node_id="test-node", status="active")

    assert msg.type == "heartbeat"
    assert msg.node_id == "test-node"
    assert msg.status == "active"


def test_workflow_request():
    """Test WorkflowRequest message."""
    msg = WorkflowRequest(
        request_id="req-123",
        workflow_id="wf-abc",
        params={"input": "test"},
        user_id="user-1",
    )

    assert msg.type == "workflow_request"
    assert msg.request_id == "req-123"
    assert msg.workflow_id == "wf-abc"
    assert msg.params["input"] == "test"
    assert msg.user_id == "user-1"


def test_workflow_request_with_graph():
    """Test WorkflowRequest with inline graph."""
    graph = {
        "nodes": [{"id": "node-1", "type": "input"}],
        "edges": [],
    }

    msg = WorkflowRequest(
        request_id="req-123",
        graph=graph,
        params={"input": "test"},
    )

    assert msg.graph is not None
    assert msg.graph["nodes"][0]["id"] == "node-1"
    assert msg.workflow_id is None


def test_workflow_response_success():
    """Test successful WorkflowResponse."""
    msg = WorkflowResponse(
        request_id="req-123",
        status="completed",
        result={"output": "result"},
        job_id="job-456",
    )

    assert msg.type == "workflow_response"
    assert msg.status == "completed"
    assert msg.result["output"] == "result"
    assert msg.error is None


def test_workflow_response_error():
    """Test error WorkflowResponse."""
    msg = WorkflowResponse(
        request_id="req-123",
        status="failed",
        error="Something went wrong",
    )

    assert msg.status == "failed"
    assert msg.error == "Something went wrong"
    assert msg.result is None


def test_workflow_update():
    """Test WorkflowUpdate message."""
    msg = WorkflowUpdate(
        request_id="req-123",
        job_id="job-456",
        update_type="node_update",
        data={"node_id": "node-1", "status": "running"},
    )

    assert msg.type == "workflow_update"
    assert msg.update_type == "node_update"
    assert msg.data["node_id"] == "node-1"


def test_command_request():
    """Test CommandRequest message."""
    msg = CommandRequest(
        request_id="req-789",
        command="list_workflows",
        args={"limit": 10},
    )

    assert msg.type == "command_request"
    assert msg.command == "list_workflows"
    assert msg.args["limit"] == 10


def test_command_response_success():
    """Test successful CommandResponse."""
    msg = CommandResponse(
        request_id="req-789",
        status="success",
        result={"workflows": []},
    )

    assert msg.type == "command_response"
    assert msg.status == "success"
    assert "workflows" in msg.result
    assert msg.error is None


def test_command_response_error():
    """Test error CommandResponse."""
    msg = CommandResponse(
        request_id="req-789",
        status="error",
        error="Command failed",
    )

    assert msg.status == "error"
    assert msg.error == "Command failed"


def test_error_message():
    """Test ErrorMessage."""
    msg = ErrorMessage(
        error="Connection failed",
        details={"reason": "timeout"},
    )

    assert msg.type == "error"
    assert msg.error == "Connection failed"
    assert msg.details["reason"] == "timeout"


def test_ack_message():
    """Test AckMessage."""
    msg = AckMessage(
        request_id="req-123",
        message="Request received",
    )

    assert msg.type == "ack"
    assert msg.request_id == "req-123"
    assert msg.message == "Request received"


def test_message_timestamp():
    """Test that all messages have timestamps."""
    msg = NodeRegistration(node_id="test")

    assert msg.timestamp is not None
    assert isinstance(msg.timestamp, datetime)


def test_message_serialization_round_trip():
    """Test message can be serialized and deserialized."""
    original = CommandRequest(
        request_id="req-123",
        command="test_command",
        args={"key": "value"},
    )

    # Serialize
    json_str = original.model_dump_json()

    # Deserialize
    data = CommandRequest.model_validate_json(json_str)

    assert data.request_id == original.request_id
    assert data.command == original.command
    assert data.args == original.args
