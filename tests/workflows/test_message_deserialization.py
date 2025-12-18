"""
Unit tests for message deserialization in job execution.
"""

import pytest

from nodetool.workflows.docker_job_execution import (
    MESSAGE_TYPE_MAP,
    _deserialize_processing_message,
)
from nodetool.workflows.subprocess_job_execution import (
    MESSAGE_TYPE_MAP as SUBPROCESS_MESSAGE_TYPE_MAP,
)
from nodetool.workflows.subprocess_job_execution import (
    _deserialize_processing_message as subprocess_deserialize,
)
from nodetool.workflows.types import (
    JobUpdate,
    LogUpdate,
    NodeUpdate,
    OutputUpdate,
    PlanningUpdate,
    PreviewUpdate,
    SaveUpdate,
    TaskUpdate,
    ToolCallUpdate,
)


def test_message_type_map_completeness():
    """Test that MESSAGE_TYPE_MAP includes all expected message types."""
    expected_types = [
        "preview_update",
        "save_update",
        "log_update",
        "task_update",
        "tool_call_update",
        "planning_update",
        "output_update",
        "node_update",
        "job_update",
        "node_progress",
        "edge_update",
        "error",
    ]

    for msg_type in expected_types:
        assert (
            msg_type in MESSAGE_TYPE_MAP
        ), f"MESSAGE_TYPE_MAP should include {msg_type}"


def test_docker_and_subprocess_maps_match():
    """Test that Docker and subprocess MESSAGE_TYPE_MAPs are identical."""
    assert set(MESSAGE_TYPE_MAP.keys()) == set(
        SUBPROCESS_MESSAGE_TYPE_MAP.keys()
    ), "Docker and subprocess MESSAGE_TYPE_MAPs should have the same keys"

    for key in MESSAGE_TYPE_MAP:
        assert (
            MESSAGE_TYPE_MAP[key] == SUBPROCESS_MESSAGE_TYPE_MAP[key]
        ), f"Message type {key} should map to the same class in both implementations"


def test_deserialize_preview_update():
    """Test deserializing a PreviewUpdate message."""
    msg_dict = {"type": "preview_update", "node_id": "test_node", "value": "test_value"}

    result = _deserialize_processing_message(msg_dict)

    assert isinstance(result, PreviewUpdate), "Should deserialize to PreviewUpdate"
    assert result.node_id == "test_node"
    assert result.value == "test_value"


def test_deserialize_save_update():
    """Test deserializing a SaveUpdate message."""
    msg_dict = {
        "type": "save_update",
        "node_id": "test_node",
        "name": "test_save",
        "value": "saved_value",
        "output_type": "string",
        "metadata": {},
    }

    result = _deserialize_processing_message(msg_dict)

    assert isinstance(result, SaveUpdate), "Should deserialize to SaveUpdate"
    assert result.node_id == "test_node"
    assert result.name == "test_save"
    assert result.value == "saved_value"


def test_deserialize_log_update():
    """Test deserializing a LogUpdate message."""
    msg_dict = {
        "type": "log_update",
        "node_id": "test_node",
        "node_name": "Test Node",
        "content": "Log message",
        "severity": "info",
    }

    result = _deserialize_processing_message(msg_dict)

    assert isinstance(result, LogUpdate), "Should deserialize to LogUpdate"
    assert result.node_id == "test_node"
    assert result.content == "Log message"
    assert result.severity == "info"


def test_deserialize_task_update():
    """Test deserializing a TaskUpdate message."""
    msg_dict = {
        "type": "task_update",
        "node_id": "test_node",
        "task": {
            "id": "task_1",
            "objective": "Test task",
            "status": "running",
            "plan": [],
            "steps": [],
        },
        "step": None,
        "event": "task_created",
    }

    result = _deserialize_processing_message(msg_dict)

    assert isinstance(result, TaskUpdate), "Should deserialize to TaskUpdate"
    assert result.node_id == "test_node"
    assert result.event == "task_created"


def test_deserialize_tool_call_update():
    """Test deserializing a ToolCallUpdate message."""
    msg_dict = {
        "type": "tool_call_update",
        "node_id": "test_node",
        "name": "test_tool",
        "args": {"arg1": "value1"},
        "message": "Tool called",
    }

    result = _deserialize_processing_message(msg_dict)

    assert isinstance(result, ToolCallUpdate), "Should deserialize to ToolCallUpdate"
    assert result.node_id == "test_node"
    assert result.name == "test_tool"
    assert result.args == {"arg1": "value1"}


def test_deserialize_planning_update():
    """Test deserializing a PlanningUpdate message."""
    msg_dict = {
        "type": "planning_update",
        "node_id": "test_node",
        "phase": "planning",
        "status": "in_progress",
        "content": "Planning content",
    }

    result = _deserialize_processing_message(msg_dict)

    assert isinstance(result, PlanningUpdate), "Should deserialize to PlanningUpdate"
    assert result.node_id == "test_node"
    assert result.phase == "planning"
    assert result.status == "in_progress"


def test_deserialize_output_update():
    """Test deserializing an OutputUpdate message."""
    msg_dict = {
        "type": "output_update",
        "node_id": "test_node",
        "node_name": "Test Node",
        "output_name": "result",
        "value": "output_value",
        "output_type": "string",
        "metadata": {},
    }

    result = _deserialize_processing_message(msg_dict)

    assert isinstance(result, OutputUpdate), "Should deserialize to OutputUpdate"
    assert result.node_id == "test_node"
    assert result.output_name == "result"
    assert result.value == "output_value"


def test_deserialize_node_update():
    """Test deserializing a NodeUpdate message (original working case)."""
    msg_dict = {
        "type": "node_update",
        "node_id": "test_node",
        "node_name": "Test Node",
        "node_type": "test.Node",
        "status": "running",
        "error": None,
        "result": None,
        "properties": None,
    }

    result = _deserialize_processing_message(msg_dict)

    assert isinstance(result, NodeUpdate), "Should deserialize to NodeUpdate"
    assert result.node_id == "test_node"
    assert result.status == "running"


def test_deserialize_job_update():
    """Test deserializing a JobUpdate message (original working case)."""
    msg_dict = {
        "type": "job_update",
        "job_id": "test_job",
        "status": "running",
        "error": None,
        "result": None,
        "workflow_id": "test_workflow",
    }

    result = _deserialize_processing_message(msg_dict)

    assert isinstance(result, JobUpdate), "Should deserialize to JobUpdate"
    assert result.job_id == "test_job"
    assert result.status == "running"


def test_deserialize_unknown_type():
    """Test deserializing an unknown message type."""
    msg_dict = {"type": "unknown_type", "data": "value"}

    result = _deserialize_processing_message(msg_dict)

    # Should return the dict as-is
    assert isinstance(result, dict), "Should return dict for unknown type"
    assert result == msg_dict


def test_deserialize_none_type():
    """Test deserializing a message with no type."""
    msg_dict = {"data": "value"}

    result = _deserialize_processing_message(msg_dict)

    assert result is None, "Should return None for message without type"


def test_deserialize_malformed_message():
    """Test deserializing a malformed message."""
    msg_dict = {
        "type": "preview_update",
        # Missing required fields
    }

    result = _deserialize_processing_message(msg_dict)

    # Should return the dict when deserialization fails
    assert isinstance(result, dict), "Should return dict when deserialization fails"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
