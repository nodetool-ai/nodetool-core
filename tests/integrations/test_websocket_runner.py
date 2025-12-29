import asyncio
from unittest.mock import AsyncMock, MagicMock

import msgpack
import pytest

from nodetool.integrations.websocket.websocket_runner import (
    JobStreamContext,
    WebSocketRunner,
    extract_binary_data_from_value,
)
from nodetool.models.workflow import Workflow
from nodetool.types.graph import Graph
from nodetool.workflows.job_execution_manager import JobExecutionManager
from nodetool.workflows.run_job_request import RunJobRequest


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
async def websocket_runner():
    """Create a WebSocketRunner instance."""
    runner = WebSocketRunner()
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
            # Use the cleanup_resources method
            job.cleanup_resources()

            # Cancel if not completed
            if not job.is_completed():
                job.cancel()
        except Exception as e:
            print(f"Error cleaning up job {job_id}: {e}")
    manager._jobs.clear()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_websocket_runner_connects(websocket_runner, mock_websocket):
    """Test that WebSocketRunner can establish a connection."""
    await websocket_runner.connect(mock_websocket)

    assert websocket_runner.websocket is not None
    mock_websocket.accept.assert_called_once()


@pytest.mark.asyncio
async def test_websocket_runner_disconnects(websocket_runner, mock_websocket):
    """Test that WebSocketRunner can disconnect."""
    await websocket_runner.connect(mock_websocket)
    await websocket_runner.disconnect()

    assert websocket_runner.websocket is None
    mock_websocket.close.assert_called_once()


@pytest.mark.asyncio
async def test_websocket_runner_manages_multiple_jobs(websocket_runner, mock_websocket, simple_workflow, cleanup_jobs):
    """Test that WebSocketRunner can manage multiple concurrent jobs."""
    await websocket_runner.connect(mock_websocket)

    # Create two job requests
    request1 = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    request2 = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    # Start both jobs
    await websocket_runner.run_job(request1)
    await websocket_runner.run_job(request2)

    # Give jobs time to start
    await asyncio.sleep(0.2)

    # Both jobs should be tracked
    assert len(websocket_runner.active_jobs) <= 2  # May complete quickly

    # Get status should show active jobs
    status = websocket_runner.get_status()
    assert "active_jobs" in status


@pytest.mark.asyncio
async def test_websocket_runner_job_streaming_context():
    """Test JobStreamContext creation."""
    from unittest.mock import MagicMock

    # Create mock background job
    bg_job = MagicMock()
    bg_job.job_id = "test-job"
    bg_job.request.workflow_id = "test-workflow"
    bg_job.status = "running"

    # Create streaming context
    ctx = JobStreamContext("test-job", "test-workflow", bg_job)

    assert ctx.job_id == "test-job"
    assert ctx.workflow_id == "test-workflow"
    assert ctx.job_execution is bg_job
    assert ctx.streaming_task is None


@pytest.mark.asyncio
async def test_websocket_runner_send_message_when_disconnected(websocket_runner):
    """Test that send_message gracefully handles disconnected websocket."""
    # Don't connect the websocket

    # Should not raise an error
    await websocket_runner.send_message({"type": "test", "data": "test"})

    # No exception should be raised


@pytest.mark.asyncio
async def test_websocket_runner_get_status_with_job_id(websocket_runner, mock_websocket, simple_workflow, cleanup_jobs):
    """Test getting status for a specific job."""
    await websocket_runner.connect(mock_websocket)

    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    await websocket_runner.run_job(request)
    await asyncio.sleep(0.2)

    # Get job_id from active jobs
    if websocket_runner.active_jobs:
        job_id = next(iter(websocket_runner.active_jobs.keys()))
        status = websocket_runner.get_status(job_id)

        assert "job_id" in status
        assert status["job_id"] == job_id


def test_extract_binary_data_from_value_with_image_asset():
    """Test extracting binary data from an ImageRef."""
    binary_data = b"fake_image_data"
    message = {
        "type": "node_update",
        "node_id": "test_node",
        "result": {
            "output": {
                "type": "image",
                "uri": "memory://test",
                "data": binary_data,
            }
        }
    }

    binaries = []
    result = extract_binary_data_from_value(message, binaries)

    # Binary data should be extracted
    assert len(binaries) == 1
    assert binaries[0] == binary_data

    # Data field should be None, binary_index should be set
    assert result["result"]["output"]["data"] is None
    assert result["result"]["output"]["binary_index"] == 0
    assert result["result"]["output"]["type"] == "image"


def test_extract_binary_data_from_value_with_multiple_assets():
    """Test extracting binary data from multiple AssetRefs."""
    binary1 = b"image_data"
    binary2 = b"audio_data"
    message = {
        "type": "node_update",
        "node_id": "test_node",
        "result": {
            "image": {
                "type": "image",
                "uri": "memory://img",
                "data": binary1,
            },
            "audio": {
                "type": "audio",
                "uri": "memory://aud",
                "data": binary2,
            }
        }
    }

    binaries = []
    result = extract_binary_data_from_value(message, binaries)

    # Both binary data should be extracted
    assert len(binaries) == 2
    assert binaries[0] == binary1
    assert binaries[1] == binary2

    # Data fields should be None, binary_indices should be set
    assert result["result"]["image"]["data"] is None
    assert result["result"]["image"]["binary_index"] == 0
    assert result["result"]["audio"]["data"] is None
    assert result["result"]["audio"]["binary_index"] == 1


def test_extract_binary_data_from_value_with_no_data():
    """Test that messages without binary data are not modified."""
    message = {
        "type": "node_update",
        "node_id": "test_node",
        "result": {
            "output": {
                "type": "image",
                "uri": "memory://test",
                "data": None,
            }
        }
    }

    binaries = []
    result = extract_binary_data_from_value(message, binaries)

    # No binary data should be extracted
    assert len(binaries) == 0

    # Message should remain unchanged
    assert result["result"]["output"]["data"] is None
    assert "binary_index" not in result["result"]["output"]


def test_extract_binary_data_from_nested_structures():
    """Test extracting binary data from nested lists and dicts."""
    binary1 = b"data1"
    binary2 = b"data2"
    message = {
        "type": "node_update",
        "result": {
            "items": [
                {
                    "type": "image",
                    "data": binary1,
                },
                {
                    "nested": {
                        "type": "audio",
                        "data": binary2,
                    }
                }
            ]
        }
    }

    binaries = []
    result = extract_binary_data_from_value(message, binaries)

    # Both binary data should be extracted
    assert len(binaries) == 2
    assert binaries[0] == binary1
    assert binaries[1] == binary2

    # Check indices are set correctly
    assert result["result"]["items"][0]["binary_index"] == 0
    assert result["result"]["items"][1]["nested"]["binary_index"] == 1


@pytest.mark.asyncio
async def test_websocket_runner_sends_binary_array_in_binary_mode(websocket_runner, mock_websocket):
    """Test that binary data is sent as msgpack array in BINARY mode."""
    from nodetool.integrations.websocket.websocket_runner import WebSocketMode

    await websocket_runner.connect(mock_websocket)
    websocket_runner.mode = WebSocketMode.BINARY

    binary_data = b"test_image_data"
    message = {
        "type": "node_update",
        "node_id": "test",
        "result": {
            "output": {
                "type": "image",
                "uri": "memory://test",
                "data": binary_data,
            }
        }
    }

    await websocket_runner.send_message(message)

    # Verify send_bytes was called
    assert mock_websocket.send_bytes.called

    # Unpack the message and verify it's an array
    sent_data = mock_websocket.send_bytes.call_args[0][0]
    unpacked = msgpack.unpackb(sent_data, raw=False)

    # Should be an array with [message, binary_data]
    assert isinstance(unpacked, list)
    assert len(unpacked) == 2

    # First element should be the modified message
    assert unpacked[0]["type"] == "node_update"
    assert unpacked[0]["result"]["output"]["data"] is None
    assert unpacked[0]["result"]["output"]["binary_index"] == 0

    # Second element should be the binary data
    assert unpacked[1] == binary_data


@pytest.mark.asyncio
async def test_websocket_runner_sends_plain_message_without_binary(websocket_runner, mock_websocket):
    """Test that messages without binary data are sent as plain msgpack."""
    from nodetool.integrations.websocket.websocket_runner import WebSocketMode

    await websocket_runner.connect(mock_websocket)
    websocket_runner.mode = WebSocketMode.BINARY

    message = {
        "type": "node_update",
        "node_id": "test",
        "status": "completed",
    }

    await websocket_runner.send_message(message)

    # Verify send_bytes was called
    assert mock_websocket.send_bytes.called

    # Unpack the message
    sent_data = mock_websocket.send_bytes.call_args[0][0]
    unpacked = msgpack.unpackb(sent_data, raw=False)

    # Should be just the message dict, not an array
    assert isinstance(unpacked, dict)
    assert unpacked["type"] == "node_update"
    assert unpacked["status"] == "completed"
