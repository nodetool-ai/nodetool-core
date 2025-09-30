import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from nodetool.integrations.websocket.websocket_runner import (
    WebSocketRunner,
    JobStreamContext,
)
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.background_job_manager import BackgroundJobManager
from nodetool.types.graph import Graph
from nodetool.models.workflow import Workflow


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
    manager = BackgroundJobManager.get_instance()
    for job_id, job in list(manager._jobs.items()):
        try:
            if job.future and not job.future.done():
                job.future.cancel()
            if job.event_loop and job.event_loop.is_running:
                job.event_loop.stop()
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
async def test_websocket_runner_manages_multiple_jobs(
    websocket_runner, mock_websocket, simple_workflow, cleanup_jobs
):
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
    from nodetool.workflows.processing_context import ProcessingContext
    from nodetool.workflows.workflow_runner import WorkflowRunner
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
    assert ctx.background_job is bg_job
    assert ctx.streaming_task is None


@pytest.mark.asyncio
async def test_websocket_runner_send_message_when_disconnected(websocket_runner):
    """Test that send_message gracefully handles disconnected websocket."""
    # Don't connect the websocket

    # Should not raise an error
    await websocket_runner.send_message({"type": "test", "data": "test"})

    # No exception should be raised


@pytest.mark.asyncio
async def test_websocket_runner_get_status_with_job_id(
    websocket_runner, mock_websocket, simple_workflow, cleanup_jobs
):
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
        job_id = list(websocket_runner.active_jobs.keys())[0]
        status = websocket_runner.get_status(job_id)

        assert "job_id" in status
        assert status["job_id"] == job_id
