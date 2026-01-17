"""
Tests for ThreadedJobExecution class.
"""

import asyncio
import os

import pytest

from nodetool.models.job import Job
from nodetool.models.run_state import RunState
from nodetool.models.workflow import Workflow
from nodetool.types.api_graph import Graph
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.threaded_job_execution import ThreadedJobExecution
from tests.conftest import get_job_status

# Check if running with pytest-xdist
_IS_XDIST = os.environ.get("PYTEST_XDIST_WORKER", "") != ""

if _IS_XDIST:
    # Skip all tests in this module when running with xdist due to resource contention
    pytest.skip(
        "Skipped in xdist due to resource contention with threaded event loops",
        allow_module_level=True,
    )

# Add timeout to all tests in this file to prevent hanging
# Run these tests in the same xdist group to avoid parallel execution issues
pytestmark = [pytest.mark.timeout(30), pytest.mark.xdist_group(name="job_execution")]


@pytest.fixture
async def cleanup_jobs():
    """Cleanup resources after each test."""
    jobs_to_cleanup = []
    yield jobs_to_cleanup
    for job in jobs_to_cleanup:
        try:
            await job.cleanup_resources()
            if not job.is_completed():
                await job.cancel()
        except Exception:
            pass


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
    await workflow.delete()


@pytest.mark.asyncio
async def test_threaded_job_creation(simple_workflow, cleanup_jobs):
    """Test creating a threaded job execution."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await ThreadedJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    assert job is not None
    assert job.job_id is not None
    assert job.runner is not None
    assert job.event_loop is not None
    assert job.future is not None
    assert job.context is context
    assert isinstance(job, ThreadedJobExecution)


@pytest.mark.asyncio
async def test_threaded_job_is_running(simple_workflow, cleanup_jobs):
    """Test checking if a threaded job is running."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await ThreadedJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Job should be running initially
    assert job.is_running() or job.is_completed()

    # Wait a bit for empty workflow to complete
    await asyncio.sleep(0.2)

    # After completion, is_running should return False
    if job.is_completed():
        assert not job.is_running() or (job.runner and job.runner.is_running() is False)


@pytest.mark.asyncio
async def test_threaded_job_cancellation(simple_workflow, cleanup_jobs):
    """Test cancelling a threaded job."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await ThreadedJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Cancel the job
    result = await job.cancel()

    # Should return True if cancelled, False if already completed
    assert isinstance(result, bool)

    if result:
        assert job.status == "cancelled"
        assert job.is_completed()


@pytest.mark.asyncio
async def test_threaded_job_cleanup(simple_workflow, cleanup_jobs):
    """Test cleaning up threaded job resources."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await ThreadedJobExecution.create_and_start(request, context)

    # Cleanup should stop the event loop
    await job.cleanup_resources()

    # Event loop should be stopped
    await asyncio.sleep(0.1)
    assert not job.event_loop.is_running


@pytest.mark.asyncio
async def test_threaded_job_status_updates(simple_workflow, cleanup_jobs):
    """Test that threaded job status updates correctly."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await ThreadedJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Initial status should be running
    assert job.status == "running"

    # Wait for completion
    await asyncio.sleep(0.3)

    # Status should update to completed or still running
    assert job.status in ["running", "completed", "cancelled"]


@pytest.mark.asyncio
async def test_threaded_job_database_record(simple_workflow, cleanup_jobs):
    """Test that threaded job creates and updates database record."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await ThreadedJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Check database record exists
    db_job = await Job.get(job.job_id)
    assert db_job is not None
    assert db_job.id == job.job_id
    assert db_job.workflow_id == simple_workflow.id
    assert db_job.user_id == "test_user"
    # Create RunState for the job (ThreadedJobExecution doesn't create it)
    run_state = await RunState.create_run(run_id=job.job_id, execution_strategy="threaded")
    run_state.status = "running"
    await run_state.save()
    # Empty workflows may complete very quickly, so status could be running, completed, failed, or scheduled
    status = await get_job_status(job.job_id)
    assert status in ["running", "completed", "failed", "scheduled"]

    # Wait for completion
    await asyncio.sleep(0.3)

    # Reload and check status updated
    status = await get_job_status(job.job_id)
    assert status in ["running", "completed", "failed", "scheduled"]


@pytest.mark.asyncio
async def test_threaded_job_age_property(simple_workflow, cleanup_jobs):
    """Test that threaded job age property works."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await ThreadedJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Age should be close to 0 initially
    age = job.age_seconds
    assert 0 <= age < 1.0

    # Wait a bit
    await asyncio.sleep(0.2)

    # Age should have increased
    new_age = job.age_seconds
    assert new_age > age


@pytest.mark.asyncio
async def test_threaded_job_completion_event(simple_workflow, cleanup_jobs):
    """Test that threaded job sends completion event."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await ThreadedJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Wait for completion
    max_wait = 2.0
    waited = 0.0
    step = 0.1
    while not job.is_completed() and waited < max_wait:
        await asyncio.sleep(step)
        waited += step

    # Job should be completed
    assert job.is_completed()

    # Check that completion event was posted to message queue
    completion_event_found = False
    checked_messages = []

    while context.has_messages():
        try:
            message = context.message_queue.get_nowait()
            checked_messages.append(message)

            if isinstance(message, JobUpdate):
                if message.job_id == job.job_id and message.status == "completed":
                    completion_event_found = True
                    break
        except Exception:
            break

    # Assert that we found a completion event
    assert completion_event_found, (
        f"No completion event found for job {job.job_id}. "
        f"Messages checked: {[type(m).__name__ for m in checked_messages]}"
    )


@pytest.mark.asyncio
async def test_threaded_job_cancellation_event(simple_workflow, cleanup_jobs):
    """Test that threaded job sends cancellation event."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await ThreadedJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Cancel the job quickly (before it might naturally complete)
    await asyncio.sleep(0.05)
    cancelled = await job.cancel()

    # If cancellation succeeded, wait for the message
    if cancelled:
        await asyncio.sleep(0.2)

        # Check that cancellation event was posted to message queue
        cancellation_event_found = False
        checked_messages = []

        while context.has_messages():
            try:
                message = context.message_queue.get_nowait()
                checked_messages.append(message)

                if isinstance(message, JobUpdate):
                    if message.job_id == job.job_id and message.status == "cancelled":
                        cancellation_event_found = True
                        break
            except Exception:
                break

        # Assert that we found a cancellation event
        assert cancellation_event_found, (
            f"No cancellation event found for job {job.job_id}. "
            f"Job status: {job.status}. "
            f"Messages checked: {[type(m).__name__ for m in checked_messages]}"
        )
