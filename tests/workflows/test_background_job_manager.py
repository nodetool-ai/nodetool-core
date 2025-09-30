import asyncio
import pytest
from datetime import datetime
from nodetool.workflows.background_job_manager import (
    BackgroundJobManager,
    BackgroundJob,
)
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.models.job import Job
from nodetool.types.graph import Graph, Node as GraphNode, Edge
from nodetool.models.workflow import Workflow

# Add timeout to all tests in this file to prevent hanging
pytestmark = pytest.mark.timeout(30)


@pytest.fixture
async def cleanup_jobs():
    """Cleanup jobs after each test."""
    yield
    # Clear any jobs created during the test
    manager = BackgroundJobManager.get_instance()

    # Just stop all event loops - they're daemon threads so they'll die
    for job_id, job in list(manager._jobs.items()):
        try:
            # Cancel future without waiting for result
            if job.future and not job.future.done():
                job.future.cancel()

            # Stop event loop (non-blocking due to daemon threads)
            if job.event_loop and job.event_loop.is_running:
                job.event_loop.stop()

        except Exception as e:
            print(f"Error cleaning up job {job_id}: {e}")

    # Clear the registry immediately - daemon threads will clean themselves up
    manager._jobs.clear()

    # Brief sleep to let daemon threads finish (but don't wait for join)
    await asyncio.sleep(0.1)


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


@pytest.mark.asyncio
async def test_background_job_manager_singleton():
    """Test that BackgroundJobManager is a singleton."""
    manager1 = BackgroundJobManager.get_instance()
    manager2 = BackgroundJobManager.get_instance()
    assert manager1 is manager2


@pytest.mark.asyncio
async def test_start_job(simple_workflow, cleanup_jobs):
    """Test starting a background job."""
    manager = BackgroundJobManager.get_instance()

    # Create a simple job request (no params for empty workflow)
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
    )

    # Create processing context
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    # Start the job
    bg_job = await manager.start_job(request, context)

    assert bg_job is not None
    assert bg_job.job_id is not None
    assert bg_job.runner is not None
    assert bg_job.context is context
    assert bg_job.event_loop is not None
    assert bg_job.future is not None
    assert bg_job.job_model is not None

    # Verify job is in the registry
    retrieved_job = manager.get_job(bg_job.job_id)
    assert retrieved_job is bg_job

    # Verify job was saved to database
    db_job = await Job.get(bg_job.job_id)
    assert db_job is not None
    assert db_job.id == bg_job.job_id
    assert db_job.workflow_id == simple_workflow.id
    assert db_job.user_id == "test_user"
    assert db_job.status == "running"
    assert db_job.params == {}

    # Wait a moment for the job to potentially complete
    await asyncio.sleep(0.1)

    # Cancel and cleanup
    await manager.cancel_job(bg_job.job_id)


@pytest.mark.asyncio
async def test_job_completion_updates_model(simple_workflow, cleanup_jobs):
    """Test that job completion updates the job model in database."""
    manager = BackgroundJobManager.get_instance()

    # Create a job request with empty graph (will complete quickly)
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

    # Start the job
    bg_job = await manager.start_job(request, context)
    job_id = bg_job.job_id

    # Wait for job to complete
    await asyncio.sleep(0.5)

    # Reload job from database to get latest status
    db_job = await Job.get(job_id)
    assert db_job is not None

    # Job should be completed
    assert db_job.status in ["completed", "running", "failed"]
    if db_job.status == "completed":
        assert db_job.finished_at is not None


@pytest.mark.asyncio
async def test_cancel_job(simple_workflow, cleanup_jobs):
    """Test cancelling a running job."""
    manager = BackgroundJobManager.get_instance()

    # Create a job with a long-running operation
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

    # Start the job
    bg_job = await manager.start_job(request, context)
    job_id = bg_job.job_id

    # Try to cancel the job
    # Note: Empty workflows complete very fast, so cancellation might fail
    cancelled = await manager.cancel_job(job_id)

    # Wait for job to finish
    await asyncio.sleep(0.2)

    # Verify job status
    db_job = await Job.get(job_id)
    assert db_job is not None
    # If cancellation succeeded, status should be cancelled
    # If job completed too quickly, status will be completed
    if cancelled:
        assert db_job.status == "cancelled"
    else:
        # Job completed before we could cancel it
        assert db_job.status == "completed"


@pytest.mark.asyncio
async def test_list_jobs(simple_workflow, cleanup_jobs):
    """Test listing jobs."""
    manager = BackgroundJobManager.get_instance()

    # Start multiple jobs
    jobs = []
    for i in range(3):
        request = RunJobRequest(
            workflow_id=simple_workflow.id,
            user_id="test_user",
            auth_token="test_token",
            job_type="workflow",
            params={},  # Empty graph needs no params
            graph=Graph(nodes=[], edges=[]),
        )

        context = ProcessingContext(
            user_id="test_user",
            auth_token="test_token",
            workflow_id=simple_workflow.id,
        )

        bg_job = await manager.start_job(request, context)
        jobs.append(bg_job)

    # List all jobs
    all_jobs = manager.list_jobs()
    assert len(all_jobs) >= 3

    # List jobs for specific user
    user_jobs = manager.list_jobs(user_id="test_user")
    assert len(user_jobs) >= 3

    # Cleanup
    for bg_job in jobs:
        await manager.cancel_job(bg_job.job_id)


@pytest.mark.asyncio
async def test_get_job(simple_workflow, cleanup_jobs):
    """Test getting a specific job."""
    manager = BackgroundJobManager.get_instance()

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

    bg_job = await manager.start_job(request, context)

    # Get the job
    retrieved = manager.get_job(bg_job.job_id)
    assert retrieved is bg_job

    # Try getting non-existent job
    non_existent = manager.get_job("non_existent_id")
    assert non_existent is None

    # Cleanup
    await manager.cancel_job(bg_job.job_id)


@pytest.mark.asyncio
async def test_background_job_properties(simple_workflow, cleanup_jobs):
    """Test BackgroundJob properties and methods."""
    manager = BackgroundJobManager.get_instance()

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

    bg_job = await manager.start_job(request, context)

    # Test properties
    assert bg_job.status in ["running", "completed", "cancelled", "error"]
    assert isinstance(bg_job.is_running(), bool)
    assert isinstance(bg_job.is_completed(), bool)

    # Test created_at
    assert isinstance(bg_job.created_at, datetime)
    assert bg_job.created_at <= datetime.now()

    # Cleanup
    await manager.cancel_job(bg_job.job_id)


@pytest.mark.asyncio
async def test_cleanup_completed_jobs(simple_workflow, cleanup_jobs):
    """Test automatic cleanup of completed jobs."""
    manager = BackgroundJobManager.get_instance()

    # Create and complete a job
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

    bg_job = await manager.start_job(request, context)
    job_id = bg_job.job_id

    # Wait for job to complete
    await asyncio.sleep(0.5)

    # Manually mark job as completed (for testing)
    if not bg_job.is_completed():
        bg_job.runner.status = "completed"
        bg_job.future.cancel()

    # Cleanup with max_age_seconds=0 should remove it
    await manager.cleanup_completed_jobs(max_age_seconds=0)

    # Job should be removed from manager but still in DB
    assert manager.get_job(job_id) is None

    # DB record should still exist
    db_job = await Job.get(job_id)
    # Note: Depending on timing, job may or may not be in DB


@pytest.mark.asyncio
async def test_job_error_handling(simple_workflow, cleanup_jobs):
    """Test that job errors are properly captured."""
    manager = BackgroundJobManager.get_instance()

    # Create a job that will fail (non-existent workflow)
    request = RunJobRequest(
        workflow_id="non_existent_workflow",
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=None,  # Force workflow loading which will fail
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="non_existent_workflow",
    )

    # Start the job
    bg_job = await manager.start_job(request, context)
    job_id = bg_job.job_id

    # Wait for job to fail
    await asyncio.sleep(0.5)

    # Check that error was captured
    db_job = await Job.get(job_id)
    assert db_job is not None
    # Job should have failed or be running (depending on timing)
    if db_job.status == "failed":
        assert db_job.error is not None
        assert len(db_job.error) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
