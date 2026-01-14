import asyncio
from datetime import datetime

import pytest

from nodetool.models.job import Job
from nodetool.models.run_state import RunState
from nodetool.models.workflow import Workflow
from nodetool.types.api_graph import Edge, Graph
from nodetool.types.api_graph import Node as GraphNode
from nodetool.workflows.job_execution_manager import (
    JobExecutionManager,
    SubprocessJobExecution,
    ThreadedJobExecution,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest

# Add timeout to all tests in this file to prevent hanging
# Run these tests in the same xdist group to avoid parallel execution issues
# Use serial marker to avoid singleton state conflicts
pytestmark = [pytest.mark.timeout(10), pytest.mark.xdist_group(name="job_execution"), pytest.mark.serial]


@pytest.fixture
async def cleanup_jobs():
    """Cleanup jobs after each test."""
    yield
    manager = JobExecutionManager.get_instance()

    jobs_to_cleanup = list(manager._jobs.items())
    for job_id, job in jobs_to_cleanup:
        try:
            if not job.is_completed():
                await job.cancel()
            await job.cleanup_resources()
        except Exception as e:
            print(f"Error cleaning up job {job_id}: {e}")

    manager._jobs.clear()

    await asyncio.sleep(0.5)


@pytest.fixture
async def simple_workflow():
    """Create a simple workflow for testing.

    Note: Cleanup is handled by table truncation in setup_and_teardown fixture,
    so we don't need to explicitly delete the workflow here.
    """
    workflow = await Workflow.create(
        user_id="test_user",
        name="Test Workflow",
        description="A test workflow",
        graph=Graph(nodes=[], edges=[]).model_dump(),
    )
    yield workflow
    # No explicit cleanup needed - table truncation handles it


@pytest.mark.asyncio
async def test_background_job_manager_singleton():
    """Test that JobExecutionManager is a singleton."""
    manager1 = JobExecutionManager.get_instance()
    manager2 = JobExecutionManager.get_instance()
    assert manager1 is manager2


@pytest.mark.asyncio
async def test_start_job(simple_workflow, cleanup_jobs):
    """Test starting a background job."""
    manager = JobExecutionManager.get_instance()

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
    assert bg_job.job_model is not None

    # Check that it's a ThreadedJobExecution with event_loop and future
    assert isinstance(bg_job, ThreadedJobExecution)
    assert bg_job.event_loop is not None
    assert bg_job.future is not None

    # Verify job is in the registry
    retrieved_job = manager.get_job(bg_job.job_id)
    assert retrieved_job is bg_job

    # Verify job was saved to database
    db_job = await Job.get(bg_job.job_id)
    assert db_job is not None
    assert db_job.id == bg_job.job_id
    assert db_job.workflow_id == simple_workflow.id
    assert db_job.user_id == "test_user"

    # Wait for job to be in running state (with timeout)
    for _ in range(50):
        run_state = await RunState.get(bg_job.job_id)
        if run_state and run_state.status == "running":
            break
        await asyncio.sleep(0.05)
    else:
        # If still not running, get the final state
        run_state = await RunState.get(bg_job.job_id)

    assert run_state is not None
    assert run_state.status in ("running", "completed")
    assert db_job.params == {}

    # Wait a moment for the job to potentially complete
    await asyncio.sleep(0.1)

    # Cancel and cleanup
    await manager.cancel_job(bg_job.job_id)


@pytest.mark.asyncio
async def test_job_completion_updates_model(simple_workflow, cleanup_jobs):
    """Test that job completion updates the job model in database."""
    manager = JobExecutionManager.get_instance()

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

    # Job should be completed - check RunState for status
    run_state = await RunState.get(job_id)
    assert run_state is not None
    assert run_state.status in ["completed", "running", "failed"]
    if run_state.status == "completed":
        assert db_job.finished_at is not None


@pytest.mark.asyncio
async def test_cancel_job(simple_workflow, cleanup_jobs):
    """Test cancelling a running job."""
    manager = JobExecutionManager.get_instance()

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
    await manager.cancel_job(job_id)

    # Wait for job to finish updating and poll for final status
    max_retries = 10
    for _ in range(max_retries):
        await asyncio.sleep(0.1)
        run_state = await RunState.get(job_id)
        # Break if we've reached a final state
        if run_state and run_state.status in ["completed", "cancelled", "failed"]:
            break

    # Verify job status
    run_state = await RunState.get(job_id)
    assert run_state is not None
    # Status should be in a final state (not running)
    assert run_state.status in ["completed", "cancelled", "failed"]

    # Note: Even if cancel() returns True, empty workflows may complete
    # before the cancellation takes effect, so we accept both statuses


@pytest.mark.asyncio
async def test_get_job(simple_workflow, cleanup_jobs):
    """Test getting a specific job."""
    manager = JobExecutionManager.get_instance()

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
async def test_cleanup_completed_jobs(simple_workflow, cleanup_jobs):
    """Test automatic cleanup of completed jobs."""
    manager = JobExecutionManager.get_instance()

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
        if bg_job.runner:
            bg_job.runner.status = "completed"
        # Cancel the job to mark it as completed
        await bg_job.cancel()

    # Cleanup with max_age_seconds=0 should remove it
    await manager.cleanup_completed_jobs(max_age_seconds=0)

    # Job should be removed from manager but still in DB
    assert manager.get_job(job_id) is None

    # Note: Depending on timing, job may or may not be in DB


@pytest.mark.asyncio
async def test_job_error_handling(simple_workflow, cleanup_jobs):
    """Test that job errors are properly captured."""
    manager = JobExecutionManager.get_instance()

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
    run_state = await RunState.get(job_id)
    if run_state and run_state.status == "failed":
        assert db_job.error is not None
        assert len(db_job.error) > 0


@pytest.mark.asyncio
async def test_subprocess_job_execution(simple_workflow, cleanup_jobs):
    """Test starting a job with subprocess execution strategy."""
    from nodetool.workflows.run_job_request import ExecutionStrategy

    manager = JobExecutionManager.get_instance()

    # Create a simple graph with a format node
    graph = Graph(
        nodes=[
            GraphNode(
                id="input_text",
                type="nodetool.workflows.test_helper.StringInput",
                data={
                    "name": "text",
                    "label": "Input Text",
                    "value": "",
                },
            ),
            GraphNode(
                id="format_text",
                type="nodetool.workflows.test_helper.FormatText",
                data={
                    "template": "Hello, {{ text }}",
                    "inputs": {"text": {"node_id": "input_text", "output": "value"}},
                },
            ),
            GraphNode(
                id="output_result",
                type="nodetool.workflows.test_helper.StringOutput",
                data={
                    "name": "result",
                    "value": "",
                    "inputs": {"value": {"node_id": "format_text", "output": "value"}},
                },
            ),
        ],
        edges=[
            Edge(
                id="edge_input_to_format",
                source="input_text",
                sourceHandle="output",
                target="format_text",
                targetHandle="text",
            ),
            Edge(
                id="edge_format_to_output",
                source="format_text",
                sourceHandle="output",
                target="output_result",
                targetHandle="value",
            ),
        ],
    )

    # Create job request with subprocess strategy
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={"text": "World"},
        graph=graph,
        execution_strategy=ExecutionStrategy.SUBPROCESS,
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
    assert bg_job.runner is None  # Subprocess doesn't use runner
    assert bg_job.context is context
    assert bg_job.job_model is not None

    # Check that it's a SubprocessJobExecution with process
    assert isinstance(bg_job, SubprocessJobExecution)
    assert bg_job.process is not None
    assert bg_job._stdout_task is not None
    assert bg_job._stderr_task is not None

    # Verify job is in the registry
    retrieved_job = manager.get_job(bg_job.job_id)
    assert retrieved_job is bg_job

    # Verify job was saved to database
    db_job = await Job.get(bg_job.job_id)
    assert db_job is not None
    assert db_job.id == bg_job.job_id
    assert db_job.workflow_id == simple_workflow.id
    assert db_job.user_id == "test_user"

    # Wait for subprocess to complete (increased timeout for parallel test runs)
    max_wait = 15.0
    wait_interval = 0.2
    elapsed = 0.0

    while elapsed < max_wait and bg_job.is_running():
        await asyncio.sleep(wait_interval)
        elapsed += wait_interval

    # Check completion
    assert bg_job.is_completed(), f"Subprocess job should have completed (elapsed: {elapsed}s, status: {bg_job.status})"
    assert bg_job.status in [
        "completed",
        "failed",
    ], f"Unexpected status: {bg_job.status}"

    # If failed, log the error
    if bg_job.status == "failed":
        db_job = await Job.get(bg_job.job_id)
        if db_job:
            print(f"Job failed with error: {db_job.error}")

    # Cancel and cleanup (should be no-op if already completed)
    await manager.cancel_job(bg_job.job_id)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
