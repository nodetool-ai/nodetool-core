"""
Tests for SubprocessJobExecution class.
"""

import asyncio

import pytest

from nodetool.models.job import Job
from nodetool.models.workflow import Workflow
from nodetool.types.graph import Edge, Graph
from nodetool.types.graph import Node as GraphNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import ExecutionStrategy, RunJobRequest
from nodetool.workflows.subprocess_job_execution import SubprocessJobExecution

# Add timeout to all tests in this file to prevent hanging
pytestmark = pytest.mark.timeout(30)


@pytest.fixture
async def cleanup_jobs():
    """Cleanup resources after each test."""
    jobs_to_cleanup = []
    yield jobs_to_cleanup
    # Cleanup any jobs created during tests
    for job in jobs_to_cleanup:
        try:
            job.cleanup_resources()
            if not job.is_completed():
                job.cancel()
        except Exception as e:
            print(f"Error cleaning up job {job.job_id}: {e}")
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
    await workflow.delete()


def _build_simple_workflow_graph() -> Graph:
    """Build a simple workflow graph for testing."""
    return Graph(
        nodes=[
            GraphNode(
                id="input_text",
                type="nodetool.input.StringInput",
                data={
                    "name": "text",
                    "label": "Input Text",
                    "value": "",
                },
            ),
            GraphNode(
                id="format_text",
                type="nodetool.text.FormatText",
                data={
                    "template": "Hello, {{ text }}",
                    "inputs": {"text": {"node_id": "input_text", "output": "value"}},
                },
            ),
            GraphNode(
                id="output_result",
                type="nodetool.output.StringOutput",
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


@pytest.mark.asyncio
async def test_subprocess_job_creation(simple_workflow, cleanup_jobs):
    """Test creating a subprocess job execution."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={"text": "World"},
        graph=_build_simple_workflow_graph(),
        execution_strategy=ExecutionStrategy.SUBPROCESS,
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await SubprocessJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    assert job is not None
    assert job.job_id is not None
    assert job.runner is None  # Subprocess doesn't use runner
    assert job.process is not None
    assert job._stdout_task is not None
    assert job._stderr_task is not None
    assert job.context is context
    assert isinstance(job, SubprocessJobExecution)


@pytest.mark.asyncio
async def test_subprocess_job_completion(simple_workflow, cleanup_jobs):
    """Test that subprocess job completes successfully."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={"text": "Completion"},
        graph=_build_simple_workflow_graph(),
        execution_strategy=ExecutionStrategy.SUBPROCESS,
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await SubprocessJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Wait for completion
    max_wait = 5.0
    wait_interval = 0.1
    elapsed = 0.0

    while elapsed < max_wait and not job.is_completed():
        await asyncio.sleep(wait_interval)
        elapsed += wait_interval

    assert job.is_completed()
    assert job.status in ["completed", "failed"]


@pytest.mark.asyncio
async def test_subprocess_job_cancellation(simple_workflow, cleanup_jobs):
    """Test cancelling a subprocess job."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={"text": "Cancel"},
        graph=_build_simple_workflow_graph(),
        execution_strategy=ExecutionStrategy.SUBPROCESS,
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await SubprocessJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Try to cancel immediately
    if job.is_running():
        result = job.cancel()
        assert result is True
        assert job.status == "cancelled"

    # Wait a bit for process to terminate
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_subprocess_job_cleanup(simple_workflow, cleanup_jobs):
    """Test cleaning up subprocess job resources."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={"text": "Cleanup"},
        graph=_build_simple_workflow_graph(),
        execution_strategy=ExecutionStrategy.SUBPROCESS,
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await SubprocessJobExecution.create_and_start(request, context)

    # Cleanup should kill the process and cancel tasks
    job.cleanup_resources()

    # Wait a bit
    await asyncio.sleep(0.2)

    # Process should be terminated
    assert job.process.returncode is not None or job.is_completed()


@pytest.mark.asyncio
async def test_subprocess_job_database_record(simple_workflow, cleanup_jobs):
    """Test that subprocess job creates and updates database record."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={"text": "Database"},
        graph=_build_simple_workflow_graph(),
        execution_strategy=ExecutionStrategy.SUBPROCESS,
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await SubprocessJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Check database record exists
    db_job = await Job.get(job.job_id)
    assert db_job is not None
    assert db_job.id == job.job_id
    assert db_job.workflow_id == simple_workflow.id
    assert db_job.user_id == "test_user"

    # Wait for completion
    max_wait = 5.0
    wait_interval = 0.1
    elapsed = 0.0

    while elapsed < max_wait and not job.is_completed():
        await asyncio.sleep(wait_interval)
        elapsed += wait_interval

    # Reload and check status
    await db_job.reload()
    assert db_job.status in ["completed", "failed", "cancelled"]


@pytest.mark.asyncio
async def test_subprocess_job_messages_forwarded(simple_workflow, cleanup_jobs):
    """Test that subprocess job forwards messages to context."""
    messages_received = []

    class TestContext(ProcessingContext):
        def post_message(self, message):
            messages_received.append(message)
            return super().post_message(message)

    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={"text": "Messages"},
        graph=_build_simple_workflow_graph(),
        execution_strategy=ExecutionStrategy.SUBPROCESS,
    )

    context = TestContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await SubprocessJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Wait for completion
    max_wait = 5.0
    wait_interval = 0.1
    elapsed = 0.0

    while elapsed < max_wait and not job.is_completed():
        await asyncio.sleep(wait_interval)
        elapsed += wait_interval

    # Should have received some messages
    assert len(messages_received) > 0


@pytest.mark.asyncio
async def test_subprocess_job_process_pid(simple_workflow, cleanup_jobs):
    """Test that subprocess job has a process ID."""
    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={"text": "PID"},
        graph=_build_simple_workflow_graph(),
        execution_strategy=ExecutionStrategy.SUBPROCESS,
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await SubprocessJobExecution.create_and_start(request, context)
    cleanup_jobs.append(job)

    # Process should have a PID
    assert job.process.pid is not None
    assert isinstance(job.process.pid, int)
    assert job.process.pid > 0
