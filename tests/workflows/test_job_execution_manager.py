"""
Tests for JobExecutionManager class.
"""

import asyncio
import subprocess
import pytest
from nodetool.workflows.job_execution_manager import JobExecutionManager
from nodetool.workflows.threaded_job_execution import ThreadedJobExecution
from nodetool.workflows.subprocess_job_execution import SubprocessJobExecution
from nodetool.workflows.docker_job_execution import DockerJobExecution
from nodetool.workflows.run_job_request import RunJobRequest, ExecutionStrategy
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Graph, Node as GraphNode, Edge
from nodetool.models.workflow import Workflow


def check_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_nodetool_image_available() -> bool:
    """Check if the nodetool Docker image is available."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "nodetool"],
            capture_output=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Add timeout to all tests in this file to prevent hanging
pytestmark = pytest.mark.timeout(30)


@pytest.fixture
async def cleanup_jobs():
    """Cleanup jobs after each test."""
    yield
    # Clear any jobs created during the test
    manager = JobExecutionManager.get_instance()

    for job_id, job in list(manager._jobs.items()):
        try:
            job.cleanup_resources()
            if not job.is_completed():
                job.cancel()
        except Exception as e:
            print(f"Error cleaning up job {job_id}: {e}")

    manager._jobs.clear()
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


@pytest.mark.asyncio
async def test_manager_singleton():
    """Test that JobExecutionManager is a singleton."""
    manager1 = JobExecutionManager.get_instance()
    manager2 = JobExecutionManager.get_instance()
    assert manager1 is manager2


@pytest.mark.asyncio
async def test_manager_start_threaded_job(simple_workflow, cleanup_jobs):
    """Test starting a threaded job via manager."""
    manager = JobExecutionManager.get_instance()

    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={},
        graph=Graph(nodes=[], edges=[]),
        execution_strategy=ExecutionStrategy.THREADED,
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await manager.start_job(request, context)

    assert job is not None
    assert isinstance(job, ThreadedJobExecution)
    assert job.job_id in manager._jobs


@pytest.mark.asyncio
async def test_manager_start_subprocess_job(simple_workflow, cleanup_jobs):
    """Test starting a subprocess job via manager."""
    manager = JobExecutionManager.get_instance()

    graph = Graph(
        nodes=[
            GraphNode(
                id="input_text",
                type="nodetool.input.StringInput",
                data={"name": "text", "label": "Input Text", "value": ""},
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
                id="edge1",
                source="input_text",
                sourceHandle="output",
                target="format_text",
                targetHandle="text",
            ),
            Edge(
                id="edge2",
                source="format_text",
                sourceHandle="output",
                target="output_result",
                targetHandle="value",
            ),
        ],
    )

    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={"text": "Manager"},
        graph=graph,
        execution_strategy=ExecutionStrategy.SUBPROCESS,
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await manager.start_job(request, context)

    assert job is not None
    assert isinstance(job, SubprocessJobExecution)
    assert job.job_id in manager._jobs


@pytest.mark.asyncio
async def test_manager_get_job(simple_workflow, cleanup_jobs):
    """Test getting a job from the manager."""
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

    job = await manager.start_job(request, context)
    job_id = job.job_id

    # Should be able to retrieve the job
    retrieved_job = manager.get_job(job_id)
    assert retrieved_job is job


@pytest.mark.asyncio
async def test_manager_list_jobs(simple_workflow, cleanup_jobs):
    """Test listing jobs from the manager."""
    manager = JobExecutionManager.get_instance()

    # Start multiple jobs
    jobs = []
    for i in range(3):
        request = RunJobRequest(
            workflow_id=simple_workflow.id,
            user_id=f"user_{i}",
            auth_token="test_token",
            job_type="workflow",
            params={},
            graph=Graph(nodes=[], edges=[]),
        )

        context = ProcessingContext(
            user_id=f"user_{i}",
            auth_token="test_token",
            workflow_id=simple_workflow.id,
        )

        job = await manager.start_job(request, context)
        jobs.append(job)

    # List all jobs
    all_jobs = manager.list_jobs()
    assert len(all_jobs) >= 3

    # List jobs for specific user
    user_jobs = manager.list_jobs(user_id="user_0")
    assert len(user_jobs) >= 1
    assert all(j.request.user_id == "user_0" for j in user_jobs)


@pytest.mark.asyncio
async def test_manager_cancel_job(simple_workflow, cleanup_jobs):
    """Test cancelling a job via manager."""
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

    job = await manager.start_job(request, context)
    job_id = job.job_id

    # Cancel the job
    result = await manager.cancel_job(job_id)

    # Should return True if cancelled (or False if already completed)
    assert isinstance(result, bool)

    # Wait a bit for job to finish
    await asyncio.sleep(0.2)

    # Job should reach a final state
    # Note: Empty workflows may complete before cancellation takes effect,
    # so we accept both "cancelled" and "completed" statuses
    assert job.status in ["cancelled", "completed", "error"]


@pytest.mark.asyncio
async def test_manager_cleanup_completed_jobs(simple_workflow, cleanup_jobs):
    """Test cleaning up completed jobs."""
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

    job = await manager.start_job(request, context)
    job_id = job.job_id

    # Wait for job to complete
    await asyncio.sleep(0.5)

    # Force cancel to mark as completed
    if not job.is_completed():
        if job.runner:
            job.runner.status = "completed"
        job.cancel()

    # Cleanup with max_age_seconds=0 should remove it
    await manager.cleanup_completed_jobs(max_age_seconds=0)

    # Job should be removed from manager
    assert job_id not in manager._jobs


@pytest.mark.asyncio
async def test_manager_validates_execution_strategy(simple_workflow, cleanup_jobs):
    """Test that execution strategy is validated by Pydantic."""
    # Pydantic validates the execution strategy enum before manager sees it
    # This test verifies that invalid strategies are rejected by Pydantic
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RunJobRequest(
            workflow_id=simple_workflow.id,
            user_id="test_user",
            auth_token="test_token",
            job_type="workflow",
            params={},
            graph=Graph(nodes=[], edges=[]),
            execution_strategy="unknown",  # type: ignore
        )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not check_docker_available() or not check_nodetool_image_available(),
    reason="Docker or nodetool image not available",
)
async def test_manager_docker_strategy(simple_workflow, cleanup_jobs):
    """Test that Docker execution strategy works when Docker is available."""
    manager = JobExecutionManager.get_instance()

    graph = Graph(
        nodes=[
            GraphNode(
                id="input_text",
                type="nodetool.input.StringInput",
                data={"name": "text", "label": "Input Text", "value": ""},
            ),
            GraphNode(
                id="format_text",
                type="nodetool.text.FormatText",
                data={
                    "template": "Docker: {{ text }}",
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

    request = RunJobRequest(
        workflow_id=simple_workflow.id,
        user_id="test_user",
        auth_token="test_token",
        job_type="workflow",
        params={"text": "Manager test"},
        graph=graph,
        execution_strategy=ExecutionStrategy.DOCKER,
    )

    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id=simple_workflow.id,
    )

    job = await manager.start_job(request, context)

    try:
        assert job is not None
        assert isinstance(job, DockerJobExecution)
        assert job.job_id in manager._jobs

        # Wait for completion with timeout
        max_wait = 30
        for _ in range(max_wait):
            if job.is_finished():
                break
            await asyncio.sleep(1)

        assert job.is_finished()
    finally:
        if job:
            await job.cleanup_resources()


@pytest.mark.asyncio
async def test_manager_shutdown(simple_workflow, cleanup_jobs):
    """Test shutting down the manager."""
    manager = JobExecutionManager.get_instance()

    # Start a few jobs
    for i in range(2):
        request = RunJobRequest(
            workflow_id=simple_workflow.id,
            user_id=f"user_{i}",
            auth_token="test_token",
            job_type="workflow",
            params={},
            graph=Graph(nodes=[], edges=[]),
        )

        context = ProcessingContext(
            user_id=f"user_{i}",
            auth_token="test_token",
            workflow_id=simple_workflow.id,
        )

        await manager.start_job(request, context)

    # Shutdown should cancel and cleanup all jobs
    await manager.shutdown()

    # All jobs should be removed
    assert len(manager._jobs) == 0
