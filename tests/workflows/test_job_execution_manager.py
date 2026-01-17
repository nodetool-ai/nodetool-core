"""
Tests for JobExecutionManager class.
"""

import asyncio
import os
import subprocess

import pytest

from nodetool.models.workflow import Workflow
from nodetool.types.api_graph import Edge, Graph
from nodetool.types.api_graph import Node as GraphNode
from nodetool.workflows.docker_job_execution import DockerJobExecution
from nodetool.workflows.job_execution_manager import JobExecutionManager
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import ExecutionStrategy, RunJobRequest
from nodetool.workflows.subprocess_job_execution import SubprocessJobExecution
from nodetool.workflows.threaded_job_execution import ThreadedJobExecution

_IS_XDIST = os.environ.get("PYTEST_XDIST_WORKER", "") != ""

if _IS_XDIST:
    pytest.skip(
        "Skipped in xdist due to resource contention with threaded event loops",
        allow_module_level=True,
    )

pytestmark = pytest.mark.xdist_group(name="database")
pytestmark = [pytestmark, pytest.mark.serial]
pytestmark = pytest.mark.timeout(30)


def check_docker_available() -> bool:
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
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "nodetool"],
            capture_output=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
async def simple_workflow():
    workflow = await Workflow.create(
        user_id="test_user",
        name="Test Workflow",
        description="A test workflow",
        graph=Graph(nodes=[], edges=[]).model_dump(),
    )
    yield workflow
    await workflow.delete()


@pytest.fixture
async def job_cleanup():
    """Track jobs created during test for cleanup."""
    manager = JobExecutionManager.get_instance()
    initial_jobs = set(manager._jobs.keys())
    yield
    jobs_to_cleanup = [jid for jid in manager._jobs if jid not in initial_jobs]
    for job_id in jobs_to_cleanup:
        try:
            job = manager._jobs.get(job_id)
            if job and not job.is_completed():
                await job.cancel()
            if job:
                await job.cleanup_resources()
        except Exception:
            pass
    for job_id in jobs_to_cleanup:
        manager._jobs.pop(job_id, None)


@pytest.mark.asyncio
async def test_manager_singleton():
    """Test that JobExecutionManager is a singleton."""
    manager1 = JobExecutionManager.get_instance()
    manager2 = JobExecutionManager.get_instance()
    assert manager1 is manager2


@pytest.mark.asyncio
async def test_manager_start_threaded_job(simple_workflow, job_cleanup):
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
async def test_manager_start_subprocess_job(simple_workflow, job_cleanup):
    """Test starting a subprocess job via manager."""
    manager = JobExecutionManager.get_instance()

    graph = Graph(
        nodes=[
            GraphNode(
                id="input_text",
                type="nodetool.workflows.test_helper.StringInput",
                data={"name": "text", "label": "Input Text", "value": ""},
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
async def test_manager_get_job(simple_workflow, job_cleanup):
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

    retrieved_job = manager.get_job(job_id)
    assert retrieved_job is job


@pytest.mark.asyncio
async def test_manager_list_jobs(simple_workflow, job_cleanup):
    """Test listing jobs from the manager."""
    manager = JobExecutionManager.get_instance()

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

        await manager.start_job(request, context)

    all_jobs = manager.list_jobs()
    assert len(all_jobs) == 3

    user_jobs = manager.list_jobs(user_id="user_0")
    assert len(user_jobs) == 1
    assert user_jobs[0].request.user_id == "user_0"


@pytest.mark.asyncio
async def test_manager_cancel_job(simple_workflow, job_cleanup):
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

    result = await manager.cancel_job(job_id)

    assert isinstance(result, bool)

    await asyncio.sleep(0.2)

    assert job.status in ["cancelled", "completed", "error"]


@pytest.mark.asyncio
async def test_manager_cleanup_completed_jobs(simple_workflow, job_cleanup):
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

    await asyncio.sleep(0.5)

    if not job.is_completed():
        if job.runner:
            job.runner.status = "completed"
        await job.cancel()

    await manager.cleanup_completed_jobs(max_age_seconds=0)

    assert job_id not in manager._jobs


@pytest.mark.asyncio
async def test_manager_validates_execution_strategy(simple_workflow, job_cleanup):
    """Test that execution strategy is validated by Pydantic."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RunJobRequest(
            workflow_id=simple_workflow.id,
            user_id="test_user",
            auth_token="test_token",
            job_type="workflow",
            params={},
            graph=Graph(nodes=[], edges=[]),
            execution_strategy="unknown",
        )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not check_docker_available() or not check_nodetool_image_available(),
    reason="Docker or nodetool image not available",
)
async def test_manager_docker_strategy(simple_workflow, job_cleanup):
    """Test that Docker execution strategy works when Docker is available."""
    manager = JobExecutionManager.get_instance()

    graph = Graph(
        nodes=[
            GraphNode(
                id="input_text",
                type="nodetool.workflows.test_helper.StringInput",
                data={"name": "text", "label": "Input Text", "value": ""},
            ),
            GraphNode(
                id="format_text",
                type="nodetool.workflows.test_helper.FormatText",
                data={
                    "template": "Docker: {{ text }}",
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
async def test_manager_shutdown(simple_workflow, job_cleanup):
    """Test shutting down the manager."""
    manager = JobExecutionManager.get_instance()

    if len(manager._jobs) > 0:
        await manager.shutdown()

    assert len(manager._jobs) == 0
