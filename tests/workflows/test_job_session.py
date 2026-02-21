"""
Tests for JobSession class.
"""

import asyncio
import os

import pytest

from nodetool.models.job import Job
from nodetool.models.workflow import Workflow
from nodetool.types.api_graph import Graph
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.job_session import JobSession
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
pytestmark = [pytest.mark.timeout(30), pytest.mark.xdist_group(name="job_session")]


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
async def test_job_session_creation():
    """Test creating a job session."""
    session = JobSession()
    
    assert session.session_id is not None
    assert not session.is_running
    assert len(session.list_active_jobs()) == 0


@pytest.mark.asyncio
async def test_job_session_start_stop():
    """Test starting and stopping a job session."""
    session = JobSession()
    
    # Initially not running
    assert not session.is_running
    
    # Start the session
    await session.start()
    assert session.is_running
    assert session._event_loop is not None
    assert session._event_loop.is_running
    
    # Stop the session
    await session.stop()
    assert not session.is_running
    assert session._event_loop is None


@pytest.mark.asyncio
async def test_job_session_context_manager():
    """Test job session as async context manager."""
    async with JobSession() as session:
        assert session.is_running
        assert len(session.list_active_jobs()) == 0
    
    # After exiting context, session should be stopped
    assert not session.is_running


@pytest.mark.asyncio
async def test_job_session_start_job(simple_workflow):
    """Test starting a job in a session."""
    async with JobSession() as session:
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
        
        job = await session.start_job(request, context)
        
        # Check job was created correctly
        assert job is not None
        assert job.job_id is not None
        assert isinstance(job, ThreadedJobExecution)
        assert job.event_loop is session._event_loop  # Shares session's event loop
        
        # Check job is tracked
        assert len(session.list_active_jobs()) == 1
        assert session.get_job(job.job_id) is job


@pytest.mark.asyncio
async def test_job_session_multiple_jobs(simple_workflow):
    """Test running multiple jobs in the same session."""
    async with JobSession() as session:
        jobs = []
        
        # Start 3 jobs
        for i in range(3):
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
            
            job = await session.start_job(request, context)
            jobs.append(job)
        
        # All jobs should be tracked
        assert len(session.list_active_jobs()) == 3
        
        # All jobs should share the same event loop
        event_loop = session._event_loop
        for job in jobs:
            assert job.event_loop is event_loop
            assert not job._owns_event_loop  # Session owns the event loop


@pytest.mark.asyncio
async def test_job_session_remove_job(simple_workflow):
    """Test removing a job from session tracking."""
    async with JobSession() as session:
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
        
        job = await session.start_job(request, context)
        job_id = job.job_id
        
        # Job should be tracked
        assert session.get_job(job_id) is job
        
        # Remove job from tracking
        removed = session.remove_job(job_id)
        assert removed is job
        assert session.get_job(job_id) is None
        assert len(session.list_active_jobs()) == 0


@pytest.mark.asyncio
async def test_job_session_cleanup_on_stop(simple_workflow):
    """Test that stopping session cleans up jobs."""
    session = JobSession()
    await session.start()
    
    # Start a slow job
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
    
    job = await session.start_job(request, context)
    
    # Stop the session
    await session.stop()
    
    # All jobs should be removed from tracking
    assert len(session.list_active_jobs()) == 0
    assert not session.is_running


@pytest.mark.asyncio
async def test_job_session_event_loop_not_stopped_by_job(simple_workflow):
    """Test that individual jobs don't stop the shared event loop."""
    async with JobSession() as session:
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
        
        job = await session.start_job(request, context)
        
        # Cleanup job resources
        await job.cleanup_resources()
        
        # Session's event loop should still be running
        # (cleanup_resources shouldn't stop it since owns_event_loop=False)
        await asyncio.sleep(0.1)
        assert session.is_running
        assert session._event_loop.is_running


@pytest.mark.asyncio
async def test_job_session_job_completion(simple_workflow):
    """Test that jobs complete successfully in a session."""
    async with JobSession() as session:
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
        
        job = await session.start_job(request, context)
        
        # Wait for completion
        max_wait = 2.0
        waited = 0.0
        step = 0.1
        while not job.is_completed() and waited < max_wait:
            await asyncio.sleep(step)
            waited += step
        
        # Job should be completed
        assert job.is_completed()
        
        # Session should still be running
        assert session.is_running
