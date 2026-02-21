"""
Job session management for workflow execution.

This module provides the JobSession class which manages a session-scoped
execution context for workflow jobs. It maintains a shared event loop
(thread) that is reused across jobs in the same session, preventing
VRAM leaks that occur when creating a new thread per job.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from uuid import uuid4

from nodetool.config.logging_config import get_logger
from nodetool.models.job import Job
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.threaded_event_loop import ThreadedEventLoop
from nodetool.workflows.threaded_job_execution import ThreadedJobExecution
from nodetool.workflows.workflow_runner import WorkflowRunner

log = get_logger(__name__)


class JobSession:
    """
    Manages a session-scoped execution context for workflow jobs.
    
    A JobSession maintains a shared ThreadedEventLoop that is reused
    across all jobs in the session. This prevents VRAM fragmentation
    that occurs when creating a new thread (with its own CUDA context)
    for each job.
    
    Usage:
        session = JobSession()
        await session.start()
        
        # Run multiple jobs - they all share the same event loop
        job1 = await session.start_job(request1, context1)
        job2 = await session.start_job(request2, context2)
        
        # When done, stop the session
        await session.stop()
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize a new job session.
        
        Args:
            session_id: Optional session identifier (generated if not provided)
        """
        self.session_id = session_id or uuid4().hex
        self._event_loop: Optional[ThreadedEventLoop] = None
        self._active_jobs: dict[str, ThreadedJobExecution] = {}
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Check if the session's event loop is running."""
        return self._event_loop is not None and self._event_loop.is_running

    async def start(self) -> None:
        """Start the session by initializing the shared event loop."""
        async with self._lock:
            if self._event_loop is None:
                self._event_loop = ThreadedEventLoop()
                self._event_loop.start()
                log.info(f"JobSession {self.session_id}: Started shared event loop")

    async def start_job(
        self,
        request: RunJobRequest,
        context: ProcessingContext,
        job_id: Optional[str] = None,
    ) -> ThreadedJobExecution:
        """
        Start a new job in this session using the shared event loop.
        
        Args:
            request: Job request with workflow details
            context: Processing context for the job
            job_id: Optional existing job ID (generated if not provided)
            
        Returns:
            ThreadedJobExecution instance with execution started
            
        Raises:
            RuntimeError: If the session has not been started
        """
        if not self.is_running:
            raise RuntimeError("JobSession not started. Call start() first.")

        job_id = job_id or uuid4().hex
        runner = WorkflowRunner(job_id=job_id)

        log.info(f"JobSession {self.session_id}: Starting job {job_id}")

        # Create job record in database
        job_model = Job(
            id=job_id,
            workflow_id=request.workflow_id,
            user_id=request.user_id,
            job_type=request.job_type,
            graph=request.graph.model_dump() if request.graph else {},
            params=request.params or {},
        )

        async with ResourceScope():
            await job_model.save()

        # Create job instance with shared event loop
        job = ThreadedJobExecution(
            job_id=job_id,
            runner=runner,
            context=context,
            request=request,
            job_model=job_model,
            event_loop=self._event_loop,
            owns_event_loop=False,  # Session owns the event loop
        )

        # Track active job
        self._active_jobs[job_id] = job

        # Start execution on shared event loop
        job.future = self._event_loop.run_coroutine(job.execute())
        job._set_status("running")

        log.info(f"JobSession {self.session_id}: Job {job_id} started")
        return job

    def get_job(self, job_id: str) -> Optional[ThreadedJobExecution]:
        """Get an active job by ID."""
        return self._active_jobs.get(job_id)

    def list_active_jobs(self) -> list[ThreadedJobExecution]:
        """List all active jobs in this session."""
        return list(self._active_jobs.values())

    def remove_job(self, job_id: str) -> Optional[ThreadedJobExecution]:
        """Remove a job from tracking (call when job completes)."""
        job = self._active_jobs.pop(job_id, None)
        if job:
            log.debug(f"JobSession {self.session_id}: Removed job {job_id}")
        return job

    async def stop(self) -> None:
        """
        Stop the session and clean up resources.
        
        Cancels all active jobs and stops the shared event loop.
        """
        log.info(f"JobSession {self.session_id}: Stopping")

        # Cancel all active jobs
        for job_id, job in list(self._active_jobs.items()):
            if job.is_running():
                log.info(f"JobSession {self.session_id}: Cancelling job {job_id}")
                await job.cancel()
        self._active_jobs.clear()

        # Stop the shared event loop
        if self._event_loop and self._event_loop.is_running:
            self._event_loop.stop()
            log.info(f"JobSession {self.session_id}: Stopped event loop")

        self._event_loop = None

    async def __aenter__(self) -> JobSession:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
