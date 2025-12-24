"""
Manager for executing workflow jobs using different execution strategies.
"""

import asyncio
from contextlib import suppress
from typing import ClassVar, Dict, Optional

from nodetool.config.logging_config import get_logger
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.docker_job_execution import DockerJobExecution
from nodetool.workflows.job_execution import JobExecution
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.subprocess_job_execution import SubprocessJobExecution
from nodetool.workflows.threaded_job_execution import ThreadedJobExecution

log = get_logger(__name__)


class JobExecutionManager:
    """
    Singleton manager for executing workflow jobs.

    This manager handles:
    - Running workflows using different execution strategies
    - Job lifecycle management (start, cancel, cleanup)
    - Job recovery for reconnecting clients
    - Automatic cleanup of completed jobs
    """

    _instance: Optional["JobExecutionManager"] = None
    _jobs: ClassVar[Dict[str, JobExecution]] = {}
    _cleanup_task: ClassVar[Optional[asyncio.Task]] = None
    _finalizing_jobs: ClassVar[set[str]] = set()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._jobs = {}
            cls._instance._cleanup_task = None
            cls._instance._finalizing_jobs = set()
        return cls._instance

    @classmethod
    def get_instance(cls) -> "JobExecutionManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = JobExecutionManager()
        return cls._instance

    async def start_job(self, request: RunJobRequest, context: ProcessingContext) -> JobExecution:
        """
        Start a new job execution using the requested execution strategy.

        Args:
            request: Job request with workflow details and execution strategy
            context: Processing context for the job

        Returns:
            JobExecution instance

        Raises:
            NotImplementedError: If the requested execution strategy is not yet implemented
        """
        from nodetool.workflows.run_job_request import ExecutionStrategy

        # Switch on execution strategy
        if request.execution_strategy == ExecutionStrategy.THREADED:
            job = await ThreadedJobExecution.create_and_start(request, context)
        elif request.execution_strategy == ExecutionStrategy.SUBPROCESS:
            job = await SubprocessJobExecution.create_and_start(request, context)
        elif request.execution_strategy == ExecutionStrategy.DOCKER:
            job = await DockerJobExecution.create_and_start(request, context)
        else:
            raise ValueError(f"Unknown execution strategy: {request.execution_strategy}")

        # Register the job in the manager
        self._jobs[job.job_id] = job

        log.info(f"Started job {job.job_id} with strategy {request.execution_strategy}")

        return job

    def get_job(self, job_id: str) -> Optional[JobExecution]:
        """Get a job by ID, scheduling persistence if it has finished."""
        job = self._jobs.get(job_id)

        if job is None:
            log.debug(
                "JobExecutionManager.get_job: job not found in memory",
                extra={"job_id": job_id},
            )
            return None

        if not job.is_running():
            log.info(
                "JobExecutionManager.get_job: job no longer running, scheduling persistence",
                extra={
                    "job_id": job_id,
                    "status": job.status,
                    "is_completed": job.is_completed(),
                },
            )
            # Persist state asynchronously to avoid blocking caller
            if job_id not in self._finalizing_jobs:
                self._finalizing_jobs.add(job_id)
                task = asyncio.create_task(self._finalize_job_state(job))
                job._finalize_task = task
            return job

        return job

    async def _finalize_job_state(self, job: JobExecution) -> None:
        """Ensure finished jobs have their status written to the database and cleanup resources."""
        job_id = job.job_id
        try:
            # Wrap in ResourceScope to ensure database adapter access for Job update
            async with ResourceScope():
                await job.finalize_state()
        finally:
            # Remove job from registry and cleanup its resources
            stored_job = self._jobs.pop(job_id, None)
            if stored_job:
                stored_job.cleanup_resources()
            self._finalizing_jobs.discard(job_id)

    def list_jobs(self, user_id: Optional[str] = None) -> list[JobExecution]:
        """
        List all jobs, optionally filtered by user.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of JobExecution instances
        """
        jobs = list(self._jobs.values())
        if user_id:
            jobs = [job for job in jobs if job.request.user_id == user_id]
        return jobs

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: ID of the job to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        job = self.get_job(job_id)
        if job and job.cancel():
            log.info(f"Cancelled background job {job_id}")
            return True
        return False

    async def cleanup_completed_jobs(self, max_age_seconds: int = 3600):
        """
        Clean up completed jobs older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds for completed jobs
        """
        jobs_to_remove = []

        for job_id, job in self._jobs.items():
            if job.is_completed() and job.age_seconds > max_age_seconds:
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            job = self._jobs.pop(job_id)
            job.cleanup_resources()
            log.info(f"Cleaned up completed job {job_id}")

    async def start_cleanup_task(self, interval_seconds: int = 300):
        """
        Start periodic cleanup task.

        Args:
            interval_seconds: Interval between cleanup runs
        """
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await self.cleanup_completed_jobs()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.error(f"Error in cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        log.info("Started background job cleanup task")

    async def shutdown(self):
        """Shutdown the job manager and all running jobs."""
        log.info("Shutting down JobExecutionManager")

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Cancel and cleanup all jobs
        for job in list(self._jobs.values()):
            if not job.is_completed():
                job.cancel()
            job.cleanup_resources()

        self._jobs.clear()
        log.info("JobExecutionManager shutdown complete")
