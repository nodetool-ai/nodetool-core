"""
Manager for executing workflow jobs using different execution strategies.
"""

import asyncio
import uuid
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Dict, Optional

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.models.condition_builder import Field
from nodetool.models.job import Job
from nodetool.models.run_state import RunState
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.docker_job_execution import DockerJobExecution
from nodetool.workflows.job_execution import JobExecution
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import ExecutionStrategy, RunJobRequest
from nodetool.workflows.subprocess_job_execution import SubprocessJobExecution
from nodetool.workflows.threaded_job_execution import ThreadedJobExecution

_HEARTBEAT_INTERVAL_SECONDS = 30
_STALE_THRESHOLD_MINUTES = 5

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
    _jobs: Dict[str, JobExecution]
    _cleanup_task: Optional[asyncio.Task]
    _heartbeat_task: Optional[asyncio.Task]
    _finalizing_jobs: set[str]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._jobs = {}
            cls._instance._cleanup_task = None
            cls._instance._heartbeat_task = None
            cls._instance._finalizing_jobs = set()
        return cls._instance

    async def initialize(self):
        """Initialize the manager, starting background tasks and reconciling state."""
        if self._cleanup_task is None:
            await self.start_cleanup_task()

        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            log.info("Started heartbeat task")

        await self._reconcile_on_startup()

    async def _reconcile_on_startup(self):
        """Reconcile in-memory state with persisted RunState.

        By default, stale "running" jobs are marked as failed instead of recovered.
        Set NODETOOL_AUTO_RECOVER_JOBS=1 to enable automatic job recovery.
        """
        import os

        auto_recover = os.environ.get("NODETOOL_AUTO_RECOVER_JOBS", "0") == "1"

        try:
            worker_id = Environment.get_worker_id()
            async with ResourceScope():
                # 1. Load active runs
                runs, _ = await RunState.query(
                    Field("status").in_list(["scheduled", "running", "suspended", "paused", "recovering"])
                )

                log.info(f"Reconciling {len(runs)} active runs on startup (auto_recover={auto_recover})")

                for run in runs:
                    if run.worker_id == worker_id:
                        # Owned by us - normally we'd rehydrate, but since we just restarted,
                        # the in-memory state is lost.
                        log.info(f"Found owned run {run.run_id} with status={run.status}")
                        if run.status == "running":
                            if auto_recover:
                                # Mark as recovering to let resume logic handle it
                                await run.mark_recovering()
                                log.info(f"Marked run {run.run_id} as recovering")
                            else:
                                # Mark as failed - server died while running
                                await run.mark_failed(error="Server shutdown while job was running")
                                log.warning(
                                    f"Marked run {run.run_id} as failed (server died during execution). "
                                    f"Set NODETOOL_AUTO_RECOVER_JOBS=1 to enable auto-recovery."
                                )
                    elif self._is_heartbeat_stale(run):
                        # Owned by dead worker
                        log.info(f"Found stale run {run.run_id} from worker {run.worker_id}")
                        if auto_recover:
                            await self.claim_and_recover(run)
                        else:
                            # Mark as failed instead of recovering
                            await run.mark_failed(error="Worker died while job was running")
                            log.warning(
                                f"Marked stale run {run.run_id} as failed. "
                                f"Set NODETOOL_AUTO_RECOVER_JOBS=1 to enable auto-recovery."
                            )
        except Exception as e:
            log.error(f"Error during startup reconciliation: {e}")

    def _is_heartbeat_stale(self, run: RunState) -> bool:
        return run.is_stale(_STALE_THRESHOLD_MINUTES)

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

        worker_id = Environment.get_worker_id()
        execution_id = str(uuid.uuid4())
        # Use client-provided job_id if available, otherwise generate a new one
        run_id = request.job_id if request.job_id else str(uuid.uuid4())

        # 1. Create RunState record (DB source of truth)
        # Using ResourceScope to ensure DB connection
        async with ResourceScope():
            run_state = await RunState.create_run(
                run_id=run_id,
                execution_strategy=request.execution_strategy.value,
                worker_id=worker_id,
            )
            run_state.execution_id = execution_id
            await run_state.save()

        # Switch on execution strategy
        if request.execution_strategy == ExecutionStrategy.THREADED:
            job = await ThreadedJobExecution.create_and_start(
                request, context, job_id=run_id, execution_id=execution_id
            )
        elif request.execution_strategy == ExecutionStrategy.SUBPROCESS:
            job = await SubprocessJobExecution.create_and_start(
                request, context, job_id=run_id, execution_id=execution_id
            )
        elif request.execution_strategy == ExecutionStrategy.DOCKER:
            job = await DockerJobExecution.create_and_start(request, context, job_id=run_id, execution_id=execution_id)
        else:
            raise ValueError(f"Unknown execution strategy: {request.execution_strategy}")

        # Update run state to running
        async with ResourceScope():
            run_state = await RunState.get(run_id)
            run_state.status = "running"
            await run_state.save()

        # Register the job in the manager
        self._jobs[job.job_id] = job

        log.info(f"Started job {job.job_id} with strategy {request.execution_strategy} (worker: {worker_id})")

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
                await stored_job.cleanup_resources()
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
        if job and await job.cancel():
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
            await job.cleanup_resources()
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

        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task

        # Cancel and cleanup all jobs
        for job in list(self._jobs.values()):
            if not job.is_completed():
                await job.cancel()
                # Also release ownership in DB
                try:
                    async with ResourceScope():
                        run = await RunState.get(job.job_id)
                        if run:
                            await run.release()
                except Exception:
                    pass
            await job.cleanup_resources()

        self._jobs.clear()
        log.info("JobExecutionManager shutdown complete")

    async def _heartbeat_loop(self):
        """Periodic heartbeat for all owned jobs."""
        while True:
            try:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)

                # Copy keys to avoid modification during iteration
                job_ids = list(self._jobs.keys())
                if not job_ids:
                    continue

                async with ResourceScope():
                    for job_id in job_ids:
                        job = self._jobs.get(job_id)
                        if job and job.is_running():
                            try:
                                run = await RunState.get(job_id)
                                if run:
                                    await run.update_heartbeat()
                            except Exception as e:
                                log.debug(f"Failed to heartbeat job {job_id}: {e}")

                    # Also check for stale runs to recover
                    await self.recover_stale_runs()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in heartbeat loop: {e}")

    async def recover_stale_runs(self):
        """Scan for and recover stale runs."""
        try:
            runs, _ = await RunState.query(
                Field("status").equals("running").and_(Field("execution_strategy").not_equals(None))
            )

            for run in runs:
                if self._is_heartbeat_stale(run) and not run.is_owned_by(Environment.get_worker_id()):
                    await self.claim_and_recover(run)

        except Exception as e:
            log.error(f"Error recovering stale runs: {e}")

    async def claim_and_recover(self, run: RunState):
        """Claim a potentially stale run and schedule recovery."""
        try:
            # Try to claim
            await run.save()
            log.info(f"Claimed stale run {run.run_id} for recovery")

            # Attempt to resume execution
            success = await self.resume_run(run.run_id)
            if not success:
                log.warning(f"Failed to resume run {run.run_id}, marking as failed")
                run.status = "failed"
                run.error_message = "Recovery failed: could not resume execution"
                await run.save()

        except Exception as e:
            log.error(f"Failed to claim run {run.run_id}: {e}")

    async def resume_run(self, run_id: str) -> bool:
        """
        Resume a suspended or recovering run.

        Reconstructs the execution environment from the persisted Job and RunState
        records and restarts execution.
        """
        try:
            log.info(f"Attempting to resume run {run_id}")

            async with ResourceScope():
                # 1. Fetch RunState to get execution strategy and metadata
                run_state = await RunState.get(run_id)
                if not run_state:
                    log.error(f"RunState not found for {run_id}")
                    return False

                # 2. Fetch Job model to get graph, params, user info
                # Note: RunState.run_id corresponds to Job.id
                job_model = await Job.get(run_id)
                if not job_model:
                    log.error(f"Job model not found for {run_id}")
                    return False

            # 3. Reconstruct RunJobRequest
            # Note: We rely on the persisted graph and params in the Job model
            try:
                execution_strategy = ExecutionStrategy(run_state.execution_strategy)
            except ValueError:
                log.error(f"Invalid execution strategy in RunState: {run_state.execution_strategy}")
                return False

            request = RunJobRequest(
                workflow_id=job_model.workflow_id,
                user_id=job_model.user_id,
                job_type=job_model.job_type,
                graph=job_model.graph,  # Graph is stored as dict, Pydantic should handle validation if needed
                params=job_model.params,
                execution_strategy=execution_strategy,
                # Contextual fields that might be lost or need defaults
                auth_token="",  # Auth token is lost, but internal recovery might not need it for all ops
            )

            # 4. create ProcessingContext (headless for recovery)
            context = ProcessingContext(
                user_id=request.user_id,
                job_id=run_id,
                workflow_id=request.workflow_id,
                # In recovery mode, we might not have active websocket clients initially
            )

            # 5. Relaunch Execution
            # We generate a NEW execution_id for this attempt, preserving the run_id
            new_execution_id = str(uuid.uuid4())
            worker_id = Environment.get_worker_id()

            async with ResourceScope():
                # Update RunState with new worker and execution ID
                run_state.worker_id = worker_id
                run_state.execution_id = new_execution_id
                run_state.status = "running"
                await run_state.save()

            if execution_strategy == ExecutionStrategy.THREADED:
                job = await ThreadedJobExecution.create_and_start(
                    request, context, job_id=run_id, execution_id=new_execution_id
                )
            elif execution_strategy == ExecutionStrategy.SUBPROCESS:
                job = await SubprocessJobExecution.create_and_start(
                    request, context, job_id=run_id, execution_id=new_execution_id
                )
            elif execution_strategy == ExecutionStrategy.DOCKER:
                job = await DockerJobExecution.create_and_start(
                    request, context, job_id=run_id, execution_id=new_execution_id
                )
            else:
                log.error(f"Unknown execution strategy: {execution_strategy}")
                return False

            self._jobs[run_id] = job
            log.info(f"Resumed run {run_id} with execution_id {new_execution_id}")
            return True

        except Exception as e:
            log.exception(f"Error resuming run {run_id}: {e}")
            return False
