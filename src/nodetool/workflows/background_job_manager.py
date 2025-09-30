import asyncio
from concurrent.futures import Future
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from nodetool.config.logging_config import get_logger
from nodetool.models.job import Job
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.threaded_event_loop import ThreadedEventLoop
from nodetool.workflows.workflow_runner import WorkflowRunner

log = get_logger(__name__)


class BackgroundJob:
    """
    Represents a single background job running in a dedicated thread.

    Attributes:
        job_id: Unique identifier for the job
        runner: WorkflowRunner instance
        context: ProcessingContext for the job
        event_loop: ThreadedEventLoop running the job
        future: Future representing the running job
        request: Original job request
        job_model: Job model instance for database updates
        created_at: When the job was created
    """

    def __init__(
        self,
        job_id: str,
        runner: WorkflowRunner,
        context: ProcessingContext,
        event_loop: ThreadedEventLoop,
        future: Future,
        request: RunJobRequest,
        job_model: Job,
    ):
        self.job_id = job_id
        self.runner = runner
        self.context = context
        self.event_loop = event_loop
        self.future = future
        self.request = request
        self.job_model = job_model
        self.created_at = datetime.now()

    @property
    def status(self) -> str:
        """Get the current status of the job."""
        return self.runner.status

    def is_running(self) -> bool:
        """Check if the job is still running."""
        return self.runner.is_running()

    def is_completed(self) -> bool:
        """Check if the job has completed (success, error, or cancelled)."""
        return self.future.done()

    def cancel(self) -> bool:
        """Cancel the running job."""
        if not self.is_completed():
            self.future.cancel()
            return True
        return False


class BackgroundJobManager:
    """
    Singleton manager for background workflow jobs.

    This manager handles:
    - Running workflows in persistent background threads
    - Job lifecycle management (start, cancel, cleanup)
    - Job recovery for reconnecting clients
    - Automatic cleanup of completed jobs
    """

    _instance: Optional["BackgroundJobManager"] = None
    _jobs: Dict[str, BackgroundJob] = {}
    _cleanup_task: Optional[asyncio.Task] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackgroundJobManager, cls).__new__(cls)
            cls._instance._jobs = {}
            cls._instance._cleanup_task = None
        return cls._instance

    @classmethod
    def get_instance(cls) -> "BackgroundJobManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = BackgroundJobManager()
        return cls._instance

    async def start_job(
        self, request: RunJobRequest, context: ProcessingContext
    ) -> BackgroundJob:
        """
        Start a new background job.

        Args:
            request: Job request with workflow details
            context: Processing context for the job

        Returns:
            BackgroundJob instance
        """
        job_id = uuid4().hex
        runner = WorkflowRunner(job_id=job_id)

        # Create persistent event loop for this job
        event_loop = ThreadedEventLoop()
        event_loop.start()

        log.info(f"Starting background job {job_id} for workflow {request.workflow_id}")

        # Create the job record in database and get the instance
        # Use base class create() since Job.create() auto-generates an ID
        job_model = Job(
            id=job_id,
            workflow_id=request.workflow_id,
            user_id=request.user_id,
            job_type=request.job_type,
            status="running",
            graph=request.graph.model_dump() if request.graph else {},
            params=request.params or {},
        )
        await job_model.save()

        # Define the execution coroutine
        async def execute():
            try:
                if request.graph is None:
                    log.info(f"Loading workflow graph for {request.workflow_id}")
                    workflow = await context.get_workflow(request.workflow_id)
                    if workflow is None:
                        raise ValueError(f"Workflow {request.workflow_id} not found")
                    request.graph = workflow.get_api_graph()

                await runner.run(request, context)

                # Update job status on completion using instance method
                await job_model.update(status="completed", finished_at=datetime.now())
                log.info(f"Background job {job_id} completed successfully")

            except asyncio.CancelledError:
                runner.status = "cancelled"
                await job_model.update(status="cancelled", finished_at=datetime.now())
                log.info(f"Background job {job_id} cancelled")
                raise
            except Exception as e:
                runner.status = "error"
                error_msg = str(e)
                await job_model.update(
                    status="failed", error=error_msg, finished_at=datetime.now()
                )
                log.error(f"Background job {job_id} failed: {error_msg}")
                context.post_message(
                    JobUpdate(job_id=job_id, status="failed", error=error_msg)
                )
                raise

        # Schedule execution on the persistent loop
        future = event_loop.run_coroutine(execute())

        # Create background job object
        bg_job = BackgroundJob(
            job_id=job_id,
            runner=runner,
            context=context,
            event_loop=event_loop,
            future=future,
            request=request,
            job_model=job_model,
        )

        self._jobs[job_id] = bg_job

        return bg_job

    def get_job(self, job_id: str) -> Optional[BackgroundJob]:
        """Get a job by ID, scheduling persistence if it has finished."""
        job = self._jobs.get(job_id)

        if job is None:
            log.debug(
                "BackgroundJobManager.get_job: job not found in memory",
                extra={"job_id": job_id},
            )
            return None

        if not job.is_running():
            log.info(
                "BackgroundJobManager.get_job: job no longer running, scheduling persistence",
                extra={
                    "job_id": job_id,
                    "status": job.status,
                    "is_completed": job.is_completed(),
                },
            )
            # Persist state asynchronously to avoid blocking caller
            asyncio.create_task(self._finalize_job_state(job))
            return None

        return job

    async def _finalize_job_state(self, job: BackgroundJob) -> None:
        """Ensure finished jobs have their status written to the database and cleanup resources."""
        job_id = job.job_id
        try:
            # Reload to ensure we operate on latest values
            await job.job_model.reload()
            update_kwargs = {}

            if job.job_model.status in {"running", "starting", "queued"}:
                update_kwargs["status"] = job.status
            if job.job_model.finished_at is None:
                update_kwargs["finished_at"] = datetime.now()

            if update_kwargs:
                await job.job_model.update(**update_kwargs)

        except Exception as e:
            log.error(
                "BackgroundJobManager._finalize_job_state: failed to persist status",
                exc_info=True,
                extra={"job_id": job_id, "error": str(e)},
            )
        finally:
            # Remove job from registry and stop its event loop if still running
            stored_job = self._jobs.pop(job_id, None)
            if (
                stored_job
                and stored_job.event_loop
                and stored_job.event_loop.is_running
            ):
                stored_job.event_loop.stop()

    def list_jobs(self, user_id: Optional[str] = None) -> list[BackgroundJob]:
        """
        List all jobs, optionally filtered by user.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of BackgroundJob instances
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
        now = datetime.now()
        jobs_to_remove = []

        for job_id, job in self._jobs.items():
            if job.is_completed():
                age = (now - job.created_at).total_seconds()
                if age > max_age_seconds:
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            job = self._jobs.pop(job_id)
            if job.event_loop and job.event_loop.is_running:
                job.event_loop.stop()
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
        log.info("Shutting down BackgroundJobManager")

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel and cleanup all jobs
        for job_id, job in list(self._jobs.items()):
            if not job.is_completed():
                job.cancel()
            if job.event_loop and job.event_loop.is_running:
                job.event_loop.stop()

        self._jobs.clear()
        log.info("BackgroundJobManager shutdown complete")
