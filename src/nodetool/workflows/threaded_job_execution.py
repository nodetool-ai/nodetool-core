"""
Threaded job execution strategy using dedicated event loops.
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from nodetool.config.logging_config import get_logger
from nodetool.models.job import Job
from nodetool.runtime.resources import ResourceScope, _current_scope, maybe_scope
from nodetool.types.job import JobUpdate
from nodetool.workflows.job_execution import JobExecution
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.threaded_event_loop import ThreadedEventLoop
from nodetool.workflows.workflow_runner import WorkflowRunner

if TYPE_CHECKING:
    from concurrent.futures import Future

log = get_logger(__name__)


class ThreadedJobExecution(JobExecution):
    """
    Job execution using a dedicated thread with an event loop.

    This is the default execution strategy that runs workflows in a
    persistent background thread with its own asyncio event loop.

    Additional Attributes:
        event_loop: ThreadedEventLoop running the job
        future: Future representing the running job
    """

    def __init__(
        self,
        job_id: str,
        runner: WorkflowRunner,
        context: ProcessingContext,
        request: RunJobRequest,
        job_model: Job,
        event_loop: ThreadedEventLoop,
        execution_id: str | None = None,
    ):
        super().__init__(job_id, context, request, job_model, runner=runner, execution_id=execution_id)
        self.event_loop = event_loop
        self.future: Future | None = None
        self._db_path: str | None = None

    def _set_status(self, status: str) -> None:
        """Update both the internal and runner status consistently."""
        self._update_status(status)
        if self.runner:
            self.runner.status = status

    def push_input_value(self, input_name: str, value: Any, source_handle: str) -> None:
        """Push an input value to the job execution."""
        assert self.runner, "Runner is not set"
        self.runner.push_input_value(input_name=input_name, value=value, source_handle=source_handle)

    def is_running(self) -> bool:
        """Check if the job is still running."""
        if self.runner:
            return self.runner.is_running()
        return not self.is_completed()

    def is_completed(self) -> bool:
        """Check if the job has completed (success, error, or cancelled)."""
        return self.future is not None and self.future.done()

    async def cancel(self) -> bool:
        """Cancel the running job."""
        if not self.is_completed():
            if self.future:
                self.future.cancel()

                # Propagate cancellation to the asyncio task running on the thread
                # The task is attached to the future by ThreadedEventLoop.run_coroutine
                task = getattr(self.future, "task", None)
                # Use private _loop access as we are tightly coupled with ThreadedEventLoop
                loop = self.event_loop._loop
                if task and loop and loop.is_running():
                    loop.call_soon_threadsafe(task.cancel)

            # Force-release the GPU lock if it's stuck from this job.
            # When a GPU node's inference thread is blocked (e.g. nunchaku hang),
            # the CancelledError may not reach the finally block that releases the lock.
            from nodetool.workflows.workflow_runner import force_release_gpu_lock
            force_release_gpu_lock()

            self._set_status("cancelled")
            return True
        return False

    async def cleanup_resources(self) -> None:
        """Clean up resources associated with this job (stop event loop)."""
        if self.event_loop and self.event_loop.is_running:
            self.event_loop.stop()
            log.debug(f"Stopped event loop for job {self.job_id}")

    async def execute(self) -> None:
        """Execute the workflow.

        Handles loading workflow graph, running the workflow, and updating
        job status throughout the execution lifecycle.

        The workflow runs within a ResourceScope to provide per-execution
        resource isolation (database adapters, settings, secrets).
        """
        assert self.runner, "Runner is not set"
        # ThreadedEventLoop propagates contextvars from the parent thread,
        # which includes _current_scope. The parent scope's DB pool uses an
        # asyncio.Queue bound to the parent's event loop. Using it from this
        # thread's event loop would cause RuntimeError. Clear the inherited
        # scope and create a fresh one with a pool for THIS event loop.
        _current_scope.set(None)

        from nodetool.runtime.db_sqlite import SQLiteConnectionPool

        pool = None
        if self._db_path:
            pool = await SQLiteConnectionPool.get_shared(self._db_path)

        try:
            async with ResourceScope(pool=pool):
                try:
                    # Load workflow graph if not already loaded
                    if self.request.graph is None:
                        log.info(f"Loading workflow graph for {self.request.workflow_id}")
                        workflow = await self.context.get_workflow(self.request.workflow_id)
                        if workflow is None:
                            raise ValueError(f"Workflow {self.request.workflow_id} not found")
                        self.request.graph = workflow.get_api_graph()

                    self._set_status("running")
                    await self.runner.run(self.request, self.context)

                    # Check if workflow was suspended (not completed)
                    if self.runner and self.runner.status == "suspended":
                        log.info("Workflow suspended, not setting completed status")
                        from nodetool.models.job import Job

                        job = await Job.get(self.job_id)
                        if job:
                            await job.mark_suspended(
                                node_id="",
                                reason="Workflow suspended",
                                state={},
                            )
                        await self.job_model.update(finished_at=datetime.now())

                        # Post suspension message BEFORE finalize_state to avoid race condition
                        self.context.post_message(
                            JobUpdate(
                                job_id=self.job_id,
                                status="suspended",
                                message="Workflow suspended",
                            )
                        )
                    elif self.runner and self.runner.status == "cancelled":
                        # Runner detected cancellation during execution
                        self._set_status("cancelled")
                        from nodetool.models.job import Job

                        job = await Job.get(self.job_id)
                        if job:
                            await job.mark_cancelled()
                        await self.job_model.update(finished_at=datetime.now())
                        # Cancelled message likely already sent by runner or will be handled by external cancellation
                    elif self.runner and self.runner.status == "completed":
                        # Runner completed successfully and ALREADY sent the completion update with results
                        self._set_status("completed")
                        from nodetool.models.job import Job

                        job = await Job.get(self.job_id)
                        if job:
                            await job.mark_completed()
                        await self.job_model.update(finished_at=datetime.now())
                        # DO NOT send another JobUpdate here as the runner already did it
                    else:
                        # Fallback for cases where runner didn't set explicit terminal status but finished without error
                        # This shouldn't happen with correct runner implementation but we keep it for safety
                        self._set_status("completed")
                        from nodetool.models.job import Job

                        job = await Job.get(self.job_id)
                        if job:
                            await job.mark_completed()
                        await self.job_model.update(finished_at=datetime.now())

                        # Post completion message BEFORE finalize_state to avoid race condition
                        self.context.post_message(
                            JobUpdate(
                                job_id=self.job_id,
                                status="completed",
                                message=f"Job {self.job_id} completed successfully",
                            )
                        )
                    log.info(f"Background job {self.job_id} finished execution loop")

                except asyncio.CancelledError:
                    self._set_status("cancelled")
                    from nodetool.models.job import Job

                    job = await Job.get(self.job_id)
                    if job:
                        await job.mark_cancelled()
                    await self.job_model.update(finished_at=datetime.now())

                    # Post cancellation message BEFORE finalize_state to avoid race condition
                    self.context.post_message(
                        JobUpdate(
                            job_id=self.job_id,
                            status="cancelled",
                            message=f"Job {self.job_id} was cancelled",
                        )
                    )
                    log.info(f"Background job {self.job_id} cancelled")
                    raise
                except Exception as e:
                    self._set_status("error")
                    import traceback

                    error_text = str(e).strip()
                    error_msg = f"{e.__class__.__name__}: {error_text}" if error_text else repr(e)
                    tb_text = traceback.format_exc()
                    self._error = error_msg
                    from nodetool.models.job import Job

                    job = await Job.get(self.job_id)
                    if job:
                        await job.mark_failed(error=error_msg)
                    await self.job_model.update(error=error_msg, finished_at=datetime.now())
                    log.exception("Background job %s failed: %s", self.job_id, error_msg)
                    self.context.post_message(
                        JobUpdate(
                            job_id=self.job_id,
                            status="failed",
                            error=error_msg,
                            traceback=tb_text,
                        )
                    )
                    raise
                finally:
                    await self.finalize_state()
        finally:
            # Close the per-thread pool to release SQLite connections before
            # the threaded event loop is stopped, preventing WAL lock hangs.
            if pool is not None:
                try:
                    await pool.close_all()
                    loop_id = id(asyncio.get_running_loop())
                    SQLiteConnectionPool._pools.pop((loop_id, pool.db_path), None)
                except Exception as e:
                    log.warning(f"Error closing thread-local pool: {e}")

    @classmethod
    async def create_and_start(
        cls,
        request: RunJobRequest,
        context: ProcessingContext,
        job_id: str | None = None,
        execution_id: str | None = None,
    ) -> "ThreadedJobExecution":
        """
        Create and start a new background job.

        This factory method handles all initialization:
        - Creates job ID (if not provided) and database record
        - Sets up workflow runner
        - Creates dedicated event loop
        - Starts execution in background

        Args:
            request: Job request with workflow details
            context: Processing context for the job
            job_id: Optional existing job ID (if pre-generated)
            execution_id: Optional execution ID for tracking specific attempts

        Returns:
            ThreadedJobExecution instance with execution already started
        """
        job_id = job_id or uuid4().hex
        runner = WorkflowRunner(job_id=job_id)

        # Create persistent event loop for this job
        event_loop = ThreadedEventLoop()
        event_loop.start()

        log.info(f"Starting background job {job_id} for workflow {request.workflow_id}")

        # Capture db_path from current scope so the threaded execution can
        # create its own pool for the same database on its own event loop.
        db_path: str | None = None
        scope = maybe_scope()
        if scope and scope.db:
            pool_obj = getattr(scope.db, "pool", None)
            if pool_obj and hasattr(pool_obj, "db_path"):
                db_path = pool_obj.db_path

        # Create the job record in database
        # We need a temporary ResourceScope for the initial Job.save()
        job_model = Job(
            id=job_id,
            workflow_id=request.workflow_id,
            user_id=request.user_id,
            job_type=request.job_type,
            graph=request.graph.model_dump() if request.graph else {},
            params=request.params or {},
        )

        # Use a temporary ResourceScope for the initial database operation
        # In test mode, inherit db_path from current scope if available
        async with ResourceScope():
            await job_model.save()

        # Create the job instance
        job_instance = cls(
            job_id=job_id,
            runner=runner,
            context=context,
            request=request,
            job_model=job_model,
            event_loop=event_loop,
            execution_id=execution_id,
        )
        job_instance._db_path = db_path

        # Schedule execution on the persistent loop
        job_instance.future = event_loop.run_coroutine(job_instance.execute())
        job_instance._set_status("running")

        return job_instance
