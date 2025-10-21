"""
Threaded job execution strategy using dedicated event loops.
"""

import asyncio
from concurrent.futures import Future
from datetime import datetime
from typing import Any
from uuid import uuid4

from nodetool.config.logging_config import get_logger
from nodetool.models.job import Job
from nodetool.types.job import JobUpdate
from nodetool.workflows.job_execution import JobExecution
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.threaded_event_loop import ThreadedEventLoop
from nodetool.workflows.workflow_runner import WorkflowRunner

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
    ):
        super().__init__(job_id, context, request, job_model, runner=runner)
        self.event_loop = event_loop
        self.future: Future | None = None

    def push_input_value(self, input_name: str, value: Any, source_handle: str) -> None:
        """Push an input value to the job execution."""
        assert self.runner, "Runner is not set"
        self.runner.push_input_value(
            input_name=input_name, value=value, source_handle=source_handle
        )

    def is_running(self) -> bool:
        """Check if the job is still running."""
        if self.runner:
            return self.runner.is_running()
        return not self.is_completed()

    def is_completed(self) -> bool:
        """Check if the job has completed (success, error, or cancelled)."""
        return self.future is not None and self.future.done()

    def cancel(self) -> bool:
        """Cancel the running job."""
        if not self.is_completed():
            if self.future:
                self.future.cancel()
            if self.runner:
                self.runner.status = "cancelled"
            self._status = "cancelled"
            return True
        return False

    def cleanup_resources(self) -> None:
        """Clean up resources associated with this job (stop event loop)."""
        if self.event_loop and self.event_loop.is_running:
            self.event_loop.stop()
            log.debug(f"Stopped event loop for job {self.job_id}")

    async def execute(self) -> None:
        """Execute the workflow.

        Handles loading workflow graph, running the workflow, and updating
        job status throughout the execution lifecycle.
        """
        try:
            # Load workflow graph if not already loaded
            if self.request.graph is None:
                log.info(f"Loading workflow graph for {self.request.workflow_id}")
                workflow = await self.context.get_workflow(self.request.workflow_id)
                if workflow is None:
                    raise ValueError(f"Workflow {self.request.workflow_id} not found")
                self.request.graph = workflow.get_api_graph()

            self._status = "running"
            await self.runner.run(self.request, self.context)

            # Update job status on completion
            self._status = "completed"
            await self.job_model.update(status="completed", finished_at=datetime.now())
            log.info(f"Background job {self.job_id} completed successfully")

        except asyncio.CancelledError:
            self.runner.status = "cancelled"
            self._status = "cancelled"
            await self.job_model.update(status="cancelled", finished_at=datetime.now())
            log.info(f"Background job {self.job_id} cancelled")
            raise
        except Exception as e:
            self.runner.status = "error"
            self._status = "error"
            error_text = str(e).strip()
            error_msg = (
                f"{e.__class__.__name__}: {error_text}"
                if error_text
                else repr(e)
            )
            await self.job_model.update(
                status="failed", error=error_msg, finished_at=datetime.now()
            )
            log.exception("Background job %s failed: %s", self.job_id, error_msg)
            self.context.post_message(
                JobUpdate(job_id=self.job_id, status="failed", error=error_msg)
            )
            raise

    @classmethod
    async def create_and_start(
        cls, request: RunJobRequest, context: ProcessingContext
    ) -> "ThreadedJobExecution":
        """
        Create and start a new background job.

        This factory method handles all initialization:
        - Creates job ID and database record
        - Sets up workflow runner
        - Creates dedicated event loop
        - Starts execution in background

        Args:
            request: Job request with workflow details
            context: Processing context for the job

        Returns:
            ThreadedJobExecution instance with execution already started
        """
        job_id = uuid4().hex
        runner = WorkflowRunner(job_id=job_id)

        # Create persistent event loop for this job
        event_loop = ThreadedEventLoop()
        event_loop.start()

        log.info(f"Starting background job {job_id} for workflow {request.workflow_id}")

        # Create the job record in database
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

        # Create the job instance
        job_instance = cls(
            job_id=job_id,
            runner=runner,
            context=context,
            request=request,
            job_model=job_model,
            event_loop=event_loop,
        )

        # Schedule execution on the persistent loop
        job_instance.future = event_loop.run_coroutine(job_instance.execute())
        job_instance._status = "running"

        return job_instance
