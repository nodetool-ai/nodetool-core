"""
Base abstract class for job execution strategies.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from nodetool.config.logging_config import get_logger
from nodetool.models.job import Job
from nodetool.workflows.job_log_handler import JobLogHandler
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.workflow_runner import WorkflowRunner

if TYPE_CHECKING:
    import asyncio

log = get_logger(__name__)


class JobExecution(ABC):
    """
    Abstract base class representing an executing job.

    This class defines the interface and common behavior for different
    job execution strategies (threaded, subprocess, docker, etc.).

    Attributes:
        job_id: Unique identifier for the job
        context: ProcessingContext for the job
        request: Original job request
        job_model: Job model instance for database updates
        created_at: When the job was created
        _status: Internal mutable status field
        runner: Optional WorkflowRunner instance (only for threaded execution)
    """

    def __init__(
        self,
        job_id: str,
        context: ProcessingContext,
        request: RunJobRequest,
        job_model: Job,
        runner: WorkflowRunner | None = None,
    ):
        self.job_id = job_id
        self.context = context
        self.request = request
        self.job_model = job_model
        self.runner = runner
        self.created_at = datetime.now()
        self._status: str = "starting"
        self._result: dict[str, Any] | None = None
        self._error: str | None = None
        self._log_handler: JobLogHandler | None = None
        self._finalize_task: asyncio.Task | None = None

        # Install log handler to capture logs
        self._install_log_handler()

    @property
    def status(self) -> str:
        """Get the current status of the job."""
        return self._status

    @property
    def result(self) -> dict[str, Any] | None:
        """Get the job result if completed."""
        return self._result

    @property
    def error(self) -> str | None:
        """Get the job error message if failed."""
        return self._error

    @property
    def age_seconds(self) -> float:
        """Get the age of the job in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the job is still running."""
        pass

    @abstractmethod
    def is_completed(self) -> bool:
        """Check if the job has completed (success, error, or cancelled)."""
        pass

    def is_finished(self) -> bool:
        """Check if the job has finished (completed, error, or cancelled)."""
        return self._status in ("completed", "error", "cancelled")

    @abstractmethod
    def cancel(self) -> bool:
        """Cancel the running job. Returns True if cancelled, False otherwise."""
        pass

    @abstractmethod
    def cleanup_resources(self) -> None:
        """Clean up resources associated with this job."""
        pass

    @abstractmethod
    def push_input_value(self, input_name: str, value: Any, source_handle: str) -> None:
        """Push an input value to the job execution."""
        pass

    def _install_log_handler(self) -> None:
        """Install log handler to capture logs for this job."""
        try:
            self._log_handler = JobLogHandler.install_handler(
                job_id=self.job_id,
                max_logs=1000,
            )
            log.info(f"Installed log handler for job {self.job_id}")
        except Exception as e:
            log.error(f"Failed to install log handler for job {self.job_id}: {e}")

    def _uninstall_log_handler(self) -> list[dict]:
        """Uninstall and persist logs for this job."""
        try:
            if self._log_handler:
                # Get captured logs
                captured_logs = self._log_handler.get_logs()

                # Uninstall the handler
                JobLogHandler.uninstall_handler(self.job_id)
                self._log_handler = None

                log.info(f"Uninstalled log handler for job {self.job_id}, captured {len(captured_logs)} logs")

                # Return logs for persistence
                return captured_logs
        except Exception as e:
            log.error(f"Failed to uninstall log handler for job {self.job_id}: {e}")
        return []

    def get_live_logs(self, limit: int | None = None) -> list[dict]:
        """
        Get logs from the active log handler (for running jobs).

        Args:
            limit: Optional limit on number of logs to return

        Returns:
            List of log entries
        """
        if self._log_handler:
            return self._log_handler.get_logs(limit=limit)
        return []

    async def finalize_state(self) -> None:
        """
        Ensure finished jobs have their status written to the database.

        This method updates the database if the job is in a transient state
        or missing a finished_at timestamp.
        """
        try:
            # Capture and persist logs before finalizing
            captured_logs = self._uninstall_log_handler()

            # Reload to ensure we operate on latest values
            await self.job_model.reload()
            update_kwargs = {}

            if self.job_model.status in {"running", "starting", "queued"}:
                update_kwargs["status"] = self._status
            if self.job_model.finished_at is None:
                update_kwargs["finished_at"] = datetime.now()

            # Add logs to the update
            if captured_logs:
                update_kwargs["logs"] = captured_logs

            if update_kwargs:
                await self.job_model.update(**update_kwargs)

        except Exception as e:
            log.exception(
                "JobExecution.finalize_state: failed to persist status",
                extra={"job_id": self.job_id, "error": str(e)},
            )
