"""
Subprocess job execution strategy for isolated workflow execution.
"""

import asyncio
import json
import sys
from asyncio import subprocess as aio_subprocess
from contextlib import suppress
from datetime import datetime
from typing import Any
from uuid import uuid4

from nodetool.config.logging_config import get_logger
from nodetool.models.job import Job
from nodetool.types.job import JobUpdate
from nodetool.types.prediction import Prediction
from nodetool.workflows.job_execution import JobExecution
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import (
    Chunk,
    EdgeUpdate,
    Error as WorkflowError,
    LogUpdate,
    NodeProgress,
    NodeUpdate,
    Notification,
    OutputUpdate,
    PlanningUpdate,
    PreviewUpdate,
    SubTaskResult,
    TaskUpdate,
    ToolCallUpdate,
    ToolResultUpdate,
)

log = get_logger(__name__)


MESSAGE_TYPE_MAP: dict[str, Any] = {
    "job_update": JobUpdate,
    "node_update": NodeUpdate,
    "edge_update": EdgeUpdate,
    "node_progress": NodeProgress,
    "chunk": Chunk,
    "notification": Notification,
    "log_update": LogUpdate,
    "task_update": TaskUpdate,
    "tool_call_update": ToolCallUpdate,
    "tool_result_update": ToolResultUpdate,
    "planning_update": PlanningUpdate,
    "output_update": OutputUpdate,
    "preview_update": PreviewUpdate,
    "subtask_result": SubTaskResult,
    "prediction": Prediction,
    "error": WorkflowError,
}


def _deserialize_processing_message(payload: dict[str, Any]) -> Any:
    """Deserialize a processing message from a JSON payload."""
    msg_type: str | None = payload.get("type")
    if msg_type is None:
        return None
    cls = MESSAGE_TYPE_MAP.get(msg_type, None)
    if cls is None:
        return None
    try:
        if hasattr(cls, "model_validate"):
            return cls.model_validate(payload)  # type: ignore[attr-defined]
        return cls(**payload)
    except Exception:
        log.exception(
            "Failed to deserialize message from subprocess", extra={"payload": payload}
        )
        return None


class SubprocessJobExecution(JobExecution):
    """
    Job execution using a subprocess.

    This execution strategy runs workflows in a separate subprocess,
    providing better isolation and resource management.

    Additional Attributes:
        process: subprocess.Process instance
        _stdout_task: asyncio.Task streaming stdout JSONL
        _stderr_task: asyncio.Task streaming stderr logs
        _completed_event: asyncio.Event signalling process exit
    """

    def __init__(
        self,
        job_id: str,
        context: ProcessingContext,
        request: RunJobRequest,
        job_model: Job,
        process: asyncio.subprocess.Process,
    ):
        super().__init__(job_id, context, request, job_model, runner=None)
        self.process = process
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._completed_event = asyncio.Event()
        self._status = "running"

    def push_input_value(self, input_name: str, value: Any, source_handle: str) -> None:
        """Push an input value to the job execution."""
        raise NotImplementedError(
            "SubprocessJobExecution does not support push_input_value"
        )

    def is_running(self) -> bool:
        """Check if the subprocess is still running."""
        return self.process.returncode is None

    def is_completed(self) -> bool:
        """Check if the subprocess has completed."""
        return self.process.returncode is not None

    def cancel(self) -> bool:
        """Cancel the running subprocess."""
        if self.is_completed():
            return False

        with suppress(ProcessLookupError):
            self.process.terminate()
        self._status = "cancelled"
        return True

    def cleanup_resources(self) -> None:
        """Clean up subprocess resources."""
        # Ensure subprocess terminated
        if self.process.returncode is None:
            with suppress(ProcessLookupError):
                self.process.kill()

        # Cancel background tasks
        for task in (self._stdout_task, self._stderr_task):
            if task is not None:
                task.cancel()

    async def _stream_stdout(self):
        """Stream and parse JSONL messages from subprocess stdout."""
        assert self.process.stdout is not None

        try:
            while True:
                line_bytes = await self.process.stdout.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8").strip()
                if not line or not line.startswith("{"):
                    continue

                try:
                    payload = json.loads(line)
                    msg = _deserialize_processing_message(payload)
                    if msg is not None:
                        # Update internal status from job_update messages
                        if isinstance(msg, JobUpdate):
                            self._status = msg.status
                        # Forward message to context
                        self.context.post_message(msg)
                except json.JSONDecodeError:
                    log.warning(
                        f"Failed to parse JSON from subprocess stdout: {line[:100]}"
                    )
                except Exception:
                    log.exception("Error processing subprocess message")

        except asyncio.CancelledError:
            log.debug(f"Subprocess stdout streaming cancelled for job {self.job_id}")
        except Exception:
            log.exception(f"Error streaming stdout for job {self.job_id}")

    async def _stream_stderr(self):
        """Stream subprocess stderr for logging."""
        assert self.process.stderr is not None

        try:
            while True:
                line_bytes = await self.process.stderr.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8").rstrip()
                if line:
                    log.debug(f"Subprocess stderr [{self.job_id}]: {line}")

        except asyncio.CancelledError:
            log.debug(f"Subprocess stderr streaming cancelled for job {self.job_id}")
        except Exception:
            log.exception(f"Error streaming stderr for job {self.job_id}")

    async def _monitor_completion(self):
        """Monitor subprocess completion and update job status."""
        try:
            returncode = await self.process.wait()

            # Update status based on return code
            if returncode == 0:
                self._status = "completed"
                await self.job_model.update(
                    status="completed", finished_at=datetime.now()
                )
            else:
                self._status = "failed"
                error_msg = f"Subprocess exited with code {returncode}"
                await self.job_model.update(
                    status="failed", error=error_msg, finished_at=datetime.now()
                )
                self.context.post_message(
                    JobUpdate(job_id=self.job_id, status="failed", error=error_msg)
                )

            self._completed_event.set()
            log.info(f"Subprocess job {self.job_id} completed with code {returncode}")

        except asyncio.CancelledError:
            self._status = "cancelled"
            await self.job_model.update(status="cancelled", finished_at=datetime.now())
            self._completed_event.set()
            log.info(f"Subprocess job {self.job_id} was cancelled")
        except Exception as e:
            self._status = "failed"
            error_msg = f"Error monitoring subprocess: {str(e)}"
            await self.job_model.update(
                status="failed", error=error_msg, finished_at=datetime.now()
            )
            self._completed_event.set()
            log.error(f"Error monitoring subprocess job {self.job_id}: {e}")

    @classmethod
    async def create_and_start(
        cls, request: RunJobRequest, context: ProcessingContext
    ) -> "SubprocessJobExecution":
        """
        Create and start a new subprocess-based job.

        This factory method:
        - Creates job ID and database record
        - Spawns subprocess running run_workflow_cli
        - Writes request JSON to subprocess stdin
        - Starts stdout/stderr streaming
        - Returns SubprocessJobExecution instance

        Args:
            request: Job request with workflow details
            context: Processing context for the job

        Returns:
            SubprocessJobExecution instance with execution started
        """
        job_id = uuid4().hex

        log.info(f"Starting subprocess job {job_id} for workflow {request.workflow_id}")

        # Create job record in database
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

        # Prepare request JSON
        request_dict = request.model_dump()
        # Convert Graph to dict if needed
        if request.graph:
            request_dict["graph"] = request.graph.model_dump()

        request_json = json.dumps(request_dict)

        # Spawn subprocess using 'nodetool run --stdin --jsonl' CLI command
        # This will read the request JSON from stdin and output JSONL
        cmd = ["nodetool", "run", "--stdin", "--jsonl"]

        process = await aio_subprocess.create_subprocess_exec(
            *cmd,
            stdout=aio_subprocess.PIPE,
            stderr=aio_subprocess.PIPE,
            stdin=aio_subprocess.PIPE,
        )

        # Write request JSON to stdin and close it
        if process.stdin:
            process.stdin.write(request_json.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

        # Create the job instance
        job_instance = cls(
            job_id=job_id,
            context=context,
            request=request,
            job_model=job_model,
            process=process,
        )

        # Start streaming tasks
        job_instance._stdout_task = asyncio.create_task(job_instance._stream_stdout())
        job_instance._stderr_task = asyncio.create_task(job_instance._stream_stderr())

        # Start monitoring completion
        asyncio.create_task(job_instance._monitor_completion())

        log.info(f"Subprocess job {job_id} started with PID {process.pid}")

        return job_instance
