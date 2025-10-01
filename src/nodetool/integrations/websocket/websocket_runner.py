import asyncio
from datetime import datetime
from enum import Enum
import json
from fastapi.websockets import WebSocketState
import msgpack
from typing import AsyncGenerator, Dict, Optional
from nodetool.types.wrap_primitive_types import wrap_primitive_types
from pydantic import BaseModel
from fastapi import WebSocket, WebSocketDisconnect
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.ml.core.model_manager import ModelManager
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import ExecutionStrategy, RunJobRequest
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.types import Chunk, Error
from nodetool.workflows.job_execution_manager import (
    JobExecutionManager,
    JobExecution,
)
from nodetool.models.job import Job
import gc

log = get_logger(__name__)

"""
WebSocket-based workflow execution manager for Node Tool.

This module provides WebSocket communication and workflow execution management, enabling real-time
bidirectional communication between clients and the workflow engine. It supports both binary 
(MessagePack) and text (JSON) protocols for message exchange.

Key components:
- WebSocketRunner: Main class handling WebSocket connections and workflow execution
- CommandType: Enum defining supported WebSocket commands (run_job, cancel_job, get_status, set_mode)
- Message Processing: Utilities for processing and streaming workflow execution messages
- Error Handling: Comprehensive error handling and status reporting

The module supports:
- Starting and canceling workflow jobs
- Real-time status updates and message streaming
- Dynamic switching between binary and text protocols
- Graceful connection and resource management
- Multiple concurrent jobs per WebSocket connection
"""


class CommandType(str, Enum):
    RUN_JOB = "run_job"
    RECONNECT_JOB = "reconnect_job"
    CANCEL_JOB = "cancel_job"
    GET_STATUS = "get_status"
    SET_MODE = "set_mode"
    CLEAR_MODELS = "clear_models"
    STREAM_INPUT = "stream_input"  # Push a streaming input item
    END_INPUT_STREAM = "end_input_stream"  # Close a streaming input


class WebSocketCommand(BaseModel):
    command: CommandType
    data: dict


class WebSocketMode(str, Enum):
    BINARY = "binary"
    TEXT = "text"


async def process_message(context: ProcessingContext, explicit_types: bool = False):
    """
    Helper method to process and send individual messages.
    Yields the message to the caller.

    Args:
        context (ProcessingContext): The processing context
        req (RunJobRequest): The request object for the job.
    """
    msg = await context.pop_message_async()
    if isinstance(msg, Error):
        raise Exception(msg.error)
    else:

        if isinstance(msg, dict):
            msg_dict = msg
        else:
            msg_dict = msg.model_dump()

        if explicit_types and "result" in msg_dict:
            msg_dict["result"] = wrap_primitive_types(msg_dict["result"])

        yield msg_dict


async def process_workflow_messages(
    job_execution: JobExecution,
    sleep_interval: float = 0.01,
    explicit_types: bool = False,
) -> AsyncGenerator[dict, None]:
    """
    Process messages from a running workflow.

    Args:
        context (ProcessingContext): The processing context
        runner (WorkflowRunner): The workflow runner
        message_handler: Async function to handle messages
        sleep_interval (float): Time to sleep between message checks
        explicit_types (bool): Whether to wrap primitive types in explicit types
    """
    try:
        log.debug("Starting workflow message processing")
        while job_execution.is_running():
            if job_execution.context.has_messages():
                async for msg in process_message(job_execution.context, explicit_types):
                    yield msg
            else:
                await asyncio.sleep(sleep_interval)

        # Process remaining messages
        while job_execution.context.has_messages():
            async for msg in process_message(job_execution.context, explicit_types):
                yield msg

        log.debug("Finished processing workflow messages")
    except Exception as e:
        log.exception(e)
        raise


async def execute_workflow(
    context: ProcessingContext, runner: WorkflowRunner, req: RunJobRequest
):
    """
    Execute a workflow with the given context and request.

    Args:
        context (ProcessingContext): The processing context
        runner (WorkflowRunner): The workflow runner
        req (RunJobRequest): The job request
    """
    try:
        if req.graph is None:
            log.info(f"Loading workflow graph for {req.workflow_id}")
            workflow = await context.get_workflow(req.workflow_id)
            if workflow is None:
                raise ValueError(f"Workflow {req.workflow_id} not found")
            req.graph = workflow.get_api_graph()
        assert runner, "Runner is not set"
        await runner.run(req, context)
    except Exception as e:
        log.exception(e)
        context.post_message(
            JobUpdate(job_id=runner.job_id, status="failed", error=str(e))
        )
        raise


class JobStreamContext:
    """Context for streaming a specific job's messages"""

    def __init__(self, job_id: str, workflow_id: str, job_execution: JobExecution):
        self.job_id = job_id
        self.workflow_id = workflow_id
        self.job_execution = job_execution
        self.streaming_task: asyncio.Task | None = None


class WebSocketRunner:
    """
    Runs multiple workflows using a single WebSocket connection.

    Attributes:
        websocket (WebSocket | None): The WebSocket connection.
        mode (WebSocketMode): The current mode for WebSocket communication.
        active_jobs (Dict[str, JobStreamContext]): Active job streaming contexts by job_id
    """

    websocket: WebSocket | None = None
    mode: WebSocketMode = WebSocketMode.BINARY
    active_jobs: Dict[str, JobStreamContext] = {}

    def __init__(
        self,
    ):
        """
        Initializes a new instance of the WebSocketRunner class.
        """
        self.mode = WebSocketMode.BINARY
        self.active_jobs = {}

    async def connect(self, websocket: WebSocket):
        """
        Establishes the WebSocket connection.

        Args:
            websocket (WebSocket): The WebSocket connection.
        """
        await websocket.accept()
        self.websocket = websocket
        log.info("WebSocket connection established")
        log.debug(f"Client connected: {websocket.client}")

    async def disconnect(self):
        """
        Closes the WebSocket connection.
        Note: Jobs continue running in the background via JobExecutionManager.
        """
        log.info("WebSocketRunner: Disconnecting")

        # Stop all streaming tasks
        for job_ctx in self.active_jobs.values():
            if job_ctx.streaming_task and not job_ctx.streaming_task.done():
                job_ctx.streaming_task.cancel()

        self.active_jobs.clear()

        # Only attempt to close if websocket exists and is not already closed
        if (
            self.websocket
            and not self.websocket.client_state == WebSocketState.DISCONNECTED
        ):
            try:
                await self.websocket.close()
                log.info("WebSocketRunner: WebSocket closed successfully")
            except Exception as e:
                log.error(f"WebSocketRunner: Error closing WebSocket: {e}")

        self.websocket = None
        log.info("WebSocketRunner: Disconnected (jobs continue in background)")

    async def run_job(self, req: RunJobRequest):
        """Start a new job in the background and stream messages to the client."""
        try:
            if not self.websocket:
                raise ValueError("WebSocket is not connected")

            log.debug(f"Run job request: {req.model_dump(exclude={'graph'})}")

            log.info(
                "WebSocketRunner.run_job starting",
                extra={
                    "workflow_id": req.workflow_id,
                    "user_id": req.user_id,
                    "job_type": req.job_type,
                    "has_graph": req.graph is not None,
                },
            )

            # Create processing context
            context = ProcessingContext(
                user_id=req.user_id,
                auth_token=req.auth_token,
                workflow_id=req.workflow_id,
                encode_assets_as_base64=self.mode == WebSocketMode.TEXT,
            )

            # Start job in background via JobExecutionManager
            job_manager = JobExecutionManager.get_instance()
            job_execution = await job_manager.start_job(req, context)
            job_id = job_execution.job_id

            log.info(
                "WebSocketRunner.run_job job info",
                extra={
                    "job_id": job_id,
                    "workflow_id": req.workflow_id,
                    "user_id": req.user_id,
                    "job_type": req.job_type,
                },
            )

            # Create streaming context
            job_ctx = JobStreamContext(job_id, req.workflow_id, job_execution)
            self.active_jobs[job_id] = job_ctx

            # Start streaming task
            job_ctx.streaming_task = asyncio.create_task(
                self._stream_job_messages(job_ctx, req.explicit_types or False)
            )

        except Exception as e:
            log.exception(f"Error starting job: {e}")
            await self.send_message(
                JobUpdate(
                    status="failed",
                    error=str(e),
                    job_id=req.workflow_id,  # Use workflow_id as fallback
                    workflow_id=req.workflow_id,
                ).model_dump()
            )

    async def _stream_job_messages(
        self, job_ctx: JobStreamContext, explicit_types: bool
    ):
        """Stream messages from a background job to the client."""
        try:
            # Send initial job update
            await self.send_message(
                JobUpdate(
                    status="running",
                    job_id=job_ctx.job_id,
                    workflow_id=job_ctx.workflow_id,
                ).model_dump()
            )

            # Stream messages from the job execution
            try:
                async for msg in process_workflow_messages(
                    job_execution=job_ctx.job_execution,
                    explicit_types=explicit_types,
                ):
                    # Add job_id and workflow_id to all messages
                    msg["job_id"] = job_ctx.job_id
                    msg["workflow_id"] = job_ctx.workflow_id
                    await self.send_message(msg)

                    if (
                        not self.websocket
                        or self.websocket.client_state == WebSocketState.DISCONNECTED
                    ):
                        log.warning(
                            "WebSocketRunner: websocket lost during stream",
                            extra={"job_id": job_ctx.job_id},
                        )
                        raise WebSocketDisconnect()
            except WebSocketDisconnect:
                log.warning(
                    "WebSocketRunner: websocket disconnected mid-stream",
                    extra={"job_id": job_ctx.job_id},
                )
                # Keep job running but stop streaming
                return
            except Exception as e:
                log.exception(e)
                await self.send_message(
                    JobUpdate(
                        status="failed",
                        error=str(e),
                        job_id=job_ctx.job_id,
                        workflow_id=job_ctx.workflow_id,
                    ).model_dump()
                )

            # Check final job status
            if job_ctx.job_execution.is_completed():
                final_status = job_ctx.job_execution.status
                if final_status == "cancelled":
                    await self.send_message(
                        JobUpdate(
                            status="cancelled",
                            job_id=job_ctx.job_id,
                            workflow_id=job_ctx.workflow_id,
                        ).model_dump()
                    )
                elif final_status == "error":
                    await self.send_message(
                        JobUpdate(
                            status="failed",
                            job_id=job_ctx.job_id,
                            workflow_id=job_ctx.workflow_id,
                        ).model_dump()
                    )
                else:
                    await self.send_message(
                        JobUpdate(
                            status="completed",
                            job_id=job_ctx.job_id,
                            workflow_id=job_ctx.workflow_id,
                        ).model_dump()
                    )

            log.info(f"Job streaming completed: {job_ctx.job_id}")

        finally:
            # Clean up
            self.active_jobs.pop(job_ctx.job_id, None)

    async def reconnect_job(self, job_id: str, workflow_id: Optional[str] = None):
        """Reconnect to an existing background job and stream remaining messages."""
        try:
            if not self.websocket:
                raise ValueError("WebSocket is not connected")

            log.info(f"WebSocketRunner: Reconnecting to job: {job_id}")

            # Get the job execution from the manager
            job_manager = JobExecutionManager.get_instance()
            job_execution = job_manager.get_job(job_id)

            if job_execution is None:
                db_job = await Job.get(job_id)
                if db_job and db_job.status in {"running", "starting", "queued"}:
                    log.warning(
                        "WebSocketRunner: Job missing from manager; marking as failed",
                        extra={"job_id": job_id},
                    )
                    await db_job.update(
                        status="failed",
                        error="Job worker unavailable during reconnect",
                        finished_at=datetime.now(),
                    )
                log.warning(
                    "WebSocketRunner: Job not found during reconnect",
                    extra={"job_id": job_id},
                )
                raise ValueError(f"Job {job_id} not found")

            log.info(
                "WebSocketRunner.reconnect_job obtained job execution",
                extra={
                    "job_id": job_id,
                    "workflow_id": job_execution.request.workflow_id,
                    "user_id": job_execution.request.user_id,
                    "status": job_execution.status,
                    "is_running": job_execution.is_running(),
                    "is_completed": job_execution.is_completed(),
                },
            )

            # Use workflow_id from the job execution if not provided
            if not workflow_id:
                workflow_id = job_execution.request.workflow_id

            if not job_execution.is_running():
                final_status = job_execution.status
                try:
                    # Ensure job model reflects final state if still marked running
                    await job_execution.job_model.reload()
                    if job_execution.job_model.status in {
                        "running",
                        "starting",
                        "queued",
                    }:
                        await job_execution.job_model.update(
                            status=final_status,
                            finished_at=datetime.now(),
                        )
                except Exception as persist_error:
                    log.error(
                        "WebSocketRunner: Failed to persist finalized job state",
                        extra={"job_id": job_id, "error": str(persist_error)},
                    )

                await self.send_message(
                    JobUpdate(
                        status=final_status,
                        job_id=job_id,
                        workflow_id=workflow_id,
                    ).model_dump()
                )
                log.info(
                    "WebSocketRunner: Job already completed during reconnect",
                    extra={"job_id": job_id, "status": final_status},
                )
                return

            # Create streaming context
            job_ctx = JobStreamContext(job_id, workflow_id, job_execution)
            self.active_jobs[job_id] = job_ctx

            # Send current job status
            await self.send_message(
                JobUpdate(
                    status=job_execution.status,
                    job_id=job_id,
                    workflow_id=workflow_id,
                ).model_dump()
            )

            # Only replay statuses if job is actually running
            # If it's about to complete, we don't want to replay edge statuses
            # that would be immediately cleared
            if job_execution.is_running():
                # Replay current status for all nodes and edges
                node_count = len(job_execution.context.node_statuses)
                edge_count = len(job_execution.context.edge_statuses)
                log.info(
                    f"Replaying status for {node_count} nodes and {edge_count} edges for job {job_id}"
                )

                # Replay node statuses
                for node_status in job_execution.context.node_statuses.values():
                    msg_dict = node_status.model_dump()
                    msg_dict["job_id"] = job_id
                    msg_dict["workflow_id"] = workflow_id
                    await self.send_message(msg_dict)

                # Replay edge statuses
                for edge_status in job_execution.context.edge_statuses.values():
                    msg_dict = edge_status.model_dump()
                    msg_dict["job_id"] = job_id
                    msg_dict["workflow_id"] = workflow_id
                    await self.send_message(msg_dict)
            else:
                log.info(
                    f"Job {job_id} completed during reconnect setup, skipping status replay"
                )

            # Start streaming remaining messages
            job_ctx.streaming_task = asyncio.create_task(
                self._stream_job_messages(job_ctx, False)
            )

            log.info(f"Reconnected to job {job_id}")

        except Exception as e:
            log.exception(f"Error reconnecting to job {job_id}: {e}")
            await self.send_message(
                JobUpdate(
                    status="failed",
                    error=str(e),
                    job_id=job_id,
                    workflow_id=workflow_id or "",
                ).model_dump()
            )

    async def send_message(self, message: dict):
        """Send a message using the current mode."""
        if (
            not self.websocket
            or self.websocket.client_state == WebSocketState.DISCONNECTED
        ):
            log.warning(
                "WebSocketRunner.send_message skipped because websocket is not connected",
                extra={"message_type": message.get("type")},
            )
            return

        try:
            if self.mode == WebSocketMode.BINARY:
                packed_message = msgpack.packb(message, use_bin_type=True)
                await self.websocket.send_bytes(packed_message)  # type: ignore
            else:
                await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            log.error(f"Error sending message: {e}")

    async def cancel_job(self, job_id: str, workflow_id: Optional[str] = None):
        """
        Cancels the specified job.

        Returns:
            dict: A dictionary with a message indicating the job was cancelled, or an error if not found.
        """
        if not job_id:
            log.warning("No job_id provided for cancel")
            return {"error": "No job_id provided"}

        log.info(f"Attempting to cancel job: {job_id}")
        job_manager = JobExecutionManager.get_instance()
        cancelled = await job_manager.cancel_job(job_id)

        # Stop streaming task if active
        job_ctx = self.active_jobs.get(job_id)
        if job_ctx and job_ctx.streaming_task and not job_ctx.streaming_task.done():
            job_ctx.streaming_task.cancel()
            self.active_jobs.pop(job_id, None)

        if cancelled:
            return {
                "message": "Job cancellation requested",
                "job_id": job_id,
                "workflow_id": workflow_id or "",
            }
        else:
            return {
                "error": "Job not found or already completed",
                "job_id": job_id,
                "workflow_id": workflow_id or "",
            }

    def get_status(self, job_id: str | None = None):
        """
        Gets the current status of job execution.

        Returns:
            dict: A dictionary with the status and job information.
        """
        if job_id:
            job_ctx = self.active_jobs.get(job_id)
            if job_ctx:
                return {
                    "status": job_ctx.job_execution.status,
                    "job_id": job_id,
                    "workflow_id": job_ctx.workflow_id,
                }
            return {"status": "not_found", "job_id": job_id}
        else:
            # Return status of all active jobs
            return {
                "active_jobs": [
                    {
                        "job_id": ctx.job_id,
                        "workflow_id": ctx.workflow_id,
                        "status": ctx.job_execution.status,
                    }
                    for ctx in self.active_jobs.values()
                ]
            }

    async def clear_models(self):
        """
        Clears unused models from the model manager.
        """
        if not Environment.is_production():
            ModelManager.clear()
            gc.collect()
            return {"message": "Unused models cleared"}
        return {"message": "Model clearing is disabled in production"}

    async def handle_command(self, command: WebSocketCommand):
        """
        Handles incoming WebSocket commands.

        Args:
            command (WebSocketCommand): The WebSocket command to handle.

        Returns:
            dict: A dictionary with the response to the command.
        """
        log.info(f"Handling command: {command.command}")

        # Extract common fields
        job_id = command.data.get("job_id")
        workflow_id = command.data.get("workflow_id")

        if command.command == CommandType.CLEAR_MODELS:
            return await self.clear_models()
        elif command.command == CommandType.RUN_JOB:
            req = RunJobRequest(**command.data)
            if Environment.get_execution_strategy() == ExecutionStrategy.DOCKER:
                req.execution_strategy = ExecutionStrategy.DOCKER
            else:
                req.execution_strategy = ExecutionStrategy.THREADED
            log.info(f"Starting workflow: {req.workflow_id}")
            asyncio.create_task(self.run_job(req))
            log.debug("Run job command scheduled")
            return {"message": "Job started", "workflow_id": req.workflow_id}
        elif command.command == CommandType.RECONNECT_JOB:
            if not job_id:
                return {"error": "job_id is required"}
            log.info(f"Reconnecting to job: {job_id}")
            asyncio.create_task(self.reconnect_job(job_id, workflow_id))
            return {
                "message": f"Reconnecting to job {job_id}",
                "job_id": job_id,
                "workflow_id": workflow_id,
            }
        elif command.command == CommandType.STREAM_INPUT:
            if not job_id:
                return {"error": "job_id is required"}
            job_ctx = self.active_jobs.get(job_id)
            if not job_ctx:
                return {"error": "No active job/context"}
            input_name = command.data.get("input")
            if not isinstance(input_name, str) or input_name.strip() == "":
                return {"error": "Invalid input name"}
            value = command.data.get("value")
            handle = command.data.get("handle")
            try:
                log.debug(
                    f"STREAM_INPUT received: input={input_name} handle={handle} type={type(value)}"
                )
                if value and value.get("type") == "chunk":
                    value = Chunk(
                        content=value["content"],
                        done=value["done"],
                        content_type=value["content_type"],
                    )
                job_ctx.job_execution.push_input_value(input_name=input_name, value=value, source_handle=handle)  # type: ignore[arg-type]
                log.debug("STREAM_INPUT enqueued to runner input queue")
                return {
                    "message": "Input item streamed",
                    "job_id": job_id,
                    "workflow_id": workflow_id,
                }
            except Exception as e:
                log.exception(e)
                return {"error": str(e), "job_id": job_id, "workflow_id": workflow_id}
        elif command.command == CommandType.END_INPUT_STREAM:
            if not job_id:
                return {"error": "job_id is required"}
            job_ctx = self.active_jobs.get(job_id)
            if not job_ctx:
                return {"error": "No active job/context"}
            input_name = command.data.get("input")
            if not isinstance(input_name, str) or input_name.strip() == "":
                return {"error": "Invalid input name"}
            handle = command.data.get("handle")
            try:
                log.debug(
                    f"END_INPUT_STREAM received: input={input_name} handle={handle}"
                )
                assert job_ctx.job_execution.runner, "Runner is not set"
                job_ctx.job_execution.runner.finish_input_stream(
                    input_name=input_name, source_handle=handle
                )
                log.debug("END_INPUT_STREAM enqueued to runner input queue")
                return {
                    "message": "Input stream ended",
                    "job_id": job_id,
                    "workflow_id": workflow_id,
                }
            except Exception as e:
                log.exception(e)
                return {"error": str(e), "job_id": job_id, "workflow_id": workflow_id}
        elif command.command == CommandType.CANCEL_JOB:
            if not job_id:
                return {"error": "job_id is required"}
            return await self.cancel_job(job_id, workflow_id)
        elif command.command == CommandType.GET_STATUS:
            return self.get_status(job_id)
        elif command.command == CommandType.SET_MODE:
            new_mode = WebSocketMode(command.data["mode"])
            self.mode = new_mode
            log.info(f"WebSocket mode set to: {new_mode}")
            return {"message": f"Mode set to {new_mode}"}
        else:
            log.warning(f"Unknown command received: {command.command}")
            return {"error": "Unknown command"}

    async def run(self, websocket: WebSocket):
        """
        Main method to run the WorkflowRunner.

        Args:
            websocket (WebSocket): The WebSocket connection.
        """
        try:
            await self.connect(websocket)
            log.debug("WebSocketRunner loop started")
            while True:
                assert self.websocket, "WebSocket is not connected"
                try:
                    message = await self.websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        log.info("Received websocket disconnect message")
                        break
                    if "bytes" in message:
                        data = msgpack.unpackb(message["bytes"])  # type: ignore
                        log.debug("Received binary message")
                    elif "text" in message:
                        data = json.loads(message["text"])
                        log.debug("Received text message")
                    else:
                        log.warning("Received message with unknown format")
                        continue

                    command = WebSocketCommand(**data)
                    response = await self.handle_command(command)
                    await self.send_message(response)
                    log.debug(f"Handled command {command.command}")
                except WebSocketDisconnect:
                    log.info("WebSocket disconnected")
                    break
        except Exception as e:
            log.error(f"WebSocket error: {str(e)}")
            log.exception(e)
        finally:
            await self.disconnect()
            log.debug("WebSocketRunner loop finished")
