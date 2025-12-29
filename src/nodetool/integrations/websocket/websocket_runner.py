import asyncio
import gc
import json
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Optional

import msgpack
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import BaseModel

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.ml.core.model_manager import ModelManager
from nodetool.models.job import Job
from nodetool.runtime.resources import ResourceScope, get_user_auth_provider
from nodetool.types.job import JobUpdate
from nodetool.types.wrap_primitive_types import wrap_primitive_types
from nodetool.workflows.job_execution_manager import (
    JobExecution,
    JobExecutionManager,
)
from nodetool.workflows.processing_context import (
    AssetOutputMode,
    ProcessingContext,
)
from nodetool.workflows.run_job_request import ExecutionStrategy, RunJobRequest
from nodetool.workflows.types import Chunk, Error

log = get_logger(__name__)


def extract_binary_data_from_value(value: Any, binaries: list[bytes]) -> Any:
    """
    Recursively extract binary data from AssetRef objects.
    
    Replaces binary data in AssetRef.data fields with an index reference,
    and appends the binary data to the binaries list.
    
    Args:
        value: The value to process (can be dict, list, or AssetRef)
        binaries: List to collect binary data
        
    Returns:
        Modified value with binary data replaced by indices
    """
    if isinstance(value, dict):
        # Check if this looks like an AssetRef
        if "type" in value and value.get("type") in ["image", "audio", "video", "text", "asset"]:
            # Check if it has binary data
            if "data" in value and value["data"] is not None:
                data = value["data"]
                # Handle bytes data
                if isinstance(data, bytes):
                    binaries.append(data)
                    # Replace data with index reference
                    value = value.copy()
                    value["data"] = None
                    value["binary_index"] = len(binaries) - 1
                    return value
        
        # Recursively process dict values
        return {k: extract_binary_data_from_value(v, binaries) for k, v in value.items()}
    elif isinstance(value, list):
        # Recursively process list items
        return [extract_binary_data_from_value(item, binaries) for item in value]
    elif isinstance(value, tuple):
        # Recursively process tuple items
        return tuple(extract_binary_data_from_value(item, binaries) for item in value)
    else:
        return value

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


async def process_message(
    context: ProcessingContext, explicit_types: bool = False
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Helper method to process and send individual messages.
    Yields the message to the caller.

    Args:
        context (ProcessingContext): The processing context
        explicit_types (bool): Whether primitive results should be wrapped
    """
    msg = await context.pop_message_async()
    if isinstance(msg, Error):
        raise RuntimeError(msg.error)
    msg_dict: dict[str, Any] = msg if isinstance(msg, dict) else msg.model_dump()

    if explicit_types and "result" in msg_dict:
        msg_dict["result"] = wrap_primitive_types(msg_dict["result"])

    yield msg_dict


async def process_workflow_messages(
    job_execution: JobExecution,
    sleep_interval: float = 0.01,
    explicit_types: bool = False,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Process messages from a running workflow.

    Args:
        job_execution (JobExecution): Active job execution to stream from
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


class JobStreamContext:
    """Context for streaming a specific job's messages"""

    def __init__(self, job_id: str, workflow_id: str, job_execution: JobExecution):
        self.job_id = job_id
        self.workflow_id = workflow_id
        self.job_execution = job_execution
        self.streaming_task: asyncio.Task[None] | None = None


class WebSocketRunner:
    """
    Runs multiple workflows using a single WebSocket connection.

    Attributes:
        websocket (WebSocket | None): The WebSocket connection.
        mode (WebSocketMode): The current mode for WebSocket communication.
        active_jobs (Dict[str, JobStreamContext]): Active job streaming contexts by job_id
    """

    def __init__(self, auth_token: str | None = None, user_id: str | None = None):
        """
        Initializes a new instance of the WebSocketRunner class.
        """
        self.websocket: WebSocket | None = None
        self.mode = WebSocketMode.BINARY
        self.active_jobs: dict[str, JobStreamContext] = {}
        self._run_job_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self.auth_token = auth_token or ""
        self.user_id = user_id or ""

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str | None = None,
        auth_token: str | None = None,
    ):
        """
        Establishes the WebSocket connection.

        Args:
            websocket (WebSocket): The WebSocket connection.
        """
        if auth_token is not None:
            self.auth_token = auth_token
        if user_id is not None:
            self.user_id = user_id

        if Environment.enforce_auth():
            if not self.user_id:
                token = self.auth_token
                if not token:
                    await websocket.close(code=1008, reason="Missing authentication")
                    log.warning("WebSocketRunner connection rejected: Missing authentication token")
                    return
                user_provider = get_user_auth_provider()
                if not user_provider:
                    await websocket.close(code=1008, reason="Authentication provider not configured")
                    log.warning("WebSocketRunner connection rejected: Remote auth not configured")
                    return
                result = await user_provider.verify_token(token)
                if not result.ok or not result.user_id:
                    await websocket.close(code=1008, reason="Invalid authentication")
                    log.warning("WebSocketRunner connection rejected: Invalid authentication")
                    return
                self.user_id = result.user_id
        else:
            self.user_id = self.user_id or "1"

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
        if self.websocket and self.websocket.client_state != WebSocketState.DISCONNECTED:
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
            log.info(f"Input params: {req.params}")

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
            asset_mode = AssetOutputMode.DATA_URI if self.mode == WebSocketMode.TEXT else AssetOutputMode.RAW
            context = ProcessingContext(
                user_id=req.user_id,
                auth_token=req.auth_token,
                workflow_id=req.workflow_id,
                encode_assets_as_base64=self.mode == WebSocketMode.TEXT,
                asset_output_mode=asset_mode,
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

    async def _stream_job_messages(self, job_ctx: JobStreamContext, explicit_types: bool):
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

            # Track if we received a terminal job_update during streaming
            received_terminal_update = False

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

                    # Track if we received a terminal status update
                    if msg.get("type") == "job_update":
                        status = msg.get("status")
                        if status in ("completed", "failed", "cancelled", "error"):
                            received_terminal_update = True

                    if not self.websocket or self.websocket.client_state == WebSocketState.DISCONNECTED:
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
                received_terminal_update = True

            # Only send terminal status if we didn't already receive one during streaming
            # This fallback is for older execution modes that don't send proper JobUpdates
            if not received_terminal_update:
                # Wait briefly to allow any final messages to be posted by the job thread
                # This handles the race condition where the job sets status before posting
                # the terminal JobUpdate message
                await asyncio.sleep(0.1)

                # Process any remaining messages that arrived during the wait
                while job_ctx.job_execution.context.has_messages():
                    async for msg in process_message(job_ctx.job_execution.context, explicit_types):
                        msg["job_id"] = job_ctx.job_id
                        msg["workflow_id"] = job_ctx.workflow_id
                        await self.send_message(msg)
                        if msg.get("type") == "job_update":
                            status = msg.get("status")
                            if status in ("completed", "failed", "cancelled", "error"):
                                received_terminal_update = True

            # If still no terminal update, check job status directly (not just is_completed)
            if not received_terminal_update:
                final_status = job_ctx.job_execution.status
                # Check if job has a terminal status even if future isn't marked done yet
                if (
                    final_status in ("completed", "cancelled", "error", "failed")
                    or job_ctx.job_execution.is_completed()
                ):
                    log.info(f"Sending fallback terminal status for job {job_ctx.job_id}: {final_status}")
                    if final_status == "cancelled":
                        await self.send_message(
                            JobUpdate(
                                status="cancelled",
                                job_id=job_ctx.job_id,
                                workflow_id=job_ctx.workflow_id,
                            ).model_dump()
                        )
                    elif final_status in ("error", "failed"):
                        # Try to include detailed error information from the job
                        err = (
                            getattr(job_ctx.job_execution, "error", None)
                            or getattr(job_ctx.job_execution.job_model, "error", None)
                            or "Unknown error"
                        )
                        await self.send_message(
                            JobUpdate(
                                status="failed",  # Normalize error to failed for terminal update
                                job_id=job_ctx.job_id,
                                workflow_id=job_ctx.workflow_id,
                                error=str(err),
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
                async with ResourceScope():
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

                # Include error details when reporting a finalized failed job
                err_detail = (
                    getattr(job_execution, "error", None) or getattr(job_execution.job_model, "error", None) or None
                )
                await self.send_message(
                    JobUpdate(
                        status=final_status,
                        job_id=job_id,
                        workflow_id=workflow_id,
                        error=str(err_detail) if err_detail else None,
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
                log.info(f"Replaying status for {node_count} nodes and {edge_count} edges for job {job_id}")

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
                log.info(f"Job {job_id} completed during reconnect setup, skipping status replay")

            # Start streaming remaining messages
            job_ctx.streaming_task = asyncio.create_task(self._stream_job_messages(job_ctx, False))

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
        if not self.websocket or self.websocket.client_state == WebSocketState.DISCONNECTED:
            log.warning(
                "WebSocketRunner.send_message skipped because websocket is not connected",
                extra={"message_type": message.get("type")},
            )
            return

        try:
            if self.mode == WebSocketMode.BINARY:
                # Extract binary data from AssetRefs in the message
                binaries: list[bytes] = []
                processed_message = extract_binary_data_from_value(message, binaries)
                
                # If we have binary data, send as array [message, binary1, binary2, ...]
                if binaries:
                    payload = [processed_message] + binaries
                    packed_message = msgpack.packb(payload, use_bin_type=True)
                else:
                    # No binary data, send just the message
                    packed_message = msgpack.packb(processed_message, use_bin_type=True)
                    
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
            if self.user_id:
                command.data.setdefault("user_id", self.user_id)
            if self.auth_token:
                command.data.setdefault("auth_token", self.auth_token)
            req = RunJobRequest(**command.data)
            strategy = Environment.get_default_execution_strategy()
            try:
                if isinstance(strategy, ExecutionStrategy):
                    req.execution_strategy = strategy
                else:
                    req.execution_strategy = ExecutionStrategy(strategy)
            except ValueError:
                log.error(
                    "Invalid execution strategy from environment: %r. Falling back to THREADED",
                    strategy,
                )
                req.execution_strategy = ExecutionStrategy.THREADED
            log.info(f"Starting workflow: {req.workflow_id} with strategy: {req.execution_strategy}")
            self._run_job_task = asyncio.create_task(self.run_job(req))
            log.debug("Run job command scheduled")
            return {"message": "Job started", "workflow_id": req.workflow_id}
        elif command.command == CommandType.RECONNECT_JOB:
            if not job_id:
                return {"error": "job_id is required"}
            log.info(f"Reconnecting to job: {job_id}")
            self._reconnect_task = asyncio.create_task(self.reconnect_job(job_id, workflow_id))
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
                log.debug(f"STREAM_INPUT received: input={input_name} handle={handle} type={type(value)}")
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
                log.debug(f"END_INPUT_STREAM received: input={input_name} handle={handle}")
                assert job_ctx.job_execution.runner, "Runner is not set"
                job_ctx.job_execution.runner.finish_input_stream(input_name=input_name, source_handle=handle)
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
            await self.connect(
                websocket,
                user_id=self.user_id or None,
                auth_token=self.auth_token or None,
            )
            log.debug("WebSocketRunner loop started")
            while True:
                assert self.websocket, "WebSocket is not connected"
                try:
                    message = await self.websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        log.info("Received websocket disconnect message")
                        break
                    try:
                        if "bytes" in message:
                            data = msgpack.unpackb(message["bytes"])  # type: ignore
                            log.debug("Received binary message")
                        elif "text" in message:
                            data = json.loads(message["text"])
                            log.debug("Received text message")
                        else:
                            log.warning("Received message with unknown format")
                            continue
                    except Exception as decode_error:
                        log.warning("Failed to decode client message: %s", decode_error)
                        await self.send_message({"error": "invalid_message"})
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
