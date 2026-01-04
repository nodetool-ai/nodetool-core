"""
Unified WebSocket runner for handling both workflow execution and chat communications.

This module provides a single WebSocket endpoint that handles both workflow execution
(previously /ws/predict) and chat communications (previously /ws/chat), enabling:
- Bidirectional workflow and chat updates through a single channel
- Workflow updates during chat conversations
- Chat updates during workflow execution
- More efficient resource usage with a single connection

The runner routes messages based on their command/type field and supports all
functionality from previous legacy runners.

Message Routing:
- Workflow commands: run_job, cancel_job, get_status, set_mode, clear_models, etc.
- Chat messages: Messages with type="message" or type="chat" are routed to chat processing
- Control messages: stop, ping, client_tools_manifest, tool_result

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    UnifiedWebSocketRunner                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────────┐  │
    │  │   Incoming   │───>│  Message Router  │───>│  Workflow Handler     │  │
    │  │   Message    │    │                  │    │  (run_job, cancel,    │  │
    │  └──────────────┘    └────────┬─────────┘    │   reconnect, etc.)    │  │
    │                               │              └───────────────────────┘  │
    │                               │                                          │
    │                               ▼              ┌───────────────────────┐  │
    │                        [Route by type]  ───>│  Chat Handler         │  │
    │                                              │  (via BaseChatRunner) │  │
    │                                              └───────────────────────┘  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import gc
import json
import time
from contextlib import suppress
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Optional

import msgpack
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import BaseModel

from nodetool.chat.base_chat_runner import BaseChatRunner
from nodetool.chat.token_counter import count_json_tokens
from nodetool.config.env_guard import RUNNING_PYTEST
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.ml.core.model_manager import ModelManager
from nodetool.models.job import Job
from nodetool.runtime.resources import ResourceScope, get_user_auth_provider
from nodetool.types.job import JobUpdate, RunStateInfo
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


class CommandType(str, Enum):
    """
    Supported WebSocket commands for the unified runner.

    All messages must be wrapped in a command structure with a valid reference
    (job_id for workflow operations, thread_id for chat operations).

    Workflow Commands:
        - RUN_JOB: Start a new workflow job
        - RECONNECT_JOB: Reconnect to an existing job
        - CANCEL_JOB: Cancel a running job
        - GET_STATUS: Get status of active jobs
        - SET_MODE: Switch between binary and text modes
        - CLEAR_MODELS: Clear ML models from memory
        - STREAM_INPUT: Push streaming input to a job
        - END_INPUT_STREAM: Close a streaming input

    Chat Commands:
        - CHAT_MESSAGE: Send a chat message for processing (requires thread_id)

    Control Commands:
        - STOP: Stop current operation (requires job_id or thread_id)
    """

    # Workflow commands
    RUN_JOB = "run_job"
    RECONNECT_JOB = "reconnect_job"
    RESUME_JOB = "resume_job"
    CANCEL_JOB = "cancel_job"
    GET_STATUS = "get_status"
    SET_MODE = "set_mode"
    CLEAR_MODELS = "clear_models"
    STREAM_INPUT = "stream_input"
    END_INPUT_STREAM = "end_input_stream"

    # Chat command - explicit command for chat messages
    CHAT_MESSAGE = "chat_message"

    # Control commands
    STOP = "stop"


class WebSocketCommand(BaseModel):
    """WebSocket command structure for explicit command messages."""

    command: CommandType
    data: dict


class WebSocketMode(str, Enum):
    """WebSocket protocol mode."""

    BINARY = "binary"
    TEXT = "text"


async def build_run_state_info(job_id: str) -> RunStateInfo | None:
    """Build RunStateInfo from persisted RunState for WebSocket messages."""
    try:
        async with ResourceScope():
            from nodetool.models.run_state import RunState

            run_state = await RunState.get(job_id)
            if run_state:
                log.info(f"build_run_state_info: Found RunState for job {job_id}, status={run_state.status}")
                return RunStateInfo(
                    status=run_state.status,
                    suspended_node_id=run_state.suspended_node_id,
                    suspension_reason=run_state.suspension_reason,
                    error_message=run_state.error_message,
                    execution_strategy=run_state.execution_strategy,
                    is_resumable=run_state.is_resumable(),
                )
            else:
                log.warning(f"build_run_state_info: No RunState found for job {job_id}")
    except Exception as e:
        log.error(f"build_run_state_info: Failed to build run state info for job {job_id}: {e}")
    return None


class ToolBridge:
    """
    Manages waiting for frontend tool results from WebSocket messages.
    """

    def __init__(self):
        self._futures: dict[str, asyncio.Future] = {}

    def create_waiter(self, tool_call_id: str) -> asyncio.Future:
        """Create a future that will be resolved when tool result arrives."""
        fut = asyncio.get_running_loop().create_future()
        self._futures[tool_call_id] = fut
        return fut

    def resolve_result(self, tool_call_id: str, payload: dict):
        """Resolve the waiting future with the tool result payload."""
        fut = self._futures.pop(tool_call_id, None)
        if fut and not fut.done():
            fut.set_result(payload)

    def cancel_all(self):
        """Cancel all pending tool result waiters."""
        for fut in self._futures.values():
            if not fut.done():
                fut.cancel()
        self._futures.clear()


class JobStreamContext:
    """Context for streaming a specific job's messages."""

    def __init__(self, job_id: str, workflow_id: str, job_execution: JobExecution):
        self.job_id = job_id
        self.workflow_id = workflow_id
        self.job_execution = job_execution
        self.streaming_task: asyncio.Task[None] | None = None


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


class UnifiedWebSocketRunner(BaseChatRunner):
    """
    Unified WebSocket runner that handles both workflow execution and chat communications.

    This runner combines the functionality of previous legacy runners,
    providing a single WebSocket endpoint for all real-time communications.

    Features:
    - Workflow job execution and management
    - Chat message processing with AI providers
    - Binary (MessagePack) and text (JSON) protocol support
    - Heartbeat for connection keep-alive
    - Tool bridge for frontend-executed tools
    - Concurrent job management

    Attributes:
        websocket: The active WebSocket connection
        mode: Current protocol mode (binary or text)
        active_jobs: Dictionary of active job streaming contexts
        tool_bridge: Bridge for handling frontend tool execution
        client_tools_manifest: Manifest of available client-side tools
        heartbeat_task: Background task for connection keep-alive
    """

    def __init__(
        self,
        auth_token: str | None = None,
        user_id: str | None = None,
        default_model: str = "gpt-oss:20b",
        default_provider: str = "ollama",
    ):
        """
        Initialize the UnifiedWebSocketRunner.

        Args:
            auth_token: Authentication token for the connection
            user_id: User ID (may be set from auth or provided directly)
            default_model: Default model for chat completions
            default_provider: Default AI provider for chat
        """
        super().__init__(auth_token, default_model, default_provider)
        self.websocket: WebSocket | None = None
        self.mode = WebSocketMode.BINARY

        # Workflow job management
        self.active_jobs: dict[str, JobStreamContext] = {}
        self._run_job_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None

        # Chat-related
        self.heartbeat_task: asyncio.Task | None = None
        self.tool_bridge = ToolBridge()
        self.client_tools_manifest: dict[str, dict] = {}

        # Store user_id if provided
        if user_id:
            self.user_id = user_id

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str | None = None,
        auth_token: str | None = None,
        **kwargs,
    ) -> None:
        """
        Establish the WebSocket connection with authentication.

        Args:
            websocket: The FastAPI WebSocket object
            user_id: Optional pre-authenticated user ID (e.g., from middleware)
            auth_token: Optional authentication token
            **kwargs: Additional keyword arguments for compatibility.
        """
        if auth_token is not None:
            self.auth_token = auth_token
        if user_id is not None:
            self.user_id = user_id

        log.debug("Initializing unified WebSocket connection")

        # Check if authentication is enforced
        if Environment.enforce_auth():
            if self.user_id:
                log.debug("Using pre-authenticated user_id for unified WebSocket")
            else:
                token = self.auth_token
                if not token:
                    await websocket.close(code=1008, reason="Missing authentication")
                    log.warning("UnifiedWebSocketRunner connection rejected: Missing authentication token")
                    return

                user_provider = get_user_auth_provider()
                if not user_provider:
                    await websocket.close(code=1008, reason="Authentication provider not configured")
                    log.warning("UnifiedWebSocketRunner connection rejected: Auth provider not configured")
                    return

                result = await user_provider.verify_token(token)
                if not result.ok or not result.user_id:
                    await websocket.close(code=1008, reason="Invalid authentication")
                    log.warning("UnifiedWebSocketRunner connection rejected: Invalid authentication")
                    return
                self.user_id = result.user_id
        else:
            # In local development without enforced auth, set a default user ID
            self.user_id = self.user_id or "1"
            log.debug("Skipping authentication in local development mode")

        if not self.user_id:
            self.user_id = "1"

        await websocket.accept()
        self.websocket = websocket
        log.info("Unified WebSocket connection established")

        # Start heartbeat to keep idle connections alive (skip in tests to avoid leaked tasks)
        if not RUNNING_PYTEST:
            if not self.heartbeat_task or self.heartbeat_task.done():
                self.heartbeat_task = asyncio.create_task(self._heartbeat())

    async def disconnect(self):
        """
        Close the WebSocket connection and clean up resources.

        Note: Jobs continue running in the background via JobExecutionManager.
        """
        log.info("UnifiedWebSocketRunner: Disconnecting")

        # Stop any ongoing chat processing task
        if self.current_task and not self.current_task.done():
            log.debug("Stopping current chat task during disconnect")
            self.current_task.cancel()

        # Cancel any pending tool result waiters
        self.tool_bridge.cancel_all()

        # Stop heartbeat task
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.heartbeat_task
        self.heartbeat_task = None

        # Stop all streaming tasks for jobs
        for job_ctx in self.active_jobs.values():
            if job_ctx.streaming_task and not job_ctx.streaming_task.done():
                job_ctx.streaming_task.cancel()

        self.active_jobs.clear()

        # Only attempt to close if websocket exists and is not already closed
        if self.websocket and self.websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await self.websocket.close()
                log.info("UnifiedWebSocketRunner: WebSocket closed successfully")
            except Exception as e:
                log.debug(f"UnifiedWebSocketRunner: WebSocket close ignored: {e}")

        self.websocket = None
        self.current_task = None
        log.info("UnifiedWebSocketRunner: Disconnected (jobs continue in background)")

    async def send_message(self, message: dict):
        """
        Send a message to the connected WebSocket client.

        The message is encoded in binary (MessagePack) or text (JSON) format
        based on the established mode for the connection.

        Args:
            message: The message payload to send
        """
        if not self.websocket:
            log.debug("Skipping send: WebSocket is not connected")
            return

        # Guard against sending after close
        if (
            getattr(self.websocket, "client_state", None) == WebSocketState.DISCONNECTED
            or getattr(self.websocket, "application_state", None) == WebSocketState.DISCONNECTED
        ):
            log.debug("Skipping send: WebSocket is disconnected")
            return

        try:
            if self.mode == WebSocketMode.BINARY:
                packed_message = msgpack.packb(message, use_bin_type=True)
                await self.websocket.send_bytes(packed_message)  # type: ignore
            else:
                json_text = json.dumps(message)
                await self.websocket.send_text(json_text)
        except Exception as e:
            log.error(f"Error sending message: {e}", exc_info=True)

    async def receive_message(self) -> Optional[dict]:
        """
        Receive a message from the WebSocket client.

        Automatically detects binary (MessagePack) or text (JSON) format
        and sets the mode accordingly.

        Returns:
            The received message data or None if connection is closed

        Raises:
            RuntimeError: If WebSocket is not connected
        """
        if self.websocket is None:
            raise RuntimeError("WebSocket is not connected")

        try:
            message = await self.websocket.receive()
            log.debug(f"Received WebSocket message: {message}")

            if message["type"] == "websocket.disconnect":
                log.info("Received websocket disconnect message")
                return None

            if "bytes" in message:
                raw_bytes = message["bytes"]
                data = msgpack.unpackb(raw_bytes)
                self.mode = WebSocketMode.BINARY
                log.debug(f"Received binary WebSocket message: {data}")
                return data
            elif "text" in message:
                raw_text = message["text"]
                data = json.loads(raw_text)
                self.mode = WebSocketMode.TEXT
                log.debug(f"Received text WebSocket message: {data}")
                return data
            else:
                log.warning(f"Received message with unknown format: {message}")
                return None

        except Exception as e:
            log.error(f"Error receiving message: {str(e)}", exc_info=True)
            raise

    # =========================================================================
    # Workflow Job Management (from WebSocketRunner)
    # =========================================================================

    async def run_job(self, req: RunJobRequest):
        """Start a new job in the background and stream messages to the client."""
        try:
            if not self.websocket:
                raise ValueError("WebSocket is not connected")

            log.debug(f"Run job request: {req.model_dump(exclude={'graph'})}")
            log.info(f"Input params: {req.params}")

            log.info(
                "UnifiedWebSocketRunner.run_job starting",
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
                "UnifiedWebSocketRunner.run_job job info",
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
            # Send initial job update with run_state
            run_state_info = await build_run_state_info(job_ctx.job_id)
            await self.send_message(
                JobUpdate(
                    status="running",
                    job_id=job_ctx.job_id,
                    workflow_id=job_ctx.workflow_id,
                    run_state=run_state_info,
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
                        if status in ("completed", "failed", "cancelled", "error", "suspended"):
                            received_terminal_update = True

                    if not self.websocket or self.websocket.client_state == WebSocketState.DISCONNECTED:
                        log.warning(
                            "UnifiedWebSocketRunner: websocket lost during stream",
                            extra={"job_id": job_ctx.job_id},
                        )
                        raise WebSocketDisconnect()
            except WebSocketDisconnect:
                log.warning(
                    "UnifiedWebSocketRunner: websocket disconnected mid-stream",
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
            if not received_terminal_update:
                # Wait briefly to allow any final messages to be posted by the job thread
                await asyncio.sleep(0.1)

                # Process any remaining messages that arrived during the wait
                while job_ctx.job_execution.context.has_messages():
                    async for msg in process_message(job_ctx.job_execution.context, explicit_types):
                        msg["job_id"] = job_ctx.job_id
                        msg["workflow_id"] = job_ctx.workflow_id
                        await self.send_message(msg)
                        if msg.get("type") == "job_update":
                            status = msg.get("status")
                            if status in ("completed", "failed", "cancelled", "error", "suspended"):
                                received_terminal_update = True

            # If still no terminal update, check job status directly
            if not received_terminal_update:
                final_status = job_ctx.job_execution.status
                if (
                    final_status in ("completed", "cancelled", "error", "failed", "suspended")
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
                        err = (
                            getattr(job_ctx.job_execution, "error", None)
                            or getattr(job_ctx.job_execution.job_model, "error", None)
                            or "Unknown error"
                        )
                        await self.send_message(
                            JobUpdate(
                                status="failed",
                                job_id=job_ctx.job_id,
                                workflow_id=job_ctx.workflow_id,
                                error=str(err),
                            ).model_dump()
                        )
                    elif final_status == "suspended":
                        # Should have been sent by runner, but send here if missed
                        await self.send_message(
                            JobUpdate(
                                status="suspended",
                                job_id=job_ctx.job_id,
                                workflow_id=job_ctx.workflow_id,
                                message="Workflow suspended (fallback update)",
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

            log.info(f"UnifiedWebSocketRunner: Reconnecting to job: {job_id}")

            # Get the job execution from the manager
            job_manager = JobExecutionManager.get_instance()
            job_execution = job_manager.get_job(job_id)

            if job_execution is None:
                async with ResourceScope():
                    db_job = await Job.get(job_id)
                    from nodetool.models.run_state import RunState

                    run_state = await RunState.get(job_id)
                    current_status = run_state.status if run_state else None

                    if current_status in {"running", "scheduled", "queued", None}:
                        log.warning(
                            "UnifiedWebSocketRunner: Job missing from manager; marking as failed",
                            extra={"job_id": job_id},
                        )
                        if run_state:
                            await run_state.mark_failed(error="Job worker unavailable during reconnect")
                        if db_job:
                            await db_job.update(
                                error="Job worker unavailable during reconnect",
                                finished_at=datetime.now(),
                            )
                log.warning(
                    "UnifiedWebSocketRunner: Job not found during reconnect",
                    extra={"job_id": job_id},
                )
                raise ValueError(f"Job {job_id} not found")

            log.info(
                "UnifiedWebSocketRunner.reconnect_job obtained job execution",
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
                    await job_execution.job_model.reload()
                    from nodetool.models.run_state import RunState

                    run_state = await RunState.get(job_id)
                    current_status = run_state.status if run_state else None

                    if current_status in {"running", "scheduled", "queued"}:
                        if final_status == "completed":
                            if run_state:
                                await run_state.mark_completed()
                        elif final_status in {"error", "failed"}:
                            err_detail = getattr(job_execution, "error", None) or getattr(
                                job_execution.job_model, "error", None
                            )
                            if run_state:
                                await run_state.mark_failed(error=str(err_detail) if err_detail else "Unknown error")
                        elif final_status == "cancelled":
                            if run_state:
                                await run_state.mark_cancelled()
                        await job_execution.job_model.update(
                            finished_at=datetime.now(),
                        )
                except Exception as persist_error:
                    log.error(
                        "UnifiedWebSocketRunner: Failed to persist finalized job state",
                        extra={"job_id": job_id, "error": str(persist_error)},
                    )

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
                    "UnifiedWebSocketRunner: Job already completed during reconnect",
                    extra={"job_id": job_id, "status": final_status},
                )
                return

            # Create streaming context
            job_ctx = JobStreamContext(job_id, workflow_id, job_execution)
            self.active_jobs[job_id] = job_ctx

            # Send current job status with run_state
            run_state_info = await build_run_state_info(job_id)
            await self.send_message(
                JobUpdate(
                    status=job_execution.status,
                    job_id=job_id,
                    workflow_id=workflow_id,
                    run_state=run_state_info,
                ).model_dump()
            )

            # Replay current status for all nodes and edges if job is running
            if job_execution.is_running():
                node_count = len(job_execution.context.node_statuses)
                edge_count = len(job_execution.context.edge_statuses)
                log.info(f"Replaying status for {node_count} nodes and {edge_count} edges for job {job_id}")

                for node_status in job_execution.context.node_statuses.values():
                    msg_dict = node_status.model_dump()
                    msg_dict["job_id"] = job_id
                    msg_dict["workflow_id"] = workflow_id
                    await self.send_message(msg_dict)

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

    async def resume_job(self, job_id: str, workflow_id: Optional[str] = None):
        """Resume a suspended or recovering job from persistence."""
        try:
            if not self.websocket:
                raise ValueError("WebSocket is not connected")

            log.info(f"UnifiedWebSocketRunner: Resuming job: {job_id}")

            # Check if job is already active in manager (in-memory)
            job_manager = JobExecutionManager.get_instance()
            existing_job = job_manager.get_job(job_id)

            if existing_job and existing_job.is_running():
                log.info(f"Job {job_id} is already running in memory. Reconnecting instead.")
                return await self.reconnect_job(job_id, workflow_id)

            # Trigger resumption
            success = await job_manager.resume_run(job_id)

            if not success:
                raise ValueError(f"Failed to resume job {job_id}. Check server logs for details.")

            # Get the new job execution (resumed job)
            job_execution = job_manager.get_job(job_id)
            if not job_execution:
                raise ValueError(f"Job {job_id} resumed but not found in manager")

            # Create streaming context
            if not workflow_id:
                workflow_id = job_execution.request.workflow_id

            job_ctx = JobStreamContext(job_id, workflow_id, job_execution)
            self.active_jobs[job_id] = job_ctx

            # Send current job status
            await self.send_message(
                JobUpdate(
                    status="running",
                    job_id=job_id,
                    workflow_id=workflow_id,
                    message="Job resumed successfully",
                ).model_dump()
            )

            # Start streaming messages
            job_ctx.streaming_task = asyncio.create_task(self._stream_job_messages(job_ctx, False))

            log.info(f"Resumed and streaming job {job_id}")

        except Exception as e:
            log.exception(f"Error resuming job {job_id}: {e}")
            await self.send_message(
                JobUpdate(
                    status="failed",
                    error=str(e),
                    job_id=job_id,
                    workflow_id=workflow_id or "",
                ).model_dump()
            )

    async def cancel_job(self, job_id: str, workflow_id: Optional[str] = None):
        """
        Cancel the specified job.

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
        Get the current status of job execution.

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
        """Clear unused models from the model manager."""
        if not Environment.is_production():
            ModelManager.clear()
            gc.collect()
            return {"message": "Unused models cleared"}
        return {"message": "Model clearing is disabled in production"}

    # =========================================================================
    # Chat Message Handling (from ChatWebSocketRunner)
    # =========================================================================

    async def handle_chat_message(self, data: dict):
        """
        Handle an incoming chat message by saving to DB and processing using chat history from DB.
        """
        log.debug(
            f"[handle_chat_message] Received data: {data.get('type')}, workflow_target={data.get('workflow_target')}, workflow_id={data.get('workflow_id')}"
        )

        # Wrap database operations in ResourceScope for per-execution isolation
        async with ResourceScope():
            try:
                # Extract thread_id from message data and ensure thread exists
                thread_id = data.get("thread_id")
                thread_id = await self.ensure_thread_exists(thread_id)

                # Update message data with the thread_id (in case it was created)
                data["thread_id"] = thread_id

                # Apply defaults if not specified
                if not data.get("model"):
                    data["model"] = self.default_model
                if not data.get("provider"):
                    data["provider"] = self.default_provider

                log.debug(
                    f"[handle_chat_message] Data before save: workflow_target={data.get('workflow_target')}, workflow_id={data.get('workflow_id')}"
                )

                # Save message to database asynchronously
                await self._save_message_to_db_async(data)

                # Load history from database
                chat_history = await self.get_chat_history_from_db(thread_id)
                log.debug(f"[handle_chat_message] Loaded {len(chat_history)} messages from history")
                if chat_history:
                    last_msg = chat_history[-1]
                    log.debug(
                        f"[handle_chat_message] Last message in history: workflow_target={getattr(last_msg, 'workflow_target', 'N/A')}, workflow_id={getattr(last_msg, 'workflow_id', 'N/A')}"
                    )

                # Call the implementation method with the loaded messages
                await self.handle_message_impl(chat_history)
            except asyncio.CancelledError:
                log.info("Chat message processing cancelled by user")
                with suppress(Exception):
                    await self.send_message(
                        {
                            "type": "generation_stopped",
                            "message": "Generation stopped by user",
                        }
                    )
            except Exception as e:
                log.error(f"Error processing chat message: {str(e)}", exc_info=True)
                error_message = {"type": "error", "message": str(e)}
                with suppress(Exception):
                    await self.send_message(error_message)

    # =========================================================================
    # Unified Command Handling
    # =========================================================================

    async def handle_command(self, command: WebSocketCommand):
        """
        Handle incoming WebSocket commands (workflow-related).

        Args:
            command: The WebSocket command to handle

        Returns:
            dict: A dictionary with the response to the command
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

        elif command.command == CommandType.RESUME_JOB:
            if not job_id:
                return {"error": "job_id is required"}
            log.info(f"Resuming job: {job_id}")
            # Use asyncio task to handle resumption in background
            self._resume_task = asyncio.create_task(self.resume_job(job_id, workflow_id))
            return {
                "message": f"Resumption initiated for job {job_id}",
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
                    value = Chunk(**value)
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
                if not job_ctx.job_execution.runner:
                    raise RuntimeError("Runner is not set for this job execution")
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

        elif command.command == CommandType.CHAT_MESSAGE:
            # Handle chat message through command interface - requires thread_id
            thread_id = command.data.get("thread_id")
            if not thread_id:
                return {"error": "thread_id is required for chat_message command"}
            self.current_task = asyncio.create_task(self.handle_chat_message(command.data))
            return {"message": "Chat message processing started", "thread_id": thread_id}

        elif command.command == CommandType.STOP:
            # Stop current operation - requires job_id or thread_id
            thread_id = command.data.get("thread_id")
            if not job_id and not thread_id:
                return {"error": "job_id or thread_id is required for stop command"}

            log.debug(f"Received stop command for job_id={job_id}, thread_id={thread_id}")

            # Cancel current chat processing task if thread_id matches
            if thread_id and self.current_task and not self.current_task.done():
                log.debug("Stopping current chat processor")
                self.current_task.cancel()

            # Cancel job if job_id is provided
            if job_id:
                job_ctx = self.active_jobs.get(job_id)
                if job_ctx and job_ctx.streaming_task and not job_ctx.streaming_task.done():
                    job_ctx.streaming_task.cancel()

            # Cancel any pending tool result waiters
            self.tool_bridge.cancel_all()

            await self.send_message(
                {
                    "type": "generation_stopped",
                    "message": "Generation stopped by user",
                    "job_id": job_id,
                    "thread_id": thread_id,
                }
            )
            log.info(f"Generation stopped by user command for job_id={job_id}, thread_id={thread_id}")
            return {"message": "Stop command processed", "job_id": job_id, "thread_id": thread_id}

        else:
            log.warning(f"Unknown command received: {command.command}")
            return {"error": "Unknown command"}

    async def _heartbeat(self):
        """Periodically send a lightweight heartbeat message to keep the WebSocket alive."""
        while True:
            try:
                # Sleep first to avoid sending immediately upon connect
                await asyncio.sleep(25)
                if not self.websocket:
                    break
                await self.send_message({"type": "ping", "ts": time.time()})
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug(f"Heartbeat send failed: {e}")
                break

    # =========================================================================
    # Main Run Loop
    # =========================================================================

    async def run(self, websocket: WebSocket):
        """
        Main method to run the UnifiedWebSocketRunner.

        This handles the message receive loop and routes messages to appropriate handlers.

        Args:
            websocket: The FastAPI WebSocket connection
        """
        await self.connect(
            websocket,
            user_id=self.user_id or None,
            auth_token=self.auth_token or None,
        )

        if not self.websocket:
            return  # Connection was rejected

        try:
            # Main message receive loop
            await self._receive_messages()
        finally:
            # Clean up any running tasks
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.current_task
            # Ensure we fully disconnect
            with suppress(Exception):
                await self.disconnect()

    async def _receive_messages(self):
        """
        Continuously receive messages from the WebSocket and handle them.

        All messages must be wrapped in a command structure with a 'command' field.
        Special message types (ping, client_tools_manifest, tool_result) are handled
        as control messages for backward compatibility with frontend tooling.
        """
        while True:
            try:
                data = await self.receive_message()

                if data is None:
                    # Connection closed
                    break

                # Handle control messages first
                if isinstance(data, dict):
                    msg_type = data.get("type")

                    # Client tools manifest for chat (control message)
                    if msg_type == "client_tools_manifest":
                        tools = data.get("tools", [])
                        self.client_tools_manifest = {tool["name"]: tool for tool in tools}
                        try:
                            manifest_tokens = count_json_tokens(tools)
                            log.debug(
                                "Received client tools manifest with %d tools (tokens=%d)",
                                len(tools),
                                manifest_tokens,
                            )
                        except Exception:
                            log.debug(
                                "Received client tools manifest with %d tools",
                                len(tools),
                            )
                        continue

                    # Tool result from frontend (control message)
                    if msg_type == "tool_result":
                        tool_call_id = data.get("tool_call_id")
                        if tool_call_id:
                            self.tool_bridge.resolve_result(tool_call_id, data)
                            log.debug(f"Resolved tool result for call_id: {tool_call_id}")
                        continue

                    # Ping-pong for connection keepalive (control message)
                    if msg_type == "ping":
                        await self.send_message({"type": "pong", "ts": time.time()})
                        continue

                    # All other messages must be commands
                    if "command" in data:
                        try:
                            command = WebSocketCommand(**data)
                            response = await self.handle_command(command)
                            await self.send_message(response)
                            log.debug(f"Handled command {command.command}")
                        except Exception as decode_error:
                            log.warning("Failed to decode command message: %s", decode_error)
                            await self.send_message({"error": "invalid_command", "details": str(decode_error)})
                        continue

                    # Unknown message type - must use command structure
                    log.warning(f"Message missing 'command' field: {data}")
                    await self.send_message(
                        {
                            "error": "invalid_message",
                            "message": "All messages must include a 'command' field. Use 'chat_message' command for chat.",
                        }
                    )

            except asyncio.CancelledError:
                log.info("Message receiving cancelled")
                break
            except Exception as e:
                log.error(f"Error in receive loop: {str(e)}", exc_info=True)
                error_message = {"type": "error", "message": str(e)}
                with suppress(Exception):
                    await self.send_message(error_message)
                continue
