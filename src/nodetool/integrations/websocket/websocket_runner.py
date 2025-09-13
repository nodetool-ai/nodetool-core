import asyncio
from enum import Enum
import json
import uuid
from fastapi.websockets import WebSocketState
import msgpack
from typing import AsyncGenerator
from concurrent.futures import Future
from nodetool.types.wrap_primitive_types import wrap_primitive_types
from pydantic import BaseModel
from fastapi import WebSocket, WebSocketDisconnect
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.ml.core.model_manager import ModelManager
from nodetool.types.job import JobUpdate
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.threaded_event_loop import ThreadedEventLoop
from nodetool.workflows.types import Chunk, Error
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
"""

TORCH_AVAILABLE = False
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    pass


class CommandType(str, Enum):
    RUN_JOB = "run_job"
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
        msg_dict = msg.model_dump()

        if explicit_types and "result" in msg_dict:
            msg_dict["result"] = wrap_primitive_types(msg_dict["result"])

        log.debug(f"Processing workflow message: {msg_dict.get('type', msg_dict)}")

        yield msg_dict


async def process_workflow_messages(
    context: ProcessingContext,
    runner: WorkflowRunner,
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
        while runner.is_running():
            if context.has_messages():
                async for msg in process_message(context, explicit_types):
                    yield msg
            else:
                await asyncio.sleep(sleep_interval)

        # Process remaining messages
        while context.has_messages():
            async for msg in process_message(context, explicit_types):
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


class WebSocketRunner:
    """
    Runs a workflow using a WebSocket connection.

    Attributes:
        websocket (WebSocket | None): The WebSocket connection.
        context (ProcessingContext | None): The processing context for job execution.
        job_id (str | None): The ID of the current job.
        runner (WorkflowRunner | None): The workflow runner for job execution.
        mode (WebSocketMode): The current mode for WebSocket communication.
    """

    websocket: WebSocket | None = None
    context: ProcessingContext | None = None
    active_job: RunJobRequest | None = None
    event_loop: ThreadedEventLoop | None = None
    job_id: str | None = None
    runner: WorkflowRunner | None = None
    mode: WebSocketMode = WebSocketMode.BINARY
    nodes: dict[str, BaseNode] = {}
    run_future: Future | None = None

    def __init__(
        self,
    ):
        """
        Initializes a new instance of the WebSocketRunner class.
        """
        self.mode = WebSocketMode.BINARY

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
        # Start a persistent threaded event loop for this connection (preserve caches)
        if not self.event_loop:
            self.event_loop = ThreadedEventLoop()
        if not self.event_loop.is_running:
            self.event_loop.start()

    async def disconnect(self):
        """
        Closes the WebSocket connection and cancels any active job.
        """
        log.info("WebSocketRunner: Disconnecting")
        log.debug(f"Active event loop: {self.event_loop is not None}")
        if self.event_loop:
            try:
                self.event_loop.stop()
            except Exception as e:
                log.error(f"Error cancelling active job during disconnect: {e}")

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
        self.event_loop = None
        self.job_id = None
        log.debug("WebSocket connection and resources cleared")
        log.info("WebSocketRunner: Disconnected and resources cleaned up")

    async def run_job(self, req: RunJobRequest):
        try:
            if not self.websocket:
                raise ValueError("WebSocket is not connected")

            log.debug(f"Run job request: {req.model_dump(exclude={'graph'})}")

            self.job_id = uuid.uuid4().hex
            self.runner = WorkflowRunner(job_id=self.job_id)
            log.info(f"WebSocketRunner: Starting job execution: {self.job_id}")

            if self.context is None:
                self.context = ProcessingContext(
                    user_id=req.user_id,
                    auth_token=req.auth_token,
                    workflow_id=req.workflow_id,
                    encode_assets_as_base64=self.mode == WebSocketMode.TEXT,
                )
                log.debug("Processing context created")
            # Ensure persistent event loop is running
            if not self.event_loop:
                self.event_loop = ThreadedEventLoop()
            if not self.event_loop.is_running:
                self.event_loop.start()
            log.debug("Threaded event loop ready")

            # Schedule workflow execution on the persistent loop
            self.run_future = self.event_loop.run_coroutine(
                execute_workflow(self.context, self.runner, req)
            )

            try:
                async for msg in process_workflow_messages(
                    context=self.context,
                    runner=self.runner,
                    explicit_types=req.explicit_types or False,
                ):
                    await self.send_message(msg)
            except Exception as e:
                log.exception(e)
                if self.run_future and not self.run_future.done():
                    self.run_future.cancel()
                await self.send_job_update("failed", str(e))

            # Propagate completion status, distinguishing cancellation
            try:
                if self.run_future:
                    self.run_future.result()
            except asyncio.CancelledError:
                log.info(f"Workflow {self.job_id} cancelled")
                await self.send_job_update("cancelled")
            except Exception as e:
                log.error(f"An error occurred during workflow execution: {e}")
                await self.send_job_update("failed", str(e))
            else:
                log.debug("Workflow execution completed")

        except Exception as e:
            log.exception(f"Error in job {self.job_id}: {e}")
            await self.send_job_update("failed", str(e))

        self.active_job = None
        self.runner = None
        self.run_future = None
        log.info("Job resources cleaned up")

    async def send_message(self, message: dict):
        """Send a message using the current mode."""
        assert self.websocket, "WebSocket is not connected"
        try:
            if self.mode == WebSocketMode.BINARY:
                packed_message = msgpack.packb(message, use_bin_type=True)
                await self.websocket.send_bytes(packed_message)  # type: ignore
                log.debug(f"Sent binary message: {message.get('type', message)}")
            else:
                await self.websocket.send_text(json.dumps(message))
                log.debug(f"Sent text message: {message.get('type', message)}")
        except Exception as e:
            log.error(f"Error sending message: {e}")

    async def send_job_update(self, status: str, error: str | None = None):
        msg = {
            "type": "job_update",
            "status": status,
            "error": error,
            "job_id": self.job_id,
        }
        log.debug(f"Sending job update: {msg}")
        await self.send_message(msg)

    async def cancel_job(self):
        """
        Cancels the active job if one exists.

        Returns:
            dict: A dictionary with a message indicating the job was cancelled, or an error if no active job exists.
        """
        log.info(f"Attempting to cancel job: {self.job_id}")
        # Cancel only the running job; keep the persistent event loop alive
        if self.run_future and not self.run_future.done():
            self.run_future.cancel()
            return {"message": "Job cancellation requested"}
        log.warning("No active job to cancel")
        return {"error": "No active job to cancel"}

    def get_status(self):
        """
        Gets the current status of job execution.

        Returns:
            dict: A dictionary with the status ("running" or "idle") and the job ID.
        """
        return {
            "status": self.runner.status if self.runner else "idle",
            "job_id": self.job_id,
        }

    async def clear_models(self):
        """
        Clears unused models from the model manager.
        """
        if not Environment.is_production():
            ModelManager.clear()
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        if command.command == CommandType.CLEAR_MODELS:
            return await self.clear_models()
        elif command.command == CommandType.RUN_JOB:
            if self.run_future and not self.run_future.done():
                log.warning("Attempted to start a job while another is running")
                return {"error": "A job is already running"}
            req = RunJobRequest(**command.data)
            log.info(f"Starting workflow: {req.workflow_id}")
            self.active_job = req
            asyncio.create_task(self.run_job(req))
            log.debug("Run job command scheduled")
            return {"message": "Job started"}
        elif command.command == CommandType.STREAM_INPUT:
            # Expected data: { input: str, value: Any, handle?: str }
            if not self.runner or not self.context:
                return {"error": "No active runner/context"}
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
                self.runner.push_input_value(input_name=input_name, value=value, source_handle=handle)  # type: ignore[arg-type]
                log.debug("STREAM_INPUT enqueued to runner input queue")
                return {"message": "Input item streamed"}
            except Exception as e:
                log.exception(e)
                return {"error": str(e)}
        elif command.command == CommandType.END_INPUT_STREAM:
            # Expected data: { input: str, handle?: str }
            if not self.runner or not self.context:
                return {"error": "No active runner/context"}
            input_name = command.data.get("input")
            if not isinstance(input_name, str) or input_name.strip() == "":
                return {"error": "Invalid input name"}
            handle = command.data.get("handle")
            try:
                log.debug(
                    f"END_INPUT_STREAM received: input={input_name} handle={handle}"
                )
                self.runner.finish_input_stream(
                    input_name=input_name, source_handle=handle
                )
                log.debug("END_INPUT_STREAM enqueued to runner input queue")
                return {"message": "Input stream ended"}
            except Exception as e:
                log.exception(e)
                return {"error": str(e)}
        elif command.command == CommandType.CANCEL_JOB:
            return await self.cancel_job()
        elif command.command == CommandType.GET_STATUS:
            status = self.get_status()
            log.debug(f"Current status: {status}")
            return status
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
