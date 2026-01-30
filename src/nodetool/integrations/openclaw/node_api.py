"""OpenClaw Node API endpoints.

This module implements the REST API endpoints that allow this node to
function as an OpenClaw node, including registration, task execution,
and status reporting.
"""

import asyncio
import platform
import time
from datetime import datetime
from typing import Any, Optional

import psutil
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from nodetool.config.logging_config import get_logger
from nodetool.integrations.openclaw.config import OpenClawConfig
from nodetool.integrations.openclaw.gateway_client import GatewayClient
from nodetool.integrations.openclaw.schemas import (
    NodeCapability,
    NodeRegistration,
    NodeRegistrationResponse,
    NodeStatus,
    NodeStatusResponse,
    TaskExecutionRequest,
    TaskExecutionResponse,
    TaskStatus,
)

log = get_logger(__name__)

# Create router for OpenClaw endpoints
router = APIRouter(prefix="/openclaw", tags=["openclaw"])

# Global state
_gateway_client: Optional[GatewayClient] = None
_active_tasks: dict[str, asyncio.Task] = {}
_task_results: dict[str, dict[str, Any]] = {}
_stats = {
    "total_tasks_completed": 0,
    "total_tasks_failed": 0,
}


def get_gateway_client() -> GatewayClient:
    """Get or create the Gateway client instance."""
    global _gateway_client
    if _gateway_client is None:
        config = OpenClawConfig.get_instance()
        _gateway_client = GatewayClient(config)
    return _gateway_client


def get_node_capabilities() -> list[NodeCapability]:
    """Define the capabilities this node provides.

    These capabilities map to nodetool-core functionality.
    """
    return [
        NodeCapability(
            name="workflow_execution",
            description="Execute AI workflows defined in nodetool format",
            input_schema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "ID of workflow to execute"},
                    "workflow_data": {
                        "type": "object",
                        "description": "Workflow graph definition"
                    },
                    "params": {
                        "type": "object",
                        "description": "Input parameters for the workflow"
                    },
                },
                "required": ["workflow_data"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "status": {"type": "string"},
                    "result": {"type": "object"},
                },
            },
        ),
        NodeCapability(
            name="chat_completion",
            description="Generate text completions using various AI models",
            input_schema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Chat messages in OpenAI format",
                    },
                    "model": {"type": "string", "description": "Model to use"},
                    "temperature": {"type": "number", "description": "Sampling temperature"},
                },
                "required": ["messages"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "model": {"type": "string"},
                    "usage": {"type": "object"},
                },
            },
        ),
        NodeCapability(
            name="asset_processing",
            description="Process and transform media assets (images, audio, video)",
            input_schema={
                "type": "object",
                "properties": {
                    "asset_url": {"type": "string", "description": "URL of asset to process"},
                    "operation": {
                        "type": "string",
                        "enum": ["resize", "convert", "transform"],
                        "description": "Operation to perform",
                    },
                    "parameters": {"type": "object", "description": "Operation parameters"},
                },
                "required": ["asset_url", "operation"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result_url": {"type": "string"},
                    "metadata": {"type": "object"},
                },
            },
        ),
    ]


def get_system_info() -> dict[str, Any]:
    """Get system resource information."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_percent": memory.percent,
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_percent": disk.percent,
        }
    except Exception as e:
        log.error("Failed to get system info: %s", e)
        return {"error": str(e)}


@router.post("/register", response_model=NodeRegistrationResponse)
async def register_node(request: Request) -> NodeRegistrationResponse:
    """Register this node with the OpenClaw Gateway.

    This endpoint initiates registration with the Gateway, providing
    information about this node's capabilities and how to reach it.
    """
    config = OpenClawConfig.get_instance()

    if not config.enabled:
        raise HTTPException(
            status_code=503,
            detail="OpenClaw integration is not enabled. Set OPENCLAW_ENABLED=true",
        )

    # Build registration request
    registration = NodeRegistration(
        node_id=config.node_id,
        node_name=config.node_name,
        node_version=config.node_version,
        capabilities=get_node_capabilities(),
        endpoint=config.node_endpoint,
        metadata={
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "started_at": datetime.utcnow().isoformat(),
        },
    )

    try:
        client = get_gateway_client()
        response = await client.register(registration)

        # Start heartbeat if registration successful
        if response.success and config.heartbeat_interval > 0:

            async def get_status():
                status = await get_node_status()
                return status.model_dump()

            await client.start_heartbeat(get_status)

        return response

    except Exception as e:
        log.error("Failed to register with Gateway: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Registration failed: {str(e)}"
        ) from e


@router.post("/execute", response_model=TaskExecutionResponse)
async def execute_task(task_request: TaskExecutionRequest) -> TaskExecutionResponse:
    """Execute a task on this node.

    This endpoint receives task execution requests from the Gateway,
    validates them, and initiates execution.
    """
    config = OpenClawConfig.get_instance()

    if not config.enabled:
        raise HTTPException(
            status_code=503, detail="OpenClaw integration is not enabled"
        )

    # Check if we're at max capacity
    if len(_active_tasks) >= config.max_concurrent_tasks:
        return TaskExecutionResponse(
            task_id=task_request.task_id,
            status=TaskStatus.PENDING,
            message=f"Node at capacity ({config.max_concurrent_tasks} tasks), task queued",
        )

    # Validate capability
    capabilities = get_node_capabilities()
    capability_names = [c.name for c in capabilities]

    if task_request.capability_name not in capability_names:
        return TaskExecutionResponse(
            task_id=task_request.task_id,
            status=TaskStatus.FAILED,
            message=f"Unknown capability: {task_request.capability_name}",
        )

    # Create async task for execution
    async def run_task():
        try:
            log.info(
                "Executing task %s with capability %s",
                task_request.task_id,
                task_request.capability_name,
            )

            # This is where we would integrate with actual nodetool-core functionality
            # For now, simulate task execution
            await asyncio.sleep(2)  # Simulate work

            result = {
                "status": "completed",
                "capability": task_request.capability_name,
                "parameters": task_request.parameters,
                "executed_at": datetime.utcnow().isoformat(),
            }

            _task_results[task_request.task_id] = {
                "status": TaskStatus.COMPLETED,
                "result": result,
            }
            _stats["total_tasks_completed"] += 1

            # Send result to callback URL if provided
            if task_request.callback_url:
                try:
                    import aiohttp

                    async with aiohttp.ClientSession() as session:
                        await session.post(
                            task_request.callback_url,
                            json={
                                "task_id": task_request.task_id,
                                "status": "completed",
                                "result": result,
                            },
                        )
                except Exception as e:
                    log.error("Failed to send callback: %s", e)

        except Exception as e:
            log.error("Task execution failed: %s", e)
            _task_results[task_request.task_id] = {
                "status": TaskStatus.FAILED,
                "message": str(e),
            }
            _stats["total_tasks_failed"] += 1
        finally:
            # Clean up
            if task_request.task_id in _active_tasks:
                del _active_tasks[task_request.task_id]

    # Start the task
    task = asyncio.create_task(run_task())
    _active_tasks[task_request.task_id] = task

    return TaskExecutionResponse(
        task_id=task_request.task_id,
        status=TaskStatus.RUNNING,
        message="Task execution started",
    )


@router.get("/capabilities", response_model=list[NodeCapability])
async def get_capabilities() -> list[NodeCapability]:
    """Get the list of capabilities this node provides.

    Returns detailed information about each capability including
    input and output schemas.
    """
    return get_node_capabilities()


@router.get("/status", response_model=NodeStatusResponse)
async def get_node_status() -> NodeStatusResponse:
    """Get the current status and health of this node.

    Returns information about uptime, resource usage, and task statistics.
    """
    config = OpenClawConfig.get_instance()

    # Determine current status
    if len(_active_tasks) >= config.max_concurrent_tasks or _active_tasks:
        status = NodeStatus.BUSY
    else:
        status = NodeStatus.ONLINE

    return NodeStatusResponse(
        node_id=config.node_id,
        status=status,
        uptime_seconds=OpenClawConfig.get_uptime(),
        active_tasks=len(_active_tasks),
        total_tasks_completed=_stats["total_tasks_completed"],
        total_tasks_failed=_stats["total_tasks_failed"],
        system_info=get_system_info(),
        timestamp=datetime.utcnow(),
    )


@router.get("/tasks/{task_id}", response_model=TaskExecutionResponse)
async def get_task_status(task_id: str) -> TaskExecutionResponse:
    """Get the status of a specific task.

    Args:
        task_id: Unique identifier for the task.

    Returns:
        Current status and result (if completed) of the task.
    """
    # Check if task is still running
    if task_id in _active_tasks:
        return TaskExecutionResponse(
            task_id=task_id, status=TaskStatus.RUNNING, message="Task is executing"
        )

    # Check if we have a result
    if task_id in _task_results:
        result_data = _task_results[task_id]
        return TaskExecutionResponse(
            task_id=task_id,
            status=result_data.get("status", TaskStatus.COMPLETED),
            message=result_data.get("message"),
            result=result_data.get("result"),
        )

    # Task not found
    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.get("/health")
async def health_check():
    """Health check endpoint for OpenClaw monitoring.

    Returns basic health status of the node.
    """
    config = OpenClawConfig.get_instance()
    return JSONResponse(
        {
            "status": "healthy" if config.enabled else "disabled",
            "node_id": config.node_id,
            "uptime_seconds": OpenClawConfig.get_uptime(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
