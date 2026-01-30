"""Pydantic schemas for OpenClaw Gateway Protocol.

These schemas define the message formats and data structures used for
communication with the OpenClaw Gateway according to the protocol specification.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    """Node status enumeration."""

    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeCapability(BaseModel):
    """Represents a capability that this node can perform."""

    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of what this capability does")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for input parameters"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for output data"
    )


class NodeRegistration(BaseModel):
    """Node registration request to OpenClaw Gateway."""

    node_id: str = Field(..., description="Unique identifier for this node")
    node_name: str = Field(..., description="Human-readable name for this node")
    node_version: str = Field(..., description="Version of the node software")
    capabilities: list[NodeCapability] = Field(
        default_factory=list, description="List of capabilities this node provides"
    )
    endpoint: Optional[str] = Field(
        None, description="Base URL endpoint where this node can be reached"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the node"
    )


class NodeRegistrationResponse(BaseModel):
    """Response from Gateway after successful registration."""

    success: bool = Field(..., description="Whether registration was successful")
    node_id: str = Field(..., description="Assigned node ID")
    token: Optional[str] = Field(None, description="Authentication token for the node")
    message: Optional[str] = Field(None, description="Additional message or error details")


class TaskExecutionRequest(BaseModel):
    """Request to execute a task on this node."""

    task_id: str = Field(..., description="Unique identifier for this task")
    capability_name: str = Field(..., description="Name of the capability to execute")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Input parameters for the task"
    )
    callback_url: Optional[str] = Field(
        None, description="URL to send results back to"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional task metadata"
    )


class TaskExecutionResponse(BaseModel):
    """Response after initiating task execution."""

    task_id: str = Field(..., description="ID of the task")
    status: TaskStatus = Field(..., description="Current status of the task")
    message: Optional[str] = Field(None, description="Status message or error details")
    result: Optional[dict[str, Any]] = Field(
        None, description="Task result (for completed tasks)"
    )


class NodeStatusResponse(BaseModel):
    """Response containing the current status of this node."""

    node_id: str = Field(..., description="Node identifier")
    status: NodeStatus = Field(..., description="Current node status")
    uptime_seconds: float = Field(..., description="How long the node has been running")
    active_tasks: int = Field(0, description="Number of currently executing tasks")
    total_tasks_completed: int = Field(
        0, description="Total number of tasks completed since startup"
    )
    total_tasks_failed: int = Field(
        0, description="Total number of tasks that have failed"
    )
    system_info: dict[str, Any] = Field(
        default_factory=dict, description="System resource information"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Time of status report"
    )


class GatewayMessage(BaseModel):
    """Generic message format for Gateway communication."""

    message_id: str = Field(..., description="Unique message identifier")
    message_type: str = Field(..., description="Type of message")
    source_node_id: Optional[str] = Field(None, description="ID of the source node")
    target_node_id: Optional[str] = Field(None, description="ID of the target node")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Message payload"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Message timestamp"
    )
