"""
Gateway protocol message definitions for OpenClaw Gateway.

Defines message types and structures for communication between
NodeTool nodes and the gateway server.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class GatewayMessage(BaseModel):
    """Base class for all gateway messages."""

    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class NodeRegistration(GatewayMessage):
    """Node registration message sent to gateway on connection."""

    type: Literal["node_registration"] = "node_registration"
    node_id: str
    capabilities: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeHeartbeat(GatewayMessage):
    """Periodic heartbeat to maintain connection."""

    type: Literal["heartbeat"] = "heartbeat"
    node_id: str
    status: str = "active"


class WorkflowRequest(GatewayMessage):
    """Workflow execution request from gateway."""

    type: Literal["workflow_request"] = "workflow_request"
    request_id: str
    workflow_id: Optional[str] = None
    graph: Optional[dict[str, Any]] = None
    params: dict[str, Any] = Field(default_factory=dict)
    user_id: str = "1"


class WorkflowResponse(GatewayMessage):
    """Workflow execution response to gateway."""

    type: Literal["workflow_response"] = "workflow_response"
    request_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    job_id: Optional[str] = None


class WorkflowUpdate(GatewayMessage):
    """Streaming update during workflow execution."""

    type: Literal["workflow_update"] = "workflow_update"
    request_id: str
    job_id: Optional[str] = None
    update_type: str
    data: dict[str, Any]


class CommandRequest(GatewayMessage):
    """Command request from gateway (similar to MCP)."""

    type: Literal["command_request"] = "command_request"
    request_id: str
    command: str
    args: dict[str, Any] = Field(default_factory=dict)


class CommandResponse(GatewayMessage):
    """Command response to gateway."""

    type: Literal["command_response"] = "command_response"
    request_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class ErrorMessage(GatewayMessage):
    """Error message."""

    type: Literal["error"] = "error"
    error: str
    details: Optional[dict[str, Any]] = None


class AckMessage(GatewayMessage):
    """Acknowledgment message."""

    type: Literal["ack"] = "ack"
    request_id: str
    message: Optional[str] = None
