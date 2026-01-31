"""
Gateway module for OpenClaw Gateway protocol support.

This module provides WebSocket client functionality to allow NodeTool
to act as a node in a distributed gateway architecture, receiving and
executing workflows from a central gateway server.
"""

from nodetool.gateway.client import GatewayClient
from nodetool.gateway.protocol import (
    CommandRequest,
    CommandResponse,
    GatewayMessage,
    NodeRegistration,
    WorkflowRequest,
    WorkflowResponse,
)

__all__ = [
    "CommandRequest",
    "CommandResponse",
    "GatewayClient",
    "GatewayMessage",
    "NodeRegistration",
    "WorkflowRequest",
    "WorkflowResponse",
]
