"""
Webhook Trigger Node
====================

This module provides the WebhookTrigger node that receives HTTP webhook
requests and triggers workflow execution with the request data.

The webhook trigger:
1. Accepts incoming HTTP POST/GET requests at a specified path
2. Parses the request body and headers
3. Emits the webhook data to downstream nodes

Usage:
    When deployed, the workflow with a WebhookTrigger will expose an HTTP
    endpoint that can receive webhook calls from external services.
"""

from typing import Any, Literal, TypedDict

from pydantic import Field

from nodetool.metadata.types import BaseType, Datetime
from nodetool.nodes.triggers.base import TriggerNode
from nodetool.workflows.processing_context import ProcessingContext


class WebhookEvent(BaseType):
    """
    Represents data received from an HTTP webhook request.
    
    Attributes:
        method: The HTTP method (GET, POST, PUT, DELETE, etc.)
        path: The request path
        headers: Dictionary of HTTP headers
        query_params: Query string parameters
        body: The request body (parsed as JSON if possible, otherwise raw string)
        content_type: The Content-Type header value
    """
    type: Literal["webhook_event"] = "webhook_event"
    timestamp: Datetime = Field(default_factory=Datetime)
    method: str = Field(default="POST", description="HTTP method of the request")
    path: str = Field(default="", description="Request path")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    query_params: dict[str, str] = Field(default_factory=dict, description="Query string parameters")
    body: dict[str, Any] = Field(default_factory=dict, description="Request body (parsed JSON or raw string)")
    content_type: str = Field(default="application/json", description="Content-Type header")


class WebhookTrigger(TriggerNode):
    """
    Trigger node that receives HTTP webhook requests.
    
    This node acts as an HTTP endpoint that can receive webhook calls from
    external services. When a webhook is received, it emits the request data
    to downstream nodes.
    
    The webhook endpoint path is determined by the workflow configuration.
    When deployed, webhooks can be sent to:
    `POST /api/v1/workflows/{workflow_id}/webhook`
    
    webhook, http, request, api, event, trigger
    
    Attributes:
        secret: Optional secret token for webhook verification
        allowed_methods: HTTP methods to accept (default: POST only)
    """
    
    secret: str = Field(
        default="",
        description="Optional secret token for webhook signature verification"
    )
    allowed_methods: list[str] = Field(
        default=["POST"],
        description="HTTP methods to accept (e.g., ['POST', 'PUT'])"
    )
    
    # Input fields that will be populated when the webhook is called
    method: str = Field(default="POST", description="HTTP method of the incoming request")
    path: str = Field(default="", description="Request path")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    query_params: dict[str, str] = Field(default_factory=dict, description="Query parameters")
    body: dict[str, Any] = Field(default_factory=dict, description="Request body as JSON")
    content_type: str = Field(default="application/json", description="Content-Type")

    class OutputType(TypedDict):
        event: WebhookEvent

    async def process(self, context: ProcessingContext) -> OutputType:
        """
        Process the webhook trigger and emit the event data.
        
        The input fields (method, path, headers, etc.) are expected to be
        populated by the workflow runner when a webhook request is received.
        """
        from nodetool.metadata.types import Datetime
        from datetime import datetime
        
        event = WebhookEvent(
            timestamp=Datetime.from_datetime(datetime.now()),
            method=self.method,
            path=self.path,
            headers=self.headers,
            query_params=self.query_params,
            body=self.body,
            content_type=self.content_type,
        )
        
        return {"event": event}
