"""
Observability Module for NodeTool.

This module provides comprehensive observability features including:
- Distributed tracing for workflow execution
- API call tracing
- WebSocket activity tracing
- Node execution tracing
- Provider execution tracing with cost tracking
- Agent execution tracing

All tracing is implemented via Python context managers for unobtrusive
instrumentation. See OBSERVABILITY.md for full documentation.

Example:
    from nodetool.observability import trace_workflow, trace_node

    async with trace_workflow(job_id="job-123") as span:
        async with trace_node("node-1", "ImageGenerate") as node_span:
            result = await node.process(inputs)
"""

from .tracing import (
    NoOpSpan,
    NoOpTracer,
    Span,
    SpanContext,
    SpanKind,
    SpanStatus,
    TraceContext,
    TracingConfig,
    WorkflowTracer,
    configure_tracing,
    get_current_trace_context,
    get_or_create_tracer,
    get_tracer,
    get_tracing_config,
    init_tracing,
    is_tracing_enabled,
    record_cost,
    remove_tracer,
    set_current_trace_context,
    set_response_attributes,
    trace_agent_task,
    trace_api_call,
    trace_node,
    trace_provider_call,
    trace_sync,
    trace_websocket_message,
    trace_workflow,
)

__all__ = [
    "NoOpSpan",
    "NoOpTracer",
    "Span",
    "SpanContext",
    "SpanKind",
    "SpanStatus",
    "TraceContext",
    "TracingConfig",
    "WorkflowTracer",
    "configure_tracing",
    "get_current_trace_context",
    "get_or_create_tracer",
    "get_tracer",
    "get_tracing_config",
    "init_tracing",
    "is_tracing_enabled",
    "record_cost",
    "remove_tracer",
    "set_current_trace_context",
    "set_response_attributes",
    "trace_agent_task",
    "trace_api_call",
    "trace_node",
    "trace_provider_call",
    "trace_sync",
    "trace_websocket_message",
    "trace_workflow",
]
