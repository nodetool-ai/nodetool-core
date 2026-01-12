"""
Observability Module for NodeTool.

This module provides observability features including:
- Distributed tracing for workflow execution
- Performance profiling
- Event logging and metrics

Submodules:
- tracing: OpenTelemetry-compatible distributed tracing
"""

from .tracing import (
    Span,
    WorkflowTracer,
    get_tracer,
    init_tracing,
    is_tracing_enabled,
)

__all__ = [
    "Span",
    "WorkflowTracer",
    "get_tracer",
    "init_tracing",
    "is_tracing_enabled",
]
