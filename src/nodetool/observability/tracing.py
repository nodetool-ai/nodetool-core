"""
OpenTelemetry Tracing Integration for NodeTool Workflows.

This module provides distributed tracing capabilities for workflow execution,
enabling better observability, debugging, and performance analysis.

Usage:
    from nodetool.observability.tracing import WorkflowTracer, init_tracing

    # Initialize tracing (optional)
    init_tracing(service_name="nodetool-worker")

    # Create tracer for a workflow run
    tracer = WorkflowTracer(job_id="job-123")
    async with tracer.start_span("execute_workflow") as span:
        # Run workflow with tracing
        await runner.run(req, context, tracer=tracer)
"""

import contextvars
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from nodetool.config.env_guard import get_system_env_value
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class SpanContext:
    """Context information for a tracing span."""

    trace_id: str
    span_id: str
    parent_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "unset"


class Span:
    """Represents a single trace span."""

    def __init__(self, name: str, tracer: "WorkflowTracer", parent: Optional["Span"] = None):
        self.name = name
        self.tracer = tracer
        self.parent = parent
        self.context = SpanContext(
            trace_id=tracer.trace_id,
            span_id=f"span-{len(tracer._spans)}",
            parent_id=parent.context.span_id if parent else None,
            start_time=time.time(),
        )
        self._children: list[Span] = []

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on this span."""
        self.context.attributes[key] = value
        log.debug(f"Span attribute set: {self.name}.{key}={value}")

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Add an event to this span."""
        self.context.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )
        log.debug(f"Span event added: {self.name}::{name}")

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Set the span status."""
        self.context.status = status
        if description:
            self.set_attribute("status_description", description)

    def end(self) -> None:
        """End this span."""
        self.context.end_time = time.time()
        duration_ms = (self.context.end_time - self.context.start_time) * 1000
        self.set_attribute("duration_ms", duration_ms)
        log.debug(f"Span ended: {self.name} ({duration_ms:.2f}ms)")

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            self.set_status("error", str(exc_val))
            self.add_event(
                "exception",
                {
                    "type": exc_type.__name__,
                    "message": str(exc_val),
                },
            )
        self.end()


class WorkflowTracer:
    """
    Distributed tracer for NodeTool workflows.

    Provides OpenTelemetry-compatible tracing for workflow execution,
    enabling distributed tracing across multiple nodes and services.
    """

    def __init__(self, job_id: str, trace_id: Optional[str] = None):
        self.job_id = job_id
        self.trace_id = trace_id or f"trace-{job_id}-{int(time.time() * 1000)}"
        self._spans: list[Span] = []
        self._current_span: Optional[Span] = None
        self._span_stack: list[Span] = []
        self._enabled = True

    @property
    def active_span(self) -> Optional[Span]:
        """Get the currently active span."""
        return self._current_span

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        parent: Optional[Span] = None,
    ) -> AsyncGenerator[Span, None]:
        """Start a new span as a child of the active span."""
        if not self._enabled:
            yield Span(name, self)
            return

        parent_span = parent or self._current_span
        span = Span(name, self, parent=parent_span)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        self._span_stack.append(span)
        self._current_span = span
        self._spans.append(span)

        span.add_event("span_started", {"span_name": name})

        try:
            yield span
        except Exception as e:
            span.set_status("error", str(e))
            span.add_event(
                "exception",
                {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            )
            raise
        finally:
            span.end()
            self._span_stack.pop()
            self._current_span = self._span_stack[-1] if self._span_stack else None

    def record_exception(self, exception: Exception, span: Optional[Span] = None) -> None:
        """Record an exception in the current or specified span."""
        target_span = span or self._current_span
        if target_span:
            target_span.set_status("error", str(exception))
            target_span.add_event(
                "exception",
                {
                    "type": type(exception).__name__,
                    "message": str(exception),
                },
            )

    def get_trace_tree(self) -> dict[str, Any]:
        """Get the trace tree as a dictionary for export/visualization."""

        def span_to_dict(span: Span, depth: int = 0) -> dict[str, Any]:
            return {
                "name": span.name,
                "span_id": span.context.span_id,
                "parent_id": span.context.parent_id,
                "trace_id": span.context.trace_id,
                "start_time": span.context.start_time,
                "end_time": span.context.end_time,
                "duration_ms": (span.context.end_time - span.context.start_time) * 1000
                if span.context.end_time
                else None,
                "status": span.context.status,
                "attributes": span.context.attributes,
                "events": span.context.events,
                "children": [span_to_dict(child, depth + 1) for child in span._children],
            }

        root_spans = [s for s in self._spans if s.parent is None]
        return {
            "trace_id": self.trace_id,
            "job_id": self.job_id,
            "total_spans": len(self._spans),
            "root_spans": [span_to_dict(s) for s in root_spans],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get trace statistics."""
        if not self._spans:
            return {"total_spans": 0}

        durations = [(s.context.end_time - s.context.start_time) * 1000 for s in self._spans if s.context.end_time]

        return {
            "total_spans": len(self._spans),
            "total_duration_ms": max(durations) if durations else 0,
            "span_count_by_name": self._count_spans_by_name(),
            "error_count": sum(1 for s in self._spans if s.context.status == "error"),
        }

    def _count_spans_by_name(self) -> dict[str, int]:
        """Count spans by name."""
        counts: dict[str, int] = {}
        for span in self._spans:
            counts[span.name] = counts.get(span.name, 0) + 1
        return counts

    def disable(self) -> None:
        """Disable tracing."""
        self._enabled = False

    def enable(self) -> None:
        """Enable tracing."""
        self._enabled = True


class NoOpSpan:
    """A no-op span for when tracing is disabled."""

    def __init__(self, name: str):
        self.name = name

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        pass

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class NoOpTracer:
    """A no-op tracer for when tracing is disabled."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.trace_id = f"no-op-{job_id}"

    @property
    def active_span(self) -> None:
        return None

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        parent: Optional[Span] = None,
    ) -> AsyncGenerator[NoOpSpan, None]:
        yield NoOpSpan(name)

    def record_exception(self, exception: Exception, span: Optional[Span] = None) -> None:
        pass

    def get_trace_tree(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "job_id": self.job_id, "spans": []}

    def get_statistics(self) -> dict[str, Any]:
        return {"total_spans": 0, "disabled": True}

    def disable(self) -> None:
        pass

    def enable(self) -> None:
        pass


_tracing_initialized = False
_global_tracer: Optional["WorkflowTracer"] = None


@dataclass
class TracingConfig:
    enabled: bool = True
    exporter: Optional[str] = None
    service_version: Optional[str] = None


_tracing_config = TracingConfig()


def _is_truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _should_initialize_traceloop(exporter: Optional[str]) -> bool:
    if exporter in {"traceloop", "openllmetry"}:
        return True
    enabled_value = get_system_env_value("TRACELOOP_ENABLED")
    if enabled_value is not None and not _is_truthy(enabled_value):
        return False
    if _is_truthy(enabled_value or "true"):
        return True
    return any(
        get_system_env_value(key)
        for key in (
            "TRACELOOP_API_KEY",
            "TRACELOOP_BASE_URL",
            "TRACELOOP_HEADERS",
        )
    )


def init_tracing(
    service_name: str = "nodetool",
    exporter: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> None:
    """
    Initialize OpenTelemetry tracing and OpenLLMetry instrumentation.

    Args:
        service_name: Name of the service for trace attribution
        exporter: Optional exporter type ("traceloop", "otlp", "console", "jaeger")
        endpoint: Optional endpoint for the exporter
    """
    global _tracing_initialized

    if _tracing_initialized:
        log.warning("Tracing already initialized")
        return

    if not _tracing_config.enabled:
        log.info("Tracing disabled by configuration")
        return

    resolved_exporter = exporter or get_system_env_value("NODETOOL_TRACING_EXPORTER") or _tracing_config.exporter

    _tracing_initialized = True
    log.info(f"Tracing initialized for service: {service_name}")

    if resolved_exporter:
        log.info(f"Tracing exporter configured: {resolved_exporter}")
        if endpoint:
            log.info(f"Tracing endpoint: {endpoint}")

    if not _should_initialize_traceloop(resolved_exporter):
        log.info("Traceloop OpenLLMetry not configured; skipping SDK initialization")
        return

    if endpoint and not get_system_env_value("TRACELOOP_BASE_URL"):
        os.environ["TRACELOOP_BASE_URL"] = endpoint

    try:
        from traceloop.sdk import Traceloop
    except ImportError:
        log.warning("traceloop-sdk not installed; skipping OpenLLMetry initialization")
        return

    app_name = get_system_env_value("TRACELOOP_APP_NAME") or get_system_env_value("OTEL_SERVICE_NAME") or service_name
    env_name = get_system_env_value("ENV")
    service_version = get_system_env_value("OTEL_SERVICE_VERSION", _tracing_config.service_version)

    resource_attributes: dict[str, str] = {}
    if env_name:
        resource_attributes["deployment.environment"] = env_name
    if service_version:
        resource_attributes["service.version"] = str(service_version)

    init_kwargs: dict[str, Any] = {"app_name": app_name}
    if resource_attributes:
        init_kwargs["resource_attributes"] = resource_attributes
    if _is_truthy(get_system_env_value("TRACELOOP_DISABLE_BATCH")):
        init_kwargs["disable_batch"] = True

    if resolved_exporter == "console":
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            init_kwargs["exporter"] = ConsoleSpanExporter()
        except ImportError:
            log.warning("opentelemetry-sdk not installed; unable to set ConsoleSpanExporter")

    Traceloop.init(**init_kwargs)
    log.info("Traceloop OpenLLMetry initialized")


def get_tracer(job_id: str, enabled: bool = True) -> WorkflowTracer | NoOpTracer:
    """
    Get a tracer for a specific job.

    Args:
        job_id: The workflow job identifier
        enabled: Whether tracing is enabled

    Returns:
        A WorkflowTracer instance
    """
    if not enabled:
        return NoOpTracer(job_id)

    return WorkflowTracer(job_id)


def is_tracing_enabled() -> bool:
    """Check if tracing is globally enabled."""
    return _tracing_initialized
