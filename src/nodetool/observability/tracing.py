"""
OpenTelemetry Tracing Integration for NodeTool.

This module provides distributed tracing capabilities for:
- Workflow execution (job lifecycle)
- Node execution (individual node processing)
- WebSocket activity (bidirectional messages)
- Agent execution (LLM agent planning and tool execution)

Note: HTTP/API calls and AI provider calls are automatically instrumented
by OpenTelemetry auto-instrumentation and Traceloop/OpenLLMetry.

All tracing is implemented via Python context managers for unobtrusive
instrumentation that doesn't modify business logic.

Usage:
    from nodetool.observability.tracing import (
        init_tracing,
        trace_workflow,
        trace_node,
        trace_websocket_message,
        trace_agent_task,
    )

    # Initialize tracing (also initializes Traceloop for AI provider instrumentation)
    init_tracing(service_name="nodetool-worker", exporter="otlp")

    # Trace workflow execution
    async with trace_workflow(job_id="job-123") as span:
        await runner.run(req, context)

See OBSERVABILITY.md for full documentation.
"""

import contextvars
import os
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from nodetool.config.env_guard import get_system_env_value
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

# Context variable for trace context propagation across async boundaries
_current_trace_context: contextvars.ContextVar[Optional["TraceContext"]] = contextvars.ContextVar(
    "current_trace_context", default=None
)


def _truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length for span attributes.

    Args:
        text: The text to truncate
        max_length: Maximum length (default: 200)

    Returns:
        Truncated text if longer than max_length, otherwise original text
    """
    return text[:max_length] if len(text) > max_length else text


class SpanKind(str, Enum):
    """OpenTelemetry-compatible span kinds."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """OpenTelemetry-compatible span status codes."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class TraceContext:
    """Context for trace propagation across async boundaries."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    def child_context(self, new_span_id: str) -> "TraceContext":
        """Create a child context with this span as parent."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=new_span_id,
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
        )


def _generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return uuid.uuid4().hex


def _generate_span_id() -> str:
    """Generate a unique span ID."""
    return uuid.uuid4().hex[:16]


def get_current_trace_context() -> TraceContext | None:
    """Get the current trace context from context vars."""
    return _current_trace_context.get()


def set_current_trace_context(ctx: TraceContext | None) -> contextvars.Token:
    """Set the current trace context."""
    return _current_trace_context.set(ctx)


@dataclass
class SpanContext:
    """Context information for a tracing span."""

    trace_id: str
    span_id: str
    parent_id: str | None
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: SpanStatus = SpanStatus.UNSET
    status_description: str | None = None
    kind: SpanKind = SpanKind.INTERNAL
    name: str = ""


class Span:
    """Represents a single trace span.

    A span represents a unit of work or operation. It tracks operations
    and their timing, and can include attributes, events, and status.

    Attributes:
        name: The operation name
        tracer: The parent tracer managing this span
        parent: Optional parent span for hierarchical tracing
        context: SpanContext containing trace data
    """

    def __init__(
        self,
        name: str,
        tracer: "WorkflowTracer",
        parent: Optional["Span"] = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ):
        self.name = name
        self.tracer = tracer
        self.parent = parent
        self.context = SpanContext(
            trace_id=tracer.trace_id,
            span_id=_generate_span_id(),
            parent_id=parent.context.span_id if parent else None,
            start_time=time.time(),
            kind=kind,
            name=name,
        )
        self._children: list[Span] = []

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.context.end_time is not None:
            return (self.context.end_time - self.context.start_time) * 1000
        return None

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set an attribute on this span.

        Args:
            key: Attribute name (use dot notation for namespacing)
            value: Attribute value (should be JSON-serializable)

        Returns:
            Self for method chaining
        """
        self.context.attributes[key] = value
        log.debug(f"Span attribute set: {self.name}.{key}={value}")
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> "Span":
        """Set multiple attributes on this span.

        Args:
            attributes: Dictionary of attribute key-value pairs

        Returns:
            Self for method chaining
        """
        for key, value in attributes.items():
            self.set_attribute(key, value)
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> "Span":
        """Add an event to this span.

        Events are time-stamped annotations that can have attributes.

        Args:
            name: Event name
            attributes: Optional event attributes

        Returns:
            Self for method chaining
        """
        self.context.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )
        log.debug(f"Span event added: {self.name}::{name}")
        return self

    def set_status(self, status: SpanStatus | str, description: str | None = None) -> "Span":
        """Set the span status.

        Args:
            status: Status code (ok, error, or unset)
            description: Optional description for error status

        Returns:
            Self for method chaining
        """
        if isinstance(status, str):
            status = SpanStatus(status.lower())
        self.context.status = status
        self.context.status_description = description
        if description:
            self.set_attribute("status_description", description)
        return self

    def record_exception(self, exception: Exception) -> "Span":
        """Record an exception in this span.

        Args:
            exception: The exception to record

        Returns:
            Self for method chaining
        """
        self.set_status(SpanStatus.ERROR, str(exception))
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )
        return self

    def end(self) -> None:
        """End this span and record duration."""
        self.context.end_time = time.time()
        duration_ms = (self.context.end_time - self.context.start_time) * 1000
        self.set_attribute("duration_ms", duration_ms)
        log.debug(f"Span ended: {self.name} ({duration_ms:.2f}ms)")

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type:
            self.record_exception(exc_val)
        elif self.context.status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)
        self.end()


class WorkflowTracer:
    """
    Distributed tracer for NodeTool workflows.

    Provides OpenTelemetry-compatible tracing for workflow execution,
    enabling distributed tracing across multiple nodes and services.

    The tracer manages span lifecycle, context propagation, and provides
    statistics about traced operations.

    Attributes:
        job_id: The workflow job identifier
        trace_id: Unique trace identifier
        enabled: Whether tracing is active
    """

    def __init__(self, job_id: str, trace_id: str | None = None):
        self.job_id = job_id
        self.trace_id = trace_id or _generate_trace_id()
        self._spans: list[Span] = []
        self._current_span: Span | None = None
        self._span_stack: list[Span] = []
        self._enabled = True
        self._total_cost: float = 0.0

    @property
    def active_span(self) -> Span | None:
        """Get the currently active span."""
        return self._current_span

    @property
    def total_cost(self) -> float:
        """Get total cost tracked across all spans."""
        return self._total_cost

    def add_cost(self, cost: float) -> None:
        """Add to the total tracked cost."""
        self._total_cost += cost

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent: Span | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> AsyncGenerator[Span, None]:
        """Start a new span as a child of the active span.

        Args:
            name: Span name (use dot notation like "workflow.execute")
            attributes: Initial span attributes
            parent: Optional explicit parent span
            kind: Span kind (internal, server, client, etc.)

        Yields:
            The created span for adding attributes and events
        """
        if not self._enabled:
            yield Span(name, self)
            return

        parent_span = parent or self._current_span
        span = Span(name, self, parent=parent_span, kind=kind)

        if attributes:
            span.set_attributes(attributes)

        self._span_stack.append(span)
        self._current_span = span
        self._spans.append(span)

        # Track parent-child relationship
        if parent_span:
            parent_span._children.append(span)

        span.add_event("span_started", {"span_name": name})

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            if span.context.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
            span.end()
            self._span_stack.pop()
            self._current_span = self._span_stack[-1] if self._span_stack else None

    @contextmanager
    def start_span_sync(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent: Span | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Generator[Span, None, None]:
        """Start a new span synchronously (for non-async code).

        Args:
            name: Span name
            attributes: Initial span attributes
            parent: Optional explicit parent span
            kind: Span kind

        Yields:
            The created span
        """
        if not self._enabled:
            yield Span(name, self)
            return

        parent_span = parent or self._current_span
        span = Span(name, self, parent=parent_span, kind=kind)

        if attributes:
            span.set_attributes(attributes)

        self._span_stack.append(span)
        self._current_span = span
        self._spans.append(span)

        if parent_span:
            parent_span._children.append(span)

        span.add_event("span_started", {"span_name": name})

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            if span.context.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
            span.end()
            self._span_stack.pop()
            self._current_span = self._span_stack[-1] if self._span_stack else None

    def record_exception(self, exception: Exception, span: Span | None = None) -> None:
        """Record an exception in the current or specified span."""
        target_span = span or self._current_span
        if target_span:
            target_span.record_exception(exception)

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
                "duration_ms": span.duration_ms,
                "status": span.context.status.value,
                "kind": span.context.kind.value,
                "attributes": span.context.attributes,
                "events": span.context.events,
                "children": [span_to_dict(child, depth + 1) for child in span._children],
            }

        root_spans = [s for s in self._spans if s.parent is None]
        return {
            "trace_id": self.trace_id,
            "job_id": self.job_id,
            "total_spans": len(self._spans),
            "total_cost": self._total_cost,
            "root_spans": [span_to_dict(s) for s in root_spans],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get trace statistics."""
        if not self._spans:
            return {"total_spans": 0}

        durations = [s.duration_ms for s in self._spans if s.duration_ms is not None]

        return {
            "total_spans": len(self._spans),
            "total_duration_ms": max(durations) if durations else 0,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "span_count_by_name": self._count_spans_by_name(),
            "error_count": sum(1 for s in self._spans if s.context.status == SpanStatus.ERROR),
            "total_cost": self._total_cost,
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
    """A no-op span for when tracing is disabled.

    Implements the same interface as Span but does nothing,
    allowing instrumented code to work without overhead when tracing is off.
    """

    def __init__(self, name: str):
        self.name = name
        self.context = SpanContext(
            trace_id="no-op",
            span_id="no-op",
            parent_id=None,
            start_time=0,
            name=name,
        )

    def set_attribute(self, key: str, value: Any) -> "NoOpSpan":
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> "NoOpSpan":
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> "NoOpSpan":
        return self

    def set_status(self, status: SpanStatus | str, description: str | None = None) -> "NoOpSpan":
        return self

    def record_exception(self, exception: Exception) -> "NoOpSpan":
        return self

    def end(self) -> None:
        pass

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class NoOpTracer:
    """A no-op tracer for when tracing is disabled.

    Implements the same interface as WorkflowTracer but does nothing,
    allowing instrumented code to work without overhead when tracing is off.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.trace_id = f"no-op-{job_id}"
        self._enabled = False
        self._total_cost: float = 0.0

    @property
    def active_span(self) -> None:
        return None

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def add_cost(self, cost: float) -> None:
        pass

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent: Span | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> AsyncGenerator[NoOpSpan, None]:
        yield NoOpSpan(name)

    @contextmanager
    def start_span_sync(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent: Span | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Generator[NoOpSpan, None, None]:
        yield NoOpSpan(name)

    def record_exception(self, exception: Exception, span: Span | None = None) -> None:
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


def _is_auto_instrumentation_active() -> bool:
    """Check if OpenTelemetry auto-instrumentation is active via environment variables.

    Auto-instrumentation is considered active when standard OTEL environment variables
    are set, indicating opentelemetry-instrument is being used to run the application.
    """
    auto_otel_vars = [
        "OTEL_SERVICE_NAME",
        "OTEL_TRACES_EXPORTER",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
    ]
    return any(get_system_env_value(var) for var in auto_otel_vars)


def _is_truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _should_initialize_traceloop(exporter: str | None) -> bool:
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
    exporter: str | None = None,
    endpoint: str | None = None,
) -> None:
    """
    Initialize OpenTelemetry tracing and OpenLLMetry instrumentation.

    This function handles both manual instrumentation (Traceloop/OpenLLMetry SDK)
    and respects OpenTelemetry auto-instrumentation when available.

    Auto-instrumentation detection:
    - If OTEL_* environment variables are set, auto-instrumentation is active
    - In this case, we skip SDK initialization and let auto-instrumentation handle traces
    - Manual context managers still work but create local spans alongside auto traces

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

    auto_instrumentation_active = _is_auto_instrumentation_active()

    if auto_instrumentation_active:
        log.info(
            "OpenTelemetry auto-instrumentation detected via environment variables. "
            "Manual SDK initialization skipped. Auto-instrumentation will handle HTTP/WS traces."
        )
        if resolved_exporter:
            log.info(f"Configured exporter '{resolved_exporter}' will be used by auto-instrumentation")
        return

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TracingConfig:
    """Configuration for the tracing system.

    Attributes:
        enabled: Whether tracing is enabled globally
        exporter: Exporter type (otlp, console, jaeger, none)
        endpoint: OTLP endpoint URL
        service_name: Service name for trace attribution
        service_version: Service version
        sample_rate: Sampling rate (0.0 to 1.0)
        batch_size: Export batch size
        export_interval_ms: Export interval in milliseconds
    """

    enabled: bool = True
    exporter: str = "console"
    endpoint: str | None = None
    service_name: str = "nodetool"
    service_version: str = "0.6.0"
    sample_rate: float = 1.0
    batch_size: int = 512
    export_interval_ms: int = 5000


_tracing_config: TracingConfig = TracingConfig()


def configure_tracing(config: TracingConfig) -> None:
    """Configure the tracing system with the given config.

    Args:
        config: TracingConfig instance
    """
    global _tracing_config, _tracing_initialized
    _tracing_config = config
    _tracing_initialized = config.enabled
    log.info(f"Tracing configured: enabled={config.enabled}, exporter={config.exporter}")


def get_tracing_config() -> TracingConfig:
    """Get the current tracing configuration."""
    return _tracing_config


# ---------------------------------------------------------------------------
# Global Tracer Management
# ---------------------------------------------------------------------------

_global_tracers: dict[str, WorkflowTracer] = {}


def get_or_create_tracer(job_id: str) -> WorkflowTracer | NoOpTracer:
    """Get or create a tracer for a job, managing lifecycle globally.

    Args:
        job_id: The job identifier

    Returns:
        WorkflowTracer or NoOpTracer based on configuration
    """
    if not _tracing_config.enabled:
        return NoOpTracer(job_id)

    if job_id not in _global_tracers:
        _global_tracers[job_id] = WorkflowTracer(job_id)

    return _global_tracers[job_id]


def remove_tracer(job_id: str) -> WorkflowTracer | None:
    """Remove and return a tracer from global management.

    Args:
        job_id: The job identifier

    Returns:
        The removed tracer or None if not found
    """
    tracer = _global_tracers.pop(job_id, None)
    if tracer and _tracing_config.exporter == "console":
        _export_traces_to_console(tracer)
    return tracer


def _export_traces_to_console(tracer: WorkflowTracer) -> None:
    """Export traces to console for the console exporter."""
    import json
    from datetime import datetime

    trace_tree = tracer.get_trace_tree()
    console_output = {
        "timestamp": datetime.now().isoformat(),
        "trace_id": trace_tree["trace_id"],
        "job_id": trace_tree["job_id"],
        "total_spans": trace_tree["total_spans"],
        "total_cost": trace_tree["total_cost"],
        "spans": trace_tree["root_spans"],
    }

    print("\n" + "=" * 80)
    print("TRACES (console exporter)")
    print("=" * 80)
    print(json.dumps(console_output, indent=2, default=str))
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Convenience Context Managers for Common Tracing Scenarios
# ---------------------------------------------------------------------------


@asynccontextmanager
async def trace_websocket_message(
    command: str,
    direction: Literal["inbound", "outbound"],
    *,
    job_id: str | None = None,
    thread_id: str | None = None,
    tracer: WorkflowTracer | NoOpTracer | None = None,
) -> AsyncGenerator[Span | NoOpSpan, None]:
    """Trace a WebSocket message.

    Usage:
        async with trace_websocket_message("run_job", "inbound", job_id="123") as span:
            await handle_command(command, data)

    Args:
        command: Command type being processed
        direction: Message direction (inbound/outbound)
        job_id: Associated job ID (for workflow commands)
        thread_id: Associated thread ID (for chat commands)
        tracer: Optional tracer (uses job tracer if job_id provided)

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled:
        yield NoOpSpan("websocket.message")
        return

    # Get tracer from job_id or create temporary
    if job_id and tracer is None:
        _tracer = get_or_create_tracer(job_id)
    else:
        _tracer = tracer or WorkflowTracer(f"ws-{_generate_span_id()}")

    kind = SpanKind.CONSUMER if direction == "inbound" else SpanKind.PRODUCER

    async with _tracer.start_span(
        f"websocket.{direction}",
        attributes={
            "websocket.command": command,
            "websocket.direction": direction,
            "websocket.job_id": job_id,
            "websocket.thread_id": thread_id,
        },
        kind=kind,
    ) as span:
        yield span


@asynccontextmanager
async def trace_workflow(
    job_id: str,
    workflow_id: str | None = None,
    *,
    user_id: str | None = None,
    tracer: WorkflowTracer | NoOpTracer | None = None,
) -> AsyncGenerator[Span | NoOpSpan, None]:
    """Trace workflow execution lifecycle.

    Usage:
        async with trace_workflow(job_id="job-123", workflow_id="wf-456") as span:
            span.set_attribute("node_count", 5)
            await runner.run(req, context)

    Args:
        job_id: Unique job identifier
        workflow_id: Optional workflow ID
        user_id: Optional user ID for attribution
        tracer: Optional tracer (creates one for the job if not provided)

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled:
        yield NoOpSpan("workflow.execute")
        return

    _tracer = tracer or get_or_create_tracer(job_id)

    async with _tracer.start_span(
        "workflow.execute",
        attributes={
            "nodetool.workflow.job_id": job_id,
            "nodetool.workflow.id": workflow_id,
            "nodetool.workflow.user_id": user_id,
        },
        kind=SpanKind.INTERNAL,
    ) as span:
        yield span


@asynccontextmanager
async def trace_node(
    node_id: str,
    node_type: str,
    *,
    job_id: str | None = None,
    tracer: WorkflowTracer | NoOpTracer | None = None,
) -> AsyncGenerator[Span | NoOpSpan, None]:
    """Trace individual node execution.

    Usage:
        async with trace_node("node-123", "ImageGenerate") as span:
            span.set_attribute("inputs", list(inputs.keys()))
            result = await node.process(inputs)

    Args:
        node_id: Node identifier
        node_type: Node class type
        job_id: Optional job ID to link to workflow tracer
        tracer: Optional tracer (uses job tracer if job_id provided)

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled:
        yield NoOpSpan("workflow.node")
        return

    _tracer = get_or_create_tracer(job_id) if job_id and tracer is None else tracer or WorkflowTracer(f"node-{node_id}")

    async with _tracer.start_span(
        "workflow.node",
        attributes={
            "nodetool.node.id": node_id,
            "nodetool.node.type": node_type,
        },
        kind=SpanKind.INTERNAL,
    ) as span:
        yield span


@asynccontextmanager
async def trace_agent_task(
    agent_type: str,
    task_description: str,
    *,
    job_id: str | None = None,
    tools: list[str] | None = None,
    tracer: WorkflowTracer | NoOpTracer | None = None,
) -> AsyncGenerator[Span | NoOpSpan, None]:
    """Trace agent task execution.

    Usage:
        async with trace_agent_task("cot", "Generate image") as span:
            span.set_attribute("tools_used", ["browser", "image_gen"])
            result = await agent.execute(task)

    Args:
        agent_type: Type of agent (cot, simple, etc.)
        task_description: Brief task description (will be truncated)
        job_id: Optional job ID to link to workflow tracer
        tools: List of available tool names
        tracer: Optional tracer

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled:
        yield NoOpSpan("agent.task")
        return

    if job_id and tracer is None:
        _tracer = get_or_create_tracer(job_id)
    else:
        _tracer = tracer or WorkflowTracer(f"agent-{_generate_span_id()}")

    # Truncate task description for span attribute
    truncated_desc = task_description[:200] if len(task_description) > 200 else task_description

    async with _tracer.start_span(
        "agent.task",
        attributes={
            "nodetool.agent.type": agent_type,
            "nodetool.agent.task_description": truncated_desc,
            "nodetool.agent.available_tools": tools or [],
        },
        kind=SpanKind.INTERNAL,
    ) as span:
        yield span


@asynccontextmanager
async def trace_tool_execution(
    tool_name: str,
    *,
    job_id: str | None = None,
    step_id: str | None = None,
    params: dict[str, Any] | None = None,
    tracer: WorkflowTracer | NoOpTracer | None = None,
) -> AsyncGenerator[Span | NoOpSpan, None]:
    """Trace tool execution in an agent workflow.

    Usage:
        async with trace_tool_execution("browser", params={"url": "..."}) as span:
            result = await tool.process(context, params)

    Args:
        tool_name: Name of the tool being executed
        job_id: Optional job ID to link to workflow tracer
        step_id: Optional step ID for attribution
        params: Optional tool parameters (keys only for privacy)
        tracer: Optional tracer

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled:
        yield NoOpSpan("tool.execute")
        return

    if job_id and tracer is None:
        _tracer = get_or_create_tracer(job_id)
    else:
        _tracer = tracer or WorkflowTracer(f"tool-{_generate_span_id()}")

    async with _tracer.start_span(
        "tool.execute",
        attributes={
            "nodetool.tool.name": tool_name,
            "nodetool.tool.step_id": step_id,
            "nodetool.tool.param_keys": list(params.keys()) if params else [],
        },
        kind=SpanKind.INTERNAL,
    ) as span:
        yield span


@asynccontextmanager
async def trace_task_planning(
    objective: str,
    *,
    job_id: str | None = None,
    model: str | None = None,
    tracer: WorkflowTracer | NoOpTracer | None = None,
) -> AsyncGenerator[Span | NoOpSpan, None]:
    """Trace agent task planning phase.

    Usage:
        async with trace_task_planning("Generate image from description") as span:
            plan = await planner.create_task(context, objective)

    Args:
        objective: The objective being planned (will be truncated)
        job_id: Optional job ID to link to workflow tracer
        model: Optional model used for planning
        tracer: Optional tracer

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled:
        yield NoOpSpan("agent.planning")
        return

    if job_id and tracer is None:
        _tracer = get_or_create_tracer(job_id)
    else:
        _tracer = tracer or WorkflowTracer(f"planning-{_generate_span_id()}")

    async with _tracer.start_span(
        "agent.planning",
        attributes={
            "nodetool.planning.objective": _truncate_text(objective),
            "nodetool.planning.model": model,
        },
        kind=SpanKind.INTERNAL,
    ) as span:
        yield span


@asynccontextmanager
async def trace_task_execution(
    task_id: str,
    task_title: str,
    *,
    job_id: str | None = None,
    step_count: int | None = None,
    tracer: WorkflowTracer | NoOpTracer | None = None,
) -> AsyncGenerator[Span | NoOpSpan, None]:
    """Trace agent task execution.

    Usage:
        async with trace_task_execution("task_123", "Process files") as span:
            async for result in executor.execute_tasks(context):
                yield result

    Args:
        task_id: Unique task identifier
        task_title: Task title/description
        job_id: Optional job ID to link to workflow tracer
        step_count: Number of steps in the task
        tracer: Optional tracer

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled:
        yield NoOpSpan("agent.task_execution")
        return

    if job_id and tracer is None:
        _tracer = get_or_create_tracer(job_id)
    else:
        _tracer = tracer or WorkflowTracer(f"task-{_generate_span_id()}")

    async with _tracer.start_span(
        "agent.task_execution",
        attributes={
            "nodetool.task.id": task_id,
            "nodetool.task.title": task_title,
            "nodetool.task.step_count": step_count,
        },
        kind=SpanKind.INTERNAL,
    ) as span:
        yield span


@asynccontextmanager
async def trace_step_execution(
    step_id: str,
    step_instructions: str,
    *,
    job_id: str | None = None,
    task_id: str | None = None,
    tracer: WorkflowTracer | NoOpTracer | None = None,
) -> AsyncGenerator[Span | NoOpSpan, None]:
    """Trace agent step execution.

    Usage:
        async with trace_step_execution("step_1", "Search for documents") as span:
            async for item in step_executor.execute():
                yield item

    Args:
        step_id: Unique step identifier
        step_instructions: Step instructions (will be truncated)
        job_id: Optional job ID to link to workflow tracer
        task_id: Optional parent task ID
        tracer: Optional tracer

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled:
        yield NoOpSpan("agent.step_execution")
        return

    if job_id and tracer is None:
        _tracer = get_or_create_tracer(job_id)
    else:
        _tracer = tracer or WorkflowTracer(f"step-{_generate_span_id()}")

    async with _tracer.start_span(
        "agent.step_execution",
        attributes={
            "nodetool.step.id": step_id,
            "nodetool.step.instructions": _truncate_text(step_instructions),
            "nodetool.step.task_id": task_id,
        },
        kind=SpanKind.INTERNAL,
    ) as span:
        yield span


# ---------------------------------------------------------------------------
# Synchronous Context Managers
# ---------------------------------------------------------------------------


@contextmanager
def trace_sync(
    name: str,
    *,
    tracer: WorkflowTracer | NoOpTracer | None = None,
    attributes: dict[str, Any] | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Generator[Span | NoOpSpan, None, None]:
    """Trace a synchronous operation.

    Usage:
        with trace_sync("process_data", attributes={"size": 100}) as span:
            process_data(data)

    Args:
        name: Span name
        tracer: Optional tracer
        attributes: Initial span attributes
        kind: Span kind

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled or tracer is None:
        yield NoOpSpan(name)
        return

    with tracer.start_span_sync(name, attributes=attributes, kind=kind) as span:
        yield span
