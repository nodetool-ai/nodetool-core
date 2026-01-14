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
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator, Literal, Optional

# OpenTelemetry SDK imports
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import SpanKind as OtelSpanKind
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace import Tracer as OtelTracer

from nodetool.config.env_guard import get_system_env_value
from nodetool.config.logging_config import get_logger

if TYPE_CHECKING:
    from opentelemetry.trace import Span as OtelSpan

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

    def to_otel(self) -> OtelSpanKind:
        """Convert to OpenTelemetry SpanKind."""
        mapping = {
            SpanKind.INTERNAL: OtelSpanKind.INTERNAL,
            SpanKind.SERVER: OtelSpanKind.SERVER,
            SpanKind.CLIENT: OtelSpanKind.CLIENT,
            SpanKind.PRODUCER: OtelSpanKind.PRODUCER,
            SpanKind.CONSUMER: OtelSpanKind.CONSUMER,
        }
        return mapping.get(self, OtelSpanKind.INTERNAL)


class SpanStatus(str, Enum):
    """OpenTelemetry-compatible span status codes."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"

    def to_otel(self) -> StatusCode:
        """Convert to OpenTelemetry StatusCode."""
        mapping = {
            SpanStatus.UNSET: StatusCode.UNSET,
            SpanStatus.OK: StatusCode.OK,
            SpanStatus.ERROR: StatusCode.ERROR,
        }
        return mapping.get(self, StatusCode.UNSET)


@dataclass
class TraceContext:
    """Context for trace propagation across async boundaries."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
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


def get_current_trace_context() -> Optional[TraceContext]:
    """Get the current trace context from context vars."""
    return _current_trace_context.get()


def set_current_trace_context(ctx: Optional[TraceContext]) -> contextvars.Token:
    """Set the current trace context."""
    return _current_trace_context.set(ctx)


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
    status: SpanStatus = SpanStatus.UNSET
    status_description: Optional[str] = None
    kind: SpanKind = SpanKind.INTERNAL
    name: str = ""


class Span:
    """Represents a single trace span that bridges to OpenTelemetry.

    A span represents a unit of work or operation. It tracks operations
    and their timing, and can include attributes, events, and status.

    This class wraps an OpenTelemetry span when OTEL is available, providing
    a consistent interface while exporting spans to OTEL backends.

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
        otel_span: Optional["OtelSpan"] = None,
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
        # Store the underlying OTEL span if provided
        self._otel_span: Optional[OtelSpan] = otel_span

    @property
    def duration_ms(self) -> Optional[float]:
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
        # Also set on OTEL span if available
        if self._otel_span is not None:
            try:
                # OTEL requires specific types: str, bool, int, float, or sequences thereof
                otel_value = _convert_to_otel_attribute(value)
                if otel_value is not None:
                    self._otel_span.set_attribute(key, otel_value)
            except Exception as e:
                log.debug(f"Failed to set OTEL attribute {key}: {e}")
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

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> "Span":
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
        # Also add event to OTEL span if available
        if self._otel_span is not None:
            try:
                otel_attrs = {}
                if attributes:
                    for k, v in attributes.items():
                        otel_v = _convert_to_otel_attribute(v)
                        if otel_v is not None:
                            otel_attrs[k] = otel_v
                self._otel_span.add_event(name, otel_attrs)
            except Exception as e:
                log.debug(f"Failed to add OTEL event {name}: {e}")
        log.debug(f"Span event added: {self.name}::{name}")
        return self

    def set_status(self, status: SpanStatus | str, description: Optional[str] = None) -> "Span":
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
        # Also set status on OTEL span if available
        if self._otel_span is not None:
            try:
                self._otel_span.set_status(Status(status.to_otel(), description))
            except Exception as e:
                log.debug(f"Failed to set OTEL status: {e}")
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
        # Also record on OTEL span if available
        if self._otel_span is not None:
            try:
                self._otel_span.record_exception(exception)
            except Exception as e:
                log.debug(f"Failed to record OTEL exception: {e}")
        return self

    def end(self) -> None:
        """End this span and record duration."""
        self.context.end_time = time.time()
        duration_ms = (self.context.end_time - self.context.start_time) * 1000
        self.set_attribute("duration_ms", duration_ms)
        # End the OTEL span if available
        if self._otel_span is not None:
            try:
                self._otel_span.end()
            except Exception as e:
                log.debug(f"Failed to end OTEL span: {e}")
        log.debug(f"Span ended: {self.name} ({duration_ms:.2f}ms)")

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type:
            self.record_exception(exc_val)
        elif self.context.status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)
        self.end()


def _convert_to_otel_attribute(value: Any) -> Any:
    """Convert a value to an OTEL-compatible attribute type.

    OTEL accepts: str, bool, int, float, or sequences of these.

    Args:
        value: The value to convert

    Returns:
        OTEL-compatible value or None if conversion not possible
    """
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        # Convert sequences, filtering out None values
        converted = []
        for item in value:
            if isinstance(item, (str, bool, int, float)):
                converted.append(item)
            else:
                # Convert non-primitive items to strings
                converted.append(str(item))
        return converted
    # For other types, convert to string
    return str(value)


class WorkflowTracer:
    """
    Distributed tracer for NodeTool workflows that bridges to OpenTelemetry.

    Provides OpenTelemetry-compatible tracing for workflow execution,
    enabling distributed tracing across multiple nodes and services.

    When OTEL is configured, spans created by this tracer are automatically
    exported to the configured OTEL backend (console, OTLP, etc.).

    The tracer manages span lifecycle, context propagation, and provides
    statistics about traced operations.

    Attributes:
        job_id: The workflow job identifier
        trace_id: Unique trace identifier
        enabled: Whether tracing is active
    """

    def __init__(self, job_id: str, trace_id: Optional[str] = None):
        self.job_id = job_id
        self.trace_id = trace_id or _generate_trace_id()
        self._spans: list[Span] = []
        self._current_span: Optional[Span] = None
        self._span_stack: list[Span] = []
        self._otel_span_stack: list[OtelSpan] = []  # Stack of OTEL spans for context
        self._enabled = True
        self._total_cost: float = 0.0

    @property
    def active_span(self) -> Optional[Span]:
        """Get the currently active span."""
        return self._current_span

    @property
    def total_cost(self) -> float:
        """Get total cost tracked across all spans."""
        return self._total_cost

    def add_cost(self, cost: float) -> None:
        """Add to the total tracked cost."""
        self._total_cost += cost

    def _create_otel_span(
        self,
        name: str,
        kind: SpanKind,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Optional["OtelSpan"]:
        """Create an OTEL span if tracing is initialized.

        Args:
            name: Span name
            kind: Span kind
            attributes: Initial attributes

        Returns:
            OTEL span or None if OTEL is not available
        """
        otel_tracer = _get_otel_tracer()
        if otel_tracer is None:
            return None

        try:
            # Convert attributes to OTEL-compatible format
            otel_attrs = {}
            if attributes:
                for key, value in attributes.items():
                    otel_value = _convert_to_otel_attribute(value)
                    if otel_value is not None:
                        otel_attrs[key] = otel_value

            # Add job_id as a standard attribute
            otel_attrs["nodetool.job_id"] = self.job_id

            # Start the OTEL span
            otel_span = otel_tracer.start_span(
                name=name,
                kind=kind.to_otel(),
                attributes=otel_attrs,
            )
            return otel_span
        except Exception as e:
            log.debug(f"Failed to create OTEL span: {e}")
            return None

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        parent: Optional[Span] = None,
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

        # Create OTEL span if available
        otel_span = self._create_otel_span(name, kind, attributes)

        # Create our wrapper span with the OTEL span attached
        span = Span(name, self, parent=parent_span, kind=kind, otel_span=otel_span)

        if attributes:
            span.set_attributes(attributes)

        self._span_stack.append(span)
        self._current_span = span
        self._spans.append(span)
        if otel_span is not None:
            self._otel_span_stack.append(otel_span)

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
            if otel_span is not None and self._otel_span_stack:
                self._otel_span_stack.pop()
            self._current_span = self._span_stack[-1] if self._span_stack else None

    @contextmanager
    def start_span_sync(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        parent: Optional[Span] = None,
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

        # Create OTEL span if available
        otel_span = self._create_otel_span(name, kind, attributes)

        # Create our wrapper span with the OTEL span attached
        span = Span(name, self, parent=parent_span, kind=kind, otel_span=otel_span)

        if attributes:
            span.set_attributes(attributes)

        self._span_stack.append(span)
        self._current_span = span
        self._spans.append(span)
        if otel_span is not None:
            self._otel_span_stack.append(otel_span)

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
            if otel_span is not None and self._otel_span_stack:
                self._otel_span_stack.pop()
            self._current_span = self._span_stack[-1] if self._span_stack else None

    def record_exception(self, exception: Exception, span: Optional[Span] = None) -> None:
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

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> "NoOpSpan":
        return self

    def set_status(self, status: SpanStatus | str, description: Optional[str] = None) -> "NoOpSpan":
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
        attributes: Optional[dict[str, Any]] = None,
        parent: Optional[Span] = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> AsyncGenerator[NoOpSpan, None]:
        yield NoOpSpan(name)

    @contextmanager
    def start_span_sync(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        parent: Optional[Span] = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Generator[NoOpSpan, None, None]:
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
_otel_tracer: Optional[OtelTracer] = None
_otel_provider: Optional[TracerProvider] = None


def _get_otel_tracer() -> Optional[OtelTracer]:
    """Get the global OTEL tracer if initialized."""
    global _otel_tracer
    return _otel_tracer


def _setup_otel_provider(
    service_name: str,
    service_version: str,
    exporter_type: str,
    endpoint: Optional[str] = None,
) -> Optional[TracerProvider]:
    """Set up OpenTelemetry TracerProvider with the specified exporter.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        exporter_type: Type of exporter (console, otlp, none)
        endpoint: OTLP endpoint if using otlp exporter

    Returns:
        Configured TracerProvider or None if setup fails
    """
    try:
        # Create resource with service info
        resource = Resource.create(
            {
                SERVICE_NAME: service_name,
                SERVICE_VERSION: service_version,
            }
        )

        # Create provider
        provider = TracerProvider(resource=resource)

        # Add appropriate span processor based on exporter type
        if exporter_type == "console":
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            log.info("OTEL console exporter configured")
        elif exporter_type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                otlp_endpoint = (
                    endpoint or get_system_env_value("OTEL_EXPORTER_OTLP_ENDPOINT") or "http://localhost:4317"
                )
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                processor = BatchSpanProcessor(otlp_exporter)
                provider.add_span_processor(processor)
                log.info(f"OTEL OTLP exporter configured with endpoint: {otlp_endpoint}")
            except ImportError:
                log.warning("opentelemetry-exporter-otlp not installed; falling back to console exporter")
                processor = SimpleSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)
        elif exporter_type != "none":
            log.warning(f"Unknown exporter type '{exporter_type}'; no spans will be exported")
            return None
        else:
            # exporter_type == "none"
            log.info("OTEL exporter disabled (none)")
            return None

        # Register the provider globally
        otel_trace.set_tracer_provider(provider)
        log.info("OTEL TracerProvider registered globally")

        return provider
    except Exception as e:
        log.error(f"Failed to setup OTEL provider: {e}")
        return None


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

    This function sets up the OpenTelemetry SDK for exporting workflow and node
    spans, and optionally initializes Traceloop for AI provider instrumentation.

    The function handles several scenarios:
    1. If OTEL auto-instrumentation is detected (via env vars), we use the
       existing tracer provider and just get a tracer from it.
    2. Otherwise, we set up our own TracerProvider with the specified exporter.
    3. Traceloop/OpenLLMetry is initialized if configured for AI provider tracing.

    Args:
        service_name: Name of the service for trace attribution
        exporter: Optional exporter type ("otlp", "console", "none")
        endpoint: Optional endpoint for the exporter
    """
    global _tracing_initialized, _otel_tracer, _otel_provider

    if _tracing_initialized:
        log.warning("Tracing already initialized")
        return

    if not _tracing_config.enabled:
        log.info("Tracing disabled by configuration")
        return

    resolved_exporter = exporter or get_system_env_value("NODETOOL_TRACING_EXPORTER") or _tracing_config.exporter
    resolved_endpoint = endpoint or get_system_env_value("OTEL_EXPORTER_OTLP_ENDPOINT") or _tracing_config.endpoint
    service_version = get_system_env_value("OTEL_SERVICE_VERSION") or _tracing_config.service_version

    _tracing_initialized = True
    log.info(f"Tracing initialized for service: {service_name}")

    if resolved_exporter:
        log.info(f"Tracing exporter configured: {resolved_exporter}")
        if resolved_endpoint:
            log.info(f"Tracing endpoint: {resolved_endpoint}")

    auto_instrumentation_active = _is_auto_instrumentation_active()

    if auto_instrumentation_active:
        log.info(
            "OpenTelemetry auto-instrumentation detected via environment variables. "
            "Using existing TracerProvider for workflow spans."
        )
        # Get tracer from existing provider (set up by auto-instrumentation)
        _otel_tracer = otel_trace.get_tracer("nodetool.workflows", service_version)
        log.info("OTEL tracer obtained from auto-instrumentation provider")
    else:
        # Set up our own TracerProvider
        _otel_provider = _setup_otel_provider(
            service_name=service_name,
            service_version=service_version,
            exporter_type=resolved_exporter,
            endpoint=resolved_endpoint,
        )
        if _otel_provider is not None:
            _otel_tracer = otel_trace.get_tracer("nodetool.workflows", service_version)
            log.info("OTEL tracer created with custom provider")

    # Initialize Traceloop for AI provider instrumentation (separate from workflow tracing)
    if _should_initialize_traceloop(resolved_exporter):
        if resolved_endpoint and not get_system_env_value("TRACELOOP_BASE_URL"):
            os.environ["TRACELOOP_BASE_URL"] = resolved_endpoint

        try:
            from traceloop.sdk import Traceloop
        except ImportError:
            log.warning("traceloop-sdk not installed; skipping OpenLLMetry initialization")
            return

        app_name = (
            get_system_env_value("TRACELOOP_APP_NAME") or get_system_env_value("OTEL_SERVICE_NAME") or service_name
        )
        env_name = get_system_env_value("ENV")

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
            init_kwargs["exporter"] = ConsoleSpanExporter()

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
    endpoint: Optional[str] = None
    service_name: str = "nodetool"
    service_version: str = "0.6.0"
    sample_rate: float = 1.0
    batch_size: int = 512
    export_interval_ms: int = 5000
    cost_tracking: bool = True


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


def remove_tracer(job_id: str) -> Optional[WorkflowTracer]:
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
    job_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
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
    workflow_id: Optional[str] = None,
    *,
    user_id: Optional[str] = None,
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
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
    job_id: Optional[str] = None,
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
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
    job_id: Optional[str] = None,
    tools: Optional[list[str]] = None,
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
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
    job_id: Optional[str] = None,
    step_id: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
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
    job_id: Optional[str] = None,
    model: Optional[str] = None,
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
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
    job_id: Optional[str] = None,
    step_count: Optional[int] = None,
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
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
    job_id: Optional[str] = None,
    task_id: Optional[str] = None,
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
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
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
    attributes: Optional[dict[str, Any]] = None,
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


# ---------------------------------------------------------------------------
# API and Provider Tracing Context Managers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def trace_api_call(
    method: str,
    path: str,
    *,
    user_id: Optional[str] = None,
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
) -> AsyncGenerator[Span | NoOpSpan, None]:
    """Trace an API call.

    Note: HTTP/API calls are typically auto-instrumented by OpenTelemetry.
    This context manager is provided for manual tracing when needed.

    Usage:
        async with trace_api_call("POST", "/api/jobs", user_id="user-123") as span:
            response = await handle_request()

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        user_id: Optional user ID for attribution
        tracer: Optional tracer

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled:
        yield NoOpSpan("http.request")
        return

    _tracer = tracer or WorkflowTracer(f"api-{_generate_span_id()}")

    async with _tracer.start_span(
        "http.request",
        attributes={
            "http.method": method,
            "http.path": path,
            "http.user_id": user_id,
        },
        kind=SpanKind.SERVER,
    ) as span:
        yield span


@asynccontextmanager
async def trace_provider_call(
    provider: str,
    model: str,
    operation: str,
    *,
    job_id: Optional[str] = None,
    tracer: Optional[WorkflowTracer | NoOpTracer] = None,
) -> AsyncGenerator[Span | NoOpSpan, None]:
    """Trace an AI provider API call.

    Note: AI provider calls are typically auto-instrumented by Traceloop/OpenLLMetry.
    This context manager is provided for manual tracing when needed.

    Usage:
        async with trace_provider_call("openai", "gpt-4o", "chat") as span:
            response = await client.chat.completions.create(...)

    Args:
        provider: Provider name (openai, anthropic, etc.)
        model: Model name
        operation: Operation type (chat, completion, embedding, etc.)
        job_id: Optional job ID to link to workflow tracer
        tracer: Optional tracer

    Yields:
        Span object for adding attributes and events
    """
    if not _tracing_config.enabled:
        yield NoOpSpan("provider.call")
        return

    if job_id and tracer is None:
        _tracer = get_or_create_tracer(job_id)
    else:
        _tracer = tracer or WorkflowTracer(f"provider-{_generate_span_id()}")

    async with _tracer.start_span(
        f"provider.{operation}",
        attributes={
            "nodetool.provider.name": provider,
            "nodetool.provider.model": model,
            "nodetool.provider.operation": operation,
        },
        kind=SpanKind.CLIENT,
    ) as span:
        yield span


def record_cost(
    span: Span | NoOpSpan,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    credits: float = 0.0,
) -> None:
    """Record cost information on a span.

    This utility function adds token usage and cost attributes to a span.
    Use this to track AI provider costs for billing and analytics.

    Usage:
        async with trace_provider_call("openai", "gpt-4o", "chat") as span:
            response = await client.chat.completions.create(...)
            record_cost(
                span,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                credits=calculate_cost(response.usage),
            )

    Args:
        span: The span to record cost on
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        cached_tokens: Number of cached tokens (if applicable)
        credits: Cost in credits/currency
    """
    span.set_attribute("cost.input_tokens", input_tokens)
    span.set_attribute("cost.output_tokens", output_tokens)
    span.set_attribute("cost.cached_tokens", cached_tokens)
    span.set_attribute("cost.credits", credits)
    span.set_attribute("cost.total_tokens", input_tokens + output_tokens)


def set_response_attributes(
    span: Span | NoOpSpan,
    *,
    status_code: Optional[int] = None,
    response_size: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Set HTTP response attributes on a span.

    Utility function to record response information on an API span.

    Args:
        span: The span to set attributes on
        status_code: HTTP status code
        response_size: Response body size in bytes
        error: Error message if request failed
    """
    if status_code is not None:
        span.set_attribute("http.status_code", status_code)
    if response_size is not None:
        span.set_attribute("http.response_size", response_size)
    if error is not None:
        span.set_attribute("http.error", error)
        span.set_status(SpanStatus.ERROR, error)
