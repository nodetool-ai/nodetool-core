"""Tests for the observability tracing module."""

import pytest

from nodetool.observability.tracing import (
    NoOpTracer,
    Span,
    WorkflowTracer,
    get_tracer,
    init_tracing,
    is_tracing_enabled,
)


class TestWorkflowTracer:
    """Tests for the WorkflowTracer class."""

    def test_create_tracer(self):
        """Test creating a tracer with a job ID."""
        tracer = WorkflowTracer(job_id="test-job-123")
        assert tracer.job_id == "test-job-123"
        assert tracer.trace_id.startswith("trace-test-job-123")
        assert tracer._enabled is True

    def test_custom_trace_id(self):
        """Test creating a tracer with a custom trace ID."""
        tracer = WorkflowTracer(job_id="test-job", trace_id="custom-trace-id")
        assert tracer.trace_id == "custom-trace-id"

    @pytest.mark.asyncio
    async def test_start_span(self):
        """Test starting a span."""
        tracer = WorkflowTracer(job_id="test-job")
        async with tracer.start_span("test_span") as span:
            assert span.name == "test_span"
            assert tracer.active_span == span

    @pytest.mark.asyncio
    async def test_nested_spans(self):
        """Test nested span creation."""
        tracer = WorkflowTracer(job_id="test-job")
        async with tracer.start_span("parent") as parent:
            assert tracer.active_span == parent
            async with tracer.start_span("child") as child:
                assert tracer.active_span == child
                assert child.parent == parent
            assert tracer.active_span == parent

    @pytest.mark.asyncio
    async def test_span_attributes(self):
        """Test setting span attributes."""
        tracer = WorkflowTracer(job_id="test-job")
        async with tracer.start_span("test") as span:
            span.set_attribute("key1", "value1")
            span.set_attribute("key2", 42)
            assert span.context.attributes["key1"] == "value1"
            assert span.context.attributes["key2"] == 42

    @pytest.mark.asyncio
    async def test_span_events(self):
        """Test adding span events."""
        tracer = WorkflowTracer(job_id="test-job")
        async with tracer.start_span("test") as span:
            span.add_event("test_event", {"foo": "bar"})
            # span_started event is auto-added when span begins
            assert len(span.context.events) >= 1
            event_names = [e["name"] for e in span.context.events]
            assert "test_event" in event_names

    @pytest.mark.asyncio
    async def test_span_status(self):
        """Test setting span status."""
        tracer = WorkflowTracer(job_id="test-job")
        async with tracer.start_span("test") as span:
            span.set_status("error", "Something went wrong")
            assert span.context.status == "error"

    @pytest.mark.asyncio
    async def test_span_exception(self):
        """Test recording exceptions in spans."""
        tracer = WorkflowTracer(job_id="test-job")
        async with tracer.start_span("test") as span:
            try:
                raise ValueError("Test error")
            except ValueError:
                tracer.record_exception(ValueError("Test error"))
            assert span.context.status == "error"

    @pytest.mark.asyncio
    async def test_span_context_manager_exception(self):
        """Test span context manager handles exceptions."""
        tracer = WorkflowTracer(job_id="test-job")
        with pytest.raises(ValueError):
            async with tracer.start_span("test") as span:
                span.set_status("ok")
                raise ValueError("Test exception")
        assert tracer._spans[-1].context.status == "error"

    def test_get_trace_tree(self):
        """Test getting trace tree."""
        tracer = WorkflowTracer(job_id="test-job")
        assert tracer.get_trace_tree()["job_id"] == "test-job"
        assert tracer.get_trace_tree()["trace_id"] == tracer.trace_id

    def test_get_statistics(self):
        """Test getting trace statistics."""
        tracer = WorkflowTracer(job_id="test-job")
        stats = tracer.get_statistics()
        assert stats["total_spans"] == 0

    def test_disable_enable(self):
        """Test disabling and enabling tracer."""
        tracer = WorkflowTracer(job_id="test-job")
        tracer.disable()
        assert tracer._enabled is False
        tracer.enable()
        assert tracer._enabled is True


class TestNoOpTracer:
    """Tests for the NoOpTracer class."""

    def test_noop_tracer_creation(self):
        """Test creating a NoOpTracer."""
        tracer = NoOpTracer(job_id="test-job")
        assert tracer.job_id == "test-job"

    @pytest.mark.asyncio
    async def test_noop_span_context_manager(self):
        """Test NoOpSpan context manager."""
        tracer = NoOpTracer(job_id="test-job")
        async with tracer.start_span("test") as span:
            span.set_attribute("key", "value")
            span.add_event("event")
            span.set_status("error")
        # Should not raise any exceptions


class TestGetTracer:
    """Tests for the get_tracer function."""

    def test_get_tracer_enabled(self):
        """Test getting tracer when enabled."""
        tracer = get_tracer("test-job", enabled=True)
        assert isinstance(tracer, WorkflowTracer)
        assert tracer.job_id == "test-job"

    def test_get_tracer_disabled(self):
        """Test getting tracer when disabled."""
        tracer = get_tracer("test-job", enabled=False)
        assert isinstance(tracer, NoOpTracer)


class TestInitTracing:
    """Tests for the init_tracing function."""

    def test_init_tracing(self):
        """Test initializing tracing."""
        init_tracing(service_name="test-service")
        assert is_tracing_enabled() is True

    def test_init_tracing_with_exporter(self):
        """Test initializing tracing with exporter."""
        init_tracing(service_name="test-service", exporter="console")
        assert is_tracing_enabled() is True
