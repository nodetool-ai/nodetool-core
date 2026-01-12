"""Tests for the observability tracing module."""

import pytest

from nodetool.observability.tracing import (
    NoOpTracer,
    Span,
    SpanKind,
    SpanStatus,
    WorkflowTracer,
    get_tracer,
    init_tracing,
    is_tracing_enabled,
    trace_api_call,
    trace_websocket_message,
    trace_workflow,
    trace_node,
    trace_provider_call,
    trace_agent_task,
    TracingConfig,
    configure_tracing,
    record_cost,
)


class TestWorkflowTracer:
    """Tests for the WorkflowTracer class."""

    def test_create_tracer(self):
        """Test creating a tracer with a job ID."""
        tracer = WorkflowTracer(job_id="test-job-123")
        assert tracer.job_id == "test-job-123"
        # trace_id is now a UUID hex string
        assert len(tracer.trace_id) == 32  # UUID hex length
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
            assert span.context.status == SpanStatus.ERROR

    @pytest.mark.asyncio
    async def test_span_exception(self):
        """Test recording exceptions in spans."""
        tracer = WorkflowTracer(job_id="test-job")
        async with tracer.start_span("test") as span:
            try:
                raise ValueError("Test error")
            except ValueError:
                tracer.record_exception(ValueError("Test error"))
            assert span.context.status == SpanStatus.ERROR

    @pytest.mark.asyncio
    async def test_span_context_manager_exception(self):
        """Test span context manager handles exceptions."""
        tracer = WorkflowTracer(job_id="test-job")
        with pytest.raises(ValueError):
            async with tracer.start_span("test") as span:
                span.set_status("ok")
                raise ValueError("Test exception")
        assert tracer._spans[-1].context.status == SpanStatus.ERROR

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

    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Test cost tracking on tracer."""
        tracer = WorkflowTracer(job_id="test-job")
        assert tracer.total_cost == 0.0
        tracer.add_cost(0.5)
        tracer.add_cost(0.25)
        assert tracer.total_cost == 0.75

    @pytest.mark.asyncio
    async def test_span_kind(self):
        """Test span kind attribute."""
        tracer = WorkflowTracer(job_id="test-job")
        async with tracer.start_span("test", kind=SpanKind.CLIENT) as span:
            assert span.context.kind == SpanKind.CLIENT


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


class TestContextManagers:
    """Tests for tracing context managers."""

    @pytest.mark.asyncio
    async def test_trace_api_call(self):
        """Test trace_api_call context manager."""
        configure_tracing(TracingConfig(enabled=True))
        async with trace_api_call("POST", "/api/jobs", user_id="user-123") as span:
            span.set_attribute("custom", "value")
            assert span.context.attributes["http.method"] == "POST"
            assert span.context.attributes["http.path"] == "/api/jobs"
            assert span.context.attributes["http.user_id"] == "user-123"
            assert span.context.kind == SpanKind.SERVER

    @pytest.mark.asyncio
    async def test_trace_websocket_message(self):
        """Test trace_websocket_message context manager."""
        configure_tracing(TracingConfig(enabled=True))
        async with trace_websocket_message("run_job", "inbound", job_id="job-123") as span:
            assert span.context.attributes["websocket.command"] == "run_job"
            assert span.context.attributes["websocket.direction"] == "inbound"
            assert span.context.attributes["websocket.job_id"] == "job-123"
            assert span.context.kind == SpanKind.CONSUMER

    @pytest.mark.asyncio
    async def test_trace_workflow(self):
        """Test trace_workflow context manager."""
        configure_tracing(TracingConfig(enabled=True))
        async with trace_workflow("job-123", "wf-456", user_id="user-1") as span:
            assert span.context.attributes["nodetool.workflow.job_id"] == "job-123"
            assert span.context.attributes["nodetool.workflow.id"] == "wf-456"

    @pytest.mark.asyncio
    async def test_trace_node(self):
        """Test trace_node context manager."""
        configure_tracing(TracingConfig(enabled=True))
        async with trace_node("node-1", "ImageGenerate") as span:
            span.set_attribute("input_count", 3)
            assert span.context.attributes["nodetool.node.id"] == "node-1"
            assert span.context.attributes["nodetool.node.type"] == "ImageGenerate"

    @pytest.mark.asyncio
    async def test_trace_provider_call(self):
        """Test trace_provider_call context manager."""
        configure_tracing(TracingConfig(enabled=True))
        async with trace_provider_call("openai", "gpt-4o", "chat") as span:
            assert span.context.attributes["nodetool.provider.name"] == "openai"
            assert span.context.attributes["nodetool.provider.model"] == "gpt-4o"
            assert span.context.attributes["nodetool.provider.operation"] == "chat"
            assert span.context.kind == SpanKind.CLIENT

    @pytest.mark.asyncio
    async def test_trace_agent_task(self):
        """Test trace_agent_task context manager."""
        configure_tracing(TracingConfig(enabled=True))
        async with trace_agent_task("cot", "Generate an image", tools=["browser", "image_gen"]) as span:
            assert span.context.attributes["nodetool.agent.type"] == "cot"
            assert span.context.attributes["nodetool.agent.task_description"] == "Generate an image"
            assert span.context.attributes["nodetool.agent.available_tools"] == ["browser", "image_gen"]

    @pytest.mark.asyncio
    async def test_disabled_tracing_returns_noop(self):
        """Test that disabled tracing returns NoOp spans."""
        configure_tracing(TracingConfig(enabled=False))
        async with trace_api_call("GET", "/test") as span:
            # Should be a NoOpSpan when disabled
            span.set_attribute("test", "value")  # Should not raise


class TestRecordCost:
    """Tests for the record_cost utility."""

    @pytest.mark.asyncio
    async def test_record_cost(self):
        """Test recording cost on a span."""
        configure_tracing(TracingConfig(enabled=True, cost_tracking=True))
        tracer = WorkflowTracer(job_id="test-job")
        async with tracer.start_span("test") as span:
            record_cost(
                span,
                input_tokens=1000,
                output_tokens=500,
                cached_tokens=100,
                credits=0.05,
            )
            assert span.context.attributes["cost.input_tokens"] == 1000
            assert span.context.attributes["cost.output_tokens"] == 500
            assert span.context.attributes["cost.cached_tokens"] == 100
            assert span.context.attributes["cost.credits"] == 0.05
            assert span.context.attributes["cost.total_tokens"] == 1500


class TestTracingConfig:
    """Tests for TracingConfig."""

    def test_default_config(self):
        """Test default TracingConfig values."""
        config = TracingConfig()
        assert config.enabled is True
        assert config.exporter == "console"
        assert config.sample_rate == 1.0
        assert config.cost_tracking is True

    def test_custom_config(self):
        """Test custom TracingConfig values."""
        config = TracingConfig(
            enabled=True,
            exporter="otlp",
            endpoint="http://localhost:4317",
            sample_rate=0.5,
        )
        assert config.exporter == "otlp"
        assert config.endpoint == "http://localhost:4317"
        assert config.sample_rate == 0.5
