"""Tests for the performance analyzer module."""

import time

import pytest

from nodetool.observability.performance_analyzer import (
    CriticalPathSegment,
    OptimizationSuggestion,
    PerformanceAnalyzer,
    PerformanceReport,
    SpanAnalysis,
)
from nodetool.observability.tracing import WorkflowTracer


class TestPerformanceAnalyzer:
    """Tests for the PerformanceAnalyzer class."""

    @pytest.fixture
    def sample_tracer(self):
        """Create a tracer with sample trace data."""
        tracer = WorkflowTracer(job_id="test-job")
        return tracer

    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, sample_tracer):
        """Test initializing the analyzer with a tracer."""
        analyzer = PerformanceAnalyzer(sample_tracer)
        assert analyzer.tracer == sample_tracer
        assert analyzer._span_analysis == {}
        assert analyzer._critical_path == []
        assert analyzer._suggestions == []

    @pytest.mark.asyncio
    async def test_analyze_empty_trace(self, sample_tracer):
        """Test analyzing a trace with no spans."""
        analyzer = PerformanceAnalyzer(sample_tracer)
        report = analyzer.analyze()

        assert report.total_spans == 0
        assert report.total_duration_ms == 0
        assert report.span_analysis == []
        assert report.critical_path == []
        assert report.suggestions == []

    @pytest.mark.asyncio
    async def test_analyze_simple_trace(self):
        """Test analyzing a simple trace with a single span."""
        tracer = WorkflowTracer(job_id="test-job")
        async with tracer.start_span("test_operation", attributes={"test": "value"}):
            time.sleep(0.01)  # 10ms

        analyzer = PerformanceAnalyzer(tracer)
        report = analyzer.analyze()

        assert report.total_spans == 1
        assert report.total_duration_ms > 0
        assert len(report.span_analysis) == 1
        assert report.span_analysis[0].name == "test_operation"

    @pytest.mark.asyncio
    async def test_span_analysis_self_time(self):
        """Test that span analysis correctly calculates self time."""
        tracer = WorkflowTracer(job_id="test-job")

        async with tracer.start_span("parent"):
            time.sleep(0.02)  # 20ms parent self time
            async with tracer.start_span("child"):
                time.sleep(0.01)  # 10ms child time

        analyzer = PerformanceAnalyzer(tracer)
        report = analyzer.analyze()

        # Find parent and child analysis
        parent_analysis = next((a for a in report.span_analysis if a.name == "parent"), None)
        child_analysis = next((a for a in report.span_analysis if a.name == "child"), None)

        assert parent_analysis is not None
        assert child_analysis is not None

        # Parent self time should be roughly parent duration - child duration
        # (accounting for timing variations)
        assert parent_analysis.self_time_ms >= 0
        assert child_analysis.self_time_ms > 0

    @pytest.mark.asyncio
    async def test_critical_path_computation(self):
        """Test computing the critical path through a trace."""
        tracer = WorkflowTracer(job_id="test-job")

        async with tracer.start_span("root"):
            async with tracer.start_span("fast_branch"):
                time.sleep(0.005)  # 5ms

            async with tracer.start_span("slow_branch"):
                time.sleep(0.015)  # 15ms - this should be on critical path

        analyzer = PerformanceAnalyzer(tracer)
        report = analyzer.analyze()

        assert len(report.critical_path) > 0
        # The critical path should include slow_branch
        path_names = [s.name for s in report.critical_path]
        assert "slow_branch" in path_names

    @pytest.mark.asyncio
    async def test_bottleneck_detection(self):
        """Test detection of performance bottlenecks."""
        tracer = WorkflowTracer(job_id="test-job")

        # Create a slow operation
        async with tracer.start_span("slow_operation"):
            time.sleep(0.15)  # 150ms - above bottleneck threshold

        # Create a fast operation
        async with tracer.start_span("fast_operation"):
            time.sleep(0.005)  # 5ms

        analyzer = PerformanceAnalyzer(tracer)
        analyzer.analyze()

        # Check that slow operation is identified as a bottleneck
        bottlenecks = analyzer.get_bottlenecks(top_n=5)
        assert len(bottlenecks) > 0
        assert bottlenecks[0].name == "slow_operation"
        assert bottlenecks[0].self_time_ms > 100  # Above threshold

    @pytest.mark.asyncio
    async def test_optimization_suggestions_bottleneck(self):
        """Test that bottleneck suggestions are generated."""
        tracer = WorkflowTracer(job_id="test-job")

        async with tracer.start_span("very_slow_operation"):
            time.sleep(0.2)  # 200ms

        analyzer = PerformanceAnalyzer(tracer)
        analyzer.analyze()

        bottleneck_suggestions = analyzer.get_suggestions(category="bottleneck")
        assert len(bottleneck_suggestions) > 0
        assert "slow" in bottleneck_suggestions[0].title.lower()
        assert bottleneck_suggestions[0].priority == "high"

    @pytest.mark.asyncio
    async def test_optimization_suggestions_error(self):
        """Test that error suggestions are generated for failed spans."""
        tracer = WorkflowTracer(job_id="test-job")

        async with tracer.start_span("failed_operation"):
            tracer.active_span.record_exception(ValueError("Test error"))

        analyzer = PerformanceAnalyzer(tracer)
        analyzer.analyze()

        error_suggestions = analyzer.get_suggestions(category="error")
        assert len(error_suggestions) > 0
        assert "failed" in error_suggestions[0].title.lower() or "fix" in error_suggestions[0].title.lower()
        assert error_suggestions[0].priority == "high"

    @pytest.mark.asyncio
    async def test_get_suggestions_filtering(self):
        """Test filtering suggestions by category."""
        tracer = WorkflowTracer(job_id="test-job")

        async with tracer.start_span("slow_op"):
            time.sleep(0.15)

        async with tracer.start_span("error_op"):
            tracer.active_span.record_exception(ValueError("Error"))

        analyzer = PerformanceAnalyzer(tracer)
        analyzer.analyze()

        bottleneck_suggestions = analyzer.get_suggestions(category="bottleneck")
        error_suggestions = analyzer.get_suggestions(category="error")
        all_suggestions = analyzer.get_suggestions()

        assert len(bottleneck_suggestions) > 0
        assert len(error_suggestions) > 0
        assert len(all_suggestions) >= len(bottleneck_suggestions) + len(error_suggestions)

    @pytest.mark.asyncio
    async def test_report_summary_generation(self):
        """Test generating a human-readable report summary."""
        tracer = WorkflowTracer(job_id="test-job")

        async with tracer.start_span("test_operation"):
            time.sleep(0.01)

        analyzer = PerformanceAnalyzer(tracer)
        report = analyzer.analyze()

        summary = report.summary()

        assert "Workflow Performance Analysis Report" in summary
        assert "Total Duration:" in summary
        assert "Total Spans:" in summary
        assert "test_operation" in summary

    @pytest.mark.asyncio
    async def test_percentage_calculations(self):
        """Test that percentage calculations are correct."""
        tracer = WorkflowTracer(job_id="test-job")

        async with tracer.start_span("op1"):
            time.sleep(0.01)

        async with tracer.start_span("op2"):
            time.sleep(0.01)

        analyzer = PerformanceAnalyzer(tracer)
        report = analyzer.analyze()

        for analysis in report.span_analysis:
            if analysis.percentage_of_total is not None:
                assert 0 <= analysis.percentage_of_total <= 100


class TestDataClasses:
    """Tests for performance analyzer data classes."""

    def test_span_analysis_dataclass(self):
        """Test SpanAnalysis dataclass."""
        analysis = SpanAnalysis(
            name="test_span",
            duration_ms=100.0,
            self_time_ms=80.0,
            children_count=2,
            percentage_of_parent=50.0,
            percentage_of_total=25.0,
        )

        assert analysis.name == "test_span"
        assert analysis.duration_ms == 100.0
        assert analysis.self_time_ms == 80.0
        assert analysis.children_count == 2
        assert analysis.percentage_of_parent == 50.0
        assert analysis.percentage_of_total == 25.0

    def test_critical_path_segment_dataclass(self):
        """Test CriticalPathSegment dataclass."""
        segment = CriticalPathSegment(
            name="critical_operation",
            duration_ms=150.0,
            cumulative_ms=300.0,
        )

        assert segment.name == "critical_operation"
        assert segment.duration_ms == 150.0
        assert segment.cumulative_ms == 300.0

    def test_optimization_suggestion_dataclass(self):
        """Test OptimizationSuggestion dataclass."""
        suggestion = OptimizationSuggestion(
            category="bottleneck",
            priority="high",
            title="Optimize slow operation",
            description="This operation takes too long.",
            impact_ms=100.0,
            details={"node_name": "slow_node"},
        )

        assert suggestion.category == "bottleneck"
        assert suggestion.priority == "high"
        assert suggestion.title == "Optimize slow operation"
        assert suggestion.impact_ms == 100.0
        assert suggestion.details["node_name"] == "slow_node"
