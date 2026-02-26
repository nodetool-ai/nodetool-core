"""
Workflow Performance Analyzer

Experimental prototype for analyzing workflow execution traces to identify
bottlenecks, compute critical paths, and suggest optimizations.

This module provides post-hoc analysis of trace data collected by the
WorkflowTracer, enabling:
- Bottleneck identification (slowest operations)
- Critical path analysis (longest execution chain)
- Parallelization opportunity detection
- Actionable optimization suggestions

Usage:
    from nodetool.observability.tracing import WorkflowTracer
    from nodetool.observability.performance_analyzer import PerformanceAnalyzer

    # After workflow execution
    tracer = get_or_create_tracer(job_id)
    analyzer = PerformanceAnalyzer(tracer)

    # Get analysis
    report = analyzer.analyze()
    print(report.summary())

    # Get suggestions
    for suggestion in analyzer.get_suggestions():
        print(suggestion)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpanAnalysis:
    """Analysis results for a single span."""

    name: str
    duration_ms: float
    self_time_ms: float  # Duration excluding children
    children_count: int
    percentage_of_parent: float | None = None
    percentage_of_total: float | None = None


@dataclass
class CriticalPathSegment:
    """A segment of the critical path."""

    name: str
    duration_ms: float
    cumulative_ms: float  # Time from start to end of this segment


@dataclass
class OptimizationSuggestion:
    """A suggestion for workflow optimization."""

    category: str  # "bottleneck", "parallelization", "caching", "error"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    impact_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class PerformanceAnalyzer:
    """
    Analyzes workflow execution traces for performance insights.

    The analyzer processes trace data from WorkflowTracer to identify
    performance bottlenecks, compute the critical path, and generate
    actionable optimization suggestions.

    Attributes:
        tracer: The WorkflowTracer containing trace data
    """

    def __init__(self, tracer):
        """
        Initialize the analyzer with a tracer.

        Args:
            tracer: WorkflowTracer instance with completed trace data
        """
        self.tracer = tracer
        self._span_analysis: dict[str, SpanAnalysis] = {}
        self._critical_path: list[CriticalPathSegment] = []
        self._suggestions: list[OptimizationSuggestion] = []

    def analyze(self) -> PerformanceReport:
        """
        Run full analysis and generate a report.

        Returns:
            PerformanceReport with comprehensive analysis
        """
        trace_tree = self.tracer.get_trace_tree()
        total_spans = trace_tree["total_spans"]

        if total_spans == 0:
            return PerformanceReport(
                total_duration_ms=0,
                total_spans=0,
                span_analysis=[],
                critical_path=[],
                suggestions=[],
            )

        # Analyze each span
        self._analyze_spans(trace_tree["root_spans"])

        # Find critical path
        self._compute_critical_path(trace_tree["root_spans"])

        # Generate suggestions
        self._generate_suggestions(trace_tree)

        # Get total duration
        total_duration = self.tracer.get_statistics().get("total_duration_ms", 0)

        return PerformanceReport(
            total_duration_ms=total_duration,
            total_spans=total_spans,
            span_analysis=list(self._span_analysis.values()),
            critical_path=self._critical_path,
            suggestions=self._suggestions,
        )

    def _analyze_spans(self, root_spans: list[dict]) -> None:
        """Analyze all spans and compute metrics."""
        total_duration_ms = self.tracer.get_statistics().get("total_duration_ms", 1)

        def process_span(span: dict, parent_duration: float | None = None) -> SpanAnalysis:
            name = span["name"]
            duration_ms = span["duration_ms"] or 0

            # Calculate self time (duration excluding children)
            children_duration = sum(
                child["duration_ms"] or 0 for child in span.get("children", [])
            )
            self_time_ms = max(0, duration_ms - children_duration)

            children_count = len(span.get("children", []))

            analysis = SpanAnalysis(
                name=name,
                duration_ms=duration_ms,
                self_time_ms=self_time_ms,
                children_count=children_count,
            )

            if parent_duration and parent_duration > 0:
                analysis.percentage_of_parent = (duration_ms / parent_duration) * 100
            if total_duration_ms > 0:
                analysis.percentage_of_total = (duration_ms / total_duration_ms) * 100

            self._span_analysis[name] = analysis

            # Recursively process children
            for child in span.get("children", []):
                process_span(child, duration_ms)

            return analysis

        for root in root_spans:
            process_span(root)

    def _compute_critical_path(self, root_spans: list[dict]) -> None:
        """
        Compute the critical path (longest execution path) through the trace tree.

        Uses a modified Dijkstra's algorithm to find the path with maximum
        cumulative duration from root to leaf.
        """
        # Build adjacency list and compute max cumulative times
        def dfs(span: dict, cumulative_time: float, path: list[CriticalPathSegment]):
            """Depth-first search to find the longest path."""
            duration = span["duration_ms"] or 0
            new_cumulative = cumulative_time + duration
            segment = CriticalPathSegment(
                name=span["name"],
                duration_ms=duration,
                cumulative_ms=new_cumulative,
            )
            new_path = [*path, segment]

            children = span.get("children", [])
            if not children:
                # Leaf node - check if this is the longest path
                if not self._critical_path or new_cumulative > self._critical_path[-1].cumulative_ms:
                    self._critical_path = new_path
            else:
                # Sort children by duration (descending) to explore longest first
                sorted_children = sorted(
                    children, key=lambda s: s["duration_ms"] or 0, reverse=True
                )
                for child in sorted_children:
                    dfs(child, new_cumulative, new_path)

        for root in root_spans:
            dfs(root, 0, [])

    def _generate_suggestions(self, trace_tree: dict) -> None:
        """Generate optimization suggestions based on analysis."""
        stats = self.tracer.get_statistics()
        error_count = stats.get("error_count", 0)

        # Check for errors first (highest priority)
        if error_count > 0:
            self._suggestions.append(
                OptimizationSuggestion(
                    category="error",
                    priority="high",
                    title=f"Fix {error_count} failed operation(s)",
                    description="Some operations failed during execution. Review error logs and fix underlying issues.",
                    details={"error_count": error_count},
                )
            )

        # Find bottlenecks (spans with high self time)
        bottlenecks = [
            (name, analysis)
            for name, analysis in self._span_analysis.items()
            if analysis.self_time_ms > 100  # More than 100ms of self time
        ]

        # Sort by self time (descending)
        bottlenecks.sort(key=lambda x: x[1].self_time_ms, reverse=True)

        for name, analysis in bottlenecks[:3]:  # Top 3 bottlenecks
            if analysis.percentage_of_total and analysis.percentage_of_total > 10:
                self._suggestions.append(
                    OptimizationSuggestion(
                        category="bottleneck",
                        priority="high",
                        title=f"Optimize slow operation: {name}",
                        description=f"This operation takes {analysis.self_time_ms:.1f}ms of self time "
                        f"({analysis.percentage_of_total:.1f}% of total). Consider optimizing "
                        f"the implementation or caching results.",
                        impact_ms=analysis.self_time_ms,
                        details={
                            "span_name": name,
                            "self_time_ms": analysis.self_time_ms,
                            "percentage_of_total": analysis.percentage_of_total,
                        },
                    )
                )

        # Check for parallelization opportunities
        # Look for sibling spans that could run in parallel
        def find_sequential_siblings(spans: list[dict]) -> None:
            for span in spans:
                children = span.get("children", [])
                if len(children) > 1:
                    # Check if children are sequential (one starts after another ends)
                    # For now, suggest if total children duration >> parent duration
                    children_duration = sum(c["duration_ms"] or 0 for c in children)
                    parent_duration = span["duration_ms"] or 1

                    if children_duration > parent_duration * 1.5:
                        self._suggestions.append(
                            OptimizationSuggestion(
                                category="parallelization",
                                priority="medium",
                                title=f"Parallelize children of: {span['name']}",
                                description=f"Children of this operation have total duration "
                                f"({children_duration:.1f}ms) significantly exceeding parent duration "
                                f"({parent_duration:.1f}ms). Some children may be able to run in parallel.",
                                impact_ms=children_duration - parent_duration,
                                details={
                                    "parent_span": span["name"],
                                    "children_count": len(children),
                                    "potential_savings_ms": children_duration - parent_duration,
                                },
                            )
                        )

                # Recursively check
                find_sequential_siblings(children)

        find_sequential_siblings(trace_tree["root_spans"])

        # Check for cacheable operations
        for name, analysis in self._span_analysis.items():
            # If a node is called multiple times and takes significant time
            if stats.get("span_count_by_name", {}).get(name, 0) > 1 and analysis.duration_ms > 50:
                self._suggestions.append(
                    OptimizationSuggestion(
                        category="caching",
                        priority="low",
                        title=f"Consider caching: {name}",
                        description=f"This operation is executed multiple times and takes "
                        f"{analysis.duration_ms:.1f}ms each time. Consider caching results.",
                        impact_ms=analysis.duration_ms,
                        details={
                            "span_name": name,
                            "execution_count": stats["span_count_by_name"][name],
                            "duration_per_execution_ms": analysis.duration_ms,
                        },
                    )
                )

    def get_bottlenecks(self, top_n: int = 5) -> list[SpanAnalysis]:
        """
        Get the top N bottlenecks (highest self time).

        Args:
            top_n: Number of bottlenecks to return

        Returns:
            List of SpanAnalysis sorted by self time (descending)
        """
        return sorted(
            self._span_analysis.values(),
            key=lambda a: a.self_time_ms,
            reverse=True,
        )[:top_n]

    def get_critical_path(self) -> list[CriticalPathSegment]:
        """Get the critical path (longest execution path)."""
        return self._critical_path

    def get_suggestions(self, category: str | None = None) -> list[OptimizationSuggestion]:
        """
        Get optimization suggestions.

        Args:
            category: Optional category filter ("bottleneck", "parallelization", "caching", "error")

        Returns:
            List of suggestions, optionally filtered by category
        """
        if category:
            return [s for s in self._suggestions if s.category == category]
        return self._suggestions


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""

    total_duration_ms: float
    total_spans: int
    span_analysis: list[SpanAnalysis]
    critical_path: list[CriticalPathSegment]
    suggestions: list[OptimizationSuggestion]

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "Workflow Performance Analysis Report",
            "=" * 60,
            f"Total Duration: {self.total_duration_ms:.2f}ms",
            f"Total Spans: {self.total_spans}",
            "",
        ]

        # Bottlenecks
        bottlenecks = sorted(self.span_analysis, key=lambda a: a.self_time_ms, reverse=True)[:3]
        if bottlenecks:
            lines.append("Top Bottlenecks (by self time):")
            for i, analysis in enumerate(bottlenecks, 1):
                lines.append(
                    f"  {i}. {analysis.name}: {analysis.self_time_ms:.2f}ms "
                    f"({analysis.percentage_of_total:.1f}% of total)"
                )
            lines.append("")

        # Critical path
        if self.critical_path:
            lines.append("Critical Path (longest execution chain):")
            for segment in self.critical_path:
                lines.append(f"  → {segment.name}: {segment.duration_ms:.2f}ms")
            lines.append("")

        # Suggestions
        if self.suggestions:
            lines.append("Optimization Suggestions:")
            for suggestion in self.suggestions[:5]:
                lines.append(f"  [{suggestion.priority.upper()}] {suggestion.title}")
                if suggestion.impact_ms:
                    lines.append(f"      Potential savings: {suggestion.impact_ms:.2f}ms")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
