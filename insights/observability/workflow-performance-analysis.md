# Workflow Performance Analysis Using Trace Data

**Insight**: Post-hoc analysis of OpenTelemetry trace data can identify performance bottlenecks, compute critical paths, and suggest actionable optimizations for NodeTool workflows without requiring runtime overhead.

**Rationale**: While NodeTool has comprehensive tracing (via OpenTelemetry) and PyTorch-specific memory profiling, it lacked general performance analysis capabilities. The WorkflowTracer collects rich timing data, but this data was only used for observability/debugging. By analyzing this trace data after workflow execution, we can provide insights to help developers optimize their workflows without adding runtime instrumentation overhead.

**Example**:
```python
from nodetool.observability import WorkflowTracer, PerformanceAnalyzer

# After workflow execution
tracer = get_or_create_tracer(job_id)
analyzer = PerformanceAnalyzer(tracer)
report = analyzer.analyze()

# Get bottlenecks (operations with high self-time)
for bottleneck in analyzer.get_bottlenecks(top_n=5):
    print(f"{bottleneck.name}: {bottleneck.self_time_ms:.1f}ms self-time")

# Get critical path (longest execution chain)
for segment in analyzer.get_critical_path():
    print(f"→ {segment.name}: {segment.duration_ms:.1f}ms")

# Get optimization suggestions
for suggestion in analyzer.get_suggestions():
    print(f"[{suggestion.priority}] {suggestion.title}")
    print(f"  {suggestion.description}")
    if suggestion.impact_ms:
        print(f"  Potential savings: {suggestion.impact_ms:.1f}ms")
```

**Key Algorithms**:

1. **Self-Time Calculation**: For each span, compute "self time" (duration excluding children) to identify operations that spend significant time in their own logic rather than waiting for children.

2. **Critical Path Analysis**: Use depth-first search to find the longest execution path from root to leaf, identifying the chain of operations that determines total workflow duration.

3. **Bottleneck Detection**: Identify spans with high self-time (>100ms) that represent significant portions (>10%) of total execution time.

4. **Parallelization Detection**: Identify parent spans where children duration significantly exceeds parent duration, indicating potential for parallel execution.

**Impact**:
- **Zero runtime overhead**: Analysis is post-hoc, using existing trace data
- **Actionable insights**: Provides specific suggestions with estimated impact
- **Developer-focused**: Helps identify optimization opportunities without deep profiling knowledge

**Files**:
- `src/nodetool/observability/performance_analyzer.py` - PerformanceAnalyzer implementation
- `tests/observability/test_performance_analyzer.py` - Comprehensive tests

**Date**: 2026-02-26
