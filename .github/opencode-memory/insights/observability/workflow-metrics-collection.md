# Workflow Metrics Collection

**Insight**: Added a metrics collection module to complement the existing tracing infrastructure with quantitative telemetry.

**Rationale**: While tracing provides detailed execution traces, metrics provide aggregated, quantitative data that is essential for monitoring, alerting, and performance analysis. The new metrics system supports:
- Counter metrics for operation counts (workflows started, nodes executed)
- Histogram metrics for latency distributions (execution times)
- Gauge metrics for current state (active workflows)
- Prometheus-compatible export format

**Example**:
```python
from nodetool.observability.metrics import (
    init_metrics,
    workflow_counter,
    node_latency_histogram,
)

init_metrics(service_name="nodetool-worker")

# Record metrics
workflow_counter.increment("workflow_started", {"workflow_type": "image_generation"})
node_latency_histogram.observe(0.156, {"node_type": "ImageGenerate"})

# Export to Prometheus
metrics = format_prometheus_metrics()
```

**Impact**:
- Complements existing tracing with quantitative metrics
- Prometheus-compatible output for integration with monitoring systems
- Low overhead with no-op patterns when disabled
- Thread-safe implementation for concurrent access

**Files**: `src/nodetool/observability/metrics.py`

**Date**: 2026-01-15
