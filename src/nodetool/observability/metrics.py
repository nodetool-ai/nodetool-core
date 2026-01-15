"""
Workflow Metrics Collection for NodeTool.

This module provides metrics collection capabilities for workflow execution,
complementing the existing tracing infrastructure with quantitative telemetry.

Features:
- Counter metrics for operation counts
- Histogram metrics for latency distributions
- Gauge metrics for current state
- Prometheus-compatible export

Usage:
    from nodetool.observability.metrics import (
        init_metrics,
        workflow_counter,
        node_latency_histogram,
        get_metrics_summary,
    )

    init_metrics(service_name="nodetool-worker")

    # Record workflow start
    workflow_counter.increment("workflow_started", {"workflow_type": "image_generation"})

    # Record node execution latency
    node_latency_histogram.observe(0.156, {"node_type": "ImageGenerate"})

    # Get metrics for export
    metrics = get_metrics_summary()
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from nodetool.config.env_guard import get_system_env_value
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class MetricType(str, Enum):
    """Supported metric types."""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


@dataclass
class MetricLabel:
    """Labels for a metric."""

    name: str
    value: str


@dataclass
class CounterMetric:
    """A counter metric that only increments."""

    name: str
    description: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    _value: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, amount: int = 1, additional_labels: Optional[dict[str, str]] = None) -> None:
        """Increment the counter."""
        with self._lock:
            self._value += amount

    def get_value(self) -> int:
        """Get the current counter value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset the counter to zero."""
        with self._lock:
            self._value = 0


@dataclass
class HistogramMetric:
    """A histogram metric for tracking value distributions."""

    name: str
    description: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    buckets: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
    )
    _values: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, additional_labels: Optional[dict[str, str]] = None) -> None:
        """Record an observation."""
        with self._lock:
            self._values.append(value)

    def get_count(self) -> int:
        """Get the number of observations."""
        with self._lock:
            return len(self._values)

    def get_sum(self) -> float:
        """Get the sum of all observations."""
        with self._lock:
            return sum(self._values)

    def get_bucket_counts(self) -> dict[float, int]:
        """Get histogram bucket counts."""
        with self._lock:
            counts: dict[float, int] = dict.fromkeys(self.buckets, 0)
            for value in self._values:
                for bucket in self.buckets:
                    if value <= bucket:
                        counts[bucket] += 1
            return counts

    def get_percentiles(self, percentiles: Optional[list[float]] = None) -> dict[float, float]:
        """Calculate percentile values."""
        if percentiles is None:
            percentiles = [0.5, 0.9, 0.95, 0.99]
        with self._lock:
            if not self._values:
                return dict.fromkeys(percentiles, 0.0)
            sorted_values = sorted(self._values)
            result: dict[float, float] = {}
            for p in percentiles:
                idx = int(len(sorted_values) * p)
                result[p] = sorted_values[min(idx, len(sorted_values) - 1)]
            return result

    def reset(self) -> None:
        """Reset the histogram."""
        with self._lock:
            self._values.clear()


@dataclass
class GaugeMetric:
    """A gauge metric that can go up and down."""

    name: str
    description: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float) -> None:
        """Set the gauge to a specific value."""
        with self._lock:
            self._value = value

    def increment(self, amount: float = 1.0) -> None:
        """Increment the gauge."""
        with self._lock:
            self._value += amount

    def decrement(self, amount: float = 1.0) -> None:
        """Decrement the gauge."""
        with self._lock:
            self._value -= amount

    def get_value(self) -> float:
        """Get the current gauge value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset the gauge to zero."""
        with self._lock:
            self._value = 0.0


class MetricsRegistry:
    """Central registry for all metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, CounterMetric] = {}
        self._histograms: dict[str, HistogramMetric] = {}
        self._gauges: dict[str, GaugeMetric] = {}
        self._lock = threading.Lock()

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[dict[str, str]] = None,
    ) -> CounterMetric:
        """Get or create a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._counters:
                self._counters[key] = CounterMetric(
                    name=name,
                    description=description,
                    labels=labels or {},
                )
            return self._counters[key]

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[dict[str, str]] = None,
        buckets: Optional[list[float]] = None,
    ) -> HistogramMetric:
        """Get or create a histogram metric."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = HistogramMetric(
                    name=name,
                    description=description,
                    labels=labels or {},
                    buckets=buckets or [],
                )
            return self._histograms[key]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[dict[str, str]] = None,
    ) -> GaugeMetric:
        """Get or create a gauge metric."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = GaugeMetric(
                    name=name,
                    description=description,
                    labels=labels or {},
                )
            return self._gauges[key]

    def _make_key(self, name: str, labels: Optional[dict[str, str]] = None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get all metrics in a format suitable for export."""
        metrics: dict[str, dict[str, Any]] = {}

        with self._lock:
            for key, counter in self._counters.items():
                metrics[key] = {
                    "type": "counter",
                    "value": counter.get_value(),
                    "labels": counter.labels,
                }

            for key, histogram in self._histograms.items():
                metrics[key] = {
                    "type": "histogram",
                    "count": histogram.get_count(),
                    "sum": histogram.get_sum(),
                    "buckets": histogram.get_bucket_counts(),
                    "percentiles": histogram.get_percentiles(),
                    "labels": histogram.labels,
                }

            for key, gauge in self._gauges.items():
                metrics[key] = {
                    "type": "gauge",
                    "value": gauge.get_value(),
                    "labels": gauge.labels,
                }

        return metrics

    def reset_all(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            for histogram in self._histograms.values():
                histogram.reset()
            for gauge in self._gauges.values():
                gauge.reset()


_global_registry: Optional[MetricsRegistry] = None
_metrics_initialized = False


def _is_truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def init_metrics(
    service_name: str = "nodetool",
    enabled: Optional[bool] = None,
) -> None:
    """
    Initialize the metrics collection system.

    Args:
        service_name: Name of the service for metric attribution
        enabled: Whether to enable metrics (defaults to environment variable)
    """
    global _global_registry, _metrics_initialized

    if _metrics_initialized:
        log.warning("Metrics already initialized")
        return

    if enabled is None:
        enabled = _is_truthy(get_system_env_value("NODETOOL_METRICS_ENABLED"))

    if not enabled:
        log.info("Metrics disabled by configuration")
        return

    _global_registry = MetricsRegistry()
    _metrics_initialized = True
    log.info(f"Metrics initialized for service: {service_name}")


def get_registry() -> Optional[MetricsRegistry]:
    """Get the global metrics registry."""
    return _global_registry


def workflow_counter(name: str, description: str = "") -> CounterMetric:
    """Get or create a workflow-related counter."""
    registry = get_registry()
    if registry is None:
        raise RuntimeError("Metrics not initialized. Call init_metrics() first.")
    return registry.counter(f"workflow_{name}", description=description)


def node_counter(name: str, description: str = "") -> CounterMetric:
    """Get or create a node-related counter."""
    registry = get_registry()
    if registry is None:
        raise RuntimeError("Metrics not initialized. Call init_metrics() first.")
    return registry.counter(f"node_{name}", description=description)


def node_latency_histogram(
    name: str = "node_execution_latency",
    description: str = "Node execution latency in seconds",
) -> HistogramMetric:
    """Get or create a node latency histogram."""
    registry = get_registry()
    if registry is None:
        raise RuntimeError("Metrics not initialized. Call init_metrics() first.")
    return registry.histogram(
        name,
        description=description,
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )


def workflow_latency_histogram(
    name: str = "workflow_execution_latency",
    description: str = "Workflow execution latency in seconds",
) -> HistogramMetric:
    """Get or create a workflow latency histogram."""
    registry = get_registry()
    if registry is None:
        raise RuntimeError("Metrics not initialized. Call init_metrics() first.")
    return registry.histogram(
        name,
        description=description,
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    )


def active_workflows_gauge() -> GaugeMetric:
    """Get or create a gauge for active workflows."""
    registry = get_registry()
    if registry is None:
        raise RuntimeError("Metrics not initialized. Call init_metrics() first.")
    return registry.gauge("workflows_active", description="Number of currently active workflows")


def get_metrics_summary() -> dict[str, Any]:
    """Get all metrics as a summary dictionary."""
    registry = get_registry()
    if registry is None:
        return {"error": "Metrics not initialized"}

    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": registry.get_all_metrics(),
    }


def format_prometheus_metrics() -> str:
    """Format all metrics in Prometheus text format."""
    registry = get_registry()
    if registry is None:
        return "# Metrics not initialized\n"

    lines: list[str] = []
    metrics = registry.get_all_metrics()

    for key, data in metrics.items():
        if data["type"] == "counter":
            lines.append(f"# TYPE {key} counter")
            if data.get("description"):
                lines.append(f"# HELP {key} {data['description']}")
            lines.append(f"{key} {data['value']}")

        elif data["type"] == "histogram":
            lines.append(f"# TYPE {key} histogram")
            if data.get("description"):
                lines.append(f"# HELP {key} {data['description']}")
            for bucket, count in data.get("buckets", {}).items():
                lines.append(f'{key}_bucket{{le="{bucket}"}} {count}')
            lines.append(f'{key}_bucket{{le="+Inf"}} {data.get("count", 0)}')
            lines.append(f"{key}_count {data.get('count', 0)}")
            lines.append(f"{key}_sum {data.get('sum', 0)}")

        elif data["type"] == "gauge":
            lines.append(f"# TYPE {key} gauge")
            if data.get("description"):
                lines.append(f"# HELP {key} {data['description']}")
            lines.append(f"{key} {data['value']}")

    return "\n".join(lines) + "\n"


class MetricsContext:
    """Context manager for automatic metrics collection."""

    def __init__(
        self,
        metric_name: str,
        metric_type: MetricType = MetricType.HISTOGRAM,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.labels = labels or {}
        self._start_time: float = 0.0
        self._registry: Optional[MetricsRegistry] = None

    async def __aenter__(self) -> "MetricsContext":
        self._registry = get_registry()
        if self._registry:
            self._start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._registry and self._start_time > 0:
            elapsed = time.perf_counter() - self._start_time
            if self.metric_type == MetricType.HISTOGRAM:
                self._registry.histogram(self.metric_name, labels=self.labels).observe(elapsed)
            elif self.metric_type == MetricType.COUNTER:
                self._registry.counter(self.metric_name, labels=self.labels).increment()


def track_workflow_execution(workflow_type: str):
    """Decorator for tracking workflow execution metrics."""

    def decorator(func):
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            registry = get_registry()
            if registry:
                counter = registry.counter(
                    f"workflow_{workflow_type}_started",
                    labels={"workflow_type": workflow_type},
                )
                counter.increment()

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                if registry:
                    elapsed = time.perf_counter() - start_time
                    registry.histogram(
                        f"workflow_{workflow_type}_duration",
                        labels={"workflow_type": workflow_type},
                    ).observe(elapsed)

        return wrapper

    return decorator


__all__ = [
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    "MetricType",
    "MetricsContext",
    "active_workflows_gauge",
    "format_prometheus_metrics",
    "get_metrics_summary",
    "get_registry",
    "init_metrics",
    "node_counter",
    "node_latency_histogram",
    "track_workflow_execution",
    "workflow_counter",
    "workflow_latency_histogram",
]
