"""
Memory logging and garbage collection utilities for workflow execution.

This module provides utilities for tracking memory usage and performing
garbage collection to help diagnose memory leaks and reduce RAM usage.

===============================================================================
GPU TRACING UTILITIES (New)
===============================================================================

For debugging GPU memory growth in iterative processes, use the following:

1. GPUTraceSession - Comprehensive tracing across multiple iterations:

   trace = GPUTraceSession("my_loop", log_interval=1)
   trace.start()

   for i in range(100):
       with trace.iteration(i):
           process_batch(data[i])

   trace.finish()
   print(trace.summary())

2. GPUIterationTracer - Lightweight per-iteration tracking:

   tracer = GPUIterationTracer(report_interval=10)

   for i in range(100):
       tracer.start_iteration(i)
       process_batch(data[i])
       tracer.end_iteration(i)

       if tracer.should_report(i):
           print(tracer.get_iteration_report(i))

   print(tracer.get_summary())

3. Decorator/context manager - trace_gpu_iterations:

   with trace_gpu_iterations("my_loop") as trace:
       for i, batch in enumerate(batches):
           with trace.iteration(i):
               process(batch)

4. Manual debugging functions:
   - log_gpu_memory_breakdown(label) - Log detailed GPU memory stats
   - reset_gpu_memory_stats() - Reset PyTorch peak memory stats
   - get_gpu_memory_breakdown() - Get dict with detailed GPU memory info

The actor.py module automatically uses GPUIterationTracer for:
- GPU nodes in _run_standard_batching() (every iteration)
- GPU controlled nodes in _run_controlled_node() (every RunEvent)

Look for [GPU TRACE] log messages to see per-iteration memory changes.
===============================================================================
"""

from __future__ import annotations

import gc
import os
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import psutil

from nodetool.config.logging_config import get_logger

if TYPE_CHECKING:
    from types import FrameType

log = get_logger(__name__)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_usage_mb() -> tuple[float, float] | None:
    """
    Get current GPU memory usage in MB.

    Returns:
        Tuple of (allocated_mb, reserved_mb) or None if CUDA unavailable.
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return (allocated, reserved)
    except ImportError:
        pass
    return None


def log_memory(label: str, include_gpu: bool = True) -> None:
    """
    Log current memory usage with a label.

    Args:
        label: A descriptive label for this memory checkpoint.
        include_gpu: Whether to also log GPU memory usage.
    """
    ram_mb = get_memory_usage_mb()
    log.info(f"[MEMORY] {label}: RAM={ram_mb:.1f}MB")

    if include_gpu:
        gpu_mem = get_gpu_memory_usage_mb()
        if gpu_mem:
            allocated, reserved = gpu_mem
            log.info(f"[MEMORY] {label}: GPU allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")


def run_gc(label: str = "", log_before_after: bool = True) -> float:
    """
    Run garbage collection and optionally log memory before and after.

    Args:
        label: A descriptive label for this GC run.
        log_before_after: Whether to log memory usage before and after GC.

    Returns:
        Memory freed in MB (RAM only, approximate).
    """
    if log_before_after:
        before_mb = get_memory_usage_mb()
        log.info(f"[GC] {label} - Before GC: RAM={before_mb:.1f}MB")
    else:
        before_mb = get_memory_usage_mb()

    gc.collect()

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    after_mb = get_memory_usage_mb()
    freed_mb = before_mb - after_mb

    if log_before_after:
        log.info(f"[GC] {label} - After GC: RAM={after_mb:.1f}MB (freed {freed_mb:.1f}MB)")
        gpu_mem = get_gpu_memory_usage_mb()
        if gpu_mem:
            allocated, reserved = gpu_mem
            log.info(f"[GC] {label} - GPU after: allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")

    return freed_mb


def get_memory_uri_cache_stats() -> dict[str, int]:
    """
    Get stats about the memory URI cache.

    Returns:
        Dictionary with cache statistics.
    """
    try:
        from nodetool.runtime.resources import maybe_scope

        scope = maybe_scope()
        if scope:
            cache = scope.get_memory_uri_cache()
            if cache and hasattr(cache, "_cache"):
                count = len(cache._cache)
                return {"count": count}
    except Exception:
        pass
    return {"count": 0}


def clear_memory_uri_cache(log_stats: bool = True) -> int:
    """
    Clear the memory URI cache to free up RAM.

    Args:
        log_stats: Whether to log cache stats before clearing.

    Returns:
        Number of items cleared from cache.
    """
    try:
        from nodetool.runtime.resources import maybe_scope

        scope = maybe_scope()
        if scope:
            cache = scope.get_memory_uri_cache()
            if cache:
                count = 0
                if hasattr(cache, "_cache"):
                    count = len(cache._cache)
                if log_stats:
                    log.info(f"[MEMORY CACHE] Clearing {count} items from memory URI cache")

                # cache.clear() is synchronous - no async wrapper needed
                cache.clear()
                return count
    except Exception as e:
        log.debug(f"Failed to clear memory URI cache: {e}")
    return 0


def log_memory_summary(label: str = "Summary") -> dict:
    """
    Log a comprehensive memory summary.

    Returns:
        Dictionary with memory stats.
    """
    stats: dict = {
        "ram_mb": get_memory_usage_mb(),
    }

    gpu_mem = get_gpu_memory_usage_mb()
    if gpu_mem:
        stats["gpu_allocated_mb"] = gpu_mem[0]
        stats["gpu_reserved_mb"] = gpu_mem[1]

    cache_stats = get_memory_uri_cache_stats()
    stats["memory_cache_count"] = cache_stats.get("count", 0)

    log.info(f"[MEMORY SUMMARY] {label}")
    log.info(f"  RAM: {stats['ram_mb']:.1f}MB")
    if gpu_mem:
        log.info(f"  GPU allocated: {stats.get('gpu_allocated_mb', 0):.1f}MB")
        log.info(f"  GPU reserved: {stats.get('gpu_reserved_mb', 0):.1f}MB")
    log.info(f"  Memory URI cache items: {stats['memory_cache_count']}")

    return stats


class MemoryTracker:
    """
    Context manager for tracking memory usage during a block of code.

    Usage:
        with MemoryTracker("Loading model"):
            # ... load model code ...
    """

    def __init__(self, label: str, run_gc_after: bool = True):
        self.label = label
        self.run_gc_after = run_gc_after
        self.start_ram_mb = 0.0
        self.start_gpu: tuple[float, float] | None = None

    def __enter__(self):
        self.start_ram_mb = get_memory_usage_mb()
        self.start_gpu = get_gpu_memory_usage_mb()
        log.info(f"[MEMORY TRACK] {self.label} - START: RAM={self.start_ram_mb:.1f}MB")
        if self.start_gpu:
            log.info(f"[MEMORY TRACK] {self.label} - START: GPU allocated={self.start_gpu[0]:.1f}MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_ram_mb = get_memory_usage_mb()
        end_gpu = get_gpu_memory_usage_mb()

        ram_delta = end_ram_mb - self.start_ram_mb
        log.info(f"[MEMORY TRACK] {self.label} - END: RAM={end_ram_mb:.1f}MB (delta: {ram_delta:+.1f}MB)")

        if end_gpu and self.start_gpu:
            gpu_delta = end_gpu[0] - self.start_gpu[0]
            log.info(f"[MEMORY TRACK] {self.label} - END: GPU allocated={end_gpu[0]:.1f}MB (delta: {gpu_delta:+.1f}MB)")

        if self.run_gc_after:
            run_gc(f"{self.label} cleanup", log_before_after=True)

        return False


# =============================================================================
# DETAILED GPU TRACING UTILITIES
# =============================================================================

@dataclass
class GPUTraceSnapshot:
    """A snapshot of GPU memory state at a specific point in time."""
    timestamp: float
    label: str
    ram_mb: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_max_allocated_mb: float | None = None
    gpu_max_reserved_mb: float | None = None
    tensor_counts: dict[str, int] = field(default_factory=dict)
    frame_summary: list[str] = field(default_factory=list)


@dataclass
class IterationStats:
    """Statistics for a single iteration in a traced loop."""
    iteration: int
    start_time: float
    end_time: float = 0.0
    start_gpu_allocated: float = 0.0
    end_gpu_allocated: float = 0.0
    start_gpu_reserved: float = 0.0
    end_gpu_reserved: float = 0.0
    delta_allocated: float = 0.0
    delta_reserved: float = 0.0

    def __post_init__(self):
        if self.end_time > 0:
            self.delta_allocated = self.end_gpu_allocated - self.start_gpu_allocated
            self.delta_reserved = self.end_gpu_reserved - self.start_gpu_reserved


class GPUTraceSession:
    """
    A session for tracing GPU memory usage across multiple iterations.

    This class tracks GPU memory changes over time and helps identify
    what is causing memory growth in iterative processes.

    Usage:
        trace = GPUTraceSession("my_loop", log_interval=1)
        trace.start()

        for i in range(100):
            with trace.iteration(i):
                # Your iterative code here
                process_batch(data[i])

        trace.finish()
        print(trace.summary())
    """

    def __init__(
        self,
        name: str,
        log_interval: int = 1,
        log_callback=None,
        capture_frames: bool = False,
        track_tensors: bool = False,
    ):
        """
        Initialize a GPU trace session.

        Args:
            name: Name of the trace session for logging
            log_interval: Log every N iterations (1 = log all)
            log_callback: Optional callback function(log_msg) for custom logging
            capture_frames: Whether to capture stack frames (expensive)
            track_tensors: Whether to track tensor counts by device (expensive)
        """
        self.name = name
        self.log_interval = log_interval
        self.log_callback = log_callback or (lambda msg: log.info(msg))
        self.capture_frames = capture_frames
        self.track_tensors = track_tensors

        self.start_time: float = 0.0
        self.iterations: list[IterationStats] = []
        self.snapshots: list[GPUTraceSnapshot] = []
        self._current_iteration: int | None = None
        self._iteration_start_stats: tuple[float, float, float] | None = None

    def start(self) -> None:
        """Start the trace session."""
        self.start_time = time.time()
        self._log(f"[GPU TRACE] Starting session: {self.name}")

        # Take initial snapshot
        self._take_snapshot("session_start")

    def finish(self) -> None:
        """Finish the trace session and log summary."""
        # Take final snapshot
        self._take_snapshot("session_end")

        self._log(f"[GPU TRACE] Finished session: {self.name}")
        self._log(self.summary())

        # Log warnings if memory grew significantly
        if len(self.iterations) >= 2:
            first_iter = self.iterations[0]
            last_iter = self.iterations[-1]
            total_growth = last_iter.end_gpu_allocated - first_iter.start_gpu_allocated
            avg_growth_per_iter = total_growth / len(self.iterations)

            if avg_growth_per_iter > 10:  # More than 10MB per iteration
                self._log(
                    f"[GPU TRACE] WARNING: Average growth of {avg_growth_per_iter:.2f}MB per iteration detected!"
                )
                self._log(self._format_top_growth_iterations(5))

    @contextmanager
    def iteration(self, iteration_num: int):
        """
        Context manager for tracing a single iteration.

        Usage:
            for i in range(100):
                with trace.iteration(i):
                    process_batch(data[i])
        """
        self._current_iteration = iteration_num
        start_gpu = get_gpu_memory_usage_mb()
        start_allocated = start_gpu[0] if start_gpu else 0.0
        start_reserved = start_gpu[1] if start_gpu else 0.0

        self._iteration_start_stats = (time.time(), start_allocated, start_reserved)

        # Take snapshot at iteration start if requested
        if self._should_log_iteration(iteration_num):
            self._take_snapshot(f"iter_{iteration_num}_start")

        try:
            yield self
        finally:
            self._finish_iteration(iteration_num)

    def _finish_iteration(self, iteration_num: int) -> None:
        """Record end of iteration stats."""
        if self._iteration_start_stats is None:
            return

        start_time, start_allocated, start_reserved = self._iteration_start_stats
        end_time = time.time()

        end_gpu = get_gpu_memory_usage_mb()
        end_allocated = end_gpu[0] if end_gpu else 0.0
        end_reserved = end_gpu[1] if end_gpu else 0.0

        stats = IterationStats(
            iteration=iteration_num,
            start_time=start_time,
            end_time=end_time,
            start_gpu_allocated=start_allocated,
            end_gpu_allocated=end_allocated,
            start_gpu_reserved=start_reserved,
            end_gpu_reserved=end_reserved,
        )
        self.iterations.append(stats)

        # Take snapshot at iteration end if requested
        if self._should_log_iteration(iteration_num):
            self._take_snapshot(f"iter_{iteration_num}_end")
            self._log_iteration_stats(stats)

        self._current_iteration = None
        self._iteration_start_stats = None

    def snapshot(self, label: str) -> None:
        """Take a manual snapshot at any point."""
        self._take_snapshot(label)

    def _take_snapshot(self, label: str) -> GPUTraceSnapshot:
        """Take a GPU memory snapshot."""
        ram_mb = get_memory_usage_mb()
        gpu_mem = get_gpu_memory_usage_mb()

        gpu_allocated = gpu_mem[0] if gpu_mem else 0.0
        gpu_reserved = gpu_mem[1] if gpu_mem else 0.0

        # Get max memory stats if available
        gpu_max_allocated = None
        gpu_max_reserved = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
                gpu_max_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
        except ImportError:
            pass

        # Track tensor counts if requested
        tensor_counts = {}
        if self.track_tensors:
            tensor_counts = self._count_tensors_by_device()

        # Capture frame summary if requested
        frame_summary = []
        if self.capture_frames:
            frame_summary = self._get_frame_summary()

        snapshot = GPUTraceSnapshot(
            timestamp=time.time(),
            label=label,
            ram_mb=ram_mb,
            gpu_allocated_mb=gpu_allocated,
            gpu_reserved_mb=gpu_reserved,
            gpu_max_allocated_mb=gpu_max_allocated,
            gpu_max_reserved_mb=gpu_max_reserved,
            tensor_counts=tensor_counts,
            frame_summary=frame_summary,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def _count_tensors_by_device(self) -> dict[str, int]:
        """Count tensors by device (expensive operation)."""
        try:
            import torch
            counts = defaultdict(int)
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        device = str(obj.device)
                        counts[device] += 1
                except Exception:
                    pass
            return dict(counts)
        except ImportError:
            return {}

    def _get_frame_summary(self) -> list[str]:
        """Get a summary of the current stack frame."""
        frames = traceback.format_stack()
        # Filter out frames from this module
        filtered = [
            f for f in frames
            if "memory_utils.py" not in f and "<frozen" not in f
        ]
        # Keep only last 5 frames
        return filtered[-5:] if len(filtered) > 5 else filtered

    def _should_log_iteration(self, iteration_num: int) -> bool:
        """Check if we should log this iteration."""
        return iteration_num % self.log_interval == 0

    def _log(self, msg: str) -> None:
        """Log a message via the configured callback."""
        self.log_callback(msg)

    def _log_iteration_stats(self, stats: IterationStats) -> None:
        """Log stats for a single iteration."""
        duration_ms = (stats.end_time - stats.start_time) * 1000

        if abs(stats.delta_allocated) > 0.1 or abs(stats.delta_reserved) > 0.1:
            # Log significant changes
            self._log(
                f"[GPU TRACE] {self.name} iter={stats.iteration}: "
                f"GPU alloc={stats.end_gpu_allocated:.1f}MB ({stats.delta_allocated:+.2f}MB), "
                f"reserved={stats.end_gpu_reserved:.1f}MB ({stats.delta_reserved:+.2f}MB), "
                f"duration={duration_ms:.1f}ms"
            )
        else:
            # Brief log for stable iterations
            self._log(
                f"[GPU TRACE] {self.name} iter={stats.iteration}: "
                f"GPU alloc={stats.end_gpu_allocated:.1f}MB, "
                f"duration={duration_ms:.1f}ms"
            )

    def _format_top_growth_iterations(self, n: int) -> str:
        """Format the top N iterations with highest GPU memory growth."""
        sorted_iters = sorted(
            self.iterations,
            key=lambda x: x.delta_allocated,
            reverse=True
        )[:n]

        lines = [f"[GPU TRACE] Top {n} iterations with highest GPU growth:"]
        for stats in sorted_iters:
            lines.append(
                f"  Iter {stats.iteration}: {stats.delta_allocated:+.2f}MB "
                f"(alloc: {stats.start_gpu_allocated:.1f} -> {stats.end_gpu_allocated:.1f})"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        """Generate a summary of the trace session."""
        if not self.iterations:
            return f"[GPU TRACE] {self.name}: No iterations recorded"

        duration = time.time() - self.start_time
        first_iter = self.iterations[0]
        last_iter = self.iterations[-1]

        total_growth = last_iter.end_gpu_allocated - first_iter.start_gpu_allocated
        avg_growth = total_growth / len(self.iterations)
        max_allocated = max(iter_.end_gpu_allocated for iter_ in self.iterations)

        lines = [
            f"[GPU TRACE] {self.name} Summary:",
            f"  Iterations: {len(self.iterations)}",
            f"  Duration: {duration:.2f}s",
            f"  Initial GPU: {first_iter.start_gpu_allocated:.1f}MB",
            f"  Final GPU: {last_iter.end_gpu_allocated:.1f}MB",
            f"  Total growth: {total_growth:+.2f}MB",
            f"  Avg growth/iter: {avg_growth:+.2f}MB",
            f"  Peak GPU: {max_allocated:.1f}MB",
        ]

        # Add tensor tracking info if available
        if self.track_tensors and self.snapshots:
            last_snapshot = self.snapshots[-1]
            lines.append("  Tensor counts:")
            for device, count in last_snapshot.tensor_counts.items():
                lines.append(f"    {device}: {count}")

        return "\n".join(lines)

    def get_memory_growth_pattern(self) -> dict:
        """Analyze memory growth pattern across iterations."""
        if len(self.iterations) < 2:
            return {"error": "Not enough iterations to analyze pattern"}

        # Calculate deltas
        deltas = [iter_.delta_allocated for iter_ in self.iterations]

        # Categorize iterations
        stable = sum(1 for d in deltas if abs(d) < 1.0)  # < 1MB change
        growing = sum(1 for d in deltas if d > 1.0)       # > 1MB growth
        shrinking = sum(1 for d in deltas if d < -1.0)    # > 1MB freed

        # Calculate trend
        if len(deltas) >= 10:
            first_half_avg = sum(deltas[:len(deltas)//2]) / (len(deltas)//2)
            second_half_avg = sum(deltas[len(deltas)//2:]) / (len(deltas) - len(deltas)//2)
        else:
            first_half_avg = second_half_avg = sum(deltas) / len(deltas)

        return {
            "total_iterations": len(self.iterations),
            "stable_iterations": stable,
            "growing_iterations": growing,
            "shrinking_iterations": shrinking,
            "first_half_avg_delta_mb": round(first_half_avg, 2),
            "second_half_avg_delta_mb": round(second_half_avg, 2),
            "trend": "increasing" if second_half_avg > first_half_avg else "stable/decreasing",
            "max_single_iter_growth_mb": round(max(deltas), 2),
            "total_growth_mb": round(sum(deltas), 2),
        }


class GPUIterationTracer:
    """
    A simpler tracer for single iteration tracking.

    Usage:
        tracer = GPUIterationTracer()

        for i in range(100):
            tracer.start_iteration(i)
            process_batch(data[i])
            tracer.end_iteration(i)

            if tracer.should_report():
                print(tracer.get_iteration_report(i))
    """

    def __init__(self, report_interval: int = 10):
        self.report_interval = report_interval
        self.iteration_stats: dict[int, dict] = {}
        self._current_start: tuple[int, float, float, float] | None = None

    def start_iteration(self, iteration: int) -> None:
        """Mark the start of an iteration."""
        gpu_mem = get_gpu_memory_usage_mb()
        self._current_start = (
            iteration,
            time.time(),
            gpu_mem[0] if gpu_mem else 0.0,
            gpu_mem[1] if gpu_mem else 0.0,
        )

    def end_iteration(self, iteration: int) -> dict | None:
        """Mark the end of an iteration and return stats."""
        if self._current_start is None or self._current_start[0] != iteration:
            return None

        _, start_time, start_alloc, start_res = self._current_start
        end_time = time.time()

        gpu_mem = get_gpu_memory_usage_mb()
        end_alloc = gpu_mem[0] if gpu_mem else 0.0
        end_res = gpu_mem[1] if gpu_mem else 0.0

        stats = {
            "iteration": iteration,
            "duration_ms": (end_time - start_time) * 1000,
            "gpu_allocated_start_mb": start_alloc,
            "gpu_allocated_end_mb": end_alloc,
            "gpu_allocated_delta_mb": end_alloc - start_alloc,
            "gpu_reserved_start_mb": start_res,
            "gpu_reserved_end_mb": end_res,
            "gpu_reserved_delta_mb": end_res - start_res,
        }
        self.iteration_stats[iteration] = stats
        self._current_start = None
        return stats

    def should_report(self, iteration: int | None = None) -> bool:
        """Check if we should report for this iteration."""
        if iteration is None:
            iteration = len(self.iteration_stats) - 1
        return iteration % self.report_interval == 0

    def get_iteration_report(self, iteration: int) -> str:
        """Get a report for a specific iteration."""
        stats = self.iteration_stats.get(iteration)
        if not stats:
            return f"[GPU TRACE] No stats for iteration {iteration}"

        delta = stats["gpu_allocated_delta_mb"]
        delta_str = f"{delta:+.2f}MB" if abs(delta) >= 0.01 else "stable"

        return (
            f"[GPU TRACE] Iter {iteration}: "
            f"GPU={stats['gpu_allocated_end_mb']:.1f}MB ({delta_str}), "
            f"time={stats['duration_ms']:.1f}ms"
        )

    def get_summary(self) -> str:
        """Get a summary of all iterations."""
        if not self.iteration_stats:
            return "[GPU TRACE] No iterations recorded"

        iterations = sorted(self.iteration_stats.keys())
        first_stats = self.iteration_stats[iterations[0]]
        last_stats = self.iteration_stats[iterations[-1]]

        total_growth = last_stats["gpu_allocated_end_mb"] - first_stats["gpu_allocated_start_mb"]
        avg_iter_time = sum(s["duration_ms"] for s in self.iteration_stats.values()) / len(self.iteration_stats)

        return (
            f"[GPU TRACE] Summary: {len(iterations)} iters, "
            f"{total_growth:+.2f}MB GPU growth ({total_growth/len(iterations):+.3f}MB/iter), "
            f"{avg_iter_time:.1f}ms avg time"
        )


def trace_gpu_iterations(
    name: str = "gpu_loop",
    log_interval: int = 1,
    capture_frames: bool = False,
    track_tensors: bool = False,
):
    """
    Decorator/context manager factory for tracing GPU memory in loops.

    Usage as decorator:
        @trace_gpu_iterations("my_loop", log_interval=10)
        async def process_batches(batches):
            for batch in batches:
                yield process(batch)

    Usage as context manager:
        with trace_gpu_iterations("my_loop") as trace:
            for i, batch in enumerate(batches):
                with trace.iteration(i):
                    process(batch)

    Args:
        name: Name for the trace session
        log_interval: Log every N iterations
        capture_frames: Whether to capture stack frames (expensive)
        track_tensors: Whether to track tensor counts (expensive)

    Returns:
        GPUTraceSession instance
    """
    return GPUTraceSession(
        name=name,
        log_interval=log_interval,
        capture_frames=capture_frames,
        track_tensors=track_tensors,
    )


# =============================================================================
# DEBUG UTILITIES FOR GPU MEMORY ANALYSIS
# =============================================================================

def get_gpu_memory_breakdown() -> dict:
    """
    Get a detailed breakdown of GPU memory usage.

    Returns:
        Dictionary with detailed GPU memory stats.
    """
    result = {
        "available": False,
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "max_reserved_mb": 0.0,
        "devices": [],
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return result

        result["available"] = True
        result["allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        result["reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        result["max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        result["max_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)

        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            result["devices"].append({
                "id": i,
                "name": device_props.name,
                "total_memory_mb": device_props.total_memory / (1024 * 1024),
                "allocated_mb": torch.cuda.memory_allocated(i) / (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved(i) / (1024 * 1024),
            })
    except ImportError:
        pass

    return result


def log_gpu_memory_breakdown(label: str = "GPU Memory Breakdown") -> None:
    """Log a detailed breakdown of GPU memory usage."""
    breakdown = get_gpu_memory_breakdown()

    if not breakdown["available"]:
        log.info(f"[GPU BREAKDOWN] {label}: CUDA not available")
        return

    log.info(f"[GPU BREAKDOWN] {label}:")
    log.info(f"  Allocated: {breakdown['allocated_mb']:.1f}MB")
    log.info(f"  Reserved: {breakdown['reserved_mb']:.1f}MB")
    log.info(f"  Max Allocated: {breakdown['max_allocated_mb']:.1f}MB")
    log.info(f"  Max Reserved: {breakdown['max_reserved_mb']:.1f}MB")

    for device in breakdown["devices"]:
        log.info(f"  Device {device['id']} ({device['name']}):")
        log.info(f"    Total: {device['total_memory_mb']:.1f}MB")
        log.info(f"    Allocated: {device['allocated_mb']:.1f}MB")
        log.info(f"    Reserved: {device['reserved_mb']:.1f}MB")


def reset_gpu_memory_stats() -> None:
    """Reset PyTorch GPU memory stats (max values)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            log.info("[GPU] Reset peak memory stats")
    except ImportError:
        pass


def cleanup_gpu_memory(force: bool = False) -> dict[str, float]:
    """
    Perform comprehensive GPU memory cleanup after node execution.

    This function should be called after GPU-intensive operations to ensure
    intermediate tensors and cache are properly freed. It:
    1. Synchronizes CUDA to ensure all operations complete
    2. Runs Python garbage collection
    3. Empties CUDA cache
    4. Collects IPC memory (if available)

    Args:
        force: If True, performs more aggressive cleanup (slower but more thorough)

    Returns:
        Dictionary with memory stats before/after cleanup
    """
    stats = {
        "allocated_before_mb": 0.0,
        "allocated_after_mb": 0.0,
        "freed_mb": 0.0,
    }

    try:
        import gc

        import torch

        if not torch.cuda.is_available():
            log.debug("GPU cleanup: CUDA not available")
            return stats

        # Record memory before cleanup
        torch.cuda.synchronize()
        stats["allocated_before_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)

        # Run Python GC first to free unreachable objects that might hold GPU tensors
        gc.collect()

        # Empty CUDA cache - this frees memory PyTorch's caching allocator is holding
        torch.cuda.empty_cache()

        if force:
            # IPC collect is only needed in multi-process scenarios but is harmless
            with suppress(Exception):
                torch.cuda.ipc_collect()
            # Force synchronization again after aggressive cleanup
            torch.cuda.synchronize()

        stats["allocated_after_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        stats["freed_mb"] = stats["allocated_before_mb"] - stats["allocated_after_mb"]

        log.debug(
            f"GPU cleanup: before={stats['allocated_before_mb']:.1f}MB, "
            f"after={stats['allocated_after_mb']:.1f}MB, freed={stats['freed_mb']:.1f}MB"
        )

    except ImportError:
        log.debug("GPU cleanup: torch not available")
    except Exception as e:
        log.debug(f"Error during GPU cleanup: {e}")

    return stats


__all__ = [
    "GPUIterationTracer",
    "GPUTraceSession",
    "GPUTraceSnapshot",
    "IterationStats",
    "MemoryTracker",
    "cleanup_gpu_memory",
    "get_gpu_memory_breakdown",
    "get_gpu_memory_usage_mb",
    "get_memory_usage_mb",
    "log_gpu_memory_breakdown",
    "log_memory",
    "log_memory_summary",
    "reset_gpu_memory_stats",
    "run_gc",
    "trace_gpu_iterations",
]
