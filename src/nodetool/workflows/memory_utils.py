"""
Memory logging and garbage collection utilities for workflow execution.

This module provides utilities for tracking memory usage and performing
garbage collection to help diagnose memory leaks and reduce RAM usage.
"""

from __future__ import annotations

import gc
import os

import psutil

from nodetool.config.logging_config import get_logger

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
