"""
Manages ML model instances in non-production environments.

This module provides the ModelManager class, a central repository for storing,
retrieving, and managing machine learning models during development or testing.
It associates models with specific nodes and handles their lifecycle, preventing
resource leaks by clearing unused models. This functionality is disabled in
production environments.
"""

import asyncio
import gc
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, ClassVar, Dict, NamedTuple

import psutil

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

logger = get_logger(__name__)


class MemorySnapshot(NamedTuple):
    """Light-weight container for system/process memory telemetry.

    Attributes:
        percent: Percentage of system RAM currently in use.
        available_gb: Free system memory in gigabytes.
        total_gb: Total system memory in gigabytes.
        process_rss_gb: Current process Resident Set Size in gigabytes.
    """

    percent: float
    available_gb: float
    total_gb: float
    process_rss_gb: float


class ModelManager:
    """Manages ML model instances and their associations with nodes.

    This class provides a centralized way to store, retrieve, and manage machine learning
    models in non-production environments. It maintains mappings between models and nodes
    and provides utilities for model lifecycle management.

    Attributes:
        _models (Dict[str, Any]): Storage for model instances keyed by model_id, task, and path
        _models_by_node (Dict[str, str]): Mapping of node IDs to model keys
        _locks (Dict[str, asyncio.Lock]): Per-model locks for thread-safe access
        _lock_creation_lock (asyncio.Lock): Lock for safely creating new per-model locks
    """

    _models: ClassVar[Dict[str, Any]] = {}
    _models_by_node: ClassVar[Dict[str, str]] = {}
    _locks: ClassVar[Dict[str, asyncio.Lock]] = {}
    _lock_creation_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _last_memory_cleanup: ClassVar[float] = 0.0
    _DEFAULT_MAX_MEMORY_PERCENT: ClassVar[float] = 92.0
    _DEFAULT_MIN_AVAILABLE_GB: ClassVar[float] = 1.0
    _DEFAULT_MEMORY_COOLDOWN_SECONDS: ClassVar[float] = 30.0

    @classmethod
    def get_model(cls, model_id: str, task: str, path: str | None = None) -> Any:
        """Retrieves a model instance based on the given parameters.

        Args:
            model_id (str): Identifier for the model
            task (str): Task associated with the model
            path (str | None): Optional path parameter

        Returns:
            Any: The stored model instance if found in non-production environment, None otherwise
        """
        if not Environment.is_production():
            key = f"{model_id}_{task}_{path}"
            model = cls._models.get(key)
            if model is not None:
                logger.info(
                    f"✓ Cache HIT: Retrieved cached model for {model_id} (task: {task}, path: {path})"
                )
            else:
                logger.info(
                    f"✗ Cache MISS: No cached model found for {model_id} (task: {task}, path: {path})"
                )
            logger.debug(
                f"Model cache status - Total models: {len(cls._models)}, Key searched: {key}"
            )
            return model
        else:
            logger.debug(
                f"Production environment: Model caching disabled for {model_id}"
            )
        return None

    @classmethod
    def set_model(
        cls, node_id: str, model_id: str, task: str, model: Any, path: str | None = None
    ):
        """Stores a model instance and associates it with a node.

        Args:
            node_id (str): ID of the node associated with the model
            model_id (str): Identifier for the model
            task (str): Task associated with the model
            model (Any): The model instance to store
            path (str | None): Optional path parameter
        """
        if not Environment.is_production():
            cls._ensure_memory_capacity(
                reason=f"Preparing to cache model {model_id} (task: {task}, path: {path})"
            )

            key = f"{model_id}_{task}_{path}"
            was_existing = key in cls._models
            cls._models[key] = model
            cls._models_by_node[node_id] = key

            if was_existing:
                logger.info(
                    f"↻ Cache UPDATE: Replaced cached model for {model_id} (task: {task}, path: {path}) - Node: {node_id}"
                )
            else:
                logger.info(
                    f"+ Cache STORE: Cached new model for {model_id} (task: {task}, path: {path}) - Node: {node_id}"
                )

            logger.debug(
                f"Model cache status - Total models: {len(cls._models)}, Node associations: {len(cls._models_by_node)}"
            )
        else:
            logger.debug(
                f"Production environment: Model caching disabled, not storing {model_id} for node {node_id}"
            )

    @classmethod
    async def get_model_lock(
        cls, model_id: str, task: str, path: str | None = None
    ) -> asyncio.Lock:
        """Gets or creates a lock for a specific model.

        This method ensures thread-safe access to individual models by providing
        per-model locks. The lock creation itself is protected by a global lock
        to prevent race conditions.

        Args:
            model_id (str): Identifier for the model
            task (str): Task associated with the model
            path (str | None): Optional path parameter

        Returns:
            asyncio.Lock: The lock associated with this model

        Example:
            lock = await ModelManager.get_model_lock("gpt-4", "text-generation")
            async with lock:
                model = ModelManager.get_model("gpt-4", "text-generation")
                # ... use model safely ...
        """
        key = f"{model_id}_{task}_{path}"

        # Check if lock exists (fast path without acquiring lock)
        if key in cls._locks:
            return cls._locks[key]

        # Slow path: need to create the lock
        async with cls._lock_creation_lock:
            # Double-check after acquiring lock (another coroutine might have created it)
            if key not in cls._locks:
                cls._locks[key] = asyncio.Lock()
                logger.debug(
                    f"🔒 Created new lock for model: {model_id} (task: {task}, path: {path})"
                )
            return cls._locks[key]

    @classmethod
    @asynccontextmanager
    async def lock_model(
        cls, model_id: str, task: str, path: str | None = None
    ) -> AsyncIterator[None]:
        """Context manager for acquiring exclusive access to a model.

        This provides a convenient way to ensure thread-safe access to models
        without manually managing lock acquisition and release.

        Args:
            model_id (str): Identifier for the model
            task (str): Task associated with the model
            path (str | None): Optional path parameter

        Yields:
            None

        Example:
            async with ModelManager.lock_model("gpt-4", "text-generation"):
                model = ModelManager.get_model("gpt-4", "text-generation")
                # ... use model exclusively ...
                # Lock is automatically released when exiting the context
        """
        lock = await cls.get_model_lock(model_id, task, path)
        key = f"{model_id}_{task}_{path}"
        logger.debug(
            f"🔐 Acquiring lock for model: {model_id} (task: {task}, path: {path})"
        )
        async with lock:
            logger.debug(f"✓ Lock acquired for model: {key}")
            try:
                yield
            finally:
                logger.debug(f"🔓 Releasing lock for model: {key}")

    @classmethod
    def clear_unused(cls, node_ids: list[str]):
        """Removes models that are no longer associated with active nodes.

        Also cleans up locks associated with the removed models.

        Args:
            node_ids (list[str]): List of active node IDs to check against
        """
        cleared_count = 0
        cleared_models = []
        cleared_locks = 0

        for node_id in node_ids:
            key = cls._models_by_node.pop(node_id, None)
            if key:
                if key in cls._models:
                    # Extract model info for logging
                    parts = key.split("_", 2)
                    model_id = parts[0] if len(parts) > 0 else "unknown"
                    task = parts[1] if len(parts) > 1 else "unknown"
                    path = parts[2] if len(parts) > 2 else None

                    del cls._models[key]
                    cleared_count += 1
                    cleared_models.append(f"{model_id} (task: {task}, path: {path})")
                    logger.debug(
                        f"- Cleared cached model for node {node_id}: {model_id}"
                    )

                    # Clean up associated lock
                    if key in cls._locks:
                        del cls._locks[key]
                        cleared_locks += 1
                        logger.debug(f"🔒 Removed lock for cleared model: {key}")

        if cleared_count > 0:
            logger.info(
                f"🗑️ Cache CLEANUP: Removed {cleared_count} unused models: {', '.join(cleared_models)}"
            )
            if cleared_locks > 0:
                logger.debug(f"🔒 Removed {cleared_locks} associated locks")
            logger.debug(
                f"Model cache status after cleanup - Total models: {len(cls._models)}, Node associations: {len(cls._models_by_node)}, Locks: {len(cls._locks)}"
            )
        else:
            logger.debug("Cache cleanup: No unused models to remove")

    @classmethod
    def clear(cls):
        """Removes all stored models, node associations, and locks."""
        model_count = len(cls._models)
        node_count = len(cls._models_by_node)
        lock_count = len(cls._locks)

        # Log which models are being cleared
        if model_count > 0:
            model_info = []
            for key in cls._models:
                parts = key.split("_", 2)
                model_id = parts[0] if len(parts) > 0 else "unknown"
                task = parts[1] if len(parts) > 1 else "unknown"
                path = parts[2] if len(parts) > 2 else None
                model_info.append(f"{model_id} (task: {task}, path: {path})")

            logger.info(
                f"🧹 Cache CLEAR ALL: Removing {model_count} cached models, {node_count} node associations, {lock_count} locks"
            )
            logger.debug(f"Models being cleared: {', '.join(model_info)}")
        else:
            logger.debug("Cache clear: No models to remove")

        cls._models.clear()
        cls._models_by_node.clear()
        cls._locks.clear()

        if model_count > 0:
            logger.info(
                f"✅ Cache cleared successfully: {model_count} models removed, {lock_count} locks removed"
            )

    # ------------------------------------------------------------------
    # Memory management helpers
    # ------------------------------------------------------------------

    @classmethod
    def free_memory_if_needed(cls, reason: str = "manual request") -> None:
        """Force a cache purge regardless of current telemetry.

        Args:
            reason: Human-readable reason for the manual cleanup. This is surfaced
                in logs to correlate cache purges with upstream triggers.
        """
        cls._ensure_memory_capacity(reason=reason, aggressive=True)

    @classmethod
    def _ensure_memory_capacity(cls, *, reason: str, aggressive: bool = False) -> None:
        """Check current memory pressure and clear cached models if needed.

        Args:
            reason: Short description, propagated to log messages when the cache
                is purged so operators can attribute the cleanup.
            aggressive: When True, bypasses thresholds/cooldowns and clears
                models even if telemetry is unavailable.
        """
        if Environment.is_production():
            return

        snapshot = cls._capture_memory_snapshot()
        if snapshot is None:
            if aggressive:
                logger.warning(
                    "Memory cleanup requested (%s) but unable to capture memory stats. Clearing cached models anyway.",
                    reason,
                )
                cls.clear()
                gc.collect()
                cls._last_memory_cleanup = time.monotonic()
            return

        if not aggressive and not cls._needs_memory_cleanup(snapshot):
            return

        cooldown = cls._get_memory_cleanup_cooldown()
        now = time.monotonic()
        if (
            not aggressive
            and cooldown > 0
            and (now - cls._last_memory_cleanup) < cooldown
        ):
            remaining = cooldown - (now - cls._last_memory_cleanup)
            logger.debug(
                "Memory pressure detected but cleanup throttled for %.2fs (usage %.2f%%, %.2f GB free)",
                max(remaining, 0.0),
                snapshot.percent,
                snapshot.available_gb,
            )
            return

        cls._perform_memory_cleanup(snapshot, reason)
        cls._last_memory_cleanup = now

    @classmethod
    def _perform_memory_cleanup(cls, snapshot: MemorySnapshot, reason: str) -> None:
        """Clear cached models and collect garbage when memory is constrained.

        Args:
            snapshot: Memory telemetry captured immediately before the cleanup.
            reason: Textual justification that will be rendered in the warning log.
        """
        removed = len(cls._models)
        logger.warning(
            (
                "Memory pressure detected (usage %.2f%%, %.2f GB free of %.2f GB total, "
                "process RSS %.2f GB). Clearing %d cached model(s). Reason: %s"
            ),
            snapshot.percent,
            snapshot.available_gb,
            snapshot.total_gb,
            snapshot.process_rss_gb,
            removed,
            reason,
        )
        cls.clear()
        gc.collect()

    @classmethod
    def _needs_memory_cleanup(cls, snapshot: MemorySnapshot) -> bool:
        """Return True if current memory snapshot violates thresholds."""
        max_percent, min_available = cls._get_memory_thresholds()
        return snapshot.percent >= max_percent or snapshot.available_gb <= min_available

    @classmethod
    def _capture_memory_snapshot(cls) -> MemorySnapshot | None:
        """Capture system + process memory usage for evaluating pressure.

        Returns:
            MemorySnapshot containing usage stats, or None when psutil raises an
            unexpected error (extremely rare).
        """
        try:
            vm = psutil.virtual_memory()
            process = psutil.Process()
            mem = process.memory_info()
            available_gb = float(vm.available) / (1024**3)
            total_gb = float(vm.total) / (1024**3)
            rss_gb = float(mem.rss) / (1024**3)
            snapshot = MemorySnapshot(
                percent=float(vm.percent),
                available_gb=available_gb,
                total_gb=total_gb,
                process_rss_gb=rss_gb,
            )
            logger.debug(
                "Memory snapshot captured: %.2f%% used, %.2f GB available, process RSS %.2f GB",
                snapshot.percent,
                snapshot.available_gb,
                snapshot.process_rss_gb,
            )
            return snapshot
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unable to capture memory stats: %s", exc)
            return None

    @classmethod
    def _get_memory_thresholds(cls) -> tuple[float, float]:
        """Return thresholds for cleanup decisions (constants for now).

        Returns:
            Tuple of (max_percent, min_available_gb) representing the point at
            which the cache should be purged.
        """
        max_percent = min(max(cls._DEFAULT_MAX_MEMORY_PERCENT, 10.0), 99.0)
        min_available = max(cls._DEFAULT_MIN_AVAILABLE_GB, 0.0)
        return max_percent, min_available

    @classmethod
    def _get_memory_cleanup_cooldown(cls) -> float:
        """Seconds to wait between automatic cleanups."""
        return max(cls._DEFAULT_MEMORY_COOLDOWN_SECONDS, 0.0)
