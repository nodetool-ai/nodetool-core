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
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, ClassVar, NamedTuple

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


class VramSnapshot(NamedTuple):
    """VRAM telemetry for the current process/device."""

    percent: float
    available_gb: float
    total_gb: float
    process_allocated_gb: float


class ModelManager:
    """Manages ML model instances and their associations with nodes.

    This class provides a centralized way to store, retrieve, and manage machine learning
    models in non-production environments. It maintains mappings between models and nodes
    and provides utilities for model lifecycle management.

    Attributes:
        _models (Dict[str, Any]): Storage for model instances keyed by model_id, task, and path
        _models_by_node (Dict[str, set[str]]): Mapping of node IDs to model cache keys
        _locks (Dict[str, asyncio.Lock]): Per-model locks for thread-safe access
        _lock_creation_lock (asyncio.Lock): Lock for safely creating new per-model locks
        _model_last_used (Dict[str, float]): Last-used timestamps per cached model key
        _node_last_used (Dict[str, float]): Last-used timestamps per node ID
        _model_device (Dict[str, str]): Known device for cached models (e.g., "cpu", "cuda:0")
        _model_size_bytes (Dict[str, int]): Approximate model size in bytes when available
    """

    _models: ClassVar[dict[str, Any]] = {}
    _models_by_node: ClassVar[dict[str, set[str]]] = {}
    _locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _lock_creation_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _last_memory_cleanup: ClassVar[float] = 0.0
    _model_last_used: ClassVar[dict[str, float]] = {}
    _node_last_used: ClassVar[dict[str, float]] = {}
    _model_device: ClassVar[dict[str, str]] = {}
    _model_size_bytes: ClassVar[dict[str, int]] = {}
    _last_vram_cleanup: ClassVar[float] = 0.0
    _DEFAULT_MAX_MEMORY_PERCENT: ClassVar[float] = 92.0
    _DEFAULT_MIN_AVAILABLE_GB: ClassVar[float] = 1.0
    _DEFAULT_MEMORY_COOLDOWN_SECONDS: ClassVar[float] = 30.0
    _DEFAULT_MAX_VRAM_PERCENT: ClassVar[float] = 92.0
    _DEFAULT_MIN_VRAM_AVAILABLE_GB: ClassVar[float] = 1.0
    _DEFAULT_VRAM_COOLDOWN_SECONDS: ClassVar[float] = 30.0

    @classmethod
    def get_model(cls, cache_key: str) -> Any:
        """Retrieves a model instance based on the given parameters.

        Args:
            cache_key (str): Cache key for the model

        Returns:
            Any: The stored model instance if found in non-production environment, None otherwise
        """
        model = cls._models.get(cache_key)
        if model is not None:
            cls._update_model_metadata(cache_key, model)
            logger.info(f"✓ Cache HIT: Retrieved cached model for {cache_key}")
        else:
            logger.info(f"✗ Cache MISS: No cached model found for {cache_key}")
        logger.debug(f"Model cache status - Total models: {len(cls._models)}, Key searched: {cache_key}")
        return model

    @classmethod
    def set_model(
        cls,
        node_id: str | None,
        model_id_or_cache_key: str,
        task_or_model: Any,
        model: Any | None = None,
    ):
        """Stores a model instance and associates it with a node.

        Args:
            node_id (str | None): ID of the node associated with the model
            model_id_or_cache_key (str): Cache key, or model id (back-compat)
            task_or_model (Any): Model instance, or task name (back-compat)
            model (Any | None): Model instance when using the legacy signature
        """
        if model is None:
            cache_key = model_id_or_cache_key
            model_instance = task_or_model
        else:
            task = str(task_or_model)
            cache_key = f"{model_id_or_cache_key}_{task}" if task else model_id_or_cache_key
            model_instance = model

        cls._ensure_memory_capacity(reason=f"Preparing to cache model {cache_key}")

        was_existing = cache_key in cls._models
        cls._models[cache_key] = model_instance
        if node_id is not None:
            cls._models_by_node.setdefault(node_id, set()).add(cache_key)
        cls._update_model_metadata(cache_key, model_instance, node_id=node_id)

        if was_existing:
            logger.info(f"↻ Cache UPDATE: Replaced cached model for {cache_key} - Node: {node_id}")
        else:
            logger.info(f"+ Cache STORE: Cached new model for {cache_key} - Node: {node_id}")

        logger.debug(
            "Model cache status - Total models: %d, Node associations: %d",
            len(cls._models),
            sum(len(keys) for keys in cls._models_by_node.values()),
        )

    @classmethod
    async def get_model_lock(cls, cache_key: str) -> asyncio.Lock:
        """Gets or creates a lock for a specific model.

        This method ensures thread-safe access to individual models by providing
        per-model locks. The lock creation itself is protected by a global lock
        to prevent race conditions.

        Args:
            cache_key (str): Cache key for the model

        Returns:
            asyncio.Lock: The lock associated with this model

        Example:
            cache_key = "gpt-4_text-generation"
            lock = await ModelManager.get_model_lock(cache_key)
            async with lock:
                model = ModelManager.get_model(cache_key)
                # ... use model safely ...
        """
        # Check if lock exists (fast path without acquiring lock)
        if cache_key in cls._locks:
            return cls._locks[cache_key]

        # Slow path: need to create the lock
        async with cls._lock_creation_lock:
            # Double-check after acquiring lock (another coroutine might have created it)
            if cache_key not in cls._locks:
                cls._locks[cache_key] = asyncio.Lock()
                logger.debug(f"🔒 Created new lock for model: {cache_key}")
            return cls._locks[cache_key]

    @classmethod
    @asynccontextmanager
    async def lock_model(cls, cache_key: str) -> AsyncIterator[None]:
        """Context manager for acquiring exclusive access to a model.

        This provides a convenient way to ensure thread-safe access to models
        without manually managing lock acquisition and release.

        Args:
            cache_key (str): Cache key for the model

        Yields:
            None

        Example:
            cache_key = "gpt-4_text-generation"
            async with ModelManager.lock_model(cache_key):
                model = ModelManager.get_model(cache_key)
                # ... use model exclusively ...
                # Lock is automatically released when exiting the context
        """
        lock = await cls.get_model_lock(cache_key)
        logger.debug(f"🔐 Acquiring lock for model: {cache_key}")
        async with lock:
            logger.debug(f"✓ Lock acquired for model: {cache_key}")
            try:
                yield
            finally:
                logger.debug(f"🔓 Releasing lock for model: {cache_key}")

    # ------------------------------------------------------------------
    # Usage tracking helpers
    # ------------------------------------------------------------------

    @classmethod
    def _mark_model_used(cls, key: str, node_id: str | None = None) -> None:
        """Record the last-used timestamp for a cached model and its node(s)."""

        now = time.monotonic()
        cls._model_last_used[key] = now

        if node_id is not None:
            cls._node_last_used[node_id] = now
            return

        for mapped_node_id, mapped_keys in list(cls._models_by_node.items()):
            if key in mapped_keys:
                cls._node_last_used[mapped_node_id] = now

    @classmethod
    def _update_model_metadata(cls, key: str, model: Any, node_id: str | None = None) -> None:
        """Refresh usage and device metadata for a cached model."""

        cls._mark_model_used(key, node_id=node_id)
        needs_device = key not in cls._model_device or cls._model_device.get(key) == "unknown"
        needs_size = key not in cls._model_size_bytes

        if needs_device or needs_size:
            device, size_bytes = cls._detect_torch_model_device_and_size(model)

            if device != "unknown":
                cls._model_device[key] = device
            if size_bytes is not None:
                cls._model_size_bytes[key] = size_bytes

    @classmethod
    def get_model_last_used(cls, cache_key: str) -> float | None:
        """Return the last-used timestamp for a cached model, if available."""

        return cls._model_last_used.get(cache_key)

    @classmethod
    def get_least_recently_used_models(cls, limit: int | None = None) -> list[tuple[str, float]]:
        """Return cached model keys ordered from least to most recently used."""

        items = sorted(cls._model_last_used.items(), key=lambda item: item[1])
        if limit is None or limit < 0:
            return items
        return items[:limit]

    @classmethod
    def get_least_recently_used_nodes(cls, limit: int | None = None) -> list[tuple[str, float]]:
        """Return node IDs ordered from least to most recently used."""

        items = sorted(cls._node_last_used.items(), key=lambda item: item[1])
        if limit is None or limit < 0:
            return items
        return items[:limit]

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_torch_model_device_and_size(model: Any) -> tuple[str, int | None]:
        """Best-effort detection of model device and approximate size.

        Returns (device, size_bytes). Device is "unknown" when torch is not
        installed or the model does not expose device metadata.
        """

        try:  # pragma: no cover - optional dependency
            pass  # type: ignore
        except Exception:
            return "unknown", None

        try:
            if hasattr(model, "parameters"):
                params = list(model.parameters())  # type: ignore[attr-defined]
                if params:
                    device = str(params[0].device)
                    size_bytes = sum(p.numel() * p.element_size() for p in params)
                    return device, int(size_bytes)

            if all(hasattr(model, attr) for attr in ("device", "numel", "element_size")):
                device = str(model.device)  # type: ignore[attr-defined]
                size_bytes = int(model.numel() * model.element_size())  # type: ignore[attr-defined]
                return device, size_bytes
        except Exception:
            return "unknown", None

        return "unknown", None

    @staticmethod
    def _move_model_to_cpu(model: Any) -> None:
        """Move a model to CPU if it exposes a relevant helper."""
        with suppress(Exception):
            if hasattr(model, "to"):
                model.to("cpu")  # type: ignore[attr-defined]
                return
            if hasattr(model, "cpu"):
                model.cpu()  # type: ignore[attr-defined]

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

        for node_id in list(node_ids):
            keys = cls._models_by_node.pop(node_id, None)
            if keys:
                for key in list(keys):
                    is_still_referenced = any(
                        key in mapped_keys for mapped_keys in cls._models_by_node.values()
                    )
                    if is_still_referenced:
                        continue

                    if key in cls._models:
                        model = cls._models.pop(key, None)
                        if model is not None:
                            cls._move_model_to_cpu(model)
                        # Extract model info for logging
                        parts = key.split("_", 2)
                        model_id = parts[0] if len(parts) > 0 else "unknown"
                        task = parts[1] if len(parts) > 1 else "unknown"
                        path = parts[2] if len(parts) > 2 else None
                        cleared_count += 1
                        cleared_models.append(f"{model_id} (task: {task}, path: {path})")
                        logger.debug(f"- Cleared cached model for node {node_id}: {model_id}")

                        # Clean up associated lock
                        if key in cls._locks:
                            del cls._locks[key]
                            cleared_locks += 1
                            logger.debug(f"🔒 Removed lock for cleared model: {key}")

                        cls._model_last_used.pop(key, None)
                        cls._model_device.pop(key, None)
                        cls._model_size_bytes.pop(key, None)

            cls._node_last_used.pop(node_id, None)

        if cleared_count > 0:
            logger.info(f"🗑️ Cache CLEANUP: Removed {cleared_count} unused models: {', '.join(cleared_models)}")
            if cleared_locks > 0:
                logger.debug(f"🔒 Removed {cleared_locks} associated locks")
            logger.debug(
                "Model cache status after cleanup - Total models: %d, Node associations: %d, Locks: %d",
                len(cls._models),
                sum(len(keys) for keys in cls._models_by_node.values()),
                len(cls._locks),
            )
        else:
            logger.debug("Cache cleanup: No unused models to remove")
        if cleared_count > 0:
            gc.collect()
            cls._try_empty_cuda_cache()

    @classmethod
    def unload_model(cls, model_id: str, task: str, path: str | None = None) -> bool:
        """Explicitly remove a cached model and free associated VRAM."""
        key = f"{model_id}_{task}_{path}"
        model = cls._models.pop(key, None)
        if model is None:
            return False

        cls._move_model_to_cpu(model)
        cls._locks.pop(key, None)
        cls._model_last_used.pop(key, None)
        cls._model_device.pop(key, None)
        cls._model_size_bytes.pop(key, None)

        for node_id, mapped_keys in list(cls._models_by_node.items()):
            if key in mapped_keys:
                mapped_keys.discard(key)
                if not mapped_keys:
                    cls._models_by_node.pop(node_id, None)
                    cls._node_last_used.pop(node_id, None)

        gc.collect()
        cls._try_empty_cuda_cache()
        logger.info("Unloaded cached model: %s (task: %s, path: %s)", model_id, task, path)
        return True

    @classmethod
    def clear(cls):
        """Removes all stored models, node associations, and locks."""
        model_count = len(cls._models)
        node_count = len(cls._models_by_node)
        lock_count = len(cls._locks)
        last_used_count = len(cls._model_last_used)
        node_usage_count = len(cls._node_last_used)
        device_count = len(cls._model_device)
        size_count = len(cls._model_size_bytes)

        # Log which models are being cleared
        if model_count > 0:
            model_info = []
            for key in list(cls._models):
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

        for model in list(cls._models.values()):
            cls._move_model_to_cpu(model)

        cls._models.clear()
        cls._models_by_node.clear()
        cls._locks.clear()
        cls._model_last_used.clear()
        cls._node_last_used.clear()
        cls._model_device.clear()
        cls._model_size_bytes.clear()

        if model_count > 0:
            logger.info(
                (
                    "✅ Cache cleared successfully: %d models removed, %d locks removed,"
                    " %d usage entries removed, %d node usage entries removed,"
                    " %d device entries removed, %d size entries removed"
                ),
                model_count,
                lock_count,
                last_used_count,
                node_usage_count,
                device_count,
                size_count,
            )
        gc.collect()
        cls._try_empty_cuda_cache()

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
        if not aggressive and cooldown > 0 and (now - cls._last_memory_cleanup) < cooldown:
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

    # ------------------------------------------------------------------
    # VRAM management helpers
    # ------------------------------------------------------------------

    @classmethod
    def free_vram_if_needed(
        cls,
        *,
        reason: str = "manual request",
        required_free_gb: float | None = None,
        aggressive: bool = False,
    ) -> None:
        """Ensure sufficient VRAM is available, offloading cached GPU models first."""

        cls._ensure_vram_capacity(
            reason=reason,
            aggressive=aggressive,
            required_free_gb=required_free_gb,
        )

    @classmethod
    def _ensure_vram_capacity(
        cls,
        *,
        reason: str,
        aggressive: bool = False,
        required_free_gb: float | None = None,
    ) -> None:
        if Environment.is_production():
            return

        snapshot = cls._capture_vram_snapshot()
        if snapshot is None:
            if aggressive:
                logger.warning(
                    "VRAM cleanup requested (%s) but telemetry unavailable. Clearing cached models.",
                    reason,
                )
                cls.clear()
                gc.collect()
                cls._try_empty_cuda_cache()
                cls._last_vram_cleanup = time.monotonic()
            return

        if not aggressive and not cls._needs_vram_cleanup(snapshot, required_free_gb):
            return

        cooldown = cls._get_vram_cleanup_cooldown()
        now = time.monotonic()
        if not aggressive and cooldown > 0 and (now - cls._last_vram_cleanup) < cooldown:
            remaining = cooldown - (now - cls._last_vram_cleanup)
            logger.debug(
                "VRAM pressure detected but cleanup throttled for %.2fs (usage %.2f%%, %.2f GB free)",
                max(remaining, 0.0),
                snapshot.percent,
                snapshot.available_gb,
            )
            return

        target_free_gb = cls._target_vram_available_gb(snapshot, required_free_gb)
        succeeded = cls._offload_gpu_models_until_free(
            target_free_gb=target_free_gb,
            snapshot=snapshot,
            reason=reason,
        )
        cls._last_vram_cleanup = now

        if aggressive and not succeeded:
            logger.warning(
                "VRAM cleanup (aggressive) did not free enough space. Clearing cached models. Reason: %s",
                reason,
            )
            cls.clear()
            gc.collect()
            cls._try_empty_cuda_cache()

    @classmethod
    def _offload_gpu_models_until_free(
        cls,
        *,
        target_free_gb: float,
        snapshot: VramSnapshot,
        reason: str,
    ) -> bool:
        """Move cached GPU models to CPU until target free VRAM is reached."""

        try:  # pragma: no cover - optional dependency
            import torch  # type: ignore
        except Exception:
            logger.debug("VRAM cleanup requested but torch is unavailable. Reason: %s", reason)
            return False

        if not hasattr(torch, "cuda") or not torch.cuda.is_available():  # type: ignore[attr-defined]
            logger.debug(
                "VRAM cleanup requested but CUDA is unavailable. Reason: %s",
                reason,
            )
            return False

        start_available = snapshot.available_gb
        candidates: list[tuple[float, str, Any]] = []

        for key, model in list(cls._models.items()):
            if model is None:
                continue

            detected_device, size_bytes = cls._detect_torch_model_device_and_size(model)
            device = detected_device if detected_device != "unknown" else cls._model_device.get(key)

            if detected_device != "unknown":
                cls._model_device[key] = detected_device
            if size_bytes is not None:
                cls._model_size_bytes[key] = size_bytes

            if not cls._is_cuda_device(device):
                continue

            last_used = cls._model_last_used.get(key, 0.0)
            candidates.append((last_used, key, model))

        if not candidates:
            logger.debug(
                "VRAM cleanup requested but no GPU-resident cached models found. Reason: %s",
                reason,
            )
            return False

        candidates.sort(key=lambda item: item[0])

        available = start_available
        offloaded_keys: list[str] = []

        for _, key, model in candidates:
            if available >= target_free_gb:
                break

            try:
                if hasattr(model, "to"):
                    model.to("cpu")  # type: ignore[attr-defined]
                elif hasattr(model, "cpu"):
                    model.cpu()  # type: ignore[attr-defined]
                else:
                    continue
                cls._model_device[key] = "cpu"
                offloaded_keys.append(key)
                if key in cls._model_size_bytes:
                    available += cls._model_size_bytes[key] / (1024**3)
            except Exception as exc:
                logger.debug("Failed to offload model %s to CPU: %s", key, exc)
                continue

        cls._try_empty_cuda_cache()
        latest = cls._capture_vram_snapshot()
        if latest is not None:
            available = latest.available_gb

        if offloaded_keys:
            logger.warning(
                "VRAM cleanup: Offloaded %d cached model(s) to CPU (free %.2f GB -> %.2f GB). Reason: %s. Keys: %s",
                len(offloaded_keys),
                start_available,
                available,
                reason,
                ", ".join(offloaded_keys),
            )
        else:
            logger.debug(
                "VRAM cleanup did not offload any models. Reason: %s (available %.2f GB, target %.2f GB)",
                reason,
                available,
                target_free_gb,
            )

        return available >= target_free_gb

    @classmethod
    def _capture_vram_snapshot(cls) -> VramSnapshot | None:
        """Capture VRAM telemetry using torch when available, NVML otherwise."""

        try:  # pragma: no cover - optional dependency
            import torch  # type: ignore
        except Exception:
            return cls._capture_vram_snapshot_via_system_stats()

        try:
            if not hasattr(torch, "cuda") or not torch.cuda.is_available():  # type: ignore[attr-defined]
                fallback = cls._capture_vram_snapshot_via_system_stats()
                if fallback is None:
                    logger.debug("Torch available but CUDA unavailable and NVML fallback failed to provide stats.")
                return fallback

            torch.cuda.synchronize()

            available_gb: float
            total_gb: float

            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                available_gb = float(free_bytes) / (1024**3)
                total_gb = float(total_bytes) / (1024**3)
            except Exception:
                props = torch.cuda.get_device_properties(0)  # type: ignore[attr-defined]
                total_gb = float(props.total_memory) / (1024**3)
                allocated_bytes = float(torch.cuda.memory_allocated(0))  # type: ignore[attr-defined]
                available_gb = max(total_gb - allocated_bytes / (1024**3), 0.0)

            allocated_gb = float(torch.cuda.memory_allocated(0)) / (1024**3)  # type: ignore[attr-defined]
            used_percent = ((total_gb - available_gb) / total_gb) * 100.0 if total_gb > 0 else 0.0

            snapshot = VramSnapshot(
                percent=used_percent,
                available_gb=available_gb,
                total_gb=total_gb,
                process_allocated_gb=allocated_gb,
            )
            logger.debug(
                "VRAM snapshot captured: %.2f%% used, %.2f GB available, process allocated %.2f GB",
                snapshot.percent,
                snapshot.available_gb,
                snapshot.process_allocated_gb,
            )
            return snapshot
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unable to capture VRAM stats via torch: %s", exc)
            return cls._capture_vram_snapshot_via_system_stats()

    @classmethod
    def _capture_vram_snapshot_via_system_stats(cls) -> VramSnapshot | None:
        """Fallback VRAM telemetry using NVML via SystemStats, if available."""

        try:
            from nodetool.system.system_stats import get_system_stats
        except Exception:  # pragma: no cover - avoid hard dependency
            return None

        try:
            stats = get_system_stats()
            if stats.vram_total_gb is None or stats.vram_used_gb is None or stats.vram_percent is None:
                return None

            available_gb = max(float(stats.vram_total_gb - stats.vram_used_gb), 0.0)
            snapshot = VramSnapshot(
                percent=float(stats.vram_percent),
                available_gb=available_gb,
                total_gb=float(stats.vram_total_gb),
                process_allocated_gb=0.0,
            )
            logger.debug(
                "VRAM snapshot captured via NVML: %.2f%% used, %.2f GB available",
                snapshot.percent,
                snapshot.available_gb,
            )
            return snapshot
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unable to capture VRAM stats via NVML/SystemStats: %s", exc)
            return None

    @classmethod
    def _needs_vram_cleanup(cls, snapshot: VramSnapshot, required_free_gb: float | None) -> bool:
        max_percent, min_available = cls._get_vram_thresholds()
        if snapshot.percent >= max_percent or snapshot.available_gb <= min_available:
            return True
        return bool(required_free_gb is not None and snapshot.available_gb < required_free_gb)

    @classmethod
    def _target_vram_available_gb(cls, snapshot: VramSnapshot, required_free_gb: float | None) -> float:
        max_percent, min_available = cls._get_vram_thresholds()
        target_from_percent = snapshot.total_gb * (1 - max_percent / 100.0)
        target = max(min_available, target_from_percent)
        if required_free_gb is not None:
            target = max(target, required_free_gb)
        return max(min(target, snapshot.total_gb), 0.0)

    @classmethod
    def _get_vram_thresholds(cls) -> tuple[float, float]:
        max_percent = min(max(cls._DEFAULT_MAX_VRAM_PERCENT, 10.0), 99.0)
        min_available = max(cls._DEFAULT_MIN_VRAM_AVAILABLE_GB, 0.0)
        return max_percent, min_available

    @classmethod
    def _get_vram_cleanup_cooldown(cls) -> float:
        return max(cls._DEFAULT_VRAM_COOLDOWN_SECONDS, 0.0)

    @classmethod
    def get_vram_snapshot(cls) -> VramSnapshot | None:
        """Public helper to capture a VRAM snapshot."""

        return cls._capture_vram_snapshot()

    @staticmethod
    def _is_cuda_device(device: str | None) -> bool:
        if device is None:
            return False
        return device.startswith("cuda")

    @staticmethod
    def _try_empty_cuda_cache() -> None:
        try:  # pragma: no cover - optional dependency
            import torch  # type: ignore
        except Exception:
            return

        if not hasattr(torch, "cuda"):
            return

        with suppress(Exception):
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
