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
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from typing import Any, AsyncIterator, ClassVar, Iterator, NamedTuple

import psutil

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

logger = get_logger(__name__)

# Identifies the execution scope (see ``ModelManager.execution_scope``) that the
# current task is running in. Every cache key touched inside a scope is pinned
# for as long as that scope is open, so background reclaim can never pull a
# model out from under a node that is still using it.
_CURRENT_SCOPE: ContextVar[int | None] = ContextVar("nodetool_model_scope", default=None)


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
    """VRAM telemetry for the current process/device.

    ``reclaimable_gb`` is memory held by torch's caching allocator that is not
    backing live tensors (``reserved - allocated``). The driver reports it as
    used, but it is immediately reusable by this process, so treating only
    driver-free memory as available makes a healthy steady state look like a
    shortfall.
    """

    percent: float
    available_gb: float
    total_gb: float
    process_allocated_gb: float
    reclaimable_gb: float = 0.0

    @property
    def usable_gb(self) -> float:
        """Memory this process can actually allocate: driver-free plus our own cached-but-free blocks."""
        return self.available_gb + self.reclaimable_gb


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
        _active_scopes (Dict[int, set[str]]): Cache keys touched by each open execution scope
    """

    _models: ClassVar[dict[str, Any]] = {}
    _active_scopes: ClassVar[dict[int, set[str]]] = {}
    _scope_counter: ClassVar[int] = 0
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
        cls._pin_key_in_scope(cache_key)
        model = cls._models.get(cache_key)
        if model is not None:
            cls._update_model_metadata(cache_key, model)
            logger.info(f"✓ Cache HIT: Retrieved cached model for {cache_key}")
        else:
            logger.info(f"✗ Cache MISS: No cached model found for {cache_key}")
        logger.debug(f"Model cache status - Total models: {len(cls._models)}, Key searched: {cache_key}")
        return model

    @staticmethod
    def _make_cache_key(model_id: str, task: str | None = None, path: str | None = None) -> str:
        """Build a cache key consistently across set_model/get_model/unload_model.

        The key is ``model_id`` optionally suffixed with ``_task`` and ``_path``.
        Only non-empty ``task``/``path`` segments are appended so that, e.g.,
        ``unload_model(model_id, task)`` produces the same key that the legacy
        ``set_model(node_id, model_id, task, model)`` signature stores.
        """
        key = model_id
        if task:
            key = f"{key}_{task}"
        if path:
            key = f"{key}_{path}"
        return key

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
            cache_key = cls._make_cache_key(model_id_or_cache_key, task)
            model_instance = model

        cls._pin_key_in_scope(cache_key)
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
    def _discard_lock(cls, cache_key: str) -> bool:
        """Drop a per-model lock, but only when nobody is holding it.

        Evicting a held lock would let the next ``get_model_lock`` hand out a
        brand-new :class:`asyncio.Lock` for the same key, so two coroutines
        could enter the same critical section at once — e.g. both loading the
        model into VRAM. A still-held lock is left in place; whichever eviction
        runs next (or :meth:`clear`) collects it.

        Returns:
            True if the lock was removed.
        """
        lock = cls._locks.get(cache_key)
        if lock is None:
            return False
        if lock.locked():
            logger.debug("🔒 Keeping in-use lock for evicted model: %s", cache_key)
            return False
        del cls._locks[cache_key]
        return True

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
    # Execution scopes (in-use protection)
    # ------------------------------------------------------------------

    @classmethod
    @contextmanager
    def execution_scope(cls) -> Iterator[None]:
        """Mark a region of code as an in-flight node execution.

        Every cache key passed to :meth:`get_model` or :meth:`set_model` while
        the scope is open is pinned: reclaim paths skip it until the scope
        closes. This is what makes proactive eviction safe to run — without it,
        a reclaim pass triggered by one node could offload (or drop) a model a
        concurrently executing node is in the middle of using, turning a memory
        leak into a correctness bug.

        Scopes are tracked per :mod:`contextvars` context, so concurrent
        ``asyncio`` tasks each get their own and never see each other's keys as
        their own. Nesting is supported; the inner scope shadows the outer one
        for the duration, and both remain pinned.
        """
        cls._scope_counter += 1
        scope_id = cls._scope_counter
        cls._active_scopes[scope_id] = set()
        token = _CURRENT_SCOPE.set(scope_id)
        try:
            yield
        finally:
            _CURRENT_SCOPE.reset(token)
            cls._active_scopes.pop(scope_id, None)

    @classmethod
    def _pin_key_in_scope(cls, cache_key: str) -> None:
        """Record that the current execution scope (if any) is using ``cache_key``."""

        scope_id = _CURRENT_SCOPE.get()
        if scope_id is None:
            return
        keys = cls._active_scopes.get(scope_id)
        if keys is not None:
            keys.add(cache_key)

    @classmethod
    def is_model_in_use(cls, cache_key: str, *, include_current_scope: bool = True) -> bool:
        """Return True when an open execution scope has touched ``cache_key``.

        Args:
            cache_key: The cache key to check.
            include_current_scope: When False, only *other* scopes count. Use
                this for explicit, caller-driven eviction (the caller knows what
                it is doing with its own models) while still refusing to disturb
                concurrently executing nodes.
        """
        current = _CURRENT_SCOPE.get()
        for scope_id, keys in cls._active_scopes.items():
            if not include_current_scope and scope_id == current:
                continue
            if cache_key in keys:
                return True
        return False

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

        Handles three shapes: torch modules (``.parameters()``), bare tensors
        (``device``/``numel``/``element_size``), and diffusers pipelines, which
        expose neither directly but carry their modules in ``.components`` —
        without the last case, VRAM eviction walks straight past the pipelines
        that hold the most memory.
        """

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

            components = getattr(model, "components", None)
            if isinstance(components, dict) and components:
                device = "unknown"
                total_bytes = 0
                found_params = False
                for component in components.values():
                    if component is None or not hasattr(component, "parameters"):
                        continue
                    params = list(component.parameters())
                    if not params:
                        continue
                    found_params = True
                    total_bytes += sum(p.numel() * p.element_size() for p in params)
                    # Report the "hottest" device: any CUDA-resident component
                    # makes the pipeline a VRAM eviction candidate, even when
                    # other components are offloaded to CPU.
                    component_device = str(params[0].device)
                    if device == "unknown" or (component_device.startswith("cuda") and not device.startswith("cuda")):
                        device = component_device
                if found_params:
                    return device, int(total_bytes)
        except Exception:
            return "unknown", None

        return "unknown", None

    @staticmethod
    def _cuda_tensor_footprint(model: Any) -> dict[int, int] | None:
        """Map ``id(tensor) -> nbytes`` for every CUDA-resident tensor in ``model``.

        Keyed by tensor identity on purpose. The same weights are routinely
        cached under two keys — a transformers model on its own plus the
        pipeline that wraps it, or a transformer shared between a txt2img and an
        img2img pipeline. That aliasing is legitimate and deliberate, so the
        accounting has to be identity-aware: summing per cache key would credit
        the same bytes twice and make a reclaim loop stop while VRAM is still
        occupied.

        Handles the same three shapes as
        :meth:`_detect_torch_model_device_and_size`: torch modules, bare
        tensors, and diffusers pipelines (modules under ``.components``).

        Returns ``None`` when the object is none of those shapes, so callers can
        tell "not introspectable" apart from "introspected, nothing on CUDA" —
        an aliased model whose weights a previous offload already moved falls in
        the second case and must be credited zero, not its recorded size.
        """

        footprint: dict[int, int] = {}

        def _add(tensor: Any) -> None:
            with suppress(Exception):
                if not str(tensor.device).startswith("cuda"):
                    return
                footprint[id(tensor)] = int(tensor.numel() * tensor.element_size())

        try:
            if hasattr(model, "parameters"):
                params = list(model.parameters())
                if params:
                    for param in params:
                        _add(param)
                    return footprint

            if all(hasattr(model, attr) for attr in ("device", "numel", "element_size")):
                _add(model)
                return footprint

            components = getattr(model, "components", None)
            if isinstance(components, dict) and components:
                for component in components.values():
                    if component is None or not hasattr(component, "parameters"):
                        continue
                    for param in component.parameters():
                        _add(param)
                return footprint
        except Exception:
            return footprint or None

        return None

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
    def release_nodes(cls, node_ids: list[str]):
        """Release the models owned exclusively by the given *retired* nodes.

        ``node_ids`` are nodes that have finished and will not run again — not
        the live ones to preserve. For each of them the association is dropped,
        and any cache key that no other node still references is evicted (moved
        to CPU first, then removed from the cache along with its lock and
        metadata).

        Two things are never evicted here:

        - keys another node is still associated with, and
        - keys pinned by an open :meth:`execution_scope`, i.e. models a
          currently executing node is using.

        Args:
            node_ids (list[str]): IDs of nodes that are done executing.
        """
        cleared_count = 0
        cleared_models = []
        cleared_locks = 0

        for node_id in list(node_ids):
            keys = cls._models_by_node.pop(node_id, None)
            if keys:
                for key in list(keys):
                    is_still_referenced = any(key in mapped_keys for mapped_keys in cls._models_by_node.values())
                    if is_still_referenced:
                        continue

                    if cls.is_model_in_use(key):
                        # A concurrently executing node holds this model. Put
                        # the association back so the key is not orphaned in the
                        # cache with nobody left to release it.
                        cls._models_by_node.setdefault(node_id, set()).add(key)
                        logger.debug("Keeping in-use model %s while releasing node %s", key, node_id)
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
                        if cls._discard_lock(key):
                            cleared_locks += 1
                            logger.debug(f"🔒 Removed lock for cleared model: {key}")

                        cls._model_last_used.pop(key, None)
                        cls._model_device.pop(key, None)
                        cls._model_size_bytes.pop(key, None)

            if node_id not in cls._models_by_node:
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
    def clear_unused(cls, node_ids: list[str]):
        """Deprecated alias for :meth:`release_nodes`.

        The name and old docstring read as "keep these active nodes, drop the
        rest", but the implementation has always done the opposite: it releases
        the models owned by the node IDs you pass. ``release_nodes`` says that
        out loud. This alias preserves the behaviour existing callers already
        depend on — passing a list of *live* node IDs here evicts exactly the
        models still in use, which was never the intent.
        """
        cls.release_nodes(node_ids)

    @classmethod
    def unload_model(cls, model_id: str, task: str, path: str | None = None) -> bool:
        """Explicitly remove a cached model and free associated VRAM."""
        key = cls._make_cache_key(model_id, task, path)
        model = cls._models.pop(key, None)
        if model is None:
            return False

        cls._move_model_to_cpu(model)
        cls._discard_lock(key)
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
    def evict_all_except(cls, cache_key: str | None = None) -> list[str]:
        """Evict every cached model except ``cache_key`` and free their VRAM.

        This is the "never-coexist" load choke point: call it before loading a
        large model so at most one model family is resident. Unlike a plain
        ``model.to("cpu")``, evicted models are dropped from the cache entirely
        so their host copies can be garbage-collected too — on GPU-bound
        machines keeping them in system RAM just delays the next eviction.

        Call this on the already-loaded fast path as well: a sibling model may
        have been cached since the last load.

        Args:
            cache_key: The one key to keep resident, or None to evict everything.

        Returns:
            The list of cache keys that were evicted.
        """
        evicted: list[str] = []
        skipped_in_use: list[str] = []
        for key in list(cls._models.keys()):
            if cache_key is not None and key == cache_key:
                continue
            # The caller owns its own scope's models (that is the point of an
            # explicit choke point), but a model another in-flight node is using
            # is off limits — dropping it mid-inference is a correctness bug,
            # not a memory win.
            if cls.is_model_in_use(key, include_current_scope=False):
                skipped_in_use.append(key)
                continue
            model = cls._models.pop(key, None)
            if model is not None:
                cls._move_model_to_cpu(model)
            cls._discard_lock(key)
            cls._model_last_used.pop(key, None)
            cls._model_device.pop(key, None)
            cls._model_size_bytes.pop(key, None)
            for node_id, mapped_keys in list(cls._models_by_node.items()):
                if key in mapped_keys:
                    mapped_keys.discard(key)
                    if not mapped_keys:
                        cls._models_by_node.pop(node_id, None)
                        cls._node_last_used.pop(node_id, None)
            evicted.append(key)

        if evicted:
            gc.collect()
            cls._try_empty_cuda_cache()
            logger.info(
                "Evicted %d cached model(s) to make room (kept: %s): %s",
                len(evicted),
                cache_key,
                ", ".join(evicted),
            )
        if skipped_in_use:
            logger.info(
                "Kept %d cached model(s) in use by concurrently executing node(s): %s",
                len(skipped_in_use),
                ", ".join(skipped_in_use),
            )
        return evicted

    @classmethod
    def evict_models(
        cls,
        *,
        node_ids: list[str] | None = None,
        target_vram_gb: float | None = None,
    ) -> tuple[list[str], float]:
        """Drop loaded model weights on explicit request.

        This is the caller-driven counterpart to the reactive threshold-based
        reclaim: it serves what only the host knows — the user switched
        workflows, another process wants the GPU, the worker is idle.

        Args:
            node_ids: Evict only models registered to these nodes. Keys another
                node outside the set still references are kept, matching
                :meth:`release_nodes`. ``None`` considers every cached model.
            target_vram_gb: Stop once this much has been reclaimed, instead of
                dropping every loaded weight. Coldest models go first, so a
                partial eviction keeps the hot ones resident.

        Returns:
            ``(evicted_keys, freed_gb)``. ``freed_gb`` is a best-effort
            estimate: the CUDA-resident footprint of each evicted model where
            that can be walked, its recorded size otherwise.
        """
        if node_ids is not None:
            requested = set(node_ids)
            keep_referenced = {
                key for owner, keys in cls._models_by_node.items() if owner not in requested for key in keys
            }
            candidates = [
                key
                for owner in requested
                for key in cls._models_by_node.get(owner, set())
                if key in cls._models and key not in keep_referenced
            ]
            # Deduplicate while keeping it a list: two retired nodes can share
            # a key, and evicting it twice would double-count the bytes freed.
            candidates = list(dict.fromkeys(candidates))
        else:
            candidates = list(cls._models.keys())

        # Coldest first: with a target, whatever survives should be what the
        # next execution is most likely to want.
        candidates.sort(key=lambda key: cls._model_last_used.get(key, 0.0))

        evicted: list[str] = []
        freed_bytes = 0
        credited_tensor_ids: set[int] = set()

        for key in candidates:
            if target_vram_gb is not None and freed_bytes / (1024**3) >= target_vram_gb:
                break
            # An explicit request owns its own scope's models, but a model
            # another in-flight node is using is off limits — dropping it
            # mid-inference is a correctness bug, not a memory win.
            if cls.is_model_in_use(key, include_current_scope=False):
                logger.debug("Skipping in-use model during eviction: %s", key)
                continue

            model = cls._models.get(key)
            if model is None:
                continue

            # Snapshot before the move: afterwards the tensors report CPU.
            # Identity-keyed so weights aliased under two cache keys (a model
            # and the pipeline wrapping it) are credited once.
            footprint = cls._cuda_tensor_footprint(model)
            if footprint:
                freed_bytes += sum(
                    nbytes for tensor_id, nbytes in footprint.items() if tensor_id not in credited_tensor_ids
                )
                credited_tensor_ids.update(footprint)
            else:
                # Not CUDA-resident, or not a shape we can walk. The recorded
                # size is the only number available and cannot be deduped.
                freed_bytes += cls._model_size_bytes.get(key, 0)

            cls._models.pop(key, None)
            cls._move_model_to_cpu(model)
            cls._discard_lock(key)
            cls._model_last_used.pop(key, None)
            cls._model_device.pop(key, None)
            cls._model_size_bytes.pop(key, None)
            for owner, mapped_keys in list(cls._models_by_node.items()):
                if key in mapped_keys:
                    mapped_keys.discard(key)
                    if not mapped_keys:
                        cls._models_by_node.pop(owner, None)
                        cls._node_last_used.pop(owner, None)
            evicted.append(key)

        freed_gb = freed_bytes / (1024**3)
        if evicted:
            gc.collect()
            cls._try_empty_cuda_cache()
            logger.info(
                "Evicted %d cached model(s) on request (%.2f GB, target %s, nodes %s): %s",
                len(evicted),
                freed_gb,
                target_vram_gb,
                node_ids,
                ", ".join(evicted),
            )
        else:
            logger.debug(
                "Eviction request matched no evictable models (nodes %s, target %s)",
                node_ids,
                target_vram_gb,
            )
        return evicted, freed_gb

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
        # Locks currently held survive the purge — see _discard_lock().
        for key in list(cls._locks):
            cls._discard_lock(key)
        lock_count -= len(cls._locks)
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
            # Telemetry is genuinely unavailable here (e.g. no CUDA device, or an
            # Apple/MLX machine). There is no VRAM-pressure signal to act on, and
            # wiping the entire model cache in this case is destructive for no
            # benefit (there is no GPU memory to reclaim). Do nothing. Callers
            # that truly want to drop everything should use free_memory_if_needed
            # / clear() explicitly.
            logger.debug(
                "VRAM cleanup requested (%s) but telemetry is unavailable; "
                "skipping cache clear (no CUDA telemetry to act on).",
                reason,
            )
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

        start_available = snapshot.usable_gb
        candidates: list[tuple[float, str, Any]] = []

        for key, model in list(cls._models.items()):
            if model is None:
                continue

            if cls.is_model_in_use(key):
                logger.debug("Skipping in-use model during VRAM cleanup: %s", key)
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
        # Tensors already credited this pass, by identity. Aliased cache keys
        # (a model plus the pipeline wrapping it) share tensors; counting a
        # shared tensor once per key would credit its bytes twice and stop the
        # loop early with VRAM still occupied.
        credited_tensor_ids: set[int] = set()

        for _, key, model in candidates:
            if available >= target_free_gb:
                break

            # Snapshot before the move: afterwards the tensors report CPU.
            footprint = cls._cuda_tensor_footprint(model)

            try:
                if hasattr(model, "to"):
                    model.to("cpu")  # type: ignore[attr-defined]
                elif hasattr(model, "cpu"):
                    model.cpu()  # type: ignore[attr-defined]
                else:
                    continue
                cls._model_device[key] = "cpu"
                offloaded_keys.append(key)
            except Exception as exc:
                logger.debug("Failed to offload model %s to CPU: %s", key, exc)
                continue

            if footprint is not None:
                freed_bytes = sum(
                    nbytes for tensor_id, nbytes in footprint.items() if tensor_id not in credited_tensor_ids
                )
                credited_tensor_ids.update(footprint)
                available += freed_bytes / (1024**3)
            elif key in cls._model_size_bytes:
                # Not a shape we can walk (an exotic wrapper). Fall back to the
                # recorded size, which cannot be deduped.
                available += cls._model_size_bytes[key] / (1024**3)

        cls._try_empty_cuda_cache()
        latest = cls._capture_vram_snapshot()
        if latest is not None:
            available = latest.usable_gb

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
            try:
                reserved_gb = float(torch.cuda.memory_reserved(0)) / (1024**3)  # type: ignore[attr-defined]
            except Exception:
                reserved_gb = allocated_gb
            reclaimable_gb = max(reserved_gb - allocated_gb, 0.0)
            used_percent = ((total_gb - available_gb) / total_gb) * 100.0 if total_gb > 0 else 0.0

            snapshot = VramSnapshot(
                percent=used_percent,
                available_gb=available_gb,
                total_gb=total_gb,
                process_allocated_gb=allocated_gb,
                reclaimable_gb=reclaimable_gb,
            )
            logger.debug(
                "VRAM snapshot captured: %.2f%% used, %.2f GB available (+%.2f GB reclaimable), process allocated %.2f GB",
                snapshot.percent,
                snapshot.available_gb,
                snapshot.reclaimable_gb,
                snapshot.process_allocated_gb,
            )
            return snapshot
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unable to capture VRAM stats via torch: %s", exc)
            return cls._capture_vram_snapshot_via_system_stats()

    @classmethod
    def _capture_vram_snapshot_via_system_stats(cls) -> VramSnapshot | None:
        """Fallback VRAM telemetry.

        The previous NVML-via-``nodetool.system.system_stats`` fallback was
        removed along with that module (it now lives in the TypeScript server),
        so this method no longer depends on it. As a best-effort fallback it
        tries ``torch.cuda.mem_get_info`` directly; when torch/CUDA is not
        available it cleanly returns ``None`` so callers treat VRAM telemetry as
        unavailable (rather than crashing on a dead import).
        """

        try:  # pragma: no cover - optional dependency
            import torch  # type: ignore
        except Exception:
            return None

        try:
            if not hasattr(torch, "cuda") or not torch.cuda.is_available():  # type: ignore[attr-defined]
                return None

            free_bytes, total_bytes = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
            available_gb = float(free_bytes) / (1024**3)
            total_gb = float(total_bytes) / (1024**3)
            used_percent = ((total_gb - available_gb) / total_gb) * 100.0 if total_gb > 0 else 0.0

            snapshot = VramSnapshot(
                percent=used_percent,
                available_gb=available_gb,
                total_gb=total_gb,
                process_allocated_gb=0.0,
            )
            logger.debug(
                "VRAM snapshot captured via torch.cuda.mem_get_info: %.2f%% used, %.2f GB available",
                snapshot.percent,
                snapshot.available_gb,
            )
            return snapshot
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unable to capture VRAM stats via torch fallback: %s", exc)
            return None

    @classmethod
    def _needs_vram_cleanup(cls, snapshot: VramSnapshot, required_free_gb: float | None) -> bool:
        # Judge pressure on usable memory (driver-free + our own cached-but-free
        # allocator blocks), not raw driver-free: after a render several GB can
        # sit reserved-but-unallocated, and counting them as a shortfall makes a
        # healthy steady state look like a leak.
        max_percent, min_available = cls._get_vram_thresholds()
        used_percent = (
            ((snapshot.total_gb - snapshot.usable_gb) / snapshot.total_gb) * 100.0 if snapshot.total_gb > 0 else 0.0
        )
        if used_percent >= max_percent or snapshot.usable_gb <= min_available:
            return True
        return bool(required_free_gb is not None and snapshot.usable_gb < required_free_gb)

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
