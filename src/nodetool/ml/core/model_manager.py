"""
Manages ML model instances in non-production environments.

This module provides the ModelManager class, a central repository for storing,
retrieving, and managing machine learning models during development or testing.
It associates models with specific nodes and handles their lifecycle, preventing
resource leaks by clearing unused models. This functionality is disabled in
production environments.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, ClassVar, Dict

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

logger = get_logger(__name__)


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
