"""
Model Registry for tracking and managing loaded ML models.

This module provides a centralized registry for tracking models loaded
during workflow execution, enabling memory management and cleanup.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class RegisteredModel:
    """Metadata about a registered model."""

    id: str
    model_type: str
    device: str
    memory_mb: float = 0.0
    offloaded: bool = False
    model_id: Optional[str] = None
    # Weak reference to the actual model object (allows GC when model is no longer used elsewhere)
    _model_ref: Optional[weakref.ref] = field(default=None, repr=False)
    # Strong reference to keep model alive (use when model should not be garbage collected)
    _model: Optional[Any] = field(default=None, repr=False)
    # Cleanup callback
    _cleanup_fn: Optional[Callable[[], None]] = field(default=None, repr=False)
    # In-use flag to prevent unloading during processing
    in_use: bool = False

    def get_model(self) -> Optional[Any]:
        """Get the model object if still alive."""
        if self._model is not None:
            return self._model
        if self._model_ref is not None:
            return self._model_ref()
        return None


class ModelRegistry:
    """
    Singleton registry for tracking loaded models.

    This registry provides:
    - Centralized tracking of all loaded models
    - Memory usage estimation
    - Safe model unloading with in-use checks
    - Thread-safe operations
    """

    _instance: Optional[ModelRegistry] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """Initialize the registry state."""
        self._models: dict[str, RegisteredModel] = {}
        self._registry_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._models.clear()
            cls._instance = None

    def register(
        self,
        model_id: str,
        model: Any,
        model_type: str,
        device: str = "cpu",
        memory_mb: float = 0.0,
        offloaded: bool = False,
        hf_model_id: Optional[str] = None,
        cleanup_fn: Optional[Callable[[], None]] = None,
        keep_strong_ref: bool = False,
    ) -> None:
        """
        Register a model in the registry.

        Args:
            model_id: Unique identifier (typically node_id)
            model: The model/pipeline object
            model_type: Type name of the model
            device: Device location (cuda, cpu, mps)
            memory_mb: Estimated memory usage
            offloaded: Whether CPU offload is enabled
            hf_model_id: HuggingFace model ID if applicable
            cleanup_fn: Optional cleanup callback
            keep_strong_ref: If True, keep a strong reference to the model
        """
        with self._registry_lock:
            # Check if model is already registered
            if model_id in self._models:
                log.debug(f"Model {model_id} already registered, updating")

            registered = RegisteredModel(
                id=model_id,
                model_type=model_type,
                device=device,
                memory_mb=memory_mb,
                offloaded=offloaded,
                model_id=hf_model_id,
                _cleanup_fn=cleanup_fn,
            )

            if keep_strong_ref:
                registered._model = model
            else:
                registered._model_ref = weakref.ref(model)

            self._models[model_id] = registered
            log.info(f"Registered model {model_id} ({model_type}) on {device}, ~{memory_mb:.1f}MB")

    def unregister(self, model_id: str) -> bool:
        """
        Unregister a model from the registry.

        Args:
            model_id: The model identifier to unregister

        Returns:
            True if model was found and unregistered, False otherwise
        """
        with self._registry_lock:
            if model_id not in self._models:
                return False

            registered = self._models.pop(model_id)
            log.info(f"Unregistered model {model_id}")

            # Run cleanup callback if provided
            if registered._cleanup_fn:
                try:
                    registered._cleanup_fn()
                except Exception as e:
                    log.warning(f"Cleanup callback for {model_id} failed: {e}")

            return True

    def mark_in_use(self, model_id: str) -> bool:
        """Mark a model as in-use to prevent unloading."""
        with self._registry_lock:
            if model_id in self._models:
                self._models[model_id].in_use = True
                return True
            return False

    def mark_not_in_use(self, model_id: str) -> bool:
        """Mark a model as not in-use."""
        with self._registry_lock:
            if model_id in self._models:
                self._models[model_id].in_use = False
                return True
            return False

    def is_in_use(self, model_id: str) -> bool:
        """Check if a model is currently in use."""
        with self._registry_lock:
            if model_id in self._models:
                return self._models[model_id].in_use
            return False

    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a model by ID if it's still alive."""
        with self._registry_lock:
            if model_id not in self._models:
                return None
            return self._models[model_id].get_model()

    def get_model_info(self, model_id: str) -> Optional[RegisteredModel]:
        """Get model info by ID."""
        with self._registry_lock:
            return self._models.get(model_id)

    def list_models(self) -> list[RegisteredModel]:
        """List all registered models."""
        with self._registry_lock:
            # Clean up dead references
            dead_ids = []
            for model_id, registered in self._models.items():
                if registered._model is None and registered._model_ref is not None:
                    if registered._model_ref() is None:
                        dead_ids.append(model_id)

            for model_id in dead_ids:
                log.debug(f"Removing dead model reference: {model_id}")
                del self._models[model_id]

            return list(self._models.values())

    def get_total_memory_mb(self) -> float:
        """Get total memory used by all registered models."""
        with self._registry_lock:
            return sum(m.memory_mb for m in self._models.values())

    def clear_all(self, force: bool = False) -> tuple[int, float]:
        """
        Clear all registered models.

        Args:
            force: If True, clear even models marked as in-use

        Returns:
            Tuple of (models_cleared, memory_freed_mb)
        """
        with self._registry_lock:
            # Select models to clear: all if forced, otherwise only those not in use
            to_clear = list(self._models.keys()) if force else [k for k, v in self._models.items() if not v.in_use]

            memory_freed = 0.0
            models_cleared = 0

            for model_id in to_clear:
                registered = self._models.pop(model_id, None)
                if registered:
                    memory_freed += registered.memory_mb
                    models_cleared += 1

                    if registered._cleanup_fn:
                        try:
                            registered._cleanup_fn()
                        except Exception as e:
                            log.warning(f"Cleanup callback for {model_id} failed: {e}")

            log.info(f"Cleared {models_cleared} models, freed ~{memory_freed:.1f}MB")
            return models_cleared, memory_freed

    def count(self) -> int:
        """Get the number of registered models."""
        with self._registry_lock:
            return len(self._models)


# Convenience function to get the global registry
def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return ModelRegistry.get_instance()
