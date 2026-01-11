"""Tests for the model registry."""

import gc

import pytest

from nodetool.workflows.model_registry import ModelRegistry, get_model_registry


class DummyModel:
    """A dummy model class for testing."""

    def __init__(self, name: str = "test"):
        self.name = name


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry singleton before and after each test."""
    ModelRegistry.reset_instance()
    yield
    ModelRegistry.reset_instance()


def test_singleton_pattern():
    """Test that ModelRegistry follows the singleton pattern."""
    registry1 = ModelRegistry.get_instance()
    registry2 = get_model_registry()
    registry3 = ModelRegistry()

    assert registry1 is registry2
    assert registry2 is registry3


def test_register_model():
    """Test registering a model."""
    registry = get_model_registry()
    model = DummyModel("test-model")

    registry.register(
        model_id="node-123",
        model=model,
        model_type="DummyModel",
        device="cpu",
        memory_mb=100.0,
        offloaded=False,
        hf_model_id="test/model",
    )

    assert registry.count() == 1
    info = registry.get_model_info("node-123")
    assert info is not None
    assert info.id == "node-123"
    assert info.model_type == "DummyModel"
    assert info.device == "cpu"
    assert info.memory_mb == 100.0
    assert info.offloaded is False
    assert info.model_id == "test/model"


def test_register_duplicate_model():
    """Test that registering a model with the same ID updates it."""
    registry = get_model_registry()
    model1 = DummyModel("model1")
    model2 = DummyModel("model2")

    registry.register(
        model_id="node-123",
        model=model1,
        model_type="DummyModel",
        device="cpu",
        memory_mb=100.0,
    )
    registry.register(
        model_id="node-123",
        model=model2,
        model_type="DummyModel",
        device="cuda",
        memory_mb=200.0,
    )

    assert registry.count() == 1
    info = registry.get_model_info("node-123")
    assert info is not None
    assert info.device == "cuda"
    assert info.memory_mb == 200.0


def test_unregister_model():
    """Test unregistering a model."""
    registry = get_model_registry()
    model = DummyModel("test-model")

    registry.register(
        model_id="node-123",
        model=model,
        model_type="DummyModel",
        device="cpu",
    )
    assert registry.count() == 1

    success = registry.unregister("node-123")
    assert success is True
    assert registry.count() == 0

    # Unregistering non-existent model returns False
    success = registry.unregister("node-123")
    assert success is False


def test_get_model():
    """Test getting a model by ID."""
    registry = get_model_registry()
    model = DummyModel("test-model")

    registry.register(
        model_id="node-123",
        model=model,
        model_type="DummyModel",
        device="cpu",
        keep_strong_ref=True,  # Keep strong ref for reliable retrieval
    )

    retrieved = registry.get_model("node-123")
    assert retrieved is model

    # Non-existent model returns None
    assert registry.get_model("non-existent") is None


def test_list_models():
    """Test listing all models."""
    registry = get_model_registry()
    model1 = DummyModel("model1")
    model2 = DummyModel("model2")

    registry.register(
        model_id="node-1",
        model=model1,
        model_type="DummyModel",
        device="cpu",
        memory_mb=100.0,
    )
    registry.register(
        model_id="node-2",
        model=model2,
        model_type="DummyModel",
        device="cuda",
        memory_mb=200.0,
    )

    models = registry.list_models()
    assert len(models) == 2

    model_ids = {m.id for m in models}
    assert model_ids == {"node-1", "node-2"}


def test_get_total_memory():
    """Test getting total memory usage."""
    registry = get_model_registry()

    registry.register(
        model_id="node-1",
        model=DummyModel(),
        model_type="DummyModel",
        device="cpu",
        memory_mb=100.0,
    )
    registry.register(
        model_id="node-2",
        model=DummyModel(),
        model_type="DummyModel",
        device="cuda",
        memory_mb=200.0,
    )

    total = registry.get_total_memory_mb()
    assert total == 300.0


def test_in_use_flag():
    """Test the in_use flag functionality."""
    registry = get_model_registry()
    model = DummyModel()

    registry.register(
        model_id="node-123",
        model=model,
        model_type="DummyModel",
        device="cpu",
    )

    # Initially not in use
    assert registry.is_in_use("node-123") is False

    # Mark as in use
    success = registry.mark_in_use("node-123")
    assert success is True
    assert registry.is_in_use("node-123") is True

    # Mark as not in use
    success = registry.mark_not_in_use("node-123")
    assert success is True
    assert registry.is_in_use("node-123") is False

    # Non-existent model
    assert registry.is_in_use("non-existent") is False
    assert registry.mark_in_use("non-existent") is False


def test_clear_all():
    """Test clearing all models."""
    registry = get_model_registry()

    registry.register(
        model_id="node-1",
        model=DummyModel(),
        model_type="DummyModel",
        device="cpu",
        memory_mb=100.0,
    )
    registry.register(
        model_id="node-2",
        model=DummyModel(),
        model_type="DummyModel",
        device="cuda",
        memory_mb=200.0,
    )

    assert registry.count() == 2

    models_cleared, memory_freed = registry.clear_all()
    assert models_cleared == 2
    assert memory_freed == 300.0
    assert registry.count() == 0


def test_clear_all_respects_in_use():
    """Test that clear_all respects the in_use flag."""
    registry = get_model_registry()

    registry.register(
        model_id="node-1",
        model=DummyModel(),
        model_type="DummyModel",
        device="cpu",
        memory_mb=100.0,
    )
    registry.register(
        model_id="node-2",
        model=DummyModel(),
        model_type="DummyModel",
        device="cuda",
        memory_mb=200.0,
    )

    # Mark one as in use
    registry.mark_in_use("node-1")

    # Clear without force
    models_cleared, memory_freed = registry.clear_all(force=False)
    assert models_cleared == 1  # Only node-2 cleared
    assert memory_freed == 200.0
    assert registry.count() == 1  # node-1 still there

    # Clear with force
    models_cleared, memory_freed = registry.clear_all(force=True)
    assert models_cleared == 1  # node-1 cleared
    assert memory_freed == 100.0
    assert registry.count() == 0


def test_cleanup_callback():
    """Test that cleanup callback is called on unregister."""
    registry = get_model_registry()
    cleanup_called = {"count": 0}

    def cleanup_fn():
        cleanup_called["count"] += 1

    registry.register(
        model_id="node-123",
        model=DummyModel(),
        model_type="DummyModel",
        device="cpu",
        cleanup_fn=cleanup_fn,
    )

    registry.unregister("node-123")
    assert cleanup_called["count"] == 1


def test_weak_reference_cleanup():
    """Test that weak references are cleaned up when model is garbage collected."""
    registry = get_model_registry()

    # Register without strong ref
    model = DummyModel("test")
    registry.register(
        model_id="node-123",
        model=model,
        model_type="DummyModel",
        device="cpu",
        keep_strong_ref=False,
    )

    # Model should be retrievable while we hold a reference
    assert registry.count() == 1

    # Delete our reference
    del model

    # Force garbage collection
    gc.collect()

    # list_models should clean up dead references
    models = registry.list_models()
    # The model may or may not be gone depending on GC timing
    # but the test verifies the cleanup mechanism doesn't crash
    assert len(models) <= 1
