"""Tests for the memory management API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nodetool.api.app import app
from nodetool.workflows.model_registry import ModelRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before and after each test."""
    ModelRegistry.reset_instance()
    yield
    ModelRegistry.reset_instance()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestMemoryStats:
    """Tests for GET /api/memory endpoint."""

    def test_get_memory_stats(self, client):
        """Test getting memory statistics."""
        response = client.get("/api/memory")
        assert response.status_code == 200

        data = response.json()
        assert "ram_mb" in data
        assert "memory_cache_count" in data
        assert "loaded_models_count" in data
        assert "loaded_models_memory_mb" in data
        assert isinstance(data["ram_mb"], (int, float))
        assert data["loaded_models_count"] == 0
        assert data["loaded_models_memory_mb"] == 0.0

    def test_memory_stats_with_models(self, client):
        """Test memory stats when models are registered."""
        registry = ModelRegistry.get_instance()
        registry.register(
            model_id="test-model",
            model=MagicMock(),
            model_type="TestModel",
            device="cpu",
            memory_mb=100.0,
            keep_strong_ref=True,  # Keep strong ref to prevent GC
        )

        response = client.get("/api/memory")
        assert response.status_code == 200

        data = response.json()
        assert data["loaded_models_count"] == 1
        assert data["loaded_models_memory_mb"] == 100.0


class TestListModels:
    """Tests for GET /api/memory/models endpoint."""

    def test_list_empty_models(self, client):
        """Test listing models when none are loaded."""
        response = client.get("/api/memory/models")
        assert response.status_code == 200

        data = response.json()
        assert data["models"] == []
        assert data["total_memory_mb"] == 0.0

    def test_list_loaded_models(self, client):
        """Test listing loaded models."""
        registry = ModelRegistry.get_instance()
        registry.register(
            model_id="model-1",
            model=MagicMock(),
            model_type="TestPipeline",
            device="cuda",
            memory_mb=1000.0,
            offloaded=False,
            hf_model_id="test/model-1",
            keep_strong_ref=True,  # Keep strong ref to prevent GC
        )
        registry.register(
            model_id="model-2",
            model=MagicMock(),
            model_type="AnotherPipeline",
            device="cpu",
            memory_mb=500.0,
            offloaded=True,
            hf_model_id="test/model-2",
            keep_strong_ref=True,  # Keep strong ref to prevent GC
        )

        response = client.get("/api/memory/models")
        assert response.status_code == 200

        data = response.json()
        assert len(data["models"]) == 2
        assert data["total_memory_mb"] == 1500.0

        # Check model details
        model_ids = {m["id"] for m in data["models"]}
        assert model_ids == {"model-1", "model-2"}

        model_1 = next(m for m in data["models"] if m["id"] == "model-1")
        assert model_1["type"] == "TestPipeline"
        assert model_1["device"] == "cuda"
        assert model_1["memory_mb"] == 1000.0
        assert model_1["offloaded"] is False
        assert model_1["model_id"] == "test/model-1"


class TestUnloadModel:
    """Tests for DELETE /api/memory/models/{model_id} endpoint."""

    def test_unload_model_success(self, client):
        """Test successfully unloading a model."""
        registry = ModelRegistry.get_instance()
        registry.register(
            model_id="test-model",
            model=MagicMock(),
            model_type="TestModel",
            device="cpu",
            memory_mb=100.0,
        )

        response = client.delete("/api/memory/models/test-model")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["model_id"] == "test-model"
        assert data["memory_freed_mb"] == 100.0

        # Verify model is gone
        assert registry.count() == 0

    def test_unload_model_not_found(self, client):
        """Test unloading a non-existent model."""
        response = client.delete("/api/memory/models/non-existent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_unload_model_in_use(self, client):
        """Test unloading a model that's in use."""
        registry = ModelRegistry.get_instance()
        registry.register(
            model_id="test-model",
            model=MagicMock(),
            model_type="TestModel",
            device="cpu",
            memory_mb=100.0,
        )
        registry.mark_in_use("test-model")

        response = client.delete("/api/memory/models/test-model")
        assert response.status_code == 409
        assert "in use" in response.json()["detail"].lower()

        # Model should still be there
        assert registry.count() == 1

    def test_unload_model_in_use_with_force(self, client):
        """Test force unloading a model that's in use."""
        registry = ModelRegistry.get_instance()
        registry.register(
            model_id="test-model",
            model=MagicMock(),
            model_type="TestModel",
            device="cpu",
            memory_mb=100.0,
        )
        registry.mark_in_use("test-model")

        response = client.delete("/api/memory/models/test-model?force=true")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert registry.count() == 0


class TestClearAllModels:
    """Tests for POST /api/memory/models/clear endpoint."""

    def test_clear_all_models(self, client):
        """Test clearing all models."""
        registry = ModelRegistry.get_instance()
        registry.register(
            model_id="model-1",
            model=MagicMock(),
            model_type="TestModel",
            device="cpu",
            memory_mb=100.0,
        )
        registry.register(
            model_id="model-2",
            model=MagicMock(),
            model_type="TestModel",
            device="cuda",
            memory_mb=200.0,
        )

        response = client.post("/api/memory/models/clear")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["models_unloaded"] == 2
        assert registry.count() == 0

    def test_clear_models_respects_in_use(self, client):
        """Test that clear respects in_use flag."""
        registry = ModelRegistry.get_instance()
        registry.register(
            model_id="model-1",
            model=MagicMock(),
            model_type="TestModel",
            device="cpu",
            memory_mb=100.0,
        )
        registry.register(
            model_id="model-2",
            model=MagicMock(),
            model_type="TestModel",
            device="cuda",
            memory_mb=200.0,
        )
        registry.mark_in_use("model-1")

        response = client.post("/api/memory/models/clear")
        assert response.status_code == 200

        data = response.json()
        assert data["models_unloaded"] == 1  # Only model-2 cleared
        assert registry.count() == 1  # model-1 still there


class TestClearGPUCache:
    """Tests for POST /api/memory/gpu endpoint."""

    def test_clear_gpu_cache_no_gpu(self, client):
        """Test clearing GPU cache when no GPU is available."""
        with patch("nodetool.api.memory._clear_gpu_cache", return_value=False):
            response = client.post("/api/memory/gpu")
            assert response.status_code == 200

            data = response.json()
            assert "No GPU available" in data["message"] or data["success"] is False

    @patch("nodetool.api.memory._clear_gpu_cache")
    @patch("nodetool.api.memory.get_gpu_memory_usage_mb")
    def test_clear_gpu_cache_success(self, mock_gpu_mem, mock_clear, client):
        """Test successfully clearing GPU cache."""
        mock_gpu_mem.side_effect = [
            (1000.0, 2000.0),  # Before
            (500.0, 1000.0),  # After
        ]
        mock_clear.return_value = True

        response = client.post("/api/memory/gpu")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["message"] == "GPU cache cleared"


class TestFullCleanup:
    """Tests for POST /api/memory/all endpoint."""

    def test_full_cleanup(self, client):
        """Test full memory cleanup."""
        registry = ModelRegistry.get_instance()
        registry.register(
            model_id="test-model",
            model=MagicMock(),
            model_type="TestModel",
            device="cpu",
            memory_mb=100.0,
        )

        response = client.post("/api/memory/all")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["models_unloaded"] == 1
        assert registry.count() == 0

    def test_full_cleanup_with_force(self, client):
        """Test full cleanup with force flag."""
        registry = ModelRegistry.get_instance()
        registry.register(
            model_id="test-model",
            model=MagicMock(),
            model_type="TestModel",
            device="cpu",
            memory_mb=100.0,
        )
        registry.mark_in_use("test-model")

        # Without force, model should remain
        response = client.post("/api/memory/all")
        data = response.json()
        assert data["models_unloaded"] == 0
        assert registry.count() == 1

        # With force, model should be cleared
        response = client.post("/api/memory/all?force=true")
        data = response.json()
        assert data["success"] is True
        assert data["models_unloaded"] == 1
        assert registry.count() == 0
