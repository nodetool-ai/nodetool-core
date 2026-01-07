import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nodetool.api.comfy import router
from nodetool.api.server import create_app


@pytest.fixture
def test_app():
    app = create_app(routers=[router])
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


@pytest.fixture
def mock_comfy_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        os.makedirs(models_dir)

        checkpoints_dir = os.path.join(models_dir, "checkpoints")
        os.makedirs(checkpoints_dir)
        with open(os.path.join(checkpoints_dir, "model1.safetensors"), "w") as f:
            f.write("dummy")
        with open(os.path.join(checkpoints_dir, "model2.ckpt"), "w") as f:
            f.write("dummy")

        vae_dir = os.path.join(models_dir, "vae")
        os.makedirs(vae_dir)
        with open(os.path.join(vae_dir, "vae.safetensors"), "w") as f:
            f.write("dummy")

        loras_dir = os.path.join(models_dir, "loras")
        os.makedirs(loras_dir)
        with open(os.path.join(loras_dir, "lora1.safetensors"), "w") as f:
            f.write("dummy")
        with open(os.path.join(loras_dir, "lora2.pt"), "w") as f:
            f.write("dummy")

        empty_dir = os.path.join(models_dir, "empty_folder")
        os.makedirs(empty_dir)

        with patch("nodetool.api.comfy.Environment.get_comfy_folder", return_value=tmpdir):
            yield tmpdir


class TestListAllComfyModelTypes:
    def test_returns_empty_when_no_comfy_folder(self):
        with patch("nodetool.api.comfy.Environment.get_comfy_folder", return_value=None):
            app = create_app(routers=[router])
            client = TestClient(app)
            response = client.get("/api/comfy/models")
            assert response.status_code == 200
            assert response.json() == {}

    def test_returns_model_types(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/models")
        assert response.status_code == 200
        data = response.json()
        assert "checkpoints" in data
        assert "vae" in data
        assert "loras" in data
        assert "empty_folder" not in data


class TestListComfyModelsByType:
    def test_returns_error_when_no_comfy_folder(self, client):
        with patch("nodetool.api.comfy.Environment.get_comfy_folder", return_value=None):
            response = client.get("/api/comfy/models/checkpoints")
            assert response.status_code == 400
            assert "COMFY_FOLDER" in response.json()["detail"]

    def test_returns_checkpoints(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/models/checkpoints")
        assert response.status_code == 200
        models = response.json()
        assert "model1.safetensors" in models
        assert "model2.ckpt" in models

    def test_returns_vae_models(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/models/vae")
        assert response.status_code == 200
        models = response.json()
        assert "vae.safetensors" in models

    def test_returns_empty_list_for_missing_folder(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/models/nonexistent")
        assert response.status_code == 400
        assert "Unknown model type" in response.json()["detail"]

    def test_returns_empty_list_for_empty_folder(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/models/empty_folder")
        assert response.status_code == 200
        assert response.json() == []

    def test_validates_model_type(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/models/invalid_type")
        assert response.status_code == 400


class TestListComfyModelsWithPaths:
    def test_returns_models_with_paths(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/models/checkpoints/paths")
        assert response.status_code == 200
        models = response.json()
        assert len(models) == 2
        for model in models:
            assert "name" in model
            assert "path" in model
            assert mock_comfy_folder in model["path"]


class TestSpecificModelTypeEndpoints:
    def test_checkpoints_endpoint(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/checkpoints")
        assert response.status_code == 200
        models = response.json()
        assert "model1.safetensors" in models
        assert "model2.ckpt" in models

    def test_vae_endpoint(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/vae")
        assert response.status_code == 200
        models = response.json()
        assert "vae.safetensors" in models

    def test_loras_endpoint(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/loras")
        assert response.status_code == 200
        models = response.json()
        assert "lora1.safetensors" in models
        assert "lora2.pt" in models


class TestListModelsInFolder:
    def test_filters_by_extension(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/models/checkpoints")
        models = response.json()
        assert "model1.safetensors" in models
        assert "model2.ckpt" in models

    def test_returns_sorted_list(self, client, mock_comfy_folder):
        response = client.get("/api/comfy/models/checkpoints")
        models = response.json()
        assert models == sorted(models)


class TestComfyModelFoldersHelper:
    def test_get_comfy_model_folders_returns_dict(self):
        from nodetool.metadata.types import get_comfy_model_folders

        folders = get_comfy_model_folders()
        assert isinstance(folders, dict)
        assert "checkpoints" in folders
        assert "loras" in folders
        assert isinstance(folders["checkpoints"], list)
        assert len(folders["checkpoints"]) > 0

    def test_comfy_model_type_folders_constant_exists(self):
        from nodetool.metadata.types import COMFY_MODEL_TYPE_FOLDERS

        assert isinstance(COMFY_MODEL_TYPE_FOLDERS, dict)
        assert "checkpoints" in COMFY_MODEL_TYPE_FOLDERS
