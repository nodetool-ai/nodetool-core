from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nodetool.deploy.admin_routes import create_admin_router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(create_admin_router())
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestAdminRoutes:
    def test_hf_download_stream(self, client):
        with patch("nodetool.deploy.admin_routes.download_hf_model") as mock_dl:
            async def gen():
                yield {"status": "ok"}
            mock_dl.return_value = gen()

            with client.stream("POST", "/admin/models/huggingface/download", json={"repo_id": "r"}) as resp:
                assert resp.status_code == 200
                text = resp.read().decode()
                assert "data: " in text

    def test_hf_download_missing_repo(self, client):
        resp = client.post("/admin/models/huggingface/download", json={})
        assert resp.status_code == 400

    def test_ollama_download_stream(self, client):
        with patch("nodetool.deploy.admin_routes.download_ollama_model") as mock_dl:
            async def gen():
                yield {"status": "ok"}
            mock_dl.return_value = gen()

            with client.stream("POST", "/admin/models/ollama/download", json={"model_name": "m"}) as resp:
                assert resp.status_code == 200
                text = resp.read().decode()
                assert "data: " in text

    def test_cache_scan(self, client):
        with patch("nodetool.deploy.admin_routes.scan_hf_cache") as mock_scan:
            async def gen():
                yield {"status": "completed"}
            mock_scan.return_value = gen()

            resp = client.get("/admin/cache/scan")
            assert resp.status_code == 200
            assert resp.json()["status"] == "completed"

    def test_cache_size(self, client):
        with patch("nodetool.deploy.admin_routes.calculate_cache_size") as mock_calc:
            async def gen():
                yield {"success": True, "size_gb": 1}
            mock_calc.return_value = gen()

            resp = client.get("/admin/cache/size")
            assert resp.status_code == 200
            assert resp.json()["success"] is True

    def test_delete_hf_model(self, client):
        with patch("nodetool.deploy.admin_routes.delete_hf_model") as mock_del:
            async def gen():
                yield {"status": "completed", "repo_id": "r"}
            mock_del.return_value = gen()

            resp = client.delete("/admin/models/huggingface/r")
            assert resp.status_code == 200
            assert resp.json()["repo_id"] == "r"

    def test_admin_workflows_status(self, client):
        with patch("nodetool.deploy.admin_routes.get_workflow_registry") as mock_reg:
            mock_reg.return_value = {"a": object(), "b": object()}
            resp = client.get("/admin/workflows/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["workflow_count"] == 2

