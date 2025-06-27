import pytest
import httpx
from nodetool.common.worker_api_client import WorkerAPIClient

class DummyResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status
    def json(self):
        return self._data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError('error', request=None, response=None)

@pytest.mark.asyncio
async def test_get_installed_models(monkeypatch):
    client = WorkerAPIClient("http://worker")

    async def fake_get(path, **kwargs):
        assert path == "/huggingface_models"
        return DummyResponse([
            {"repo_id": "model1", "repo_type": "model", "size_on_disk": 123, "path": "/path/to/model1"}
        ])

    monkeypatch.setattr(client, "get", fake_get)
    models = await client.get_installed_models()
    assert len(models) == 1
    assert models[0].repo_id == "model1"

@pytest.mark.asyncio
async def test_get_recommended_models(monkeypatch):
    client = WorkerAPIClient("http://worker")

    async def fake_get(path, **kwargs):
        assert path == "/recommended_models"
        return DummyResponse([
            {"repo_id": "m2", "type": "hf.text_generation"}
        ])

    monkeypatch.setattr(client, "get", fake_get)
    models = await client.get_recommended_models()
    assert len(models) == 1
    assert models[0].repo_id == "m2"
    assert models[0].type == "hf.text_generation"

@pytest.mark.asyncio
async def test_get_system_stats(monkeypatch):
    client = WorkerAPIClient("http://worker")

    async def fake_get(path, **kwargs):
        assert path == "/system_stats"
        return DummyResponse({
            "cpu_percent": 10.0,
            "memory_total_gb": 8.0,
            "memory_used_gb": 1.0,
            "memory_percent": 12.5,
            "vram_total_gb": 4.0,
            "vram_used_gb": 0.5,
            "vram_percent": 12.5
        })

    monkeypatch.setattr(client, "get", fake_get)
    stats = await client.get_system_stats()
    assert stats.cpu_percent == 10.0
    assert stats.memory_total_gb == 8.0
    assert stats.vram_total_gb == 4.0


def test_get_url():
    client = WorkerAPIClient("http://worker")
    assert client._get_url("/foo") == "http://worker/foo"
    assert client._get_url("bar") == "http://worker/bar"
