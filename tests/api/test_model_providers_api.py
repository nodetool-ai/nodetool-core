"""Regression tests for /api/models/providers."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_get_models_providers_does_not_500(client: TestClient, headers: dict[str, str]):
    response = client.get("/api/models/providers", headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    for item in data:
        assert isinstance(item, dict)
        assert "provider" in item
        assert "capabilities" in item
