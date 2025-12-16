from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

import nodetool.api.collection as collection_api

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class _DummyCollection:
    def __init__(self, name: str, count: int) -> None:
        self.name = name
        self.metadata = {}
        self.count = AsyncMock(return_value=count)


@pytest.mark.asyncio
async def test_list_collections_awaits_counts(client: TestClient, monkeypatch):
    dummy_col = _DummyCollection(name="col1", count=7)
    dummy_client = AsyncMock()
    dummy_client.list_collections = AsyncMock(return_value=[dummy_col])

    async def _fake_get_async_chroma_client(*args: object, **kwargs: object) -> object:
        return dummy_client

    monkeypatch.setattr(
        collection_api, "get_async_chroma_client", _fake_get_async_chroma_client
    )

    response = client.get("/api/collections")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["collections"][0]["name"] == "col1"
    assert payload["collections"][0]["count"] == 7

    dummy_col.count.assert_awaited_once()
