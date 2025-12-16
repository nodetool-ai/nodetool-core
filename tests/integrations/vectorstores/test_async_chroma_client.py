import inspect

import pytest

from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    AsyncChromaClient,
)


class _DummyCollection:
    def __init__(self, name: str, metadata: dict, count: int) -> None:
        self.name = name
        self.metadata = metadata
        self._count = count

    def count(self) -> int:
        return self._count


class _DummySyncClient:
    def list_collections(self) -> list[_DummyCollection]:
        return [_DummyCollection(name="test", metadata={"a": "b"}, count=3)]


@pytest.mark.asyncio
async def test_list_collections_returns_async_wrappers():
    client = AsyncChromaClient(_DummySyncClient())  # type: ignore[arg-type]

    class _DummyExecutor:
        async def run(self, func, *args, **kwargs):
            return func(*args, **kwargs)

        def shutdown(self, wait: bool = False) -> None:
            return None

    client._executor = _DummyExecutor()  # type: ignore[assignment]

    collections = await client.list_collections()
    assert len(collections) == 1
    assert collections[0].name == "test"
    assert collections[0].metadata == {"a": "b"}

    awaitable = collections[0].count()
    assert inspect.isawaitable(awaitable)
    assert await awaitable == 3
