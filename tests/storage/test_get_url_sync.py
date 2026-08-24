"""Regression tests: AbstractStorage.get_url is synchronous.

All storage implementations return a plain ``str`` from ``get_url``. Call sites
must not await it (awaiting a str raises TypeError).
"""

import inspect
from io import BytesIO

import pytest

from nodetool.storage.abstract_storage import AbstractStorage
from nodetool.storage.file_storage import FileStorage
from nodetool.storage.memory_storage import MemoryStorage
from nodetool.workflows.processing_context import ProcessingContext


def test_get_url_is_not_a_coroutine_function(tmp_path):
    for storage in (
        FileStorage(base_path=str(tmp_path), base_url="http://localhost/files"),
        MemoryStorage(base_url="http://localhost/mem"),
    ):
        assert not inspect.iscoroutinefunction(storage.get_url)
        url = storage.get_url("abc.txt")
        assert isinstance(url, str)
        assert url.endswith("abc.txt")

    assert not inspect.iscoroutinefunction(AbstractStorage.get_url)


@pytest.mark.asyncio
async def test_asset_storage_url_returns_str(user_id: str):
    context = ProcessingContext(user_id=user_id, auth_token="t")
    url = await context.asset_storage_url("abc.txt")
    assert isinstance(url, str)
    assert url.endswith("abc.txt")


@pytest.mark.asyncio
async def test_named_asset_creation_paths_do_not_await_get_url(user_id: str):
    """Named asset creation used to raise TypeError from `await storage.get_url(...)`."""
    context = ProcessingContext(user_id=user_id, auth_token="t")

    audio = await context.audio_from_io(BytesIO(b"RIFFfake"), name="clip.wav")
    assert isinstance(audio.uri, str)
    assert audio.uri != ""
    assert audio.asset_id

    text = await context.text_from_str("hello", name="note.txt")
    assert isinstance(text.uri, str)
    assert text.uri != ""

    model3d = await context.model3d_from_io(BytesIO(b"solid\n"), name="mesh.obj")
    assert isinstance(model3d.uri, str)
    assert model3d.uri != ""
