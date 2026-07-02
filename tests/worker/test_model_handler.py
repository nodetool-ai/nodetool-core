"""Tests for the models.* bridge handler."""

import asyncio
from pathlib import Path

import msgpack
import pytest
import pytest_asyncio
import websockets

from nodetool.worker import BRIDGE_PROTOCOL_VERSION
from nodetool.worker.server import WorkerServer, start_server


def test_protocol_version_is_3():
    assert BRIDGE_PROTOCOL_VERSION == 3


@pytest_asyncio.fixture(loop_scope="function")
async def server():
    worker = WorkerServer()
    host, port, stop_event, task = await start_server(
        host="127.0.0.1", port=0, worker=worker,
    )
    yield host, port
    stop_event.set()
    await task


@pytest.mark.asyncio(loop_scope="function")
async def test_models_list_cached(server, monkeypatch):
    """models.list_cached returns the worker's cached repos as UnifiedModel[]."""
    from nodetool.types.model import UnifiedModel
    import nodetool.worker.model_handler as mh

    async def fake_read_cached():
        return [
            UnifiedModel(
                id="org/model-a", type="hf.model", name="org/model-a",
                repo_id="org/model-a", downloaded=True, size_on_disk=123,
            )
        ]

    monkeypatch.setattr(mh, "read_cached_hf_models", fake_read_cached)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb(
            {"type": "models.list_cached", "request_id": "ml-1", "data": {}}
        ))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "result"
        assert resp["request_id"] == "ml-1"
        models = resp["data"]["models"]
        assert len(models) == 1
        assert models[0]["repo_id"] == "org/model-a"
        assert models[0]["downloaded"] is True


@pytest.mark.asyncio(loop_scope="function")
async def test_models_list_cached_forces_downloaded_true(server, monkeypatch):
    """Even if a cached entry reports downloaded=False, list_cached forces True."""
    from nodetool.types.model import UnifiedModel
    import nodetool.worker.model_handler as mh

    async def fake_read_cached():
        return [
            UnifiedModel(
                id="org/model-b", type="hf.model", name="org/model-b",
                repo_id="org/model-b", downloaded=False, size_on_disk=456,
            )
        ]

    monkeypatch.setattr(mh, "read_cached_hf_models", fake_read_cached)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb(
            {"type": "models.list_cached", "request_id": "ml-2", "data": {}}
        ))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "result"
        assert resp["data"]["models"][0]["downloaded"] is True
        assert resp["data"]["models"][0]["size_on_disk"] == 456


@pytest.mark.asyncio(loop_scope="function")
async def test_models_unknown_type(server):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb(
            {"type": "models.nonexistent", "request_id": "mu-1", "data": {}}
        ))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "error"
        assert "Unknown models message type" in resp["data"]["error"]


@pytest.mark.asyncio(loop_scope="function")
async def test_models_delete(server, monkeypatch):
    """models.delete removes the repo and returns { deleted: bool }."""
    import nodetool.worker.model_handler as mh

    seen: dict[str, str] = {}

    async def fake_delete(repo_id):
        seen["repo_id"] = repo_id
        return True

    monkeypatch.setattr(mh, "delete_cached_hf_model", fake_delete)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb(
            {"type": "models.delete", "request_id": "md-del",
             "data": {"repo_id": "org/model-a"}}
        ))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "result"
        assert resp["request_id"] == "md-del"
        assert resp["data"]["deleted"] is True
        assert seen["repo_id"] == "org/model-a"


@pytest.mark.asyncio(loop_scope="function")
async def test_models_delete_missing_returns_false(server, monkeypatch):
    """A repo that isn't cached returns deleted=False."""
    import nodetool.worker.model_handler as mh

    async def fake_delete(repo_id):
        return False

    monkeypatch.setattr(mh, "delete_cached_hf_model", fake_delete)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb(
            {"type": "models.delete", "request_id": "md-del2",
             "data": {"repo_id": "org/not-there"}}
        ))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "result"
        assert resp["data"]["deleted"] is False


@pytest.mark.asyncio(loop_scope="function")
async def test_models_download_emits_progress_then_result(server, monkeypatch):
    """models.download lists repo files, downloads each, emits ordered progress."""
    import nodetool.worker.model_handler as mh

    # One-file repo; async_hf_download drives the byte-delta progress_callback.
    async def fake_list_repo_files(repo_id, token=None):
        return [("model.bin", 100)]

    async def fake_download(repo_id, filename, *, token=None, progress_callback=None,
                            cancel_event=None, **kwargs):
        progress_callback(50, 100)
        progress_callback(50, 100)
        return Path("/tmp") / filename

    async def fake_token():
        return None

    monkeypatch.setattr(mh, "_list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(mh, "async_hf_download", fake_download)
    monkeypatch.setattr(mh, "get_hf_token", fake_token)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "models.download",
            "request_id": "md-1",
            "data": {"repo_id": "org/model-a", "model_type": "hf.model"},
        }))
        frames = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            f = msgpack.unpackb(raw, raw=False)
            frames.append(f)
            if f["type"] in ("result", "error"):
                break

    progress = [f for f in frames if f["type"] == "progress"]
    assert progress, "expected at least one progress frame"
    first = progress[0]["data"]
    assert first["repo_id"] == "org/model-a"
    assert first["status"] in ("start", "progress")
    # Exact field set the web ModelDownloadStore consumes:
    for key in ("status", "repo_id", "path", "model_type", "downloaded_bytes",
                "total_bytes", "downloaded_files", "current_files", "total_files"):
        assert key in first
    # First emitted frame must be the "start" frame.
    assert progress[0]["data"]["status"] == "start"
    # downloaded_bytes accumulates the per-chunk deltas up to total_bytes.
    streamed = [
        p["data"]["downloaded_bytes"] for p in progress
        if p["data"]["status"] == "progress"
    ]
    assert streamed == sorted(streamed), "downloaded_bytes must be non-decreasing"
    assert streamed[-1] == 100
    final = frames[-1]
    assert final["type"] == "result"
    assert final["data"]["status"] == "completed"
    assert final["data"]["repo_id"] == "org/model-a"


@pytest.mark.asyncio(loop_scope="function")
async def test_models_download_accumulates_deltas(server, monkeypatch):
    """Per-chunk deltas accumulate to the file total (not treated as absolute)."""
    import nodetool.worker.model_handler as mh

    async def fake_list_repo_files(repo_id, token=None):
        return [("model.bin", 100)]

    async def fake_download(repo_id, filename, *, token=None, progress_callback=None,
                            cancel_event=None, **kwargs):
        # Uneven per-chunk deltas summing to the file size.
        progress_callback(40, 100)
        progress_callback(40, 100)
        progress_callback(20, 100)
        return Path("/tmp") / filename

    async def fake_token():
        return None

    monkeypatch.setattr(mh, "_list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(mh, "async_hf_download", fake_download)
    monkeypatch.setattr(mh, "get_hf_token", fake_token)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "models.download",
            "request_id": "md-acc",
            "data": {"repo_id": "org/model-a"},
        }))
        frames = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            f = msgpack.unpackb(raw, raw=False)
            frames.append(f)
            if f["type"] in ("result", "error"):
                break

    streamed = [
        f["data"]["downloaded_bytes"] for f in frames
        if f["type"] == "progress" and f["data"]["status"] == "progress"
    ]
    # Deltas 40/40/20 accumulate to 40, 80, 100 — NOT [40, 40, 20] (the bug
    # of treating each delta as the file's absolute progress).
    assert streamed == [40, 80, 100]


@pytest.mark.asyncio(loop_scope="function")
async def test_models_download_mid_file_cancel(server, monkeypatch):
    """A mid-stream cancel (CancelledError) yields a terminal cancelled frame."""
    import nodetool.worker.model_handler as mh

    async def fake_list_repo_files(repo_id, token=None):
        return [("model.bin", 100)]

    async def fake_download(repo_id, filename, *, token=None, progress_callback=None,
                            cancel_event=None, **kwargs):
        if progress_callback:
            progress_callback(40, 100)
        # The real async_hf_download raises CancelledError (a BaseException)
        # when cancel_event is set mid-stream. Without an explicit handler this
        # escapes `except Exception` and the bridge promise hangs forever.
        raise asyncio.CancelledError("Download cancelled")

    async def fake_token():
        return None

    monkeypatch.setattr(mh, "_list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(mh, "async_hf_download", fake_download)
    monkeypatch.setattr(mh, "get_hf_token", fake_token)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "models.download",
            "request_id": "md-cancel",
            "data": {"repo_id": "org/model-a"},
        }))
        frames = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            f = msgpack.unpackb(raw, raw=False)
            frames.append(f)
            if f["type"] in ("result", "error"):
                break

    final = frames[-1]
    assert final["type"] == "result"
    assert final["data"]["status"] == "cancelled"
    assert any(
        f["type"] == "progress" and f["data"]["status"] == "cancelled"
        for f in frames
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_models_download_resolves_token_on_worker(server, monkeypatch):
    """The HF token is resolved on the worker, never taken from the request."""
    import nodetool.worker.model_handler as mh

    received: dict[str, str | None] = {}

    async def fake_list_repo_files(repo_id, token=None):
        received["list_token"] = token
        return [("model.bin", 10)]

    async def fake_download(repo_id, filename, *, token=None, progress_callback=None,
                            cancel_event=None, **kwargs):
        received["dl_token"] = token
        if progress_callback:
            progress_callback(10, 10)
        return Path("/tmp") / filename

    async def fake_token():
        return "worker-secret"

    monkeypatch.setattr(mh, "_list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(mh, "async_hf_download", fake_download)
    monkeypatch.setattr(mh, "get_hf_token", fake_token)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "models.download",
            "request_id": "md-token",
            # A malicious client token must be ignored.
            "data": {"repo_id": "org/model-a", "token": "client-evil"},
        }))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            f = msgpack.unpackb(raw, raw=False)
            if f["type"] in ("result", "error"):
                break

    assert received["dl_token"] == "worker-secret"
    assert received["list_token"] == "worker-secret"


@pytest.mark.asyncio(loop_scope="function")
async def test_models_download_filters_by_allow_patterns(server, monkeypatch):
    """allow_patterns restricts the file set that is downloaded."""
    import nodetool.worker.model_handler as mh

    downloaded: list[str] = []

    async def fake_list_repo_files(repo_id, token=None):
        return [("model.safetensors", 100), ("model.bin", 100), ("README.md", 1)]

    async def fake_download(repo_id, filename, *, token=None, progress_callback=None,
                            cancel_event=None, **kwargs):
        downloaded.append(filename)
        if progress_callback:
            progress_callback(100, 100)
        return Path("/tmp") / filename

    async def fake_token():
        return None

    monkeypatch.setattr(mh, "_list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(mh, "async_hf_download", fake_download)
    monkeypatch.setattr(mh, "get_hf_token", fake_token)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "models.download",
            "request_id": "md-allow",
            "data": {"repo_id": "org/model-a",
                     "allow_patterns": ["*.safetensors"]},
        }))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            f = msgpack.unpackb(raw, raw=False)
            if f["type"] in ("result", "error"):
                break

    assert downloaded == ["model.safetensors"]


@pytest.mark.asyncio(loop_scope="function")
async def test_models_download_error_emits_error_frame(server, monkeypatch):
    """A failing download emits an error-status progress frame then an error frame."""
    import nodetool.worker.model_handler as mh

    async def fake_list_repo_files(repo_id, token=None):
        raise RuntimeError("hub exploded")

    async def fake_token():
        return None

    monkeypatch.setattr(mh, "_list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(mh, "get_hf_token", fake_token)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "models.download",
            "request_id": "md-err",
            "data": {"repo_id": "org/model-a"},
        }))
        frames = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            f = msgpack.unpackb(raw, raw=False)
            frames.append(f)
            if f["type"] == "error":
                break

    # An error-status progress frame precedes the terminal error frame.
    progress = [f for f in frames if f["type"] == "progress"]
    assert any(p["data"]["status"] == "error" for p in progress)
    assert frames[-1]["type"] == "error"
    assert "hub exploded" in frames[-1]["data"]["error"]
