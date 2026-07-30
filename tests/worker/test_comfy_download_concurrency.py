"""Regression tests for concurrent/corrupt comfy.models.download handling.

Two concurrent downloads of the same model used to share one `.part` path and
interleave writes, installing a byte-scrambled file that the `exists` fast path
then trusted forever. A truncated body was likewise installed unchecked, and
non-ComfyError failures escaped without an "error" progress frame.
"""

import asyncio

import pytest
from aiohttp import web

from nodetool.worker import comfy_handler
from nodetool.worker.comfy_handler import ComfyError, _handle_models_download


class Collector:
    """Captures the progress/result frames the handler emits."""

    def __init__(self) -> None:
        self.progress: list[dict] = []
        self.results: list[dict] = []

    async def send_progress(self, request_id, data):
        self.progress.append(data)

    async def send_result(self, request_id, data):
        self.results.append(data)


async def _download(data, request_id, collector):
    await _handle_models_download(
        data, request_id, {}, collector.send_progress, collector.send_result
    )


# --- stub source plumbing -------------------------------------------------------------


class _StubContent:
    def __init__(self, chunks, fail_after=None):
        self._chunks = chunks
        self._fail_after = fail_after

    async def _gen(self):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise OSError("connection reset by peer")
            yield chunk
            await asyncio.sleep(0)

    def iter_chunked(self, _size):
        return self._gen()


class _StubResponse:
    def __init__(self, chunks, headers, fail_after=None):
        self.headers = headers
        self.content = _StubContent(chunks, fail_after)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_source(monkeypatch, factory, name="model.safetensors"):
    calls = []

    async def fake_open(session, source):
        calls.append(source)
        return factory(len(calls) - 1), name

    monkeypatch.setattr(comfy_handler, "_open_model_source", fake_open)
    return calls


# --- concurrency ----------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_concurrent_downloads_of_same_target_transfer_once(tmp_path, monkeypatch):
    """Two concurrent requests for one model: one transfer, intact file, both succeed."""
    monkeypatch.setattr("nodetool.worker.comfy_handler.is_ip_private", lambda x: False)
    monkeypatch.setenv("COMFY_MODELS_DIR", str(tmp_path))
    payload = bytes(range(256)) * 4096  # 1 MiB of position-sensitive bytes
    hits = []

    async def serve(request: web.Request) -> web.StreamResponse:
        hits.append(request.path)
        resp = web.StreamResponse(headers={"Content-Length": str(len(payload))})
        await resp.prepare(request)
        # Chunked + yielding writes guarantee the two requests would interleave
        # if they were allowed to run concurrently.
        for off in range(0, len(payload), 64 * 1024):
            await resp.write(payload[off : off + 64 * 1024])
            await asyncio.sleep(0.01)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.add_routes([web.get("/model.safetensors", serve)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]

    try:
        source = {"type": "url", "url": f"http://127.0.0.1:{port}/model.safetensors"}
        a, b = Collector(), Collector()
        data_a = {"folder": "checkpoints", "filename": "model.safetensors", "source": source}
        data_b = dict(data_a)
        await asyncio.gather(_download(data_a, "d-1", a), _download(data_b, "d-2", b))
    finally:
        await runner.cleanup()

    assert len(hits) == 1, "the second request must reuse the first transfer"

    target = tmp_path / "checkpoints" / "model.safetensors"
    assert target.read_bytes() == payload
    assert not target.with_name("model.safetensors.part").exists()

    statuses = sorted(r["status"] for r in (a.results + b.results))
    assert statuses == ["completed", "exists"]
    assert all(r["size_bytes"] == len(payload) for r in a.results + b.results)


@pytest.mark.asyncio(loop_scope="function")
async def test_waiter_retries_when_the_first_download_fails(tmp_path, monkeypatch):
    """A waiter must not silently inherit the first download's failure."""
    monkeypatch.setenv("COMFY_MODELS_DIR", str(tmp_path))
    payload = b"abc" * 1000
    started = asyncio.Event()

    def factory(call_index):
        if call_index == 0:
            started.set()
            return _StubResponse([payload], {"Content-Length": str(len(payload))}, fail_after=0)
        return _StubResponse([payload], {"Content-Length": str(len(payload))})

    calls = _patch_source(monkeypatch, factory)

    first, second = Collector(), Collector()
    data = {"folder": "checkpoints", "filename": "model.safetensors", "source": {"type": "url", "url": "http://x/model.safetensors"}}

    async def run_first():
        with pytest.raises(ComfyError):
            await _download(dict(data), "f-1", first)

    async def run_second():
        await started.wait()
        await _download(dict(data), "f-2", second)

    await asyncio.gather(run_first(), run_second())

    assert len(calls) == 2, "the waiter must retry the transfer after a failure"
    assert second.results[-1]["status"] == "completed"
    target = tmp_path / "checkpoints" / "model.safetensors"
    assert target.read_bytes() == payload
    assert not target.with_name("model.safetensors.part").exists()


# --- size validation ------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_truncated_body_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_MODELS_DIR", str(tmp_path))
    _patch_source(
        monkeypatch,
        lambda _i: _StubResponse([b"short"], {"Content-Length": "5000"}),
    )

    collector = Collector()
    with pytest.raises(ComfyError, match="incomplete"):
        await _download(
            {"folder": "checkpoints", "filename": "model.safetensors", "source": {"type": "url", "url": "http://x/m"}},
            "t-1",
            collector,
        )

    target = tmp_path / "checkpoints" / "model.safetensors"
    assert not target.exists()
    assert not target.with_name("model.safetensors.part").exists()
    assert collector.progress[-1]["status"] == "error"
    assert "incomplete" in collector.progress[-1]["error"]
    assert collector.results == []


@pytest.mark.asyncio(loop_scope="function")
async def test_missing_content_length_is_accepted(tmp_path, monkeypatch):
    """Servers that omit Content-Length must still be able to complete."""
    monkeypatch.setenv("COMFY_MODELS_DIR", str(tmp_path))
    _patch_source(monkeypatch, lambda _i: _StubResponse([b"payload"], {}))

    collector = Collector()
    await _download(
        {"folder": "checkpoints", "filename": "model.safetensors", "source": {"type": "url", "url": "http://x/m"}},
        "n-1",
        collector,
    )
    assert collector.results[-1]["status"] == "completed"
    assert (tmp_path / "checkpoints" / "model.safetensors").read_bytes() == b"payload"


# --- error frames ---------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_non_comfy_failure_emits_error_frame(tmp_path, monkeypatch):
    """A transport/filesystem error must surface as an "error" progress frame."""
    monkeypatch.setenv("COMFY_MODELS_DIR", str(tmp_path))
    _patch_source(
        monkeypatch,
        lambda _i: _StubResponse([b"a" * 16], {"Content-Length": "16"}, fail_after=0),
    )

    collector = Collector()
    with pytest.raises(ComfyError) as exc:
        await _download(
            {"folder": "checkpoints", "filename": "model.safetensors", "source": {"type": "url", "url": "http://x/m"}},
            "e-1",
            collector,
        )
    assert "OSError" in str(exc.value)
    assert collector.progress[-1]["status"] == "error"
    assert "connection reset" in collector.progress[-1]["error"]
    assert not (tmp_path / "checkpoints" / "model.safetensors.part").exists()
