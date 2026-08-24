"""Regression tests: a bad frame must not kill the worker, and protocol
writes must never be truncated.

Covers:
  * stdio server survives malformed / oversized / non-map frames and answers
    with a protocol-level ``error`` frame (recovering ``request_id`` when it can)
  * stdio server always drains in-flight dispatch tasks on teardown
  * websocket server survives malformed binary frames and text frames
  * ``_ProtocolStdoutBuffer.write`` retries short writes
  * ``StdioTransport.send`` rejects a short write instead of desynchronizing
"""

import asyncio
import io
import os
import struct
from typing import Any, Awaitable, Callable, cast

import msgpack
import pytest
import pytest_asyncio
import websockets

import nodetool.worker.stdio_server as stdio_server_mod
import nodetool.worker.stdio_stdout_guard as guard_mod
from nodetool.worker.protocol import MAX_BRIDGE_FRAME_SIZE
from nodetool.worker.server import WorkerServer, start_server
from nodetool.worker.stdio_server import StdioTransport, StdioWorkerServer, salvage_request_id
from nodetool.worker.stdio_stdout_guard import _ProtocolStdoutBuffer


def _frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def _pack(msg: dict[str, Any]) -> bytes:
    return cast("bytes", msgpack.packb(msg))


def _unframe(stream: bytes) -> list[dict]:
    """Split a length-prefixed msgpack stream into decoded messages."""
    out: list[dict] = []
    pos = 0
    while pos + 4 <= len(stream):
        (length,) = struct.unpack(">I", stream[pos : pos + 4])
        pos += 4
        out.append(msgpack.unpackb(stream[pos : pos + length], raw=False))
        pos += length
    return out


class _CollectingStdout:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, data: bytes) -> int:
        self.data += data
        return len(data)

    def flush(self) -> None:
        return None


async def _run_stdio(monkeypatch, stdin_bytes: bytes, *, server: StdioWorkerServer | None = None) -> list[dict]:
    """Feed ``stdin_bytes`` to a StdioWorkerServer and return emitted frames."""
    out = _CollectingStdout()
    monkeypatch.setattr(stdio_server_mod, "get_protocol_stdout_buffer", lambda: out)

    class _FakeStdin:
        buffer = io.BytesIO(stdin_bytes)

    monkeypatch.setattr(stdio_server_mod.sys, "stdin", _FakeStdin())

    srv = server or StdioWorkerServer()
    if not srv._protocol._nodes_metadata:
        srv.set_nodes_metadata([{"node_type": "fake.Node", "properties": []}])
    await asyncio.wait_for(srv.run(), timeout=10)
    return _unframe(bytes(out.data))


# ── stdio: malformed frames ──────────────────────────────────────────────


async def test_stdio_survives_malformed_frame_and_keeps_serving(monkeypatch):
    bad = b"\xc1" + _pack({"request_id": "req-1"})[1:]  # 0xc1 is never valid msgpack
    stdin = _frame(bad) + _frame(_pack({"type": "discover", "request_id": "req-2"}))

    frames = await _run_stdio(monkeypatch, stdin)

    assert frames[0]["type"] == "error"
    assert frames[0]["request_id"] == "req-1"
    assert "Malformed msgpack frame" in frames[0]["data"]["error"]
    # The worker kept serving: the following valid frame was answered.
    assert frames[1]["type"] == "discover"
    assert frames[1]["request_id"] == "req-2"


async def test_stdio_oversized_frame_emits_error_and_stops_cleanly(monkeypatch):
    stdin = struct.pack(">I", MAX_BRIDGE_FRAME_SIZE + 1) + b"\x00" * 8

    frames = await _run_stdio(monkeypatch, stdin)

    assert len(frames) == 1
    assert frames[0]["type"] == "error"
    assert frames[0]["request_id"] is None
    assert "exceeds max size" in frames[0]["data"]["error"]


async def test_stdio_non_map_frame_is_rejected_without_dying(monkeypatch):
    stdin = _frame(cast("bytes", msgpack.packb([1, 2, 3]))) + _frame(
        _pack({"type": "discover", "request_id": "req-ok"})
    )

    frames = await _run_stdio(monkeypatch, stdin)

    assert frames[0]["type"] == "error"
    assert "msgpack map" in frames[0]["data"]["error"]
    assert frames[1]["type"] == "discover"


async def test_stdio_drains_inflight_tasks_when_a_bad_frame_ends_the_stream(monkeypatch):
    """An oversized frame ends the loop; the in-flight execute must still answer."""
    started = asyncio.Event()

    async def handle_execute(
        data: dict,
        _cancel: asyncio.Event,
        _progress: Callable[[dict[str, Any]], Awaitable[None]],
        _chunk: Callable[[dict[str, Any]], Awaitable[None]] | None,
        _update: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict:
        started.set()
        await asyncio.sleep(0.05)
        return {"outputs": {"out": 1}, "blobs": {}}

    srv = StdioWorkerServer()
    srv.set_nodes_metadata([{"node_type": "fake.Node", "properties": []}])
    srv.set_execute_handler(handle_execute)

    stdin = _frame(
        _pack({"type": "execute", "request_id": "exec-1", "data": {"node_type": "fake.Node"}})
    ) + struct.pack(">I", MAX_BRIDGE_FRAME_SIZE + 1)

    frames = await _run_stdio(monkeypatch, stdin, server=srv)

    assert started.is_set()
    results = [f for f in frames if f["type"] == "result"]
    assert results and results[0]["request_id"] == "exec-1"


def test_salvage_request_id():
    payload = _pack({"type": "execute", "request_id": "abc-123"})
    assert salvage_request_id(payload) == "abc-123"
    assert salvage_request_id(_pack({"type": "execute"})) is None
    long_id = "x" * 40  # str8 encoding
    assert salvage_request_id(_pack({"request_id": long_id})) == long_id


# ── websocket: malformed frames ──────────────────────────────────────────


@pytest_asyncio.fixture(loop_scope="function")
async def ws_server():
    worker = WorkerServer()
    worker.set_nodes_metadata([{"node_type": "fake.Node", "properties": []}])
    host, port, stop_event, task = await start_server(host="127.0.0.1", port=0, worker=worker)
    yield host, port
    stop_event.set()
    await task


async def _ws_recv(ws) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=5)
    return msgpack.unpackb(raw, raw=False)


@pytest.mark.asyncio(loop_scope="function")
async def test_websocket_survives_malformed_binary_frame(ws_server):
    host, port = ws_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(b"\xc1" + _pack({"request_id": "req-1"})[1:])
        err = await _ws_recv(ws)
        assert err["type"] == "error"
        assert err["request_id"] == "req-1"

        # Connection still usable.
        await ws.send(_pack({"type": "discover", "request_id": "req-2"}))
        reply = await _ws_recv(ws)
        assert reply["type"] == "discover"
        assert reply["request_id"] == "req-2"


@pytest.mark.asyncio(loop_scope="function")
async def test_websocket_survives_text_frame(ws_server):
    host, port = ws_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send("this is a text frame")
        err = await _ws_recv(ws)
        assert err["type"] == "error"

        await ws.send(_pack({"type": "discover", "request_id": "after-text"}))
        reply = await _ws_recv(ws)
        assert reply["type"] == "discover"


@pytest.mark.asyncio(loop_scope="function")
async def test_websocket_rejects_non_map_frame(ws_server):
    host, port = ws_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(cast("bytes", msgpack.packb([1, 2, 3])))
        err = await _ws_recv(ws)
        assert err["type"] == "error"
        assert "msgpack map" in err["data"]["error"]


# ── stdout guard: short writes ───────────────────────────────────────────


def test_protocol_stdout_buffer_retries_short_writes(tmp_path, monkeypatch):
    path = tmp_path / "protocol.bin"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT)
    real_write = os.write

    def short_write(target_fd: int, data) -> int:
        # Simulate the kernel accepting only part of a large payload.
        return real_write(target_fd, bytes(data)[:7])

    monkeypatch.setattr(guard_mod.os, "write", short_write)
    payload = bytes(range(256)) * 100
    try:
        written = _ProtocolStdoutBuffer(fd).write(payload)
    finally:
        os.close(fd)

    assert written == len(payload)
    assert path.read_bytes() == payload


async def test_stdio_transport_send_rejects_short_write(monkeypatch):
    class _ShortStdout:
        def write(self, data: bytes) -> int:
            return len(data) - 1

        def flush(self) -> None:
            return None

    monkeypatch.setattr(stdio_server_mod, "get_protocol_stdout_buffer", lambda: _ShortStdout())
    transport = StdioTransport()
    with pytest.raises(OSError, match="Short write"):
        await transport.send(b"payload")


# ── provider cache: bounded LRU ──────────────────────────────────────────


def test_provider_cache_is_bounded_and_releases_evicted_instances(monkeypatch):
    import nodetool.providers.base as providers_base
    from nodetool.worker import provider_handler

    closed: list[Any] = []

    class _FakeProvider:
        def __init__(self, secrets: dict[str, str], **_kwargs: Any) -> None:
            self.secrets = secrets

        def close(self) -> None:
            closed.append(self)

    monkeypatch.setattr(provider_handler, "_providers_imported", True)
    monkeypatch.setattr(
        providers_base, "get_registered_provider", lambda _enum: (_FakeProvider, {}), raising=False
    )
    monkeypatch.setattr(provider_handler, "PROVIDER_CACHE_MAX_SIZE", 2)
    monkeypatch.setattr(provider_handler, "_provider_cache", provider_handler.OrderedDict())

    a = provider_handler._get_provider("huggingface", {"HF_TOKEN": "a"})
    b = provider_handler._get_provider("huggingface", {"HF_TOKEN": "b"})
    # Cache hit keeps the same instance and refreshes recency for "a".
    assert provider_handler._get_provider("huggingface", {"HF_TOKEN": "a"}) is a
    c = provider_handler._get_provider("huggingface", {"HF_TOKEN": "c"})

    assert len(provider_handler._provider_cache) == 2
    assert closed == [b]  # least-recently-used was evicted and released
    assert provider_handler._get_provider("huggingface", {"HF_TOKEN": "c"}) is c
    assert provider_handler._get_provider("huggingface", {"HF_TOKEN": "a"}) is a
