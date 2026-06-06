"""Tests for the WebSocket worker server protocol parity.

These exercise the WebSocket transport end-to-end against a real
``websockets`` client, asserting the server delegates to the shared
``WorkerProtocolServer`` (discover / worker.status / execute /
execute.stream / unknown-type).
"""

import asyncio
from typing import Any, Awaitable, Callable, cast

import msgpack
import pytest
import pytest_asyncio
import websockets

from nodetool.worker import BRIDGE_PROTOCOL_VERSION
from nodetool.worker.server import WorkerServer, start_server


def _pack(msg: dict[str, Any]) -> bytes:
    return cast("bytes", msgpack.packb(msg))


async def _recv(ws) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=5)
    return msgpack.unpackb(raw, raw=False)


@pytest_asyncio.fixture(loop_scope="function")
async def configured_server():
    """Start a fully-configured server with fake metadata and a streaming
    execute handler. Yields (host, port)."""
    worker = WorkerServer()
    worker.set_nodes_metadata([{"node_type": "fake.Node", "properties": []}])
    worker.set_namespaces(["fake"])
    worker.set_load_errors([])

    async def handle_execute(
        data: dict,
        _cancel_event: asyncio.Event,
        _emit_progress: Callable[[dict[str, Any]], Awaitable[None]],
        emit_chunk: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> dict:
        if data.get("node_type") != "fake.Echo":
            raise ValueError(f"Unknown node type: {data.get('node_type')}")
        # Emit a couple of chunks when streaming is requested.
        if emit_chunk is not None:
            await emit_chunk({"outputs": {"output": "a"}, "blobs": {}})
            await emit_chunk({"outputs": {"output": "b"}, "blobs": {}})
        return {"outputs": {"output": data.get("fields", {}).get("text", "")}, "blobs": {}}

    worker.set_execute_handler(handle_execute)

    host, port, stop_event, task = await start_server(
        host="127.0.0.1", port=0, worker=worker,
    )
    yield host, port
    stop_event.set()
    await task


@pytest_asyncio.fixture(loop_scope="function")
async def cancellable_server():
    """Start a server whose streaming execute handler loops emitting chunks
    until ``cancel_event`` is set. Yields (host, port)."""
    worker = WorkerServer()
    worker.set_nodes_metadata([{"node_type": "fake.Stream", "properties": []}])
    worker.set_namespaces(["fake"])
    worker.set_load_errors([])

    async def handle_execute(
        data: dict[str, Any],
        cancel_event: asyncio.Event,
        emit_progress: Callable[[dict[str, Any]], Awaitable[None]],
        emit_chunk: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> dict[str, Any]:
        _ = data, emit_progress
        assert emit_chunk is not None
        i = 0
        while not cancel_event.is_set():
            await emit_chunk({"outputs": {"output": i}, "blobs": {}})
            i += 1
            # Yield control so the cancel message can be dispatched between
            # chunk emissions.
            await asyncio.sleep(0.01)
        return {"outputs": {"output": "done"}, "blobs": {}}

    worker.set_execute_handler(handle_execute)

    host, port, stop_event, task = await start_server(
        host="127.0.0.1", port=0, worker=worker,
    )
    yield host, port
    stop_event.set()
    await task


@pytest.mark.asyncio(loop_scope="function")
async def test_discover_returns_protocol_metadata(configured_server):
    host, port = configured_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(_pack({"type": "discover", "request_id": "d-1", "data": {}}))
        resp = await _recv(ws)
        assert resp["type"] == "discover"
        assert resp["request_id"] == "d-1"
        assert resp["data"]["protocol_version"] == BRIDGE_PROTOCOL_VERSION
        assert resp["data"]["nodes"] == [{"node_type": "fake.Node", "properties": []}]
        assert resp["data"]["load_errors"] == []


@pytest.mark.asyncio(loop_scope="function")
async def test_worker_status_reports_websocket_transport(configured_server):
    host, port = configured_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(_pack({"type": "worker.status", "request_id": "s-1", "data": {}}))
        resp = await _recv(ws)
        assert resp["type"] == "result"
        assert resp["request_id"] == "s-1"
        assert resp["data"]["transport"] == "websocket"
        assert resp["data"]["protocol_version"] == BRIDGE_PROTOCOL_VERSION
        assert resp["data"]["node_count"] == 1
        assert resp["data"]["namespaces"] == ["fake"]


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_unknown_node_returns_error(configured_server):
    host, port = configured_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(_pack({
            "type": "execute",
            "request_id": "e-err",
            "data": {"node_type": "fake.DoesNotExist", "fields": {}},
        }))
        resp = await _recv(ws)
        assert resp["type"] == "error"
        assert resp["request_id"] == "e-err"
        assert "error" in resp["data"]


@pytest.mark.asyncio(loop_scope="function")
async def test_unknown_message_type_returns_error(configured_server):
    host, port = configured_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(_pack({"type": "bogus.type", "request_id": "u-1", "data": {}}))
        resp = await _recv(ws)
        assert resp["type"] == "error"
        assert resp["request_id"] == "u-1"
        assert "Unknown message type" in resp["data"]["error"]


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_success(configured_server):
    host, port = configured_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(_pack({
            "type": "execute",
            "request_id": "e-ok",
            "data": {"node_type": "fake.Echo", "fields": {"text": "hello"}},
        }))
        resp = await _recv(ws)
        assert resp["type"] == "result"
        assert resp["request_id"] == "e-ok"
        assert resp["data"]["outputs"]["output"] == "hello"


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_stream_emits_chunks_then_result(configured_server):
    host, port = configured_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(_pack({
            "type": "execute.stream",
            "request_id": "e-stream",
            "data": {"node_type": "fake.Echo", "fields": {"text": "hi"}},
        }))

        chunks: list[dict] = []
        while True:
            resp = await _recv(ws)
            assert resp["request_id"] == "e-stream"
            if resp["type"] == "chunk":
                chunks.append(resp["data"])
                continue
            # Terminator
            assert resp["type"] == "result"
            # execute.stream result frame is an empty terminator
            assert resp["data"] == {"outputs": {}, "blobs": {}}
            break

        assert len(chunks) >= 1
        assert chunks[0]["outputs"]["output"] == "a"


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_stream_cancel_terminates_stream(cancellable_server):
    """A streaming execute that loops until cancelled stops emitting chunks
    once a ``cancel`` for its request_id arrives and sends a terminator."""
    host, port = cancellable_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(_pack({
            "type": "execute.stream",
            "request_id": "e-cancel",
            "data": {"node_type": "fake.Stream"},
        }))

        # Read at least one chunk to confirm the stream is live.
        first = await _recv(ws)
        assert first["type"] == "chunk"
        assert first["request_id"] == "e-cancel"

        # Request cancellation for the same request_id.
        await ws.send(_pack({"type": "cancel", "request_id": "e-cancel", "data": {}}))

        # Drain until the terminator. The handler loops with a small sleep, so
        # a bounded number of in-flight chunks may still arrive before it
        # observes the cancel — but the stream MUST terminate with a result.
        saw_result = False
        frames_after_first = 0
        while frames_after_first < 100:
            resp = await _recv(ws)
            assert resp["request_id"] == "e-cancel"
            frames_after_first += 1
            if resp["type"] == "result":
                saw_result = True
                break
            assert resp["type"] == "chunk"

        assert saw_result, "stream did not terminate after cancel"

        # After the terminator no further frames are sent for this request.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ws.recv(), timeout=0.2)


@pytest.mark.asyncio(loop_scope="function")
async def test_concurrent_requests_preserve_request_id_correlation(configured_server):
    """Two execute requests dispatched as concurrent tracked tasks each get a
    result frame correlated to their own request_id (no cross-talk).

    Note: both executes are non-streaming (single frame each), so this proves
    request_id correlation under concurrent dispatch rather than write-lock
    frame ordering. See test_execute_stream_cancel for the stateful/streaming
    branch.
    """
    host, port = configured_server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(_pack({
            "type": "execute",
            "request_id": "c-1",
            "data": {"node_type": "fake.Echo", "fields": {"text": "one"}},
        }))
        await ws.send(_pack({
            "type": "execute",
            "request_id": "c-2",
            "data": {"node_type": "fake.Echo", "fields": {"text": "two"}},
        }))

        seen: dict[str, str] = {}
        for _ in range(2):
            resp = await _recv(ws)
            assert resp["type"] == "result"
            seen[resp["request_id"]] = resp["data"]["outputs"]["output"]

        assert seen == {"c-1": "one", "c-2": "two"}
