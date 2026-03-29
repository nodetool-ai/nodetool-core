"""Tests for the provider bridge handler."""

import asyncio
import msgpack
import pytest
import pytest_asyncio
import websockets

from nodetool.worker.server import WorkerServer, start_server


@pytest_asyncio.fixture(loop_scope="function")
async def server():
    """Start server on random port, yield (host, port), then shut down."""
    worker = WorkerServer()
    host, port, stop_event, task = await start_server(
        host="127.0.0.1", port=0, worker=worker,
    )
    yield host, port
    stop_event.set()
    await task


@pytest.mark.asyncio(loop_scope="function")
async def test_provider_list(server):
    """provider.list returns available providers (may be empty in test env)."""
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "provider.list",
            "request_id": "pl-1",
            "data": {},
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "result"
        assert resp["request_id"] == "pl-1"
        assert "providers" in resp["data"]
        assert isinstance(resp["data"]["providers"], list)


@pytest.mark.asyncio(loop_scope="function")
async def test_provider_unknown_type(server):
    """Unknown provider.X message returns error."""
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "provider.nonexistent",
            "request_id": "pu-1",
            "data": {},
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "error"
        assert "Unknown provider message type" in resp["data"]["error"]


@pytest.mark.asyncio(loop_scope="function")
async def test_provider_models_invalid_provider(server):
    """provider.models with unknown provider returns error."""
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "provider.models",
            "request_id": "pm-1",
            "data": {"provider": "nonexistent_provider", "model_type": "language"},
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "error"
        assert resp["request_id"] == "pm-1"
