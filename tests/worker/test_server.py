import asyncio
import msgpack
import pytest
import pytest_asyncio
import websockets

from nodetool.worker.server import start_server


@pytest_asyncio.fixture(loop_scope="function")
async def server():
    """Start server on random port, yield (host, port), then shut down."""
    host, port, stop_event, task = await start_server(host="127.0.0.1", port=0)
    yield host, port
    stop_event.set()
    await task


@pytest.mark.asyncio(loop_scope="function")
async def test_discover_returns_empty_nodes(server):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        msg = msgpack.packb({
            "type": "discover",
            "request_id": "test-1",
            "data": {},
        })
        await ws.send(msg)
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        response = msgpack.unpackb(raw, raw=False)
        assert response["type"] == "discover"
        assert response["request_id"] == "test-1"
        assert "nodes" in response["data"]
        assert isinstance(response["data"]["nodes"], list)


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_without_handler_returns_error(server):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        msg = msgpack.packb({
            "type": "execute",
            "request_id": "test-2",
            "data": {
                "node_type": "fake.Node",
                "fields": {},
                "secrets": {},
                "blobs": {},
            },
        })
        await ws.send(msg)
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        response = msgpack.unpackb(raw, raw=False)
        assert response["type"] == "error"
        assert response["request_id"] == "test-2"
        assert "error" in response["data"]


@pytest.mark.asyncio(loop_scope="function")
async def test_full_execute_with_echo_node():
    """Integration test: discover + execute a real node through the server."""
    from nodetool.workflows.base_node import BaseNode, NODE_BY_TYPE
    from nodetool.workflows.processing_context import ProcessingContext
    from pydantic import Field
    from nodetool.worker.server import WorkerServer, start_server
    from nodetool.worker.node_loader import load_nodes
    from nodetool.worker.executor import execute_node

    class IntegrationEchoNode(BaseNode):
        text: str = Field(default="")

        @classmethod
        def get_node_type(cls) -> str:
            return "testns.IntegrationEchoNode"

        async def process(self, context: ProcessingContext) -> str:
            return self.text

    NODE_BY_TYPE["testns.IntegrationEchoNode"] = IntegrationEchoNode
    try:
        worker = WorkerServer()
        worker.set_nodes_metadata(load_nodes(namespaces=["testns"]))

        async def handle(data, cancel):
            return await execute_node(
                node_type=data["node_type"],
                fields=data.get("fields", {}),
                secrets=data.get("secrets", {}),
                input_blobs=data.get("blobs", {}),
                cancel_event=cancel,
            )

        worker.set_execute_handler(handle)
        host, port, stop_event, task = await start_server(
            host="127.0.0.1", port=0, worker=worker,
        )

        async with websockets.connect(f"ws://{host}:{port}") as ws:
            # Execute
            msg = msgpack.packb({
                "type": "execute",
                "request_id": "int-1",
                "data": {
                    "node_type": "testns.IntegrationEchoNode",
                    "fields": {"text": "integration test"},
                    "secrets": {},
                    "blobs": {},
                },
            })
            await ws.send(msg)
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            response = msgpack.unpackb(raw, raw=False)
            assert response["type"] == "result"
            assert response["data"]["outputs"]["output"] == "integration test"

        stop_event.set()
        await task
    finally:
        NODE_BY_TYPE.pop("testns.IntegrationEchoNode", None)


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_accepts_large_blob_payload():
    """Large execute payloads should not close the worker websocket."""
    from nodetool.worker.server import WorkerServer, start_server

    worker = WorkerServer()

    async def handle(data, cancel):
        blobs = data.get("blobs", {})
        image_blob = blobs.get("image", b"")
        return {
            "outputs": {"size": len(image_blob)},
            "blobs": {},
        }

    worker.set_execute_handler(handle)
    host, port, stop_event, task = await start_server(
        host="127.0.0.1", port=0, worker=worker,
    )

    try:
        async with websockets.connect(f"ws://{host}:{port}") as ws:
            msg = msgpack.packb({
                "type": "execute",
                "request_id": "large-1",
                "data": {
                    "node_type": "test.LargeBlobNode",
                    "fields": {},
                    "secrets": {},
                    "blobs": {
                        "image": b"x" * (2 * 1024 * 1024),
                    },
                },
            })
            await ws.send(msg)
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            response = msgpack.unpackb(raw, raw=False)
            assert response["type"] == "result"
            assert response["data"]["outputs"]["size"] == 2 * 1024 * 1024
    finally:
        stop_event.set()
        await task
