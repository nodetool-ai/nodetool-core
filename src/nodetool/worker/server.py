import asyncio
from typing import Any, Awaitable, Callable, cast

import msgpack
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed

from nodetool.worker.protocol import WorkerProtocolServer


class WebSocketTransport:
    """WorkerTransport implementation over a websockets ServerConnection.

    Writes are serialized with a lock so concurrent dispatch tasks cannot
    interleave their frames (parity with StdioTransport).
    """

    def __init__(self, websocket: ServerConnection) -> None:
        self._ws = websocket
        self._write_lock = asyncio.Lock()

    async def send_msg(self, msg: dict[str, Any]) -> None:
        """Encode and send a dict as msgpack (thread-safe)."""
        data = cast("bytes", msgpack.packb(msg))
        async with self._write_lock:
            await self._ws.send(data)


class WorkerServer:
    """WebSocket worker server that delegates to the shared protocol server."""

    def __init__(self) -> None:
        self._protocol = WorkerProtocolServer(transport_name="websocket")

    def set_nodes_metadata(self, metadata: list[dict]) -> None:
        self._protocol.set_nodes_metadata(metadata)

    def set_load_errors(self, errors: list[dict[str, Any]]) -> None:
        self._protocol.set_load_errors(errors)

    def set_namespaces(self, namespaces: list[str]) -> None:
        self._protocol.set_namespaces(namespaces)

    def set_execute_handler(
        self,
        handler: Callable[
            [
                dict,
                asyncio.Event,
                Callable[[dict[str, Any]], Awaitable[None]],
                Callable[[dict[str, Any]], Awaitable[None]] | None,
            ],
            Awaitable[dict],
        ],
    ) -> None:
        self._protocol.set_execute_handler(handler)

    async def handle_connection(self, websocket: ServerConnection) -> None:
        transport = WebSocketTransport(websocket)
        tasks: set[asyncio.Task] = set()

        try:
            async for raw_message in websocket:
                msg = msgpack.unpackb(raw_message, raw=False)
                task = asyncio.create_task(self._protocol.dispatch(msg, transport))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except ConnectionClosed:
            pass
        finally:
            # Wait for in-flight tasks to finish
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


async def start_server(
    host: str = "127.0.0.1",
    port: int = 0,
    worker: WorkerServer | None = None,
) -> tuple[str, int, asyncio.Event, asyncio.Task]:
    """Start the WebSocket server. Returns (host, port, stop_event, task)."""
    if worker is None:
        worker = WorkerServer()

    stop_event = asyncio.Event()

    server = await ws_serve(
        worker.handle_connection,
        host,
        port,
        # Workflow asset blobs routinely exceed the websockets default 1 MiB limit.
        # Let the bridge transport large execute payloads instead of closing.
        max_size=None,
    )

    # Get the actual port (important when port=0)
    sockets = list(server.sockets)
    actual_port = sockets[0].getsockname()[1]

    async def run():
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
        server.close()
        await server.wait_closed()

    task = asyncio.create_task(run())
    return host, actual_port, stop_event, task
