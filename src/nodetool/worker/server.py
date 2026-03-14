import asyncio
import traceback
from typing import Any, Callable, Awaitable

import msgpack
from websockets.asyncio.server import serve as ws_serve, ServerConnection


class WorkerServer:
    def __init__(self):
        self._nodes_metadata: list[dict] = []
        self._cancel_flags: dict[str, asyncio.Event] = {}
        self._execute_handler: Callable | None = None

    def set_nodes_metadata(self, metadata: list[dict]):
        self._nodes_metadata = metadata

    def set_execute_handler(
        self,
        handler: Callable[
            [dict, asyncio.Event],
            Awaitable[dict],
        ],
    ):
        self._execute_handler = handler

    async def handle_connection(self, websocket: ServerConnection):
        async for raw_message in websocket:
            msg = msgpack.unpackb(raw_message, raw=False)
            msg_type = msg.get("type")
            request_id = msg.get("request_id")

            if msg_type == "discover":
                response = msgpack.packb({
                    "type": "discover",
                    "request_id": request_id,
                    "data": {"nodes": self._nodes_metadata},
                })
                await websocket.send(response)

            elif msg_type == "execute":
                cancel_event = asyncio.Event()
                if request_id:
                    self._cancel_flags[request_id] = cancel_event
                try:
                    if self._execute_handler is None:
                        raise RuntimeError("No execute handler registered")
                    result = await self._execute_handler(msg["data"], cancel_event)
                    response = msgpack.packb({
                        "type": "result",
                        "request_id": request_id,
                        "data": result,
                    })
                    await websocket.send(response)
                except Exception as e:
                    response = msgpack.packb({
                        "type": "error",
                        "request_id": request_id,
                        "data": {
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    })
                    await websocket.send(response)
                finally:
                    self._cancel_flags.pop(request_id, None)

            elif msg_type == "cancel":
                cancel_event = self._cancel_flags.get(request_id)
                if cancel_event:
                    cancel_event.set()


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
