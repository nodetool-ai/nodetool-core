"""
Stdio-based transport for the Python worker.

Uses the same msgpack message protocol as the WebSocket server but
communicates over stdin/stdout with length-prefixed framing:

  [4 bytes big-endian length][msgpack payload]

This avoids WebSocket connection instability issues and is simpler
to manage from a parent process.
"""

import asyncio
import struct
import sys
import traceback
from typing import Any, Callable, Awaitable

import msgpack

from nodetool.worker import BRIDGE_PROTOCOL_VERSION


class StdioTransport:
    """Async reader/writer for length-prefixed msgpack over stdin/stdout.

    Uses run_in_executor for blocking I/O so it works on all platforms,
    including Windows where asyncio connect_read_pipe/connect_write_pipe
    fail with OSError (WinError 6) on the ProactorEventLoop.

    Writes are serialized with a lock so concurrent tasks cannot
    interleave their length-prefix + payload.
    """

    def __init__(self) -> None:
        self._write_lock = asyncio.Lock()

    async def read_message(self) -> dict[str, Any] | None:
        """Read one length-prefixed msgpack message. Returns None on EOF."""
        loop = asyncio.get_event_loop()
        header = await loop.run_in_executor(None, sys.stdin.buffer.read, 4)
        if len(header) < 4:
            return None
        length = struct.unpack(">I", header)[0]
        payload = await loop.run_in_executor(None, sys.stdin.buffer.read, length)
        if len(payload) < length:
            return None
        return msgpack.unpackb(payload, raw=False)

    async def send(self, data: bytes) -> None:
        """Write a length-prefixed msgpack payload (thread-safe)."""
        loop = asyncio.get_event_loop()
        message = struct.pack(">I", len(data)) + data
        async with self._write_lock:
            await loop.run_in_executor(None, sys.stdout.buffer.write, message)
            await loop.run_in_executor(None, sys.stdout.buffer.flush)

    async def send_msg(self, msg: dict[str, Any]) -> None:
        """Encode and send a dict as msgpack."""
        await self.send(msgpack.packb(msg))


class StdioWorkerServer:
    """Drop-in replacement for WorkerServer that uses stdio transport."""

    def __init__(self) -> None:
        self._nodes_metadata: list[dict] = []
        self._cancel_flags: dict[str, asyncio.Event] = {}
        self._execute_handler: Callable | None = None
        self._transport: StdioTransport | None = None

    def set_nodes_metadata(self, metadata: list[dict]) -> None:
        self._nodes_metadata = metadata

    def set_execute_handler(
        self,
        handler: Callable[[dict, asyncio.Event], Awaitable[dict]],
    ) -> None:
        self._execute_handler = handler

    async def run(self) -> None:
        """Main loop: read from stdin, dispatch, write to stdout."""
        self._transport = StdioTransport()
        self._tasks: set[asyncio.Task] = set()

        try:
            while True:
                try:
                    msg = await self._transport.read_message()
                except (asyncio.IncompleteReadError, ConnectionError):
                    break
                if msg is None:
                    break
                task = asyncio.create_task(self._dispatch(msg))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        except KeyboardInterrupt:
            pass

        # Wait for in-flight tasks to finish
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        transport = self._transport
        assert transport is not None

        msg_type = msg.get("type")
        request_id = msg.get("request_id")

        if msg_type == "discover":
            await transport.send_msg({
                "type": "discover",
                "request_id": request_id,
                "data": {
                    "nodes": self._nodes_metadata,
                    "protocol_version": BRIDGE_PROTOCOL_VERSION,
                },
            })

        elif msg_type == "execute":
            cancel_event = asyncio.Event()
            if request_id:
                self._cancel_flags[request_id] = cancel_event
            try:
                if self._execute_handler is None:
                    raise RuntimeError("No execute handler registered")
                result = await self._execute_handler(msg["data"], cancel_event)
                await transport.send_msg({
                    "type": "result",
                    "request_id": request_id,
                    "data": result,
                })
            except Exception as e:
                await transport.send_msg({
                    "type": "error",
                    "request_id": request_id,
                    "data": {
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                })
            finally:
                self._cancel_flags.pop(request_id, None)

        elif msg_type == "cancel":
            cancel_event = self._cancel_flags.get(request_id)
            if cancel_event:
                cancel_event.set()

        elif msg_type and msg_type.startswith("provider."):
            from nodetool.worker.provider_handler import handle_provider_message_stdio
            await handle_provider_message_stdio(
                msg_type=msg_type,
                request_id=request_id,
                data=msg.get("data", {}),
                transport=transport,
                cancel_flags=self._cancel_flags,
            )


async def run_stdio_worker(namespaces: list[str] | None = None) -> None:
    """Entry point for the stdio worker."""
    from nodetool.worker.node_loader import load_nodes
    from nodetool.worker.executor import execute_node

    print("Loading node packages...", file=sys.stderr)
    nodes_metadata = load_nodes(namespaces=namespaces)
    print(f"Loaded {len(nodes_metadata)} nodes", file=sys.stderr)

    server = StdioWorkerServer()
    server.set_nodes_metadata(nodes_metadata)

    async def handle_execute(data: dict, cancel_event: asyncio.Event) -> dict:
        return await execute_node(
            node_type=data["node_type"],
            fields=data.get("fields", {}),
            secrets=data.get("secrets", {}),
            input_blobs=data.get("blobs", {}),
            cancel_event=cancel_event,
        )

    server.set_execute_handler(handle_execute)

    # Signal readiness on stderr (stdout is for protocol only)
    print("NODETOOL_STDIO_READY", file=sys.stderr, flush=True)

    await server.run()
