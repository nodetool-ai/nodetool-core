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
from typing import Any, Awaitable, Callable

import msgpack


class StdioTransport:
    """Async reader/writer for length-prefixed msgpack over stdin/stdout.

    Writes are serialized with a lock so concurrent tasks cannot
    interleave their length-prefix + payload.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        self._reader = reader
        self._writer = writer
        self._write_lock = asyncio.Lock()

    async def read_message(self) -> dict[str, Any] | None:
        """Read one length-prefixed msgpack message. Returns None on EOF."""
        header = await self._reader.readexactly(4)
        length = struct.unpack(">I", header)[0]
        payload = await self._reader.readexactly(length)
        return msgpack.unpackb(payload, raw=False)

    async def send(self, data: bytes) -> None:
        """Write a length-prefixed msgpack payload (thread-safe)."""
        header = struct.pack(">I", len(data))
        async with self._write_lock:
            self._writer.write(header + data)
            await self._writer.drain()

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
        loop = asyncio.get_event_loop()

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

        w_transport, w_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout.buffer
        )
        writer = asyncio.StreamWriter(w_transport, w_protocol, reader, loop)

        self._transport = StdioTransport(reader, writer)
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
                "data": {"nodes": self._nodes_metadata},
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
    from nodetool.worker.executor import execute_node
    from nodetool.worker.node_loader import load_nodes

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
