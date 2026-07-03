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
from typing import Any, Awaitable, Callable

import msgpack

from nodetool.worker.executor import msgpack_default
from nodetool.worker.msgpack_codec import decode_message
from nodetool.worker.protocol import MAX_BRIDGE_FRAME_SIZE, WorkerProtocolServer
from nodetool.worker.stdio_stdout_guard import get_protocol_stdout_buffer


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
        self._protocol_stdout = get_protocol_stdout_buffer()

    async def read_message(self) -> dict[str, Any] | None:
        """Read one length-prefixed msgpack message. Returns None on EOF."""
        loop = asyncio.get_event_loop()
        header = await loop.run_in_executor(None, sys.stdin.buffer.read, 4)
        if len(header) < 4:
            return None
        length = struct.unpack(">I", header)[0]
        if length > MAX_BRIDGE_FRAME_SIZE:
            raise ValueError(f"Incoming bridge frame exceeds max size ({length} > {MAX_BRIDGE_FRAME_SIZE})")
        payload = await loop.run_in_executor(None, sys.stdin.buffer.read, length)
        if len(payload) < length:
            return None
        return decode_message(payload)

    async def send(self, data: bytes) -> None:
        """Write a length-prefixed msgpack payload (thread-safe)."""
        loop = asyncio.get_event_loop()
        if len(data) > MAX_BRIDGE_FRAME_SIZE:
            raise ValueError(f"Outgoing bridge frame exceeds max size ({len(data)} > {MAX_BRIDGE_FRAME_SIZE})")
        message = struct.pack(">I", len(data)) + data
        async with self._write_lock:
            await loop.run_in_executor(None, self._protocol_stdout.write, message)
            await loop.run_in_executor(None, self._protocol_stdout.flush)

    async def send_msg(self, msg: dict[str, Any]) -> None:
        """Encode and send a dict as msgpack."""
        await self.send(msgpack.packb(msg, default=msgpack_default, datetime=True))


class StdioWorkerServer:
    """Drop-in replacement for WorkerServer that uses stdio transport."""

    def __init__(self) -> None:
        self._protocol = WorkerProtocolServer(transport_name="stdio")
        self._transport: StdioTransport | None = None

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
                task = asyncio.create_task(self._protocol.dispatch(msg, self._transport))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        except KeyboardInterrupt:
            pass

        # Wait for in-flight tasks to finish
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


async def run_stdio_worker(namespaces: list[str] | None = None) -> None:
    """Entry point for the stdio worker."""
    from nodetool.worker.stdio_stdout_guard import install_stdio_stdout_guard

    install_stdio_stdout_guard()

    from nodetool.worker.executor import execute_node
    from nodetool.worker.node_loader import load_nodes, resolve_namespaces

    resolved_namespaces = resolve_namespaces(namespaces)

    print("Loading node packages...", file=sys.stderr)
    nodes_metadata = load_nodes(resolved_namespaces)
    print(f"Loaded {len(nodes_metadata)} nodes", file=sys.stderr)

    server = StdioWorkerServer()
    server.set_nodes_metadata(nodes_metadata)
    server.set_namespaces(resolved_namespaces)

    async def handle_execute(
        data: dict,
        cancel_event: asyncio.Event,
        emit_progress: Callable[[dict[str, Any]], Awaitable[None]],
        emit_chunk: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> dict:
        return await execute_node(
            node_type=data["node_type"],
            fields=data.get("fields", {}),
            secrets=data.get("secrets", {}),
            input_blobs=data.get("blobs", {}),
            cancel_event=cancel_event,
            emit_progress=emit_progress,
            emit_chunk=emit_chunk,
        )

    server.set_execute_handler(handle_execute)

    # Signal readiness on stderr (stdout is for protocol only) — BEFORE the
    # warm-import below, so the TS bridge startup timeout (20s) is satisfied
    # immediately. The discover request that Node sends next will sit in the
    # stdin OS buffer until server.run() starts draining it, which is fine.
    print("NODETOOL_STDIO_READY", file=sys.stderr, flush=True)

    # Warm-import heavy ML modules on the MAIN thread before the event loop
    # starts dispatching work. Importing these from a worker thread (via
    # asyncio.to_thread / run_in_executor) hangs indefinitely on Windows for
    # diffusers + nunchaku Flux. Paying ~30s once at startup is far better
    # than a per-job hang. Must happen AFTER readiness signal but BEFORE
    # server.run() so the first execute() finds the module cached.
    if "huggingface" in resolved_namespaces:
        import time as _time

        _t0 = _time.perf_counter()
        print("Warm-importing heavy ML modules...", file=sys.stderr, flush=True)
        try:
            from diffusers.pipelines.flux.pipeline_flux import FluxPipeline  # noqa: F401

            print(
                f"Warm-imported diffusers.FluxPipeline in {_time.perf_counter() - _t0:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        except Exception as _e:  # pragma: no cover — best-effort warm-up
            print(f"Warm-import failed (non-fatal): {_e}", file=sys.stderr, flush=True)

    await server.run()
