"""
Stdio-based transport for the Python worker.

Uses the same msgpack message protocol as the WebSocket server but
communicates over stdin/stdout with length-prefixed framing:

  [4 bytes big-endian length][msgpack payload]

This avoids WebSocket connection instability issues and is simpler
to manage from a parent process.

Verbs handled by the dispatcher:

- ``discover`` / ``execute`` / ``cancel`` — original per-call node
  execution protocol.
- ``provider.*`` — provider proxy verbs (chat, embed, etc.).
- ``start_session`` / ``update_parameter`` / ``push_input_frame`` /
  ``stop_session`` — realtime session protocol (PLAN-REALTIME.md item 6c).
  Server pushes ``realtime_output_frame`` events back to the client as the
  warm node yields outputs.
"""

import asyncio
import struct
import sys
import traceback
from typing import Any, Callable, Awaitable

import msgpack

from nodetool.worker.realtime_session import (
    RealtimeNodeInstance,
    RealtimeSessionError,
)
from nodetool.workflows.realtime import RealtimeSessionInfo


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
        # Realtime session state (PLAN-REALTIME.md item 6c). Keyed by the
        # session_id chosen by the TS-side RealtimeSessionManager and passed
        # in by the bridge; the worker never invents session ids.
        self._realtime_sessions: dict[str, RealtimeNodeInstance] = {}

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

        elif msg_type == "start_session":
            await self._handle_start_session(transport, request_id, msg.get("data", {}))

        elif msg_type == "update_parameter":
            await self._handle_update_parameter(
                transport, request_id, msg.get("data", {})
            )

        elif msg_type == "push_input_frame":
            await self._handle_push_input_frame(
                transport, request_id, msg.get("data", {})
            )

        elif msg_type == "stop_session":
            await self._handle_stop_session(
                transport, request_id, msg.get("data", {})
            )

    # ── Realtime session verbs (PLAN-REALTIME.md item 6c) ──────────────

    async def _handle_start_session(
        self,
        transport: "StdioTransport",
        request_id: Any,
        data: dict[str, Any],
    ) -> None:
        """Spin up a warm node for a realtime session.

        Awaits ``pre_process`` and ``on_session_start`` before responding so
        the bridge only flips the session to ``running`` once the node is
        actually ready to receive frames.
        """
        session_id = str(data.get("session_id") or "")
        if not session_id:
            await self._send_realtime_error(
                transport, request_id, session_id, "Missing session_id"
            )
            return
        if session_id in self._realtime_sessions:
            await self._send_realtime_error(
                transport,
                request_id,
                session_id,
                f"Session {session_id} is already running",
            )
            return

        node_type = str(data.get("node_type") or "")
        if not node_type:
            await self._send_realtime_error(
                transport, request_id, session_id, "Missing node_type"
            )
            return

        session_payload = data.get("session_info") or {
            "session_id": session_id,
            "workflow_id": data.get("workflow_id"),
            "transport": data.get("transport", "websocket"),
            "parameters": data.get("parameters") or {},
            "media_tracks": data.get("media_tracks") or [],
        }
        # The bridge may omit session_id from session_info; force it from the
        # outer envelope so the two views can never diverge.
        session_payload["session_id"] = session_id

        try:
            session = RealtimeSessionInfo.from_dict(session_payload)
        except (KeyError, TypeError, ValueError) as exc:
            await self._send_realtime_error(
                transport,
                request_id,
                session_id,
                f"Invalid session_info payload: {exc}",
            )
            return

        async def _emit_frame(handle: str, value: Any) -> None:
            await transport.send_msg({
                "type": "realtime_output_frame",
                "session_id": session_id,
                "handle": handle,
                "data": value,
            })

        try:
            instance = await RealtimeNodeInstance.start(
                session=session,
                node_type=node_type,
                fields=data.get("fields") or {},
                secrets=data.get("secrets") or {},
                emit_frame=_emit_frame,
            )
        except RealtimeSessionError as exc:
            await self._send_realtime_error(
                transport, request_id, session_id, str(exc)
            )
            return
        except Exception as exc:
            await self._send_realtime_error(
                transport,
                request_id,
                session_id,
                f"Unexpected error starting session: {exc}",
                trace=traceback.format_exc(),
            )
            return

        self._realtime_sessions[session_id] = instance
        await transport.send_msg({
            "type": "result",
            "request_id": request_id,
            "data": {
                "session_id": session_id,
                "status": "running",
            },
        })

    async def _handle_update_parameter(
        self,
        transport: "StdioTransport",
        request_id: Any,
        data: dict[str, Any],
    ) -> None:
        session_id = str(data.get("session_id") or "")
        instance = self._realtime_sessions.get(session_id)
        if instance is None:
            await self._send_realtime_error(
                transport, request_id, session_id, "Unknown session_id"
            )
            return

        name = str(data.get("name") or "")
        if not name:
            await self._send_realtime_error(
                transport, request_id, session_id, "Missing parameter name"
            )
            return

        routed = instance.update_parameter(name, data.get("value"))
        await transport.send_msg({
            "type": "result",
            "request_id": request_id,
            "data": {
                "session_id": session_id,
                "ok": True,
                "routed": routed,
            },
        })

    async def _handle_push_input_frame(
        self,
        transport: "StdioTransport",
        request_id: Any,
        data: dict[str, Any],
    ) -> None:
        session_id = str(data.get("session_id") or "")
        instance = self._realtime_sessions.get(session_id)
        if instance is None:
            await self._send_realtime_error(
                transport, request_id, session_id, "Unknown session_id"
            )
            return

        handle = str(data.get("handle") or "")
        if not handle:
            await self._send_realtime_error(
                transport, request_id, session_id, "Missing handle"
            )
            return

        dropped = instance.push_input_frame(handle, data.get("data"))
        await transport.send_msg({
            "type": "result",
            "request_id": request_id,
            "data": {
                "session_id": session_id,
                "ok": True,
                "dropped_count": dropped,
            },
        })

    async def _handle_stop_session(
        self,
        transport: "StdioTransport",
        request_id: Any,
        data: dict[str, Any],
    ) -> None:
        session_id = str(data.get("session_id") or "")
        instance = self._realtime_sessions.pop(session_id, None)
        if instance is None:
            await self._send_realtime_error(
                transport, request_id, session_id, "Unknown session_id"
            )
            return

        try:
            await instance.stop()
        except Exception as exc:
            await self._send_realtime_error(
                transport,
                request_id,
                session_id,
                f"Error stopping session: {exc}",
                trace=traceback.format_exc(),
            )
            return

        runner_error = instance.error
        await transport.send_msg({
            "type": "result",
            "request_id": request_id,
            "data": {
                "session_id": session_id,
                "ok": runner_error is None,
                "error": None if runner_error is None else str(runner_error),
            },
        })

    async def _send_realtime_error(
        self,
        transport: "StdioTransport",
        request_id: Any,
        session_id: str,
        message: str,
        trace: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "error": message,
        }
        if trace is not None:
            payload["traceback"] = trace
        await transport.send_msg({
            "type": "error",
            "request_id": request_id,
            "data": payload,
        })


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
