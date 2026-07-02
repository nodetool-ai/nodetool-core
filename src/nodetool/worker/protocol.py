from __future__ import annotations

import asyncio
import os
import traceback
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from nodetool.worker import BRIDGE_PROTOCOL_VERSION

MAX_BRIDGE_FRAME_SIZE = int(os.environ.get("NODETOOL_BRIDGE_MAX_FRAME_SIZE", str(256 * 1024 * 1024)))


class WorkerTransport(Protocol):
    async def send_msg(self, msg: dict[str, Any]) -> None: ...


@dataclass
class WorkerStatus:
    protocol_version: int
    node_count: int
    provider_count: int
    namespaces: list[str]
    load_errors: list[dict[str, Any]]
    transport: str
    max_frame_size: int
    comfy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "node_count": self.node_count,
            "provider_count": self.provider_count,
            "namespaces": self.namespaces,
            "load_errors": self.load_errors,
            "transport": self.transport,
            "max_frame_size": self.max_frame_size,
            "comfy": self.comfy,
        }


class WorkerProtocolServer:
    def __init__(self, *, transport_name: str):
        self._nodes_metadata: list[dict[str, Any]] = []
        self._cancel_flags: dict[str, asyncio.Event] = {}
        self._execute_handler: Callable[[
            dict[str, Any],
            asyncio.Event,
            Callable[[dict[str, Any]], Awaitable[None]],
            Callable[[dict[str, Any]], Awaitable[None]] | None,
        ], Awaitable[dict[str, Any]]] | None = None
        self._load_errors: list[dict[str, Any]] = []
        self._namespaces: list[str] = []
        self._transport_name = transport_name

    def set_nodes_metadata(self, metadata: list[dict[str, Any]]) -> None:
        self._nodes_metadata = metadata

    def set_load_errors(self, errors: list[dict[str, Any]]) -> None:
        self._load_errors = errors

    def set_namespaces(self, namespaces: list[str]) -> None:
        self._namespaces = namespaces

    def set_execute_handler(
        self,
        handler: Callable[[
            dict[str, Any],
            asyncio.Event,
            Callable[[dict[str, Any]], Awaitable[None]],
            Callable[[dict[str, Any]], Awaitable[None]] | None,
        ], Awaitable[dict[str, Any]]],
    ) -> None:
        self._execute_handler = handler

    async def dispatch(self, msg: dict[str, Any], transport: WorkerTransport) -> None:
        msg_type = msg.get("type")
        request_id = msg.get("request_id")

        if msg_type == "discover":
            await transport.send_msg({
                "type": "discover",
                "request_id": request_id,
                "data": {
                    "protocol_version": BRIDGE_PROTOCOL_VERSION,
                    "nodes": self._nodes_metadata,
                    "load_errors": self._load_errors,
                },
            })
            return

        if msg_type == "worker.status":
            from nodetool.worker.comfy_handler import get_comfy_info
            from nodetool.worker.provider_handler import get_available_providers

            status = WorkerStatus(
                protocol_version=BRIDGE_PROTOCOL_VERSION,
                node_count=len(self._nodes_metadata),
                provider_count=len(get_available_providers()),
                namespaces=list(self._namespaces),
                load_errors=list(self._load_errors),
                transport=self._transport_name,
                max_frame_size=MAX_BRIDGE_FRAME_SIZE,
                comfy=get_comfy_info(),
            )
            await transport.send_msg({
                "type": "result",
                "request_id": request_id,
                "data": status.to_dict(),
            })
            return

        if msg_type in ("execute", "execute.stream"):
            cancel_event = asyncio.Event()
            if request_id:
                self._cancel_flags[request_id] = cancel_event
            try:
                if self._execute_handler is None:
                    raise RuntimeError("No execute handler registered")

                async def emit_progress(progress: dict[str, Any]) -> None:
                    await transport.send_msg({
                        "type": "progress",
                        "request_id": request_id,
                        "data": progress,
                    })

                emit_chunk: Callable[[dict[str, Any]], Awaitable[None]] | None = None
                if msg_type == "execute.stream":
                    async def _emit_chunk(chunk: dict[str, Any]) -> None:
                        await transport.send_msg({
                            "type": "chunk",
                            "request_id": request_id,
                            "data": chunk,
                        })
                    emit_chunk = _emit_chunk

                result = await self._execute_handler(
                    msg["data"], cancel_event, emit_progress, emit_chunk
                )
                # For execute.stream, the per-chunk frames already carried the
                # outputs/blobs. The result frame is just a terminator so the
                # bridge can close the stream — keep its data empty to match
                # the pre-protocol-refactor wire contract.
                result_data = (
                    {"outputs": {}, "blobs": {}}
                    if msg_type == "execute.stream"
                    else result
                )
                await transport.send_msg({
                    "type": "result",
                    "request_id": request_id,
                    "data": result_data,
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
            return

        if msg_type == "cancel":
            cancel_event = self._cancel_flags.get(request_id)
            if cancel_event:
                cancel_event.set()
            return

        if msg_type and str(msg_type).startswith("provider."):
            from nodetool.worker.provider_handler import handle_provider_message

            await handle_provider_message(
                msg_type=str(msg_type),
                request_id=request_id,
                data=msg.get("data", {}),
                transport=transport,
                cancel_flags=self._cancel_flags,
            )
            return

        if msg_type and str(msg_type).startswith("models."):
            from nodetool.worker.model_handler import handle_models_message

            await handle_models_message(
                msg_type=str(msg_type),
                request_id=request_id,
                data=msg.get("data", {}),
                transport=transport,
                cancel_flags=self._cancel_flags,
            )
            return

        if msg_type and str(msg_type).startswith("comfy."):
            from nodetool.worker.comfy_handler import handle_comfy_message

            await handle_comfy_message(
                msg_type=str(msg_type),
                request_id=request_id,
                data=msg.get("data", {}),
                transport=transport,
                cancel_flags=self._cancel_flags,
            )
            return

        await transport.send_msg({
            "type": "error",
            "request_id": request_id,
            "data": {"error": f"Unknown message type: {msg_type}"},
        })
