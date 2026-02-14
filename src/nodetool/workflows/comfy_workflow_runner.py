"""
Backend-native ComfyUI workflow execution.

Submits a Comfy prompt, listens on the Comfy websocket for execution events,
and translates them into the standard NodeTool ``ProcessingMessage`` stream
(``job_update``, ``node_update``, ``node_progress``, ``output_update``, etc.).

This module is self-contained: it uses the synchronous ``comfy_api`` helpers
for HTTP interactions and the ``websocket-client`` library for the WS
subscription, running the blocking WS loop in a thread so the rest of the
async pipeline stays responsive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator

import websocket

from nodetool.types.job import JobUpdate
from nodetool.workflows.comfy_graph_converter import graph_to_prompt, has_comfy_nodes
from nodetool.workflows.types import (
    NodeProgress,
    NodeUpdate,
    OutputUpdate,
    ProcessingMessage,
)

if TYPE_CHECKING:
    from nodetool.types.api_graph import Graph

log = logging.getLogger(__name__)

COMFY_HOST: str = os.environ.get("COMFYUI_ADDR", "127.0.0.1:8188")


# ---------------------------------------------------------------------------
# Event translator
# ---------------------------------------------------------------------------

class ComfyEventTranslator:
    """Stateful translator that converts raw Comfy WS messages into
    ``ProcessingMessage`` instances.

    It keeps track of which nodes are currently running so that on a terminal
    event (``execution_success`` or ``executing`` with ``node=null``) any
    lingering nodes can be force-completed.
    """

    def __init__(self, prompt_id: str, workflow_id: str | None = None):
        self.prompt_id = prompt_id
        self.workflow_id = workflow_id
        self._running_nodes: set[str] = set()
        self._current_node: str | None = None
        self._completed = False

    # -- public API --

    def translate(self, message: dict[str, Any]) -> list[ProcessingMessage]:
        """Translate a single Comfy WS JSON message.

        Returns a (possibly empty) list of ``ProcessingMessage`` objects.
        """
        msg_type = message.get("type")
        data = message.get("data", {})

        # Filter by prompt_id when available
        msg_prompt_id = data.get("prompt_id")
        if msg_prompt_id is not None and msg_prompt_id != self.prompt_id:
            return []

        handler = getattr(self, f"_handle_{msg_type}", None)
        if handler is not None:
            msgs = handler(data)
            log.debug(
                "Comfy event '%s' -> %d NodeTool message(s) [prompt=%s]",
                msg_type,
                len(msgs),
                self.prompt_id,
            )
            return msgs

        log.debug("Ignoring unhandled Comfy event type '%s'", msg_type)
        return []

    @property
    def is_completed(self) -> bool:
        return self._completed

    # -- handlers --

    def _handle_execution_start(self, data: dict[str, Any]) -> list[ProcessingMessage]:
        return [
            JobUpdate(
                job_id=self.prompt_id,
                workflow_id=self.workflow_id,
                status="running",
            )
        ]

    def _handle_executing(self, data: dict[str, Any]) -> list[ProcessingMessage]:
        node_id = data.get("node")
        if node_id is None:
            # Terminal pattern: execution finished
            return self._finalize()
        self._running_nodes.add(node_id)
        self._current_node = node_id
        return [
            NodeUpdate(
                node_id=node_id,
                node_name=node_id,
                node_type="comfy",
                status="running",
                workflow_id=self.workflow_id,
            )
        ]

    def _handle_progress(self, data: dict[str, Any]) -> list[ProcessingMessage]:
        node_id = data.get("node", self._current_node) or ""
        return [
            NodeProgress(
                node_id=node_id,
                progress=data.get("value", 0),
                total=data.get("max", 0),
                workflow_id=self.workflow_id,
            )
        ]

    def _handle_executed(self, data: dict[str, Any]) -> list[ProcessingMessage]:
        node_id = data.get("node", "")
        output = data.get("output", {})
        msgs: list[ProcessingMessage] = []

        # Emit OutputUpdate for each output key
        for key, value in output.items():
            msgs.append(
                OutputUpdate(
                    node_id=node_id,
                    node_name=node_id,
                    output_name=key,
                    value=value,
                    output_type=type(value).__name__,
                    workflow_id=self.workflow_id,
                )
            )

        msgs.append(
            NodeUpdate(
                node_id=node_id,
                node_name=node_id,
                node_type="comfy",
                status="completed",
                workflow_id=self.workflow_id,
            )
        )
        self._running_nodes.discard(node_id)
        return msgs

    def _handle_execution_cached(self, data: dict[str, Any]) -> list[ProcessingMessage]:
        cached_nodes = data.get("nodes", [])
        msgs: list[ProcessingMessage] = []
        for node_id in cached_nodes:
            msgs.append(
                NodeUpdate(
                    node_id=node_id,
                    node_name=node_id,
                    node_type="comfy",
                    status="completed",
                    result={"cached": True},
                    workflow_id=self.workflow_id,
                )
            )
            self._running_nodes.discard(node_id)
        return msgs

    def _handle_execution_success(self, data: dict[str, Any]) -> list[ProcessingMessage]:
        return self._finalize()

    def _handle_execution_error(self, data: dict[str, Any]) -> list[ProcessingMessage]:
        node_id = data.get("node_id", "")
        exception_message = data.get("exception_message", "Unknown ComfyUI error")
        traceback_lines = data.get("traceback", [])
        traceback_str = "\n".join(traceback_lines) if isinstance(traceback_lines, list) else str(traceback_lines)

        msgs: list[ProcessingMessage] = []
        if node_id:
            msgs.append(
                NodeUpdate(
                    node_id=node_id,
                    node_name=node_id,
                    node_type="comfy",
                    status="error",
                    error=exception_message,
                    workflow_id=self.workflow_id,
                )
            )
        msgs.append(
            JobUpdate(
                job_id=self.prompt_id,
                workflow_id=self.workflow_id,
                status="failed",
                error=exception_message,
                traceback=traceback_str if traceback_str else None,
            )
        )
        self._completed = True
        return msgs

    def _handle_execution_interrupted(self, data: dict[str, Any]) -> list[ProcessingMessage]:
        self._completed = True
        return [
            JobUpdate(
                job_id=self.prompt_id,
                workflow_id=self.workflow_id,
                status="cancelled",
            )
        ]

    # -- helpers --

    def _finalize(self) -> list[ProcessingMessage]:
        """Force-complete lingering running nodes and emit job completion."""
        if self._completed:
            return []
        self._completed = True

        msgs: list[ProcessingMessage] = []
        for node_id in list(self._running_nodes):
            log.info(
                "Finalizing lingering running node %s [prompt=%s]",
                node_id,
                self.prompt_id,
            )
            msgs.append(
                NodeUpdate(
                    node_id=node_id,
                    node_name=node_id,
                    node_type="comfy",
                    status="completed",
                    workflow_id=self.workflow_id,
                )
            )
        self._running_nodes.clear()

        msgs.append(
            JobUpdate(
                job_id=self.prompt_id,
                workflow_id=self.workflow_id,
                status="completed",
            )
        )
        return msgs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _submit_prompt(
    prompt: dict[str, Any],
    client_id: str,
    comfy_host: str | None = None,
) -> str:
    """Submit a prompt to ComfyUI and return the ``prompt_id``."""
    from nodetool.providers.comfy_api import queue_workflow

    host = comfy_host or COMFY_HOST
    # queue_workflow uses the module-level COMFY_HOST; we temporarily patch it
    # only if a different host was requested.
    import nodetool.providers.comfy_api as _api

    original_host = _api.COMFY_HOST
    try:
        _api.COMFY_HOST = host
        result = queue_workflow(prompt, client_id)
    finally:
        _api.COMFY_HOST = original_host

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise ValueError(f"Missing prompt_id in ComfyUI queue response: {result}")
    return prompt_id


def _cancel_prompt(comfy_host: str | None = None) -> None:
    """Send an interrupt request to ComfyUI."""
    import requests

    host = comfy_host or COMFY_HOST
    try:
        requests.post(f"http://{host}/interrupt", timeout=10)
        log.info("Sent interrupt to ComfyUI at %s", host)
    except Exception as exc:
        log.warning("Failed to send interrupt to ComfyUI: %s", exc)


async def run_comfy_workflow(
    graph: Graph,
    workflow_id: str | None = None,
    job_id: str | None = None,
    comfy_host: str | None = None,
) -> AsyncGenerator[ProcessingMessage, None]:
    """Execute a ComfyUI workflow and yield ``ProcessingMessage`` events.

    This is the top-level entry point called by the routing logic in
    ``run_workflow.py`` when the workflow is detected as a Comfy workflow.

    Steps:
    1. Convert the NodeTool graph to a Comfy prompt.
    2. Open a WS connection to Comfy.
    3. Submit the prompt via HTTP.
    4. Listen for WS events and translate them to NodeTool messages.
    5. On terminal events, clean up and return.
    """
    host = comfy_host or COMFY_HOST
    client_id = uuid.uuid4().hex

    # 1. Convert
    prompt = graph_to_prompt(graph)
    if not prompt:
        yield JobUpdate(
            job_id=job_id or "",
            workflow_id=workflow_id,
            status="failed",
            error="No comfy nodes found in graph",
        )
        return

    log.info(
        "Comfy run: workflow_id=%s, job_id=%s, comfy_host=%s, prompt_nodes=%d",
        workflow_id,
        job_id,
        host,
        len(prompt),
    )

    # 2. Connect WS
    ws_url = f"ws://{host}/ws?clientId={client_id}"
    ws: websocket.WebSocket | None = None
    try:
        ws = websocket.WebSocket()
        ws.settimeout(300)  # 5 min timeout for long jobs
        ws.connect(ws_url, timeout=10)
        log.info("Comfy WS connected: %s [job=%s]", ws_url, job_id)
    except Exception as exc:
        log.error("Failed to connect to Comfy WS at %s: %s", ws_url, exc)
        yield JobUpdate(
            job_id=job_id or "",
            workflow_id=workflow_id,
            status="failed",
            error=f"Cannot connect to ComfyUI websocket at {host}: {exc}",
        )
        return

    prompt_id: str | None = None
    try:
        # 3. Submit prompt
        try:
            prompt_id = _submit_prompt(prompt, client_id, host)
            log.info(
                "Comfy prompt submitted: prompt_id=%s, job_id=%s, workflow_id=%s",
                prompt_id,
                job_id,
                workflow_id,
            )
        except Exception as exc:
            log.error("Failed to submit Comfy prompt: %s", exc)
            yield JobUpdate(
                job_id=job_id or "",
                workflow_id=workflow_id,
                status="failed",
                error=f"Failed to submit ComfyUI prompt: {exc}",
            )
            return

        # 4. Listen and translate
        translator = ComfyEventTranslator(
            prompt_id=prompt_id,
            workflow_id=workflow_id,
        )

        # Run the blocking WS recv loop in a thread
        loop = asyncio.get_event_loop()
        message_queue: asyncio.Queue[list[ProcessingMessage] | None] = asyncio.Queue()

        def _ws_listener():
            """Blocking listener that puts translated messages onto the queue."""
            try:
                while not translator.is_completed:
                    try:
                        raw = ws.recv()  # type: ignore[union-attr]
                    except websocket.WebSocketTimeoutException:
                        continue
                    except websocket.WebSocketConnectionClosedException:
                        log.warning("Comfy WS closed unexpectedly [prompt=%s]", prompt_id)
                        break
                    except Exception as exc:
                        log.error("Comfy WS error: %s", exc)
                        break

                    if isinstance(raw, bytes):
                        continue  # skip binary frames (preview images)

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning("Invalid JSON from Comfy WS")
                        continue

                    translated = translator.translate(msg)
                    if translated:
                        loop.call_soon_threadsafe(message_queue.put_nowait, translated)
            finally:
                loop.call_soon_threadsafe(message_queue.put_nowait, None)

        listener_future = loop.run_in_executor(None, _ws_listener)

        while True:
            batch = await message_queue.get()
            if batch is None:
                break
            for msg in batch:
                yield msg

        # Ensure the listener thread has finished
        await listener_future

        # If we exited without a terminal message, emit one
        if not translator.is_completed:
            for msg in translator._finalize():
                yield msg

    finally:
        # 5. Cleanup
        if ws and ws.connected:
            try:
                ws.close()
                log.info("Comfy WS closed [prompt=%s, job=%s]", prompt_id, job_id)
            except Exception:
                pass


def should_use_comfy_runner(
    run_mode: str | None,
    graph: Graph | None,
) -> bool:
    """Determine whether to use the Comfy execution path.

    Returns True when:
    - ``run_mode == "comfy"``, or
    - The graph contains any node with type starting with ``comfy.``
    """
    if run_mode == "comfy":
        return True
    return bool(graph is not None and has_comfy_nodes(graph))
