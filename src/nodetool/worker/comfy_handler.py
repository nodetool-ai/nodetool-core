"""Handle comfy.* bridge messages: proxy a co-located ComfyUI server.

The worker fronts a headless ComfyUI instance (same container or host,
``COMFYUI_URL``, default ``http://127.0.0.1:8188``) and exposes it through the
worker protocol so the TS server never talks to ComfyUI directly:

- ``comfy.execute``      — submit a workflow (API-format prompt JSON) with clean
  input/output management: input blobs are uploaded via ``POST /upload/image``
  and spliced into the workflow (``"blob:<key>"`` placeholders), execution
  events stream back as ``progress`` frames over the prompt's own
  ``client_id``-scoped WebSocket, and file outputs from ``GET /history`` are
  fetched via ``GET /view`` and returned as binary blobs in the result frame.
- ``comfy.queue`` / ``comfy.interrupt`` / ``comfy.cancel`` — queue introspection
  and cancellation (delete pending, interrupt running).
- ``comfy.upload`` / ``comfy.view`` — stage inputs / fetch files standalone.
- ``comfy.object_info`` / ``comfy.system_stats`` / ``comfy.free`` /
  ``comfy.status`` — node catalog, health, VRAM management.
- ``comfy.models.download`` / ``comfy.models.list`` / ``comfy.models.delete`` —
  manage model files on the persistent volume (``COMFY_MODELS_DIR``, default
  ``/workspace/models`` — the RunPod network volume mount), downloading from
  HuggingFace or a direct URL with streamed progress and cancellation.

Mirrors ``model_handler.handle_models_message``: transport-agnostic, it only
needs a transport exposing an async ``send_msg``, so the same code path serves
both the WebSocket and stdio workers. Each ``comfy.execute`` gets its own
ComfyUI ``client_id`` and WebSocket connection, so concurrent prompts from
different bridge requests never leak events into each other — routing back to
the caller stays keyed on the protocol-level ``request_id``.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import struct
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

import aiohttp

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_MODELS_DIR = "/workspace/models"

# ComfyUI binary WebSocket frames: 4-byte big-endian event type, then payload.
# Event type 1 = preview image, whose payload is 4-byte format + image bytes.
_BINARY_EVENT_PREVIEW_IMAGE = 1
_PREVIEW_FORMATS = {1: "jpeg", 2: "png"}

# How long to keep retrying GET /history/{prompt_id} after the WebSocket said
# the prompt finished — history is written asynchronously and can lag briefly.
_HISTORY_RETRIES = 40
_HISTORY_RETRY_DELAY = 0.25

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def comfy_url() -> str:
    return os.environ.get("COMFYUI_URL", DEFAULT_COMFYUI_URL).rstrip("/")


def comfy_enabled() -> bool:
    """Whether this worker advertises ComfyUI proxying in worker.status.

    Explicit via NODETOOL_COMFY_ENABLED, otherwise implied by COMFYUI_URL
    being configured. comfy.* messages are always dispatched regardless —
    they fail with a connection error when no ComfyUI is reachable.
    """
    flag = os.environ.get("NODETOOL_COMFY_ENABLED")
    if flag is not None:
        return flag.strip().lower() in ("1", "true", "yes", "on")
    return "COMFYUI_URL" in os.environ


def get_comfy_info() -> dict[str, Any]:
    return {"enabled": comfy_enabled(), "url": comfy_url()}


def models_dir() -> Path:
    return Path(os.environ.get("COMFY_MODELS_DIR", DEFAULT_MODELS_DIR))


class ComfyError(Exception):
    """A ComfyUI API call failed; the message carries the response detail."""


class ComfyClient:
    """Thin async client for the ComfyUI HTTP + WebSocket API."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or comfy_url()).rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> ComfyClient:
        # No total timeout: prompt executions and file transfers are long-lived.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("ComfyClient must be used as an async context manager")
        return self._session

    async def _raise_for_status(self, resp: aiohttp.ClientResponse) -> None:
        if resp.status < 400:
            return
        body = (await resp.text())[:4000]
        raise ComfyError(f"ComfyUI {resp.method} {resp.url.path} failed with HTTP {resp.status}: {body}")

    async def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        async with self.session.get(f"{self.base_url}{path}", params=params) as resp:
            await self._raise_for_status(resp)
            return await resp.json(content_type=None)

    async def post_json(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        async with self.session.post(f"{self.base_url}{path}", json=payload or {}) as resp:
            await self._raise_for_status(resp)
            if resp.content_type == "application/json":
                return await resp.json(content_type=None)
            return await resp.text()

    # --- must-have endpoints -------------------------------------------------

    async def submit_prompt(self, workflow: dict[str, Any], client_id: str) -> dict[str, Any]:
        """POST /prompt — submit an API-format workflow, returns prompt_id."""
        return await self.post_json("/prompt", {"prompt": workflow, "client_id": client_id})

    async def history(self, prompt_id: str) -> dict[str, Any]:
        """GET /history/{prompt_id} — {} until the prompt has finished."""
        return await self.get_json(f"/history/{prompt_id}")

    async def view(self, filename: str, subfolder: str = "", type_: str = "output") -> tuple[bytes, str]:
        """GET /view — fetch output/input/temp file bytes and content type."""
        params = {"filename": filename, "subfolder": subfolder, "type": type_}
        async with self.session.get(f"{self.base_url}/view", params=params) as resp:
            await self._raise_for_status(resp)
            content_type = resp.headers.get("Content-Type") or (
                mimetypes.guess_type(filename)[0] or "application/octet-stream"
            )
            return await resp.read(), content_type

    async def upload(
        self,
        data: bytes,
        filename: str,
        *,
        kind: str = "image",
        subfolder: str = "",
        overwrite: bool = True,
        original_ref: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /upload/image (or /upload/mask) — stage bytes into the input dir."""
        form = aiohttp.FormData()
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        form.add_field("image", data, filename=filename, content_type=content_type)
        form.add_field("overwrite", "true" if overwrite else "false")
        if subfolder:
            form.add_field("subfolder", subfolder)
        if kind == "mask" and original_ref is not None:
            form.add_field("original_ref", json.dumps(original_ref))
        endpoint = "/upload/mask" if kind == "mask" else "/upload/image"
        async with self.session.post(f"{self.base_url}{endpoint}", data=form) as resp:
            await self._raise_for_status(resp)
            return await resp.json(content_type=None)

    # --- queue management ----------------------------------------------------

    async def queue(self) -> dict[str, Any]:
        """GET /queue — full running + pending queue contents."""
        return await self.get_json("/queue")

    async def queue_remaining(self) -> int:
        """GET /prompt — lightweight queue depth (distinct from POST /prompt)."""
        info = await self.get_json("/prompt")
        return int(info.get("exec_info", {}).get("queue_remaining", 0))

    async def delete_queued(self, prompt_ids: list[str]) -> None:
        """POST /queue {"delete": [...]} — drop queued-but-not-running prompts."""
        await self.post_json("/queue", {"delete": prompt_ids})

    async def clear_queue(self) -> None:
        await self.post_json("/queue", {"clear": True})

    async def interrupt(self) -> None:
        """POST /interrupt — stop the currently executing prompt."""
        await self.post_json("/interrupt")

    # --- situational ----------------------------------------------------------

    async def object_info(self, node_class: str | None = None) -> dict[str, Any]:
        path = f"/object_info/{node_class}" if node_class else "/object_info"
        return await self.get_json(path)

    async def system_stats(self) -> dict[str, Any]:
        return await self.get_json("/system_stats")

    async def free(self, *, unload_models: bool = True, free_memory: bool = False) -> None:
        await self.post_json("/free", {"unload_models": unload_models, "free_memory": free_memory})

    def ws_connect(self, client_id: str):
        ws_base = self.base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
        return self.session.ws_connect(f"{ws_base}/ws?clientId={client_id}", max_msg_size=0)


def _is_prompt_running(queue_data: dict[str, Any], prompt_id: str) -> bool:
    """Queue entries are [number, prompt_id, prompt, extra_data, outputs]."""
    for entry in queue_data.get("queue_running", []):
        if isinstance(entry, (list, tuple)) and len(entry) > 1 and entry[1] == prompt_id:
            return True
    return False


async def _cancel_prompt(client: ComfyClient, prompt_id: str) -> None:
    """Best-effort cancel: drop from the pending queue, interrupt if running.

    /interrupt only stops the *active* prompt, so guard it with a queue check —
    blindly interrupting would kill another client's job.
    """
    try:
        await client.delete_queued([prompt_id])
    except ComfyError:
        pass
    try:
        if _is_prompt_running(await client.queue(), prompt_id):
            await client.interrupt()
    except ComfyError:
        pass


def _sniff_extension(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[4:8] == b"ftyp":
        return ".mp4"
    return ".bin"


def _patch_blob_refs(obj: Any, uploads: dict[str, str]) -> Any:
    """Replace ``"blob:<key>"`` string values with uploaded ComfyUI filenames."""
    if isinstance(obj, dict):
        return {k: _patch_blob_refs(v, uploads) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_patch_blob_refs(v, uploads) for v in obj]
    if isinstance(obj, str) and obj.startswith("blob:"):
        key = obj[len("blob:"):]
        if key not in uploads:
            raise ComfyError(f"Workflow references input blob '{key}' but no such blob was sent")
        return uploads[key]
    return obj


async def _upload_input_blobs(
    client: ComfyClient,
    blobs: dict[str, bytes],
) -> dict[str, str]:
    """Upload each input blob, returning blob key → ComfyUI filename reference."""
    uploads: dict[str, str] = {}
    for key, data in blobs.items():
        if not isinstance(data, (bytes, bytearray)):
            raise ComfyError(f"Input blob '{key}' must be binary, got {type(data).__name__}")
        safe_key = _SAFE_NAME_RE.sub("_", str(key)) or "input"
        filename = f"nodetool_{uuid.uuid4().hex[:12]}_{safe_key}{_sniff_extension(bytes(data))}"
        info = await client.upload(bytes(data), filename)
        name = info.get("name", filename)
        subfolder = info.get("subfolder") or ""
        uploads[str(key)] = f"{subfolder}/{name}" if subfolder else name
    return uploads


def _looks_like_file_output(values: Any) -> bool:
    return (
        isinstance(values, list)
        and len(values) > 0
        and all(isinstance(v, dict) and "filename" in v for v in values)
    )


async def _collect_outputs(
    client: ComfyClient,
    history_outputs: dict[str, Any],
    *,
    include_temp: bool = False,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Fetch every file output via GET /view; pass scalar outputs through.

    Returns (outputs, blobs) where each file entry carries a ``blob`` key into
    the blobs dict. SaveImage-style nodes write type="output"; preview nodes
    write type="temp" and are skipped unless include_temp is set.
    """
    outputs: dict[str, Any] = {}
    blobs: dict[str, bytes] = {}

    for node_id, node_outputs in history_outputs.items():
        if not isinstance(node_outputs, dict):
            outputs[node_id] = node_outputs
            continue
        node_result: dict[str, Any] = {}
        for slot, values in node_outputs.items():
            if not _looks_like_file_output(values):
                node_result[slot] = values
                continue
            entries: list[dict[str, Any]] = []
            for index, item in enumerate(values):
                file_type = item.get("type", "output")
                meta: dict[str, Any] = {
                    "filename": item.get("filename"),
                    "subfolder": item.get("subfolder", ""),
                    "type": file_type,
                }
                if file_type == "temp" and not include_temp:
                    entries.append(meta)
                    continue
                data, content_type = await client.view(
                    item["filename"], item.get("subfolder", ""), file_type
                )
                blob_key = f"{node_id}/{slot}/{index}/{item['filename']}"
                blobs[blob_key] = data
                meta["content_type"] = content_type
                meta["blob"] = blob_key
                entries.append(meta)
            node_result[slot] = entries
        outputs[node_id] = node_result

    return outputs, blobs


async def _wait_history(client: ComfyClient, prompt_id: str) -> dict[str, Any]:
    """Fetch the history entry, retrying briefly — history writes lag the ws."""
    for _ in range(_HISTORY_RETRIES):
        history = await client.history(prompt_id)
        entry = history.get(prompt_id)
        if entry is not None:
            return entry
        await asyncio.sleep(_HISTORY_RETRY_DELAY)
    raise ComfyError(f"Prompt {prompt_id} finished but never appeared in /history")


async def _poll_until_done(
    client: ComfyClient,
    prompt_id: str,
    cancel_event: asyncio.Event,
    deadline: float | None,
) -> dict[str, Any]:
    """History-polling fallback for when the WebSocket drops mid-execution."""
    loop = asyncio.get_running_loop()
    while True:
        if cancel_event.is_set():
            await _cancel_prompt(client, prompt_id)
            return {"__cancelled__": True}
        if deadline is not None and loop.time() > deadline:
            await _cancel_prompt(client, prompt_id)
            raise ComfyError(f"Prompt {prompt_id} timed out")
        history = await client.history(prompt_id)
        entry = history.get(prompt_id)
        if entry is not None:
            return entry
        queue_data = await client.queue()
        pending = [
            e[1]
            for e in queue_data.get("queue_pending", []) + queue_data.get("queue_running", [])
            if isinstance(e, (list, tuple)) and len(e) > 1
        ]
        if prompt_id not in pending:
            raise ComfyError(
                f"Prompt {prompt_id} disappeared from the queue without a history entry "
                "(ComfyUI may have crashed or restarted)"
            )
        await asyncio.sleep(1.0)


def _extract_history_error(entry: dict[str, Any]) -> str | None:
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        for event_name, event_data in status.get("messages", []) or []:
            if event_name == "execution_error":
                node = event_data.get("node_id") or event_data.get("node_type")
                message = event_data.get("exception_message", "execution error")
                return f"Execution failed at node {node}: {message}"
        return "Execution failed"
    return None


async def _handle_execute(
    data: dict[str, Any],
    request_id: str | None,
    cancel_flags: dict[str, asyncio.Event],
    send_progress: Callable,
    send_result: Callable,
) -> None:
    workflow = data.get("workflow")
    if not isinstance(workflow, dict) or not workflow:
        raise ComfyError("comfy.execute requires a non-empty 'workflow' (API-format prompt JSON)")

    blobs: dict[str, bytes] = data.get("blobs") or {}
    forward_previews = bool(data.get("previews", False))
    include_temp = bool(data.get("include_temp", False))
    timeout = data.get("timeout")

    cancel_event = asyncio.Event()
    if request_id:
        cancel_flags[request_id] = cancel_event

    loop = asyncio.get_running_loop()
    deadline = (loop.time() + float(timeout)) if timeout else None

    try:
        async with ComfyClient() as client:
            client_id = uuid.uuid4().hex

            # Connect the events socket BEFORE submitting so no event is missed.
            async with client.ws_connect(client_id) as ws:
                uploads = await _upload_input_blobs(client, blobs)
                patched = _patch_blob_refs(workflow, uploads)

                submit = await client.submit_prompt(patched, client_id)
                prompt_id = submit.get("prompt_id")
                if not prompt_id:
                    raise ComfyError(f"ComfyUI POST /prompt returned no prompt_id: {submit}")

                await send_progress(request_id, {
                    "status": "queued",
                    "prompt_id": prompt_id,
                    "queue_position": submit.get("number"),
                })

                status = await _stream_events(
                    ws=ws,
                    client=client,
                    prompt_id=prompt_id,
                    request_id=request_id,
                    cancel_event=cancel_event,
                    deadline=deadline,
                    forward_previews=forward_previews,
                    send_progress=send_progress,
                )

            if status == "cancelled":
                await send_progress(request_id, {"status": "cancelled", "prompt_id": prompt_id})
                await send_result(request_id, {"prompt_id": prompt_id, "status": "cancelled"})
                return

            entry = await _wait_history(client, prompt_id)
            error = _extract_history_error(entry)
            if error:
                raise ComfyError(error)

            outputs, output_blobs = await _collect_outputs(
                client, entry.get("outputs", {}) or {}, include_temp=include_temp
            )
            await send_progress(request_id, {"status": "completed", "prompt_id": prompt_id})
            await send_result(request_id, {
                "prompt_id": prompt_id,
                "status": "completed",
                "outputs": outputs,
                "blobs": output_blobs,
            })
    finally:
        if request_id:
            cancel_flags.pop(request_id, None)


async def _stream_events(
    *,
    ws: Any,
    client: ComfyClient,
    prompt_id: str,
    request_id: str | None,
    cancel_event: asyncio.Event,
    deadline: float | None,
    forward_previews: bool,
    send_progress: Callable,
) -> str:
    """Forward execution events until the prompt completes.

    Returns "completed" or "cancelled"; raises ComfyError on execution errors
    or timeout. Falls back to history polling if the socket drops.
    """
    loop = asyncio.get_running_loop()
    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        while True:
            recv_task = asyncio.create_task(ws.receive())
            timeout_left = None if deadline is None else max(deadline - loop.time(), 0.0)
            done, _pending = await asyncio.wait(
                {recv_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout_left,
            )

            if not done:
                recv_task.cancel()
                await _cancel_prompt(client, prompt_id)
                raise ComfyError(f"Prompt {prompt_id} timed out")

            if cancel_task in done:
                recv_task.cancel()
                await _cancel_prompt(client, prompt_id)
                return "cancelled"

            msg = recv_task.result()

            if msg.type == aiohttp.WSMsgType.TEXT:
                event = json.loads(msg.data)
                event_type = event.get("type")
                event_data = event.get("data") or {}
                event_prompt = event_data.get("prompt_id")
                if event_prompt is not None and event_prompt != prompt_id:
                    continue

                if event_type == "status":
                    remaining = (
                        event_data.get("status", {}).get("exec_info", {}).get("queue_remaining")
                    )
                    if remaining is not None:
                        await send_progress(request_id, {
                            "status": "queue",
                            "prompt_id": prompt_id,
                            "queue_remaining": remaining,
                        })
                elif event_type == "execution_start" and event_prompt == prompt_id:
                    await send_progress(request_id, {"status": "started", "prompt_id": prompt_id})
                elif event_type == "execution_cached" and event_prompt == prompt_id:
                    await send_progress(request_id, {
                        "status": "cached",
                        "prompt_id": prompt_id,
                        "nodes": event_data.get("nodes", []),
                    })
                elif event_type == "executing" and event_prompt == prompt_id:
                    if event_data.get("node") is None:
                        return "completed"
                    await send_progress(request_id, {
                        "status": "executing",
                        "prompt_id": prompt_id,
                        "node": event_data.get("node"),
                    })
                elif event_type == "progress":
                    await send_progress(request_id, {
                        "status": "progress",
                        "prompt_id": prompt_id,
                        "node": event_data.get("node"),
                        "value": event_data.get("value"),
                        "max": event_data.get("max"),
                    })
                elif event_type == "executed" and event_prompt == prompt_id:
                    await send_progress(request_id, {
                        "status": "node_output",
                        "prompt_id": prompt_id,
                        "node": event_data.get("node"),
                        "outputs": event_data.get("output"),
                    })
                elif event_type == "execution_success" and event_prompt == prompt_id:
                    return "completed"
                elif event_type == "execution_interrupted" and event_prompt == prompt_id:
                    return "cancelled"
                elif event_type == "execution_error" and event_prompt == prompt_id:
                    node = event_data.get("node_id") or event_data.get("node_type")
                    message = event_data.get("exception_message", "execution error")
                    raise ComfyError(f"Execution failed at node {node}: {message}")

            elif msg.type == aiohttp.WSMsgType.BINARY:
                # 8-byte header: 4-byte event type + 4-byte image format, then bytes.
                if not forward_previews or len(msg.data) < 8:
                    continue
                event_code = struct.unpack(">I", msg.data[:4])[0]
                if event_code != _BINARY_EVENT_PREVIEW_IMAGE:
                    continue
                image_format = struct.unpack(">I", msg.data[4:8])[0]
                await send_progress(request_id, {
                    "status": "preview",
                    "prompt_id": prompt_id,
                    "format": _PREVIEW_FORMATS.get(image_format, "unknown"),
                    "image": msg.data[8:],
                })

            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            ):
                # Socket dropped mid-run — fall back to polling /history.
                entry = await _poll_until_done(client, prompt_id, cancel_event, deadline)
                if entry.get("__cancelled__"):
                    return "cancelled"
                error = _extract_history_error(entry)
                if error:
                    raise ComfyError(error)
                return "completed"
    finally:
        cancel_task.cancel()


# --- model volume management --------------------------------------------------


def _validate_relpath(*parts: str) -> Path:
    """Join path components, rejecting traversal / absolute segments."""
    rel = Path()
    for part in parts:
        if not part:
            raise ComfyError("Empty path component")
        candidate = part.replace("\\", "/")
        p = Path(candidate)
        if p.is_absolute() or any(seg in ("..", "") for seg in p.parts) or "\x00" in candidate:
            raise ComfyError(f"Unsafe path component: {part!r}")
        rel = rel / p
    return rel


def _resolve_model_path(folder: str, filename: str) -> Path:
    from nodetool.metadata.types import comfy_model_to_folder

    folder = comfy_model_to_folder(folder)
    root = models_dir()
    target = root / _validate_relpath(folder, filename)
    resolved_root = root.resolve()
    if not target.resolve().is_relative_to(resolved_root):
        raise ComfyError(f"Path escapes the models directory: {folder}/{filename}")
    return target


async def _resolve_hf_token() -> str | None:
    from nodetool.integrations.huggingface.huggingface_models import get_hf_token

    return await get_hf_token()


def _hf_file_url(repo_id: str, path: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{path}"


async def _open_model_source(
    session: aiohttp.ClientSession, source: dict[str, Any]
) -> tuple[aiohttp.ClientResponse, str]:
    """Resolve a download source to an open streaming response + default name."""
    source_type = source.get("type")
    headers: dict[str, str] = {}

    if source_type == "huggingface":
        repo_id = source.get("repo_id")
        path = source.get("path")
        if not repo_id or not path:
            raise ComfyError("huggingface source requires 'repo_id' and 'path'")
        url = _hf_file_url(repo_id, path, source.get("revision", "main"))
        # The token is resolved on the worker — client-supplied tokens are ignored.
        token = await _resolve_hf_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        default_name = os.path.basename(path)
    elif source_type == "url":
        url = source.get("url")
        if not url or not str(url).startswith(("http://", "https://")):
            raise ComfyError("url source requires an http(s) 'url'")
        default_name = os.path.basename(str(url).split("?", 1)[0]) or "model.bin"
    else:
        raise ComfyError(f"Unknown model source type: {source_type!r} (expected 'huggingface' or 'url')")

    resp = await session.get(url, headers=headers, allow_redirects=True)
    if resp.status >= 400:
        body = (await resp.text())[:1000]
        resp.release()
        raise ComfyError(f"Model download failed with HTTP {resp.status}: {body}")
    return resp, default_name


async def _handle_models_download(
    data: dict[str, Any],
    request_id: str | None,
    cancel_flags: dict[str, asyncio.Event],
    send_progress: Callable,
    send_result: Callable,
) -> None:
    source = data.get("source") or {}
    folder = data.get("folder")
    if not folder:
        raise ComfyError("comfy.models.download requires 'folder' (e.g. 'checkpoints' or 'comfy.checkpoint_file')")

    cancel_event = asyncio.Event()
    if request_id:
        cancel_flags[request_id] = cancel_event

    def frame(status: str, downloaded: int, total: int, *, error: str | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": status,
            "folder": folder,
            "filename": data.get("filename"),
            "downloaded_bytes": downloaded,
            "total_bytes": total,
        }
        if error:
            d["error"] = error
        return d

    async def report_exists(target: Path, filename: str) -> None:
        existing = target.stat().st_size
        await send_progress(request_id, frame("exists", existing, existing))
        await send_result(request_id, {
            "status": "exists",
            "folder": folder,
            "filename": filename,
            "path": str(target),
            "size_bytes": existing,
        })

    try:
        # With an explicit filename the existence check needs no network —
        # keeps warm restarts on a populated volume instant and offline-safe.
        filename = data.get("filename")
        if filename and not data.get("force"):
            target = _resolve_model_path(str(folder), str(filename))
            if target.exists():
                await report_exists(target, str(filename))
                return

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            resp, default_name = await _open_model_source(session, source)
            async with resp:
                filename = filename or default_name
                data["filename"] = filename
                target = _resolve_model_path(str(folder), str(filename))
                total = int(resp.headers.get("Content-Length") or 0)

                if target.exists() and not data.get("force"):
                    await report_exists(target, str(filename))
                    return

                target.parent.mkdir(parents=True, exist_ok=True)
                part_path = target.with_name(target.name + ".part")
                downloaded = 0

                await send_progress(request_id, frame("start", 0, total))
                try:
                    with open(part_path, "wb") as out:
                        async for chunk in resp.content.iter_chunked(1024 * 1024):
                            if cancel_event.is_set():
                                raise asyncio.CancelledError()
                            out.write(chunk)
                            downloaded += len(chunk)
                            await send_progress(request_id, frame("progress", downloaded, total))
                    os.replace(part_path, target)
                except asyncio.CancelledError:
                    part_path.unlink(missing_ok=True)
                    await send_progress(request_id, frame("cancelled", downloaded, total))
                    await send_result(request_id, {
                        "status": "cancelled",
                        "folder": folder,
                        "filename": filename,
                    })
                    return
                except BaseException:
                    part_path.unlink(missing_ok=True)
                    raise

                await send_progress(request_id, frame("completed", downloaded, downloaded or total))
                await send_result(request_id, {
                    "status": "completed",
                    "folder": folder,
                    "filename": filename,
                    "path": str(target),
                    "size_bytes": downloaded,
                })
    except ComfyError as e:
        await send_progress(request_id, frame("error", 0, 0, error=str(e)))
        raise
    finally:
        if request_id:
            cancel_flags.pop(request_id, None)


def _list_models() -> list[dict[str, Any]]:
    root = models_dir()
    if not root.is_dir():
        return []
    result: list[dict[str, Any]] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.name.endswith(".part"):
                continue
            result.append({
                "folder": folder.name,
                "filename": str(path.relative_to(folder)),
                "size_bytes": path.stat().st_size,
            })
    return result


# --- dispatch -------------------------------------------------------------------


async def handle_comfy_message(
    msg_type: str,
    request_id: str | None,
    data: dict[str, Any],
    transport: Any,  # WorkerTransport (exposes async send_msg)
    cancel_flags: dict[str, asyncio.Event],
) -> None:
    """Handle a comfy.* message via any transport exposing ``send_msg``."""

    async def send_result(rid: str | None, d: dict) -> None:
        await transport.send_msg({"type": "result", "request_id": rid, "data": d})

    async def send_error(rid: str | None, error: str, tb: str | None = None) -> None:
        await transport.send_msg(
            {"type": "error", "request_id": rid, "data": {"error": error, "traceback": tb}}
        )

    async def send_progress(rid: str | None, d: dict) -> None:
        await transport.send_msg({"type": "progress", "request_id": rid, "data": d})

    try:
        if msg_type == "comfy.execute":
            await _handle_execute(data, request_id, cancel_flags, send_progress, send_result)

        elif msg_type == "comfy.status":
            info = get_comfy_info()
            try:
                async with ComfyClient() as client:
                    info["system_stats"] = await client.system_stats()
                    info["queue_remaining"] = await client.queue_remaining()
                    info["reachable"] = True
            except Exception as e:
                info["reachable"] = False
                info["error"] = str(e)
            await send_result(request_id, info)

        elif msg_type == "comfy.queue":
            async with ComfyClient() as client:
                queue_data = await client.queue()
            await send_result(request_id, {
                "queue_running": queue_data.get("queue_running", []),
                "queue_pending": queue_data.get("queue_pending", []),
            })

        elif msg_type == "comfy.interrupt":
            async with ComfyClient() as client:
                await client.interrupt()
            await send_result(request_id, {"interrupted": True})

        elif msg_type == "comfy.cancel":
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise ComfyError("comfy.cancel requires 'prompt_id'")
            async with ComfyClient() as client:
                await _cancel_prompt(client, str(prompt_id))
            await send_result(request_id, {"cancelled": True, "prompt_id": prompt_id})

        elif msg_type == "comfy.upload":
            blob = data.get("data")
            if not isinstance(blob, (bytes, bytearray)):
                raise ComfyError("comfy.upload requires binary 'data'")
            filename = data.get("filename") or f"nodetool_{uuid.uuid4().hex[:12]}{_sniff_extension(bytes(blob))}"
            async with ComfyClient() as client:
                info = await client.upload(
                    bytes(blob),
                    str(filename),
                    kind=str(data.get("kind", "image")),
                    subfolder=str(data.get("subfolder", "")),
                    overwrite=bool(data.get("overwrite", True)),
                    original_ref=data.get("original_ref"),
                )
            await send_result(request_id, info)

        elif msg_type == "comfy.view":
            if not data.get("filename"):
                raise ComfyError("comfy.view requires 'filename'")
            async with ComfyClient() as client:
                blob, content_type = await client.view(
                    str(data["filename"]),
                    str(data.get("subfolder", "")),
                    str(data.get("type", "output")),
                )
            await send_result(request_id, {
                "filename": data["filename"],
                "content_type": content_type,
                "data": blob,
            })

        elif msg_type == "comfy.object_info":
            async with ComfyClient() as client:
                info = await client.object_info(data.get("node_class"))
            await send_result(request_id, {"object_info": info})

        elif msg_type == "comfy.system_stats":
            async with ComfyClient() as client:
                stats = await client.system_stats()
            await send_result(request_id, stats)

        elif msg_type == "comfy.free":
            async with ComfyClient() as client:
                await client.free(
                    unload_models=bool(data.get("unload_models", True)),
                    free_memory=bool(data.get("free_memory", False)),
                )
            await send_result(request_id, {"freed": True})

        elif msg_type == "comfy.models.download":
            await _handle_models_download(data, request_id, cancel_flags, send_progress, send_result)

        elif msg_type == "comfy.models.list":
            await send_result(request_id, {
                "models_dir": str(models_dir()),
                "models": _list_models(),
            })

        elif msg_type == "comfy.models.delete":
            folder = data.get("folder")
            filename = data.get("filename")
            if not folder or not filename:
                raise ComfyError("comfy.models.delete requires 'folder' and 'filename'")
            target = _resolve_model_path(str(folder), str(filename))
            deleted = target.is_file()
            if deleted:
                target.unlink()
            await send_result(request_id, {"deleted": deleted})

        else:
            await send_error(request_id, f"Unknown comfy message type: {msg_type}")

    except Exception as e:
        await send_error(request_id, str(e), traceback.format_exc())
