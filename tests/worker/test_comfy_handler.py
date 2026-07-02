"""Tests for the comfy.* bridge handler (ComfyUI proxy).

A fake ComfyUI server (aiohttp.web) implements the endpoints the proxy uses —
POST /prompt, GET /ws, GET /history/{id}, GET /view, POST /upload/image,
GET/POST /queue, POST /interrupt, GET /object_info, GET /system_stats,
POST /free — and the tests drive the real worker WebSocket server end to end,
exactly like the TS bridge would.
"""

import asyncio
import json
import struct

import msgpack
import pytest
import pytest_asyncio
import websockets
from aiohttp import web

from nodetool.worker.comfy_handler import (
    ComfyError,
    _hf_file_url,
    _patch_blob_refs,
    _sniff_extension,
)
from nodetool.worker.server import WorkerServer, start_server

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload"
OUTPUT_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-output-image"


class FakeComfy:
    """Minimal in-process ComfyUI double covering the proxied endpoints."""

    def __init__(self) -> None:
        self.prompt_counter = 0
        self.prompts: dict[str, dict] = {}
        self.uploads: dict[str, bytes] = {}
        self.history_entries: dict[str, dict] = {}
        self.ws_by_client: dict[str, web.WebSocketResponse] = {}
        self.queue_running: list[list] = []
        self.queue_pending: list[list] = []
        self.deleted: list[str] = []
        self.interrupts = 0
        self.freed: list[dict] = []
        self.hold = False  # prompt "runs" until POST /interrupt
        self.fail_prompt: dict | None = None  # 400 body for POST /prompt
        self.error_event: dict | None = None  # execution_error payload
        self.send_preview = False
        self.model_files: dict[str, bytes] = {}
        self._interrupt_events: dict[str, asyncio.Event] = {}

        self.app = web.Application()
        self.app.add_routes([
            web.get("/ws", self.handle_ws),
            web.post("/prompt", self.handle_post_prompt),
            web.get("/prompt", self.handle_get_prompt),
            web.get("/history/{prompt_id}", self.handle_history),
            web.get("/view", self.handle_view),
            web.post("/upload/image", self.handle_upload),
            web.get("/queue", self.handle_get_queue),
            web.post("/queue", self.handle_post_queue),
            web.post("/interrupt", self.handle_interrupt),
            web.get("/system_stats", self.handle_system_stats),
            web.get("/object_info", self.handle_object_info),
            web.get("/object_info/{node_class}", self.handle_object_info),
            web.post("/free", self.handle_free),
            web.get("/files/{name}", self.handle_model_file),
        ])

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        client_id = request.rel_url.query.get("clientId", "")
        self.ws_by_client[client_id] = ws
        async for _msg in ws:
            pass
        self.ws_by_client.pop(client_id, None)
        return ws

    async def _send_event(self, client_id: str, event: dict) -> None:
        ws = self.ws_by_client.get(client_id)
        if ws is not None and not ws.closed:
            await ws.send_str(json.dumps(event))

    async def handle_post_prompt(self, request: web.Request) -> web.Response:
        if self.fail_prompt is not None:
            return web.json_response(self.fail_prompt, status=400)
        body = await request.json()
        self.prompt_counter += 1
        prompt_id = f"prompt-{self.prompt_counter}"
        client_id = body.get("client_id", "")
        self.prompts[prompt_id] = {"workflow": body.get("prompt"), "client_id": client_id}
        asyncio.get_running_loop().create_task(self._run(prompt_id, client_id))
        return web.json_response({"prompt_id": prompt_id, "number": self.prompt_counter})

    async def _run(self, prompt_id: str, client_id: str) -> None:
        await asyncio.sleep(0.05)
        await self._send_event(client_id, {"type": "execution_start", "data": {"prompt_id": prompt_id}})

        if self.error_event is not None:
            await self._send_event(client_id, {
                "type": "execution_error",
                "data": {"prompt_id": prompt_id, **self.error_event},
            })
            return

        if self.hold:
            self.queue_running.append([1, prompt_id, {}, {}, []])
            await self._send_event(client_id, {
                "type": "executing", "data": {"prompt_id": prompt_id, "node": "1"},
            })
            event = self._interrupt_events.setdefault(prompt_id, asyncio.Event())
            await event.wait()
            self.queue_running = [e for e in self.queue_running if e[1] != prompt_id]
            await self._send_event(client_id, {
                "type": "execution_interrupted", "data": {"prompt_id": prompt_id},
            })
            return

        await self._send_event(client_id, {
            "type": "executing", "data": {"prompt_id": prompt_id, "node": "1"},
        })
        await self._send_event(client_id, {
            "type": "progress", "data": {"prompt_id": prompt_id, "node": "1", "value": 1, "max": 2},
        })
        if self.send_preview:
            ws = self.ws_by_client.get(client_id)
            if ws is not None and not ws.closed:
                # 4-byte event type (1 = preview) + 4-byte format (2 = png) + bytes
                await ws.send_bytes(struct.pack(">II", 1, 2) + b"preview-bytes")
        node_output = {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
        self.history_entries[prompt_id] = {
            "outputs": {"9": node_output},
            "status": {"status_str": "success", "completed": True, "messages": []},
        }
        await self._send_event(client_id, {
            "type": "executed", "data": {"prompt_id": prompt_id, "node": "9", "output": node_output},
        })
        await self._send_event(client_id, {
            "type": "executing", "data": {"prompt_id": prompt_id, "node": None},
        })

    async def handle_get_prompt(self, request: web.Request) -> web.Response:
        return web.json_response({"exec_info": {"queue_remaining": len(self.queue_pending)}})

    async def handle_history(self, request: web.Request) -> web.Response:
        prompt_id = request.match_info["prompt_id"]
        entry = self.history_entries.get(prompt_id)
        return web.json_response({prompt_id: entry} if entry is not None else {})

    async def handle_view(self, request: web.Request) -> web.Response:
        filename = request.rel_url.query.get("filename", "")
        if filename == "out.png":
            return web.Response(body=OUTPUT_BYTES, content_type="image/png")
        if filename in self.uploads:
            return web.Response(body=self.uploads[filename], content_type="image/png")
        return web.Response(status=404, text="not found")

    async def handle_upload(self, request: web.Request) -> web.Response:
        reader = await request.multipart()
        filename = ""
        data = b""
        field = await reader.next()
        while field is not None:
            if field.name == "image":
                filename = field.filename or "upload.bin"
                data = await field.read()
            else:
                await field.read()
            field = await reader.next()
        self.uploads[filename] = data
        return web.json_response({"name": filename, "subfolder": "", "type": "input"})

    async def handle_get_queue(self, request: web.Request) -> web.Response:
        return web.json_response({
            "queue_running": self.queue_running,
            "queue_pending": self.queue_pending,
        })

    async def handle_post_queue(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.deleted.extend(body.get("delete", []))
        return web.json_response({})

    async def handle_interrupt(self, request: web.Request) -> web.Response:
        self.interrupts += 1
        for event in self._interrupt_events.values():
            event.set()
        return web.Response()

    async def handle_system_stats(self, request: web.Request) -> web.Response:
        return web.json_response({"system": {"os": "fake", "comfyui_version": "0.0.0"}, "devices": []})

    async def handle_object_info(self, request: web.Request) -> web.Response:
        node_class = request.match_info.get("node_class")
        catalog = {"KSampler": {"input": {"required": {}}}}
        if node_class:
            return web.json_response({node_class: catalog.get(node_class, {})})
        return web.json_response(catalog)

    async def handle_free(self, request: web.Request) -> web.Response:
        self.freed.append(await request.json())
        return web.Response()

    async def handle_model_file(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if name not in self.model_files:
            return web.Response(status=404, text="no such file")
        return web.Response(body=self.model_files[name])


@pytest_asyncio.fixture(loop_scope="function")
async def fake_comfy(monkeypatch):
    fake = FakeComfy()
    runner = web.AppRunner(fake.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    fake.base_url = f"http://{host}:{port}"
    monkeypatch.setenv("COMFYUI_URL", fake.base_url)
    yield fake
    await runner.cleanup()


@pytest_asyncio.fixture(loop_scope="function")
async def server():
    worker = WorkerServer()
    host, port, stop_event, task = await start_server(
        host="127.0.0.1", port=0, worker=worker,
    )
    yield host, port
    stop_event.set()
    await task


async def _request(ws, msg: dict) -> list[dict]:
    """Send one bridge message, collect frames until result/error."""
    await ws.send(msgpack.packb(msg))
    frames = []
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        frame = msgpack.unpackb(raw, raw=False)
        frames.append(frame)
        if frame["type"] in ("result", "error"):
            return frames


# --- pure helpers ----------------------------------------------------------------


def test_sniff_extension():
    assert _sniff_extension(PNG_BYTES) == ".png"
    assert _sniff_extension(b"\xff\xd8\xff\xe0rest") == ".jpg"
    assert _sniff_extension(b"RIFF1234WEBPdata") == ".webp"
    assert _sniff_extension(b"RIFF1234WAVEdata") == ".wav"
    assert _sniff_extension(b"arbitrary") == ".bin"


def test_patch_blob_refs_replaces_placeholders():
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "blob:img", "extra": ["blob:img"]}},
        "2": {"inputs": {"text": "not a blob"}},
    }
    patched = _patch_blob_refs(workflow, {"img": "sub/nodetool_x.png"})
    assert patched["1"]["inputs"]["image"] == "sub/nodetool_x.png"
    assert patched["1"]["inputs"]["extra"] == ["sub/nodetool_x.png"]
    assert patched["2"]["inputs"]["text"] == "not a blob"
    # original untouched
    assert workflow["1"]["inputs"]["image"] == "blob:img"


def test_patch_blob_refs_unknown_key_raises():
    with pytest.raises(ComfyError, match="missing"):
        _patch_blob_refs({"1": {"inputs": {"image": "blob:missing"}}}, {})


def test_hf_file_url():
    assert (
        _hf_file_url("org/repo", "unet/model.safetensors")
        == "https://huggingface.co/org/repo/resolve/main/unet/model.safetensors"
    )
    assert (
        _hf_file_url("org/repo", "m.bin", revision="fp16")
        == "https://huggingface.co/org/repo/resolve/fp16/m.bin"
    )


# --- worker.status capability flag -------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_worker_status_reports_comfy(server, fake_comfy):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {"type": "worker.status", "request_id": "st-1", "data": {}})
    status = frames[-1]["data"]
    assert status["comfy"]["enabled"] is True
    assert status["comfy"]["url"] == fake_comfy.base_url


# --- comfy.execute ------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_end_to_end(server, fake_comfy):
    """Blobs upload → workflow patched → events stream → outputs fetched as blobs."""
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "blob:img1"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.execute",
            "request_id": "ex-1",
            "data": {"workflow": workflow, "blobs": {"img1": PNG_BYTES}},
        })

    result = frames[-1]
    assert result["type"] == "result", result
    assert result["data"]["status"] == "completed"
    assert result["data"]["prompt_id"] == "prompt-1"

    # Input management: the blob was uploaded and spliced into the workflow.
    submitted = fake_comfy.prompts["prompt-1"]["workflow"]
    uploaded_name = submitted["1"]["inputs"]["image"]
    assert uploaded_name.startswith("nodetool_")
    assert uploaded_name.endswith(".png")
    assert fake_comfy.uploads[uploaded_name] == PNG_BYTES

    # Progress stream carries the ComfyUI execution lifecycle.
    statuses = [f["data"]["status"] for f in frames if f["type"] == "progress"]
    assert statuses[0] == "queued"
    for expected in ("started", "executing", "progress", "node_output", "completed"):
        assert expected in statuses

    # Output management: the SaveImage file came back as a binary blob.
    images = result["data"]["outputs"]["9"]["images"]
    assert images[0]["filename"] == "out.png"
    assert images[0]["content_type"] == "image/png"
    blob_key = images[0]["blob"]
    assert result["data"]["blobs"][blob_key] == OUTPUT_BYTES


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_forwards_previews_when_enabled(server, fake_comfy):
    fake_comfy.send_preview = True
    workflow = {"9": {"class_type": "SaveImage", "inputs": {}}}
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.execute",
            "request_id": "ex-prev",
            "data": {"workflow": workflow, "previews": True},
        })
    previews = [f["data"] for f in frames if f["type"] == "progress" and f["data"]["status"] == "preview"]
    assert len(previews) == 1
    assert previews[0]["format"] == "png"
    assert previews[0]["image"] == b"preview-bytes"


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_skips_previews_by_default(server, fake_comfy):
    fake_comfy.send_preview = True
    workflow = {"9": {"class_type": "SaveImage", "inputs": {}}}
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.execute",
            "request_id": "ex-noprev",
            "data": {"workflow": workflow},
        })
    assert frames[-1]["type"] == "result"
    assert not any(
        f["type"] == "progress" and f["data"]["status"] == "preview" for f in frames
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_submit_rejection_is_an_error(server, fake_comfy):
    fake_comfy.fail_prompt = {"error": {"message": "invalid prompt"}, "node_errors": {}}
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.execute",
            "request_id": "ex-bad",
            "data": {"workflow": {"1": {"class_type": "Nope", "inputs": {}}}},
        })
    assert frames[-1]["type"] == "error"
    assert "invalid prompt" in frames[-1]["data"]["error"]
    assert "400" in frames[-1]["data"]["error"]


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_execution_error_event(server, fake_comfy):
    fake_comfy.error_event = {"node_id": "5", "node_type": "KSampler", "exception_message": "CUDA OOM"}
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.execute",
            "request_id": "ex-err",
            "data": {"workflow": {"5": {"class_type": "KSampler", "inputs": {}}}},
        })
    assert frames[-1]["type"] == "error"
    assert "CUDA OOM" in frames[-1]["data"]["error"]
    assert "5" in frames[-1]["data"]["error"]


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_missing_blob_reference(server, fake_comfy):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.execute",
            "request_id": "ex-miss",
            "data": {"workflow": {"1": {"inputs": {"image": "blob:nope"}}}},
        })
    assert frames[-1]["type"] == "error"
    assert "nope" in frames[-1]["data"]["error"]


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_requires_workflow(server, fake_comfy):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.execute", "request_id": "ex-empty", "data": {},
        })
    assert frames[-1]["type"] == "error"
    assert "workflow" in frames[-1]["data"]["error"]


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_cancel_interrupts_running_prompt(server, fake_comfy):
    """The protocol-level cancel maps to queue-delete + interrupt on ComfyUI."""
    fake_comfy.hold = True
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "comfy.execute",
            "request_id": "ex-cancel",
            "data": {"workflow": {"1": {"class_type": "KSampler", "inputs": {}}}},
        }))
        # Wait until the prompt is executing, then cancel it.
        while True:
            frame = msgpack.unpackb(await asyncio.wait_for(ws.recv(), timeout=10), raw=False)
            if frame["type"] == "progress" and frame["data"]["status"] == "executing":
                break
        await ws.send(msgpack.packb({"type": "cancel", "request_id": "ex-cancel"}))
        while True:
            frame = msgpack.unpackb(await asyncio.wait_for(ws.recv(), timeout=10), raw=False)
            if frame["type"] in ("result", "error"):
                break

    assert frame["type"] == "result"
    assert frame["data"]["status"] == "cancelled"
    assert fake_comfy.interrupts == 1
    assert fake_comfy.deleted == ["prompt-1"]


# --- passthrough endpoints -----------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_comfy_status(server, fake_comfy):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {"type": "comfy.status", "request_id": "cs-1", "data": {}})
    data = frames[-1]["data"]
    assert data["reachable"] is True
    assert data["enabled"] is True
    assert data["queue_remaining"] == 0
    assert data["system_stats"]["system"]["os"] == "fake"


@pytest.mark.asyncio(loop_scope="function")
async def test_comfy_status_unreachable(server, monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "http://127.0.0.1:1")
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {"type": "comfy.status", "request_id": "cs-2", "data": {}})
    data = frames[-1]["data"]
    assert frames[-1]["type"] == "result"
    assert data["reachable"] is False
    assert "error" in data


@pytest.mark.asyncio(loop_scope="function")
async def test_comfy_queue(server, fake_comfy):
    fake_comfy.queue_pending = [[2, "prompt-x", {}, {}, []]]
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {"type": "comfy.queue", "request_id": "cq-1", "data": {}})
    data = frames[-1]["data"]
    assert data["queue_running"] == []
    assert data["queue_pending"][0][1] == "prompt-x"


@pytest.mark.asyncio(loop_scope="function")
async def test_comfy_interrupt(server, fake_comfy):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {"type": "comfy.interrupt", "request_id": "ci-1", "data": {}})
    assert frames[-1]["data"]["interrupted"] is True
    assert fake_comfy.interrupts == 1


@pytest.mark.asyncio(loop_scope="function")
async def test_comfy_cancel_pending_prompt(server, fake_comfy):
    """comfy.cancel deletes from the queue but does NOT interrupt someone else's job."""
    fake_comfy.queue_running = [[1, "other-prompt", {}, {}, []]]
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.cancel", "request_id": "cc-1", "data": {"prompt_id": "pending-1"},
        })
    assert frames[-1]["data"]["cancelled"] is True
    assert fake_comfy.deleted == ["pending-1"]
    assert fake_comfy.interrupts == 0  # running prompt belongs to another client


@pytest.mark.asyncio(loop_scope="function")
async def test_comfy_upload_and_view(server, fake_comfy):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.upload",
            "request_id": "cu-1",
            "data": {"data": PNG_BYTES, "filename": "staged.png"},
        })
        assert frames[-1]["data"]["name"] == "staged.png"
        assert fake_comfy.uploads["staged.png"] == PNG_BYTES

        frames = await _request(ws, {
            "type": "comfy.view",
            "request_id": "cv-1",
            "data": {"filename": "staged.png", "type": "input"},
        })
    assert frames[-1]["data"]["data"] == PNG_BYTES
    assert frames[-1]["data"]["content_type"] == "image/png"


@pytest.mark.asyncio(loop_scope="function")
async def test_comfy_object_info_and_system_stats_and_free(server, fake_comfy):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {"type": "comfy.object_info", "request_id": "co-1", "data": {}})
        assert "KSampler" in frames[-1]["data"]["object_info"]

        frames = await _request(ws, {"type": "comfy.system_stats", "request_id": "cst-1", "data": {}})
        assert frames[-1]["data"]["system"]["os"] == "fake"

        frames = await _request(ws, {
            "type": "comfy.free", "request_id": "cf-1", "data": {"free_memory": True},
        })
    assert frames[-1]["data"]["freed"] is True
    assert fake_comfy.freed == [{"unload_models": True, "free_memory": True}]


@pytest.mark.asyncio(loop_scope="function")
async def test_comfy_unknown_type(server, fake_comfy):
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {"type": "comfy.nonexistent", "request_id": "cx-1", "data": {}})
    assert frames[-1]["type"] == "error"
    assert "Unknown comfy message type" in frames[-1]["data"]["error"]


# --- model volume management ----------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_models_download_from_url(server, fake_comfy, tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_MODELS_DIR", str(tmp_path))
    payload = b"x" * 64
    fake_comfy.model_files["sd.safetensors"] = payload

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.models.download",
            "request_id": "cmd-1",
            "data": {
                "folder": "checkpoints",
                "source": {"type": "url", "url": f"{fake_comfy.base_url}/files/sd.safetensors"},
            },
        })

    result = frames[-1]
    assert result["type"] == "result"
    assert result["data"]["status"] == "completed"
    assert result["data"]["filename"] == "sd.safetensors"
    assert result["data"]["size_bytes"] == 64

    target = tmp_path / "checkpoints" / "sd.safetensors"
    assert target.read_bytes() == payload
    assert not target.with_name("sd.safetensors.part").exists()

    statuses = [f["data"]["status"] for f in frames if f["type"] == "progress"]
    assert statuses[0] == "start"
    assert statuses[-1] == "completed"
    assert "progress" in statuses
    progress_frames = [f["data"] for f in frames if f["type"] == "progress" and f["data"]["status"] == "progress"]
    assert progress_frames[-1]["downloaded_bytes"] == 64
    assert progress_frames[-1]["total_bytes"] == 64


@pytest.mark.asyncio(loop_scope="function")
async def test_models_download_existing_is_skipped_without_network(server, tmp_path, monkeypatch):
    """A file already on the volume short-circuits — no ComfyUI/PyPI/HF needed."""
    monkeypatch.setenv("COMFY_MODELS_DIR", str(tmp_path))
    target = tmp_path / "loras" / "style.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"y" * 10)

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.models.download",
            "request_id": "cmd-2",
            "data": {
                "folder": "loras",
                "filename": "style.safetensors",
                # Unreachable source proves no network round-trip happens.
                "source": {"type": "url", "url": "http://127.0.0.1:1/nope.safetensors"},
            },
        })
    assert frames[-1]["type"] == "result"
    assert frames[-1]["data"]["status"] == "exists"
    assert frames[-1]["data"]["size_bytes"] == 10


@pytest.mark.asyncio(loop_scope="function")
async def test_models_download_maps_comfy_type_to_folder(server, fake_comfy, tmp_path, monkeypatch):
    """comfy.checkpoint_file-style type names map to the ComfyUI folder layout."""
    monkeypatch.setenv("COMFY_MODELS_DIR", str(tmp_path))
    fake_comfy.model_files["model.ckpt"] = b"z" * 8

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {
            "type": "comfy.models.download",
            "request_id": "cmd-3",
            "data": {
                "folder": "comfy.checkpoint_file",
                "source": {"type": "url", "url": f"{fake_comfy.base_url}/files/model.ckpt"},
            },
        })
    assert frames[-1]["data"]["status"] == "completed"
    assert (tmp_path / "checkpoints" / "model.ckpt").exists()


@pytest.mark.asyncio(loop_scope="function")
async def test_models_download_rejects_path_traversal(server, tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_MODELS_DIR", str(tmp_path))
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        for folder, filename in (
            ("../evil", "model.bin"),
            ("checkpoints", "../../etc/passwd"),
            ("/abs", "model.bin"),
        ):
            frames = await _request(ws, {
                "type": "comfy.models.download",
                "request_id": f"cmd-trav-{folder}",
                "data": {
                    "folder": folder,
                    "filename": filename,
                    "source": {"type": "url", "url": "http://127.0.0.1:1/x"},
                },
            })
            assert frames[-1]["type"] == "error"
            assert "Unsafe path" in frames[-1]["data"]["error"]
    assert not (tmp_path / "model.bin").exists()


@pytest.mark.asyncio(loop_scope="function")
async def test_models_list_and_delete(server, tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_MODELS_DIR", str(tmp_path))
    (tmp_path / "checkpoints").mkdir(parents=True)
    (tmp_path / "checkpoints" / "a.safetensors").write_bytes(b"a" * 3)
    (tmp_path / "loras" / "sub").mkdir(parents=True)
    (tmp_path / "loras" / "sub" / "b.safetensors").write_bytes(b"b" * 5)
    (tmp_path / "loras" / "partial.safetensors.part").write_bytes(b"junk")

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _request(ws, {"type": "comfy.models.list", "request_id": "cml-1", "data": {}})
        models = frames[-1]["data"]["models"]
        assert {(m["folder"], m["filename"], m["size_bytes"]) for m in models} == {
            ("checkpoints", "a.safetensors", 3),
            ("loras", "sub/b.safetensors", 5),
        }

        frames = await _request(ws, {
            "type": "comfy.models.delete",
            "request_id": "cml-2",
            "data": {"folder": "checkpoints", "filename": "a.safetensors"},
        })
        assert frames[-1]["data"]["deleted"] is True
        assert not (tmp_path / "checkpoints" / "a.safetensors").exists()

        frames = await _request(ws, {
            "type": "comfy.models.delete",
            "request_id": "cml-3",
            "data": {"folder": "checkpoints", "filename": "a.safetensors"},
        })
    assert frames[-1]["data"]["deleted"] is False
