"""Tests for the provider bridge handler."""

import asyncio
import sys

import msgpack
import pytest
import pytest_asyncio
import websockets

from nodetool.worker.provider_handler import _text_to_audio_kwargs
from nodetool.worker.server import WorkerServer, start_server


def test_text_to_audio_bridge_envelope_is_normalized():
    """Camel-case bridge params become provider-compatible keyword arguments."""
    context = object()
    assert _text_to_audio_kwargs(
        {
            "params": {
                "prompt": "cinematic score",
                "model": "music-model",
                "lyrics": "hello",
                "durationSeconds": 12,
                "guidanceScale": 3.5,
                "numInferenceSteps": 20,
                "seed": 7,
            }
        },
        context,
    ) == {
        "prompt": "cinematic score",
        "model": "music-model",
        "lyrics": "hello",
        "audio_duration": 12,
        "guidance_scale": 3.5,
        "num_inference_steps": 20,
        "seed": 7,
        "context": context,
    }


@pytest_asyncio.fixture(loop_scope="function")
async def server():
    """Start server on random port, yield (host, port), then shut down."""
    worker = WorkerServer()
    host, port, stop_event, task = await start_server(
        host="127.0.0.1", port=0, worker=worker,
    )
    yield host, port
    stop_event.set()
    await task


@pytest.mark.asyncio(loop_scope="function")
async def test_provider_list(server):
    """provider.list returns available providers (may be empty in test env)."""
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "provider.list",
            "request_id": "pl-1",
            "data": {},
        }))
        # First provider discovery imports optional ML stacks on cold CI hosts.
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "result"
        assert resp["request_id"] == "pl-1"
        assert "providers" in resp["data"]
        assert isinstance(resp["data"]["providers"], list)
        for provider in resp["data"]["providers"]:
            assert provider["access"] == "in_process"
            assert provider["display_name"]


@pytest.mark.asyncio(loop_scope="function")
async def test_provider_unknown_type(server):
    """Unknown provider.X message returns error."""
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "provider.nonexistent",
            "request_id": "pu-1",
            "data": {},
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "error"
        assert "Unknown provider message type" in resp["data"]["error"]


@pytest.mark.asyncio(loop_scope="function")
async def test_provider_models_invalid_provider(server):
    """provider.models with unknown provider returns error."""
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "provider.models",
            "request_id": "pm-1",
            "data": {"provider": "nonexistent_provider", "model_type": "language"},
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        resp = msgpack.unpackb(raw, raw=False)
        assert resp["type"] == "error"
        assert resp["request_id"] == "pm-1"


@pytest.mark.asyncio(loop_scope="function")
async def test_command_backed_video_provider(server, monkeypatch, tmp_path):
    """An image-owned interpreter can expose models and video generation."""
    adapter = tmp_path / "provider_adapter.py"
    output = tmp_path / "generated.mp4"
    adapter.write_text(
        """
import json
import pathlib
import sys

request = json.loads(sys.stdin.readline())
operation = request["operation"]
if operation == "models":
    data = {"models": [{"id": "demo", "name": "Demo", "provider": "external"}]}
else:
    if operation == "image_to_video":
        assert pathlib.Path(request["image_path"]).read_bytes() == b"image-data"
    print(json.dumps({"type": "progress", "data": {"progress": 50}}), flush=True)
    path = pathlib.Path(sys.argv[1])
    path.write_bytes(b"video-data")
    data = {"path": str(path)}
print(json.dumps({"type": "result", "data": data}), flush=True)
""".strip()
        + "\n"
    )
    monkeypatch.setenv(
        "NODETOOL_PROVIDER_ADAPTER_COMMAND_EXTERNAL",
        f"{sys.executable} {adapter} {output}",
    )
    monkeypatch.setenv(
        "NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_EXTERNAL",
        "text_to_video,image_to_video",
    )
    monkeypatch.setenv("NODETOOL_PROVIDER_ADAPTER_DISPLAY_NAME_EXTERNAL", "External Video")

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(
            msgpack.packb(
                {"type": "provider.list", "request_id": "list", "data": {}}
            )
        )
        listed = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        external = next(p for p in listed["data"]["providers"] if p["id"] == "external")
        assert external == {
            "id": "external",
            "capabilities": ["image_to_video", "text_to_video"],
            "required_secrets": [],
            "access": "in_process",
            "display_name": "External Video",
        }

        await ws.send(
            msgpack.packb(
                {
                    "type": "provider.models",
                    "request_id": "models",
                    "data": {"provider": "external", "model_type": "video"},
                }
            )
        )
        models = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        assert models["data"]["models"][0]["id"] == "demo"

        await ws.send(
            msgpack.packb(
                {
                    "type": "provider.image_to_video",
                    "request_id": "video",
                    "data": {
                        "provider": "external",
                        "image": b"image-data",
                        "params": {"model": "demo", "prompt": "move"},
                        "blob_transfer": "chunked-v1",
                    },
                }
            )
        )
        progress = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        blob_start = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        blob_chunk = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        blob_end = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        result = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        assert progress["type"] == "progress"
        assert progress["data"] == {"progress": 50}
        assert blob_start["type"] == "blob.start"
        assert blob_start["data"] == {"name": "video", "size": 10}
        assert blob_chunk["type"] == "blob.chunk"
        assert blob_chunk["data"]["bytes"] == b"video-data"
        assert blob_end["type"] == "blob.end"
        assert result["type"] == "result"
        assert result["data"]["blobs"] == {}


@pytest.mark.parametrize(
    ("message_type", "capability", "blob_name", "extra_data"),
    [
        ("provider.text_to_image", "text_to_image", "image", {}),
        (
            "provider.image_to_image",
            "image_to_image",
            "image",
            {"image": b"image-data"},
        ),
        ("provider.text_to_audio", "text_to_audio", "audio", {}),
        (
            "provider.tts_encoded",
            "text_to_speech_encoded",
            "audio",
            {"params": {"referenceAudio": b"reference-audio"}},
        ),
    ],
)
@pytest.mark.asyncio(loop_scope="function")
async def test_command_backed_provider_media_operations(
    server,
    monkeypatch,
    tmp_path,
    message_type,
    capability,
    blob_name,
    extra_data,
):
    """External adapters can return image and encoded audio files."""
    adapter = tmp_path / "media_provider.py"
    output = tmp_path / "generated.bin"
    adapter.write_text(
        """
import json
import pathlib
import sys

request = json.loads(sys.stdin.readline())
if request["operation"].startswith("image_to_"):
    assert pathlib.Path(request["image_path"]).read_bytes() == b"image-data"
if request["operation"] == "tts_encoded":
    assert pathlib.Path(request["reference_audio_path"]).read_bytes() == b"reference-audio"
    assert "referenceAudio" not in request["params"]
path = pathlib.Path(sys.argv[1])
path.write_bytes(request["operation"].encode())
print(json.dumps({"type": "result", "data": {"path": str(path)}}), flush=True)
""".strip()
        + "\n"
    )
    monkeypatch.setenv(
        "NODETOOL_PROVIDER_ADAPTER_COMMAND_EXTERNAL",
        f"{sys.executable} {adapter} {output}",
    )
    monkeypatch.setenv(
        "NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_EXTERNAL", capability
    )

    data = {
        "provider": "external",
        "params": {"model": "demo", "prompt": "test"},
        "blob_transfer": "chunked-v1",
        **extra_data,
    }
    if "params" in extra_data:
        data["params"] = {
            "model": "demo",
            "text": "hello",
            **extra_data["params"],
        }

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(
            msgpack.packb(
                {
                    "type": message_type,
                    "request_id": "media",
                    "data": data,
                }
            )
        )
        frames = []
        while not frames or frames[-1]["type"] != "result":
            frames.append(
                msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
            )

    start = next(frame for frame in frames if frame["type"] == "blob.start")
    chunk = next(frame for frame in frames if frame["type"] == "blob.chunk")
    assert start["data"]["name"] == blob_name
    assert chunk["data"]["bytes"] == message_type.removeprefix("provider.").encode()


@pytest.mark.asyncio(loop_scope="function")
async def test_command_backed_provider_cancel_stops_adapter(server, monkeypatch, tmp_path):
    adapter = tmp_path / "slow_provider.py"
    adapter.write_text(
        """
import json
import sys
import time

json.loads(sys.stdin.readline())
print(json.dumps({"type": "progress", "data": {"progress": 1}}), flush=True)
time.sleep(30)
""".strip()
        + "\n"
    )
    monkeypatch.setenv(
        "NODETOOL_PROVIDER_ADAPTER_COMMAND_EXTERNAL", f"{sys.executable} {adapter}"
    )
    monkeypatch.setenv(
        "NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_EXTERNAL", "text_to_video"
    )

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(
            msgpack.packb(
                {
                    "type": "provider.text_to_video",
                    "request_id": "cancel-me",
                    "data": {
                        "provider": "external",
                        "params": {"model": "demo", "prompt": "move"},
                    },
                }
            )
        )
        progress = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        assert progress["type"] == "progress"
        await ws.send(
            msgpack.packb(
                {"type": "cancel", "request_id": "cancel-me", "data": {}}
            )
        )
        error = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        assert error["type"] == "error"
        assert error["data"]["error"] == "Provider operation cancelled"
