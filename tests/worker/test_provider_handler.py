"""Tests for the provider bridge handler."""

import asyncio
import json
import sys
from pathlib import Path

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
        assert progress["type"] == "progress", progress
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

@pytest.mark.asyncio(loop_scope="function")
async def test_command_backed_reference_video_stages_ordered_media(server, monkeypatch, tmp_path):
    """Reference media is staged with detected suffixes and ordered path lists."""
    adapter = tmp_path / "reference_adapter.py"
    output = tmp_path / "generated.mp4"
    paths_file = tmp_path / "paths.json"
    adapter.write_text(
        r"""
import json
import pathlib
import sys
request = json.loads(sys.stdin.readline())
assert request["operation"] == "reference_to_video"
images = request["reference_image_paths"]
videos = request["reference_video_paths"]
assert [pathlib.Path(p).read_bytes() for p in images] == [bytes.fromhex("89504e470d0a1a0a696d616765"), bytes.fromhex("ffd8ff70686f746f")]
assert pathlib.Path(videos[0]).read_bytes() == bytes.fromhex("1a45dfa3766964656f")
assert pathlib.Path(images[0]).suffix == ".png"
assert pathlib.Path(images[1]).suffix == ".jpg"
assert pathlib.Path(videos[0]).suffix == ".webm"
print(json.dumps({"type": "progress", "data": {"progress": 50}}), flush=True)
pathlib.Path(sys.argv[2]).write_text(json.dumps(images + videos))
path = pathlib.Path(sys.argv[1])
path.write_bytes(b"video-data")
print(json.dumps({"type": "result", "data": {"path": str(path)}}), flush=True)
""".strip()
        + "\n"
    )
    monkeypatch.setenv(
        "NODETOOL_PROVIDER_ADAPTER_COMMAND_EXTERNAL",
        f"{sys.executable} {adapter} {output} {paths_file}",
    )
    monkeypatch.setenv(
        "NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_EXTERNAL",
        "reference_to_video",
    )

    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        await ws.send(msgpack.packb({
            "type": "provider.reference_to_video",
            "request_id": "reference",
            "data": {
                "provider": "external",
                "reference_images": [bytes.fromhex("89504e470d0a1a0a696d616765"), bytes.fromhex("ffd8ff70686f746f")],
                "reference_videos": [bytes.fromhex("1a45dfa3766964656f")],
                "params": {"model": "demo", "prompt": "keep identity"},
                "blob_transfer": "chunked-v1",
            },
        }))
        progress = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        assert progress["type"] == "progress", progress
        assert progress["data"] == {"progress": 50}
        assert msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)["type"] == "blob.start"
        assert msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)["type"] == "blob.chunk"
        assert msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)["type"] == "blob.end"
        result = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 5), raw=False)
        assert result["type"] == "result"
        assert result["data"]["blobs"] == {}
    staged_paths = [Path(path) for path in json.loads(paths_file.read_text())]
    assert staged_paths
    assert all(not path.exists() for path in staged_paths)

@pytest.mark.asyncio(loop_scope="function")
async def test_reference_validation_happens_before_adapter(monkeypatch, tmp_path):
    """Malformed and oversized references never start the configured adapter."""
    from nodetool.worker import provider_handler

    marker = tmp_path / "started"
    adapter = tmp_path / "adapter.py"
    adapter.write_text(f"import pathlib; pathlib.Path({str(marker)!r}).write_text('yes')\n")
    monkeypatch.setenv("NODETOOL_PROVIDER_ADAPTER_COMMAND_EXTERNAL", f"{sys.executable} {adapter}")
    monkeypatch.setenv("NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_EXTERNAL", "reference_to_video")

    async def progress(_request_id, _data):
        return None

    with pytest.raises(ValueError, match="must be arrays"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": b"not-array", "reference_videos": []},
            "malformed-arrays", {}, progress,
        )
    with pytest.raises(ValueError, match="non-empty image"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": ["not-binary"], "reference_videos": []},
            "non-binary", {}, progress,
        )
    monkeypatch.setenv("NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_EXTERNAL", "image_to_video")
    with pytest.raises(ValueError, match="does not support reference_to_video"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": [], "reference_videos": []},
            "unsupported-capability", {}, progress,
        )
    monkeypatch.setenv("NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_EXTERNAL", "reference_to_video")

    with pytest.raises(ValueError, match="at least one"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": [], "reference_videos": []},
            "empty", {}, progress,
        )
    with pytest.raises(ValueError, match="non-empty image"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": [b""], "reference_videos": []},
            "empty-buffer", {}, progress,
        )
    with pytest.raises(ValueError, match="unsupported image media format"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": [b"unknown"], "reference_videos": []},
            "unknown-format", {}, progress,
        )
    with pytest.raises(ValueError, match="unsupported image media format"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": [bytes.fromhex("1a45dfa3")], "reference_videos": []},
            "wrong-kind-image", {}, progress,
        )
    with pytest.raises(ValueError, match="unsupported video media format"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": [], "reference_videos": [bytes.fromhex("89504e470d0a1a0a")]},
            "wrong-kind-video", {}, progress,
        )
    with pytest.raises(ValueError, match="unsupported video media format"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": [], "reference_videos": [b"....ftypavif"]},
            "avif-video", {}, progress,
        )
    with pytest.raises(ValueError, match="maximum"):
        monkeypatch.setattr(provider_handler, "_REFERENCE_INPUT_LIMIT", 4)
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": [b"12345"], "reference_videos": []},
            "too-large", {}, progress,
        )
    assert not marker.exists()


@pytest.mark.asyncio(loop_scope="function")
async def test_reference_cancel_during_staging_removes_files(monkeypatch, tmp_path):
    """Cancellation observed after an input write prevents adapter startup."""
    from nodetool.worker import provider_handler

    marker = tmp_path / "started"
    adapter = tmp_path / "adapter.py"
    adapter.write_text(f"import pathlib; pathlib.Path({str(marker)!r}).write_text('yes')\n")
    monkeypatch.setenv("NODETOOL_PROVIDER_ADAPTER_COMMAND_EXTERNAL", f"{sys.executable} {adapter}")
    monkeypatch.setenv("NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_EXTERNAL", "reference_to_video")
    original_to_thread = asyncio.to_thread
    flags: dict[str, asyncio.Event] = {}
    staged_paths: list[Path] = []

    async def staged_to_thread(func, *args):
        result = await original_to_thread(func, *args)
        if isinstance(getattr(func, "__self__", None), Path):
            staged_paths.append(func.__self__)
        if flags.get("during-stage") is not None:
            flags["during-stage"].set()
        return result

    monkeypatch.setattr(provider_handler.asyncio, "to_thread", staged_to_thread)
    flags["during-stage"] = asyncio.Event()
    with pytest.raises(RuntimeError, match="cancelled"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": [bytes.fromhex("89504e470d0a1a0a696d616765")], "reference_videos": []},
            "during-stage", flags, lambda *_args: None,
        )
    assert not marker.exists()
    assert "during-stage" not in flags
    assert staged_paths
    assert all(not path.exists() for path in staged_paths)


@pytest.mark.asyncio(loop_scope="function")
async def test_reference_adapter_failure_cleans_staged_files(monkeypatch, tmp_path):
    from nodetool.worker import provider_handler
    path_file = tmp_path / "path"
    adapter = tmp_path / "failing.py"
    adapter.write_text("""import json, pathlib, sys
request = json.loads(sys.stdin.readline())
pathlib.Path(sys.argv[1]).write_text(request[\"reference_image_paths\"][0])
raise SystemExit(3)
""")
    monkeypatch.setenv("NODETOOL_PROVIDER_ADAPTER_COMMAND_EXTERNAL", f"{sys.executable} {adapter} {path_file}")
    monkeypatch.setenv("NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_EXTERNAL", "reference_to_video")
    with pytest.raises(RuntimeError, match="Provider adapter failed"):
        await provider_handler._handle_adapter_media(
            "provider.reference_to_video",
            {"provider": "external", "reference_images": [bytes.fromhex("89504e470d0a1a0a696d616765")], "reference_videos": []},
            "adapter-failure", {}, lambda *_args: None,
        )
    assert path_file.exists()
    assert not Path(path_file.read_text()).exists()
