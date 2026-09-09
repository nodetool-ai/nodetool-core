"""Tests for the blender.* bridge handler (headless Blender jobs).

Fake ``blender`` executables (shell scripts behind ``BLENDER_PATH``) stand
in for the real binary everywhere except the end-to-end render test, which
runs the real Blender and the real shipped op scripts and is skipped only
when either is absent.
"""

from __future__ import annotations

import asyncio
import json
import stat
import struct
import sys
from pathlib import Path
from typing import Any

import msgpack
import pytest
import pytest_asyncio
import websockets

import nodetool.worker.blender_handler as bh
from nodetool.worker.blender_handler import (
    BlenderBinary,
    get_blender_info,
    handle_blender_message,
    resolve_blender_binary,
)
from nodetool.worker.protocol import WorkerProtocolServer
from nodetool.worker.server import WorkerServer, start_server

REPO_ROOT = Path(__file__).resolve().parents[2]
SIBLING_OPS = REPO_ROOT.parent / "nodetool" / "packages" / "blender-nodes" / "blender_ops"
REAL_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")

TINY_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-render-bytes"

# A pure-Python stand-in for the shipped op script: proves the worker runs
# exactly the code the client sent, with no Blender needed.
TINY_RUN_JOB = """\
import json
import os

here = os.getcwd()
with open(os.path.join(here, "job.json"), encoding="utf-8") as handle:
    job = json.load(handle)
outputs = job["outputs"]
with open(os.path.join(here, outputs["image"]), "wb") as handle:
    handle.write(b"TINY-SHIPPED-OP")
result = {
    "ok": True,
    "produced": ["image"],
    "stats": {
        "blender_version": "test-shipped",
        "render_seconds": 0.25,
        "seen_op": job["job"]["op"],
    },
}
with open(os.path.join(here, "result.json"), "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"""


def _triangle_glb() -> bytes:
    """One-triangle GLB, mirroring the sibling's createTriangleGlb fixture."""
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{
            "bufferView": 0,
            "componentType": 5126,
            "count": 3,
            "type": "VEC3",
            "min": [0, 0, 0],
            "max": [1, 1, 0],
        }],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions)}],
        "buffers": [{"byteLength": len(positions)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(positions)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<I", len(js)) + b"JSON" + js
    out += struct.pack("<I", len(positions)) + b"BIN\x00" + positions
    return out


def _image_job(**overrides: Any) -> dict[str, Any]:
    job: dict[str, Any] = {
        "version": 1,
        "inputs": {"model": "model.glb"},
        "outputs": {"image": "render.png"},
        "job": {"op": "render_image", "params": {}},
    }
    job.update(overrides)
    return job


def _request(
    job: dict[str, Any],
    blobs: dict[str, bytes],
    inputs: dict[str, str] | None = None,
    timeout: Any = 60,
) -> dict[str, Any]:
    if inputs is None:
        inputs = {"model": "model"}
    return {"job": job, "inputs": inputs, "blobs": blobs, "timeout": timeout}


def _success_blobs(extra: dict[str, bytes] | None = None) -> dict[str, bytes]:
    blobs: dict[str, bytes] = {"model": _triangle_glb(), "run_job.py": b"placeholder"}
    if extra:
        blobs.update(extra)
    return blobs


class RecordingTransport:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def send_msg(self, msg: dict[str, Any]) -> None:
        self.frames.append(msg)


@pytest.fixture(autouse=True)
def clean_blender_env(monkeypatch):
    monkeypatch.delenv("BLENDER_PATH", raising=False)
    bh.reset_blender_binary_cache()
    yield
    bh.reset_blender_binary_cache()


def _write_exe(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _version_script(path: Path, banner: str) -> Path:
    return _write_exe(path, f'#!/bin/sh\necho "{banner}"\nexit 0\n')


SUCCESS_RESULT = {
    "ok": True,
    "produced": ["image"],
    "stats": {"blender_version": "5.2.1", "render_seconds": 0.5, "objects": 1},
}


def _script_with_result(path: Path, result: dict[str, Any], outfile: str, outbytes: str, extra: str = "") -> Path:
    """Fake blender: answer --version, else write result.json + one output file."""
    return _write_exe(
        path,
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "Blender 5.2.1 LTS"; exit 0; fi\n'
        + extra
        + "cat > result.json <<'RESULT_EOF'\n"
        + json.dumps(result)
        + "\nRESULT_EOF\n"
        + f"printf '%s' '{outbytes}' > {outfile}\n"
        + "exit 0\n",
    )


def _success_script(path: Path) -> Path:
    return _script_with_result(path, SUCCESS_RESULT, "render.png", "FAKEPNGDATA")


def _sleeper_script(path: Path, marker_dir: Path) -> Path:
    return _write_exe(
        path,
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "Blender 5.2.1 LTS"; exit 0; fi\n'
        f'echo started > "{marker_dir}/started"\n'
        "sleep 60 &\n"
        "pid=$!\n"
        f'trap \'kill $pid 2>/dev/null; echo termed > "{marker_dir}/termed"; exit 0\' TERM\n'
        "wait $pid\n",
    )


def _progress_script(path: Path) -> Path:
    body = (
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "Blender 5.2.1 LTS"; exit 0; fi\n'
        'echo "Blender 5.2.1 LTS" >&2\n'
        'echo "Fra:1 Mem:12.00M" >&2\n'
        "sleep 0.1\n"
        'echo "Fra:2" >&2\n'
        'echo "a line mentioning Fra: without anchoring" >&2\n'
        "sleep 0.1\n"
        'echo "Fra:3 " >&2\n'
        "cat > result.json <<'RESULT_EOF'\n"
        + json.dumps(SUCCESS_RESULT)
        + "\nRESULT_EOF\nprintf 'X' > render.png\nexit 0\n"
    )
    return _write_exe(path, body)


def _passthrough_script(path: Path) -> Path:
    """A fake blender that runs the SHIPPED run_job.py with system python."""
    return _write_exe(
        path,
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "Blender 5.2.1 LTS"; exit 0; fi\n'
        'script=""\nprev=""\n'
        'for arg in "$@"; do\n'
        '  if [ "$prev" = "--python" ]; then script="$arg"; fi\n'
        '  prev="$arg"\n'
        "done\n"
        f'exec "{sys.executable}" "$script" -- job.json\n',
    )


async def _call(
    data: dict[str, Any],
    msg_type: str = "blender.execute",
    request_id: str = "r1",
    cancel_flags: dict[str, asyncio.Event] | None = None,
) -> list[dict[str, Any]]:
    transport = RecordingTransport()
    await handle_blender_message(
        msg_type=msg_type,
        request_id=request_id,
        data=data,
        transport=transport,
        cancel_flags=cancel_flags if cancel_flags is not None else {},
    )
    return transport.frames


def _terminal(frames: list[dict[str, Any]]) -> dict[str, Any]:
    assert frames, "handler must answer every request with a frame"
    assert frames[-1]["type"] in ("result", "error"), frames[-1]
    return frames[-1]


def _record_scratch(monkeypatch, created: list[Path]):
    import tempfile

    orig = tempfile.mkdtemp

    def spy(*args: Any, **kwargs: Any) -> str:
        path = orig(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", spy)


def _isolate_discovery(monkeypatch, tmp_path: Path) -> None:
    """Hide every real Blender: empty PATH, no well-known locations."""
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(bh, "_well_known_locations", lambda: [])


# --- discovery -----------------------------------------------------------------


def test_enabled_false_with_no_binary(monkeypatch, tmp_path):
    _isolate_discovery(monkeypatch, tmp_path)
    monkeypatch.setenv("BLENDER_PATH", str(tmp_path / "no-such-blender"))
    assert resolve_blender_binary_raises()
    assert get_blender_info() == {"enabled": False}


def resolve_blender_binary_raises() -> bool:
    try:
        resolve_blender_binary()
    except Exception:
        return True
    return False


def test_enabled_false_with_old_binary(monkeypatch, tmp_path):
    _isolate_discovery(monkeypatch, tmp_path)
    monkeypatch.setenv("BLENDER_PATH", str(_version_script(tmp_path / "blender42", "Blender 4.2.9 LTS")))
    assert get_blender_info() == {"enabled": False}


def test_enabled_true_with_supported_binary(monkeypatch, tmp_path):
    _isolate_discovery(monkeypatch, tmp_path)
    monkeypatch.setenv("BLENDER_PATH", str(_version_script(tmp_path / "blender", "Blender 5.2.1 LTS")))
    binary = resolve_blender_binary()
    assert isinstance(binary, BlenderBinary)
    assert binary.version == (5, 2, 1)
    assert get_blender_info() == {"enabled": True, "version": "5.2.1"}


def test_first_runnable_candidate_wins_even_when_old(monkeypatch, tmp_path):
    """An old BLENDER_PATH shadows a newer blender on PATH (TS parity)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _version_script(bindir / "blender", "Blender 5.9.0 LTS")
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setattr(bh, "_well_known_locations", lambda: [])
    monkeypatch.setenv("BLENDER_PATH", str(_version_script(tmp_path / "old", "Blender 4.2.0 LTS")))
    assert get_blender_info() == {"enabled": False}


def test_blender_binary_reports_path_and_version(monkeypatch, tmp_path):
    _isolate_discovery(monkeypatch, tmp_path)
    script = _version_script(tmp_path / "blender", "Blender 5.3.0 LTS")
    monkeypatch.setenv("BLENDER_PATH", str(script))
    binary = resolve_blender_binary()
    assert binary.path == str(script)
    assert binary.version == (5, 3, 0)


# --- request validation ----------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_message_type_is_an_error():
    frames = await _call({}, msg_type="blender.frobnicate")
    terminal = _terminal(frames)
    assert terminal["type"] == "error"
    assert "blender.frobnicate" in terminal["data"]["error"]


@pytest.mark.asyncio
async def test_missing_job_is_an_error():
    frames = await _call({"inputs": {}, "blobs": {}, "timeout": 5})
    terminal = _terminal(frames)
    assert terminal["type"] == "error"
    assert "job" in terminal["data"]["error"]


@pytest.mark.asyncio
async def test_non_numeric_timeout_is_an_error():
    frames = await _call(_request(_image_job(), _success_blobs(), timeout="600"))
    terminal = _terminal(frames)
    assert terminal["type"] == "error"
    assert "timeout" in terminal["data"]["error"]


@pytest.mark.asyncio
async def test_missing_entry_point_is_an_error():
    blobs = {"model": _triangle_glb()}
    frames = await _call(_request(_image_job(), blobs))
    terminal = _terminal(frames)
    assert terminal["type"] == "error"
    assert "run_job.py" in terminal["data"]["error"]


@pytest.mark.asyncio
async def test_unsafe_output_filename_is_an_error():
    job = _image_job(outputs={"image": "../evil.png"})
    frames = await _call(_request(job, _success_blobs()))
    terminal = _terminal(frames)
    assert terminal["type"] == "error"
    assert "unsafe" in terminal["data"]["error"]


@pytest.mark.asyncio
async def test_traversal_op_blob_key_is_an_error():
    blobs = _success_blobs({"../escape.py": b"evil"})
    frames = await _call(_request(_image_job(), blobs))
    terminal = _terminal(frames)
    assert terminal["type"] == "error"
    assert "safe relative path" in terminal["data"]["error"]


@pytest.mark.asyncio
async def test_unknown_version_is_bad_job(monkeypatch):
    created: list[Path] = []
    _record_scratch(monkeypatch, created)
    job = _image_job(version=999)
    frames = await _call(_request(job, _success_blobs()))
    terminal = _terminal(frames)
    assert terminal["type"] == "result"
    assert terminal["data"]["ok"] is False
    assert terminal["data"]["error"]["code"] == "bad_job"
    assert created == []


# --- execute ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_returns_declared_outputs_with_honest_sizes(monkeypatch, tmp_path):
    monkeypatch.setenv("BLENDER_PATH", str(_success_script(tmp_path / "blender")))
    created: list[Path] = []
    _record_scratch(monkeypatch, created)
    frames = await _call(_request(_image_job(), _success_blobs()))
    terminal = _terminal(frames)
    assert terminal["type"] == "result"
    data = terminal["data"]
    assert data["ok"] is True
    assert data["produced"] == ["image"]
    assert data["stats"]["blender_version"] == "5.2.1"
    assert data["blobs"]["image"] == b"FAKEPNGDATA"
    assert data["sizes"] == {"image": len(b"FAKEPNGDATA")}
    assert created != []
    assert all(not p.exists() for p in created)


@pytest.mark.asyncio
async def test_worker_runs_the_shipped_op_script(monkeypatch, tmp_path):
    """The staged run_job.py is what executes — no vendored copy exists."""
    monkeypatch.setenv("BLENDER_PATH", str(_passthrough_script(tmp_path / "blender")))
    blobs = {"model": _triangle_glb(), "run_job.py": TINY_RUN_JOB.encode("utf-8")}
    frames = await _call(_request(_image_job(), blobs))
    data = _terminal(frames)["data"]
    assert data["ok"] is True
    assert data["blobs"]["image"] == b"TINY-SHIPPED-OP"
    assert data["sizes"] == {"image": len(b"TINY-SHIPPED-OP")}
    assert data["stats"]["seen_op"] == "render_image"


@pytest.mark.asyncio
async def test_undeclared_output_ignored_and_paths_never_opened(monkeypatch, tmp_path):
    """A produced name the job did not declare brings no bytes; a path in
    result.json is never read. File reads are recorded so the test fails if
    the handler opens the sentinel."""
    sentinel = tmp_path / "secret.bin"
    sentinel.write_bytes(b"top-secret-bytes")

    import builtins
    import pathlib

    opened: list[str] = []
    orig_open = builtins.open
    orig_read_bytes = pathlib.Path.read_bytes
    orig_read_text = pathlib.Path.read_text

    def spy_open(file, *args, **kwargs):
        opened.append(str(file))
        return orig_open(file, *args, **kwargs)

    def spy_read_bytes(self):
        opened.append(str(self))
        return orig_read_bytes(self)

    def spy_read_text(self, *args, **kwargs):
        opened.append(str(self))
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(pathlib.Path, "read_bytes", spy_read_bytes)
    monkeypatch.setattr(pathlib.Path, "read_text", spy_read_text)

    evil_result = {
        "ok": True,
        "produced": ["image", "evil"],
        "stats": {
            "blender_version": "5.2.1",
            "render_seconds": 0.5,
            "trace": str(sentinel),
        },
    }
    monkeypatch.setenv(
        "BLENDER_PATH",
        str(_script_with_result(tmp_path / "evil-blender", evil_result, "render.png", "FAKEPNGDATA")),
    )
    frames = await _call(_request(_image_job(), _success_blobs()))
    data = _terminal(frames)["data"]
    assert data["ok"] is True
    assert "evil" not in data["blobs"]
    assert "evil" not in data["sizes"]
    assert data["blobs"]["image"] == b"FAKEPNGDATA"
    assert not any(str(sentinel) in entry for entry in opened)


@pytest.mark.asyncio
async def test_op_failure_passes_code_through(monkeypatch, tmp_path):
    opfail = {"ok": False, "error": {"code": "render_failed", "message": "Eevee complained"}}
    body = (
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "Blender 5.2.1 LTS"; exit 0; fi\n'
        "cat > result.json <<'RESULT_EOF'\n"
        + json.dumps(opfail)
        + "\nRESULT_EOF\nexit 64\n"
    )
    monkeypatch.setenv("BLENDER_PATH", str(_write_exe(tmp_path / "blender", body)))
    frames = await _call(_request(_image_job(), _success_blobs()))
    terminal = _terminal(frames)
    assert terminal["type"] == "result"
    assert terminal["data"]["ok"] is False
    assert terminal["data"]["error"]["code"] == "render_failed"
    assert "Eevee complained" in terminal["data"]["error"]["message"]


@pytest.mark.asyncio
async def test_crash_without_result_is_bad_result(monkeypatch, tmp_path):
    script = (
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "Blender 5.2.1 LTS"; exit 0; fi\n'
        'echo "font not found, aborting" >&2\n'
        "exit 3\n"
    )
    monkeypatch.setenv("BLENDER_PATH", str(_write_exe(tmp_path / "blender", script)))
    created: list[Path] = []
    _record_scratch(monkeypatch, created)
    frames = await _call(_request(_image_job(), _success_blobs()))
    terminal = _terminal(frames)
    assert terminal["type"] == "result"
    assert terminal["data"]["ok"] is False
    assert terminal["data"]["error"]["code"] == "bad_result"
    assert "font not found, aborting" in terminal["data"]["error"]["message"]
    assert all(not p.exists() for p in created)


@pytest.mark.asyncio
async def test_timeout_kills_blender_and_cleans_up(monkeypatch, tmp_path):
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    monkeypatch.setenv("BLENDER_PATH", str(_sleeper_script(tmp_path / "blender", marker_dir)))
    created: list[Path] = []
    _record_scratch(monkeypatch, created)
    frames = await _call(_request(_image_job(), _success_blobs(), timeout=1))
    terminal = _terminal(frames)
    assert terminal["type"] == "result"
    assert terminal["data"]["ok"] is False
    assert terminal["data"]["error"]["code"] == "timeout"
    assert (marker_dir / "termed").is_file()
    assert all(not p.exists() for p in created)


@pytest.mark.asyncio
async def test_cancel_kills_blender_and_cleans_up(monkeypatch, tmp_path):
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    monkeypatch.setenv("BLENDER_PATH", str(_sleeper_script(tmp_path / "blender", marker_dir)))
    created: list[Path] = []
    _record_scratch(monkeypatch, created)
    cancel_flags: dict[str, asyncio.Event] = {}
    task = asyncio.create_task(
        _call(_request(_image_job(), _success_blobs(), timeout=120), request_id="cancel-1", cancel_flags=cancel_flags)
    )
    try:
        for _ in range(100):
            if (marker_dir / "started").is_file():
                break
            await asyncio.sleep(0.1)
        assert (marker_dir / "started").is_file(), "fake blender never started"
        cancel_flags["cancel-1"].set()
        frames = await asyncio.wait_for(task, timeout=30)
    finally:
        if not task.done():
            task.cancel()
    terminal = _terminal(frames)
    assert terminal["type"] == "result"
    assert terminal["data"]["ok"] is False
    assert terminal["data"]["error"]["code"] == "cancelled"
    assert (marker_dir / "termed").is_file()
    assert "cancel-1" not in cancel_flags
    assert all(not p.exists() for p in created)


@pytest.mark.asyncio
async def test_fra_lines_become_progress_events_with_animation_total(monkeypatch, tmp_path):
    monkeypatch.setenv("BLENDER_PATH", str(_progress_script(tmp_path / "blender")))
    job = _image_job(job={"op": "render_animation", "params": {"frame_start": 1, "frame_end": 3}})
    frames = await _call(_request(job, _success_blobs()))
    events = [f["data"] for f in frames if f["type"] == "blender.event"]
    assert [(e["frame"], e["total"]) for e in events] == [(1, 3), (2, 3), (3, 3)]
    assert all(e["event"] == "progress" for e in events)
    assert _terminal(frames)["type"] == "result"


@pytest.mark.asyncio
async def test_fra_total_falls_back_to_frame_without_animation_range(monkeypatch, tmp_path):
    monkeypatch.setenv("BLENDER_PATH", str(_progress_script(tmp_path / "blender")))
    frames = await _call(_request(_image_job(), _success_blobs()))
    events = [f["data"] for f in frames if f["type"] == "blender.event"]
    assert [(e["frame"], e["total"]) for e in events] == [(1, 1), (2, 2), (3, 3)]


# --- dispatcher wiring -----------------------------------------------------------


@pytest_asyncio.fixture(loop_scope="function")
async def server():
    worker = WorkerServer()
    host, port, stop_event, task = await start_server(host="127.0.0.1", port=0, worker=worker)
    yield host, port
    stop_event.set()
    await task


async def _ws_request(ws, msg: dict) -> list[dict]:
    await ws.send(msgpack.packb(msg))
    frames = []
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=60)
        frame = msgpack.unpackb(raw, raw=False)
        frames.append(frame)
        if frame["type"] in ("result", "error"):
            return frames


@pytest.mark.asyncio(loop_scope="function")
async def test_worker_status_carries_blender_block(server, monkeypatch, tmp_path):
    monkeypatch.setenv("BLENDER_PATH", str(_version_script(tmp_path / "blender", "Blender 5.2.1 LTS")))
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _ws_request(ws, {"type": "worker.status", "request_id": "st-1", "data": {}})
    status = frames[-1]["data"]
    assert frames[-1]["type"] == "result"
    assert status["blender"]["enabled"] is True
    assert status["blender"]["version"] == "5.2.1"


@pytest.mark.asyncio(loop_scope="function")
async def test_execute_end_to_end_over_websocket(server, monkeypatch, tmp_path):
    """The blender.* dispatcher branch answers through the real transport,
    with input bytes crossing msgpack as binary blobs."""
    monkeypatch.setenv("BLENDER_PATH", str(_success_script(tmp_path / "blender")))
    host, port = server
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        frames = await _ws_request(
            ws,
            {
                "type": "blender.execute",
                "request_id": "ws-1",
                "data": _request(_image_job(), _success_blobs()),
            },
        )
    terminal = frames[-1]
    assert terminal["type"] == "result"
    assert terminal["data"]["ok"] is True
    assert terminal["data"]["blobs"]["image"] == b"FAKEPNGDATA"
    assert terminal["data"]["sizes"] == {"image": len(b"FAKEPNGDATA")}


@pytest.mark.asyncio
async def test_blender_status_message():
    frames = await _call({}, msg_type="blender.status", request_id="bs-1")
    assert frames[-1]["type"] == "result"
    assert isinstance(frames[-1]["data"]["enabled"], bool)


@pytest.mark.asyncio
async def test_worker_frames_validate_against_bridge_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from pathlib import Path as _Path

    schema = json.loads((_Path(__file__).parent / "fixtures" / "bridge-frames.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)

    server = WorkerProtocolServer(transport_name="test")
    transport = RecordingTransport()
    await server.dispatch({"type": "worker.status", "request_id": "w1", "data": {}}, transport)
    for frame in transport.frames:
        errors = list(validator.iter_errors(frame))
        assert not errors, [e.message for e in errors]

    good_result = {
        "type": "result",
        "request_id": "b1",
        "data": {
            "ok": True,
            "produced": ["image"],
            "stats": {"blender_version": "5.2.1", "render_seconds": 1.5},
            "sizes": {"image": 11},
            "blobs": {"image": b"bytes-go-here"},
        },
    }
    assert not list(validator.iter_errors(good_result))

    event = {"type": "blender.event", "request_id": "b1", "data": {"event": "progress", "frame": 2, "total": 3}}
    assert not list(validator.iter_errors(event))

    bad_event = {"type": "blender.event", "request_id": "b1", "data": {"event": "progress", "frame": "x"}}
    assert list(validator.iter_errors(bad_event)), "a malformed blender.event must fail validation"


# --- real render -------------------------------------------------------------------


def _real_ops_blobs() -> dict[str, bytes]:
    """Every file of the sibling's blender_ops/ keyed by relative path."""
    blobs: dict[str, bytes] = {}
    for path in sorted(SIBLING_OPS.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        blobs[path.relative_to(SIBLING_OPS).as_posix()] = path.read_bytes()
    return blobs


def _render_image_params() -> dict[str, Any]:
    return {
        "camera_mode": "orbit",
        "azimuth": 45,
        "elevation": 25,
        "fov": 35,
        "zoom": 1,
        "lighting": "studio",
        "light_intensity": 1,
        "background_color": "#102030",
        "transparent": False,
        "engine": "eevee",
        "samples": 8,
        "denoise": True,
        "resolution_percentage": 100,
        "width": 64,
        "height": 64,
    }


def _parse_png_size(png: bytes) -> tuple[int, int]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "output is not a PNG"
    (length,) = struct.unpack(">I", png[8:12])
    assert png[12:16] == b"IHDR", "first PNG chunk is not IHDR"
    assert length == 13
    width, height = struct.unpack(">II", png[16:24])
    return width, height


needs_real_blender = pytest.mark.skipif(
    not REAL_BLENDER.is_file() or not (SIBLING_OPS / "run_job.py").is_file(),
    reason="needs installed Blender and the sibling blender_ops/ checkout",
)


@pytest.mark.asyncio
@needs_real_blender
async def test_real_render_end_to_end(monkeypatch, tmp_path):
    """A real triangle GLB through real Blender and the real shipped ops."""
    monkeypatch.setenv("BLENDER_PATH", str(REAL_BLENDER))
    ops = _real_ops_blobs()
    assert len(ops) > 5, f"expected the full op tree, got {sorted(ops)}"
    assert "run_job.py" in ops and "ops/render_image.py" in ops
    job = _image_job(job={"op": "render_image", "params": _render_image_params()})
    blobs = {"model": _triangle_glb(), **ops}
    frames = await _call(_request(job, blobs, timeout=300), request_id="real-1")
    data = _terminal(frames)["data"]
    assert data["ok"] is True, data.get("error")
    assert data["produced"] == ["image"]
    png = data["blobs"]["image"]
    assert data["sizes"] == {"image": len(png)}
    assert _parse_png_size(png) == (64, 64)
    assert len(png) > 500
    idat = png.split(b"IDAT")
    assert len(set(b"".join(idat[1:]))) > 8, "render is uniform, the triangle is missing"
    assert data["stats"]["blender_version"].startswith("5.2")
    assert isinstance(data["stats"]["render_seconds"], float)
