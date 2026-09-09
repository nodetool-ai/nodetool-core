"""Handle blender.* bridge messages: run headless Blender jobs in a scratch dir.

A GPU worker answers ``blender.execute`` the way it already answers
``comfy.execute``: the request carries the verbatim ``BlenderJob``, an
``inputs`` manifest mapping each logical input name to its bridge blob key,
the input bytes under those keys, and a worker-side ``timeout`` in seconds::

    { job, inputs: {logicalName: blobKey}, blobs: {blobKey: bytes}, timeout }

The op script travels with the job inside ``blobs``. Every blob key the
``inputs`` manifest does not name is staged into the per-request scratch
directory as a relative path, and the entry point ``run_job.py`` among them
is what Blender runs::

    blender -b --factory-startup --disable-autoexec --python-exit-code 64
        --python <scratch>/run_job.py -- job.json

No copy of the op script is vendored here on purpose: the worker runs
exactly the ops the client shipped, so a worker image and a NodeTool
release cannot drift apart. On the security of that choice: the bridge
client is the trusted NodeTool server (Bearer [REDACTED] when configured),
and the worker is a provisioned container the operator reaps per run — the
same trust boundary ``execute`` already assumes, since a workflow graph
reaches the worker's filesystem and network through ordinary nodes. The
D4 invariant still holds worker-side: only paths the job declared are ever
opened, and a path appearing anywhere in ``result.json`` is never read, so
a buggy or compromised op cannot turn result collection into an arbitrary
file read. Do not mix inputs from untrusted third parties into one shared
worker without that container boundary.

Mirrors ``comfy_handler.handle_comfy_message``: transport-agnostic, it only
needs a transport exposing an async ``send_msg``, so the same code path
serves both the WebSocket and stdio workers. Progress (``Fra:<n>`` lines on
stderr) streams back as dedicated ``blender.event`` frames
(``{event: "progress", frame, total}``), and the terminal ``result`` frame
carries ``{ok, produced, stats, sizes, blobs}`` with output bytes keyed by
the job's logical output names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

#: Version of the job contract this handler speaks (mirrors BLENDER_JOB_VERSION).
BLENDER_JOB_VERSION = 1

#: Minimum supported Blender (mirrors BLENDER_MIN_VERSION): the ops use
#: ``scene.compositing_node_group`` and ``image_settings.media_type``,
#: neither of which exists below 5.2.
BLENDER_MIN_VERSION: tuple[int, int, int] = (5, 2, 0)

#: Exit code ``run_job.py`` uses for "the script raised" (D5).
PYTHON_EXIT_CODE = 64

#: The shipped op script entry point: the blob keyed with this name is what
#: Blender runs. Fixed by D5, not negotiated per request.
ENTRY_POINT = "run_job.py"

#: A bare file name: no separator, no ``..``, no leading dot (jobFileNameSchema).
_JOB_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Worker-side wall clock in seconds when the request names no timeout.
DEFAULT_TIMEOUT = 600.0

#: Grace between SIGTERM and SIGKILL when stopping the child.
_KILL_GRACE = 5.0

#: Last bytes of stderr carried on a ``bad_result`` (mirrors STDERR_TAIL_BYTES).
_STDERR_TAIL_BYTES = 4 * 1024

#: Cap on buffered stderr text; progress-heavy animation renders print long.
_STDERR_BUFFER_CAP = 32 * 1024

#: Env keys the Blender child inherits (mirrors ALLOWED_ENV_KEYS). The
#: ``BLENDER_USER_*`` redirects below join this list per run.
ALLOWED_ENV_KEYS = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "SYSTEMROOT",
    "CUDA_VISIBLE_DEVICES",
)

#: ``Fra:<n>`` lines Blender prints on stderr during animation renders.
_FRA_RE = re.compile(r"^Fra:(\d+)\b")


class BlenderError(Exception):
    """A blender.* request failed before or around the Blender run."""


@dataclass
class BlenderBinary:
    path: str
    version: tuple[int, int, int]


# --- discovery ---------------------------------------------------------------


def _well_known_locations() -> list[str]:
    if sys.platform == "darwin":
        return ["/Applications/Blender.app/Contents/MacOS/Blender"]
    if sys.platform == "win32":
        roots = [os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)")]
        found: list[str] = []
        for root in roots:
            base = Path(root) / "Blender Foundation" if root else None
            if base is None or not base.is_dir():
                continue
            for entry in sorted(p.name for p in base.iterdir()):
                if entry.startswith("Blender"):
                    found.append(str(base / entry / "blender.exe"))
        return found
    return ["/usr/bin/blender", "/snap/bin/blender"]


def _candidates() -> list[str]:
    env_path = os.environ.get("BLENDER_PATH")
    ordered = [env_path] if env_path else []
    ordered.append("blender")
    ordered.extend(_well_known_locations())
    return ordered


def _parse_version(output: str) -> tuple[int, int, int] | None:
    match = re.search(r"Blender\s+(\d+)\.(\d+)\.(\d+)", output)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _probe(candidate: str) -> BlenderBinary | None:
    """Run ``candidate --version``; None when it does not run or parse."""
    try:
        proc = subprocess.run(
            [candidate, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    version = _parse_version(f"{proc.stdout}\n{proc.stderr}")
    if version is None:
        return None
    return BlenderBinary(path=candidate, version=version)


_cached_env: str | None = None
_cached_binary: BlenderBinary | None = None
_cached_error: str | None = None


def reset_blender_binary_cache() -> None:
    """Forget the cached resolution. Tests use this after mutating the env."""
    global _cached_env, _cached_binary, _cached_error
    _cached_env = None
    _cached_binary = None
    _cached_error = None


def resolve_blender_binary() -> BlenderBinary:
    """Resolve the Blender executable per D3. First runnable candidate wins.

    Raises BlenderError naming the found version when it is below the 5.2
    floor, or when nothing runs.
    """
    global _cached_env, _cached_binary, _cached_error
    env_value = os.environ.get("BLENDER_PATH")
    if _cached_env == env_value and (_cached_binary is not None or _cached_error is not None):
        if _cached_binary is not None:
            return _cached_binary
        raise BlenderError(_cached_error or "blender was not found")

    failures: list[str] = []
    result: BlenderBinary | None = None
    error: str | None = None
    for candidate in _candidates():
        binary = _probe(candidate)
        if binary is None:
            failures.append(candidate)
            continue
        if binary.version < BLENDER_MIN_VERSION:
            floor = ".".join(str(v) for v in BLENDER_MIN_VERSION)
            found = ".".join(str(v) for v in binary.version)
            error = (
                f"Blender {found} is too old: NodeTool needs Blender "
                f"{floor} or newer. Install a newer Blender or point "
                f"BLENDER_PATH at one."
            )
        else:
            result = binary
        break

    _cached_env = env_value
    if result is not None:
        _cached_binary, _cached_error = result, None
        return result
    if error is None:
        probe_note = ""
        if env_value and env_value in failures:
            probe_note = f" BLENDER_PATH={env_value!r} did not run."
        error = (
            "blender was not found. Install Blender 5.2 or newer and add it "
            f"to PATH, or set BLENDER_PATH to the Blender executable.{probe_note}"
        )
    _cached_binary, _cached_error = None, error
    raise BlenderError(error)


def blender_enabled() -> bool:
    """Whether this worker advertises Blender in worker.status.

    Absent or too old means False, never a crash.
    """
    try:
        resolve_blender_binary()
    except BlenderError:
        return False
    return True


def get_blender_info() -> dict[str, Any]:
    """The ``blender`` block for worker.status: ``{enabled}`` plus version."""
    try:
        binary = resolve_blender_binary()
    except BlenderError:
        return {"enabled": False}
    return {
        "enabled": True,
        "version": ".".join(str(v) for v in binary.version),
    }


# --- request validation ------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_bare_filename(name: str) -> bool:
    return bool(_JOB_FILENAME_RE.match(name))


def _check_rel_path(key: str) -> bool:
    """Op blob keys stage as relative paths: no absolute, ``..`` or NUL."""
    if not key or "\x00" in key or "\\" in key or key.endswith("/"):
        return False
    path = Path(key)
    if path.is_absolute() or key.startswith("/"):
        return False
    return all(seg not in ("", ".", "..") for seg in path.parts)


def _progress_total(job: dict[str, Any], frame: int) -> int:
    """Total for a progress frame, mirroring the local tier's progressTotal."""
    inner = job.get("job")
    if isinstance(inner, dict) and inner.get("op") == "render_animation":
        params = inner.get("params")
        if isinstance(params, dict):
            start, end = params.get("frame_start"), params.get("frame_end")
            if (
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and end >= start
            ):
                return end - start + 1
    return frame


# --- scratch staging ---------------------------------------------------------


def _stage_request(
    scratch: Path,
    job: dict[str, Any],
    inputs: dict[str, Any],
    blobs: dict[str, Any],
) -> None:
    """Write op blobs, input files and job.json into the scratch directory.

    Blob keys the ``inputs`` manifest does not name are op files staged as
    relative paths; manifest blobs land under the bare file name the job
    declares. Op files go first so a colliding input always wins.
    """
    staged_inputs = {str(name): str(key) for name, key in inputs.items()}
    input_keys = set(staged_inputs.values())

    for key, data in blobs.items():
        if str(key) in input_keys:
            continue
        target = scratch / Path(str(key))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(data))

    declared_inputs = job["inputs"]
    for name, key in staged_inputs.items():
        filename = declared_inputs[name]
        (scratch / filename).write_bytes(bytes(blobs[key]))

    (scratch / "job.json").write_text(json.dumps(job), encoding="utf-8")


def _child_env(scratch: Path) -> dict[str, str]:
    """The allowlisted environment plus BLENDER_USER_* redirects."""
    user_dir = scratch / "blender-user"
    for sub in ("config", "scripts", "extensions"):
        (user_dir / sub).mkdir(parents=True, exist_ok=True)
    env: dict[str, str] = {}
    for key in ALLOWED_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    env["BLENDER_USER_CONFIG"] = str(user_dir / "config")
    env["BLENDER_USER_SCRIPTS"] = str(user_dir / "scripts")
    env["BLENDER_USER_EXTENSIONS"] = str(user_dir / "extensions")
    return env


# --- child supervision -------------------------------------------------------


def _read_crash_log(scratch: Path, run_start: float) -> str | None:
    """Contents of a fresh ``.crash.txt``, the temp directory first.

    Blender writes crash logs into its temp directory (``$TMPDIR``, which the
    env allowlist forwards), never next to the scratch files. Only a log
    written at or after the run started counts.
    """
    candidates: list[Path] = []
    tmpdir = os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP")
    if tmpdir:
        candidates.append(Path(tmpdir))
    candidates.append(scratch)
    for directory in candidates:
        try:
            entries = sorted(p.name for p in directory.iterdir() if p.name.endswith(".crash.txt"))
        except OSError:
            continue
        for name in entries:
            path = directory / name
            try:
                if path.stat().st_mtime < run_start:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            return text[:_STDERR_TAIL_BYTES]
    return None


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM then SIGKILL after a grace period; never raises on timing."""
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=_KILL_GRACE)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            return
        await proc.wait()


async def _run_blender(
    binary: str,
    scratch: Path,
    timeout: float,
    cancel_event: asyncio.Event,
    on_progress: Callable[[int], None],
) -> tuple[int, str]:
    """Spawn Blender, stream ``Fra:`` progress, enforce timeout and cancel.

    Returns (exit code, full stderr text). Raises TimeoutError on
    timeout and asyncio.CancelledError on cancel, after killing the child.
    """
    argv = [
        binary,
        "-b",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        str(PYTHON_EXIT_CODE),
        "--python",
        str(scratch / ENTRY_POINT),
        "--",
        "job.json",
    ]
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(scratch),
        env=_child_env(scratch),
    )
    assert proc.stdout is not None and proc.stderr is not None
    stderr_chunks: list[str] = []
    stderr_len = 0

    async def drain_stdout() -> None:
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                return

    async def drain_stderr() -> None:
        nonlocal stderr_len
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace")
            if stderr_len < _STDERR_BUFFER_CAP:
                stderr_chunks.append(text)
                stderr_len += len(text)
            match = _FRA_RE.match(text.strip())
            if match:
                on_progress(int(match.group(1)))

    stdout_task = asyncio.create_task(drain_stdout())
    stderr_task = asyncio.create_task(drain_stderr())
    cancel_task = asyncio.create_task(cancel_event.wait())
    wait_task = asyncio.create_task(proc.wait())
    try:
        done, _pending = await asyncio.wait(
            {wait_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout,
        )
        if not done:
            wait_task.cancel()
            await _terminate(proc)
            raise TimeoutError()
        if cancel_task in done:
            wait_task.cancel()
            await _terminate(proc)
            raise asyncio.CancelledError()
    finally:
        cancel_task.cancel()
        wait_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
    stderr_text = "".join(stderr_chunks)
    if stderr_len >= _STDERR_BUFFER_CAP:
        stderr_text = stderr_text[-_STDERR_TAIL_BYTES * 2 :]
    return proc.returncode or 0, stderr_text


# --- result collection -------------------------------------------------------


def _bad_result_message(stderr: str, crash: str | None) -> str:
    message = (
        "Blender finished without a parsable result.json. "
        f"Stderr tail: {stderr[-_STDERR_TAIL_BYTES:]}"
    )
    if crash:
        message += f" Crash log: {crash}"
    return message


def _collect_outputs(
    scratch: Path,
    job: dict[str, Any],
    parsed: dict[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, int], dict[str, bytes]]:
    """Read back only the outputs the job declared.

    Returns (produced, stats, sizes, blobs). A name the op invented is
    ignored at warn; a declared output missing from disk or ``produced`` is
    ``missing_output``. Sizes are real byte sizes from stat, taken before
    any byte is read.
    """
    produced = parsed["produced"]
    stats = parsed["stats"]
    declared: dict[str, str] = job["outputs"]
    produced_set = set(produced)
    for name in produced:
        if name not in declared:
            log.warning('Blender produced undeclared output "%s"; ignoring.', name)

    sizes: dict[str, int] = {}
    for name, filename in declared.items():
        if name not in produced_set:
            raise BlenderError(
                f'Blender did not produce declared output "{name}" '
                f'(file "{filename}").'
            )
        try:
            sizes[name] = (scratch / filename).stat().st_size
        except OSError:
            raise BlenderError(
                f'Blender did not produce declared output "{name}": '
                f'file "{filename}" is missing.'
            ) from None

    blobs: dict[str, bytes] = {}
    for name, filename in declared.items():
        try:
            blobs[name] = (scratch / filename).read_bytes()
        except OSError:
            raise BlenderError(
                f'Blender did not produce declared output "{name}": '
                f'file "{filename}" is missing.'
            ) from None
    return produced, stats, sizes, blobs


# --- dispatch ------------------------------------------------------------------


async def _handle_execute(
    data: dict[str, Any],
    request_id: str | None,
    cancel_flags: dict[str, asyncio.Event],
    send_event: Callable,
    send_result: Callable,
) -> None:
    job = data.get("job")
    if not isinstance(job, dict):
        raise BlenderError("blender.execute requires a 'job' object (BlenderJob)")
    if not _is_number(job.get("version")):
        raise BlenderError("blender.execute requires a numeric job.version")
    if not isinstance(job.get("inputs"), dict) or not isinstance(job.get("outputs"), dict):
        raise BlenderError("blender.execute requires job.inputs and job.outputs maps")

    inputs = data.get("inputs")
    if not isinstance(inputs, dict):
        raise BlenderError("blender.execute requires an 'inputs' blob-key manifest")
    blobs = data.get("blobs") or {}
    if not isinstance(blobs, dict):
        raise BlenderError("blender.execute requires 'blobs' input bytes")
    for name, key in inputs.items():
        if not isinstance(key, str):
            raise BlenderError(f'blender.execute input "{name}" must name a blob key')
        if key not in blobs:
            raise BlenderError(f'blender.execute input "{name}" names missing blob "{key}"')
        if not isinstance(blobs[key], (bytes, bytearray)):
            raise BlenderError(f'blender.execute blob "{key}" must be binary')
        if name not in job["inputs"]:
            raise BlenderError(f'Input "{name}" is not declared in job.inputs.')
    for name, filename in job["inputs"].items():
        if name not in inputs:
            raise BlenderError(f'blender.execute is missing declared input "{name}"')
        if not isinstance(filename, str) or not _check_bare_filename(filename):
            raise BlenderError(f'blender.execute input "{name}" names an unsafe file')
    for name, filename in job["outputs"].items():
        if not isinstance(filename, str) or not _check_bare_filename(filename):
            raise BlenderError(f'blender.execute output "{name}" names an unsafe file')
    input_keys = {str(k) for k in inputs.values()}
    for key, value in blobs.items():
        if key in input_keys:
            continue
        if not _check_rel_path(str(key)):
            raise BlenderError(f'blender.execute blob key "{key}" is not a safe relative path')
        if not isinstance(value, (bytes, bytearray)):
            raise BlenderError(f'blender.execute blob "{key}" must be binary')

    timeout = data.get("timeout", DEFAULT_TIMEOUT)
    if not _is_number(timeout) or float(timeout) <= 0:
        raise BlenderError("blender.execute requires a numeric 'timeout'")
    # Whole seconds like the client sends; sub-second values clamp to 1s.
    timeout_s = max(1.0, float(timeout))

    if job["version"] != BLENDER_JOB_VERSION:
        await send_result(request_id, {
            "ok": False,
            "error": {
                "code": "bad_job",
                "message": (
                    f"unsupported job version {job['version']!r}: "
                    f"this worker speaks version {BLENDER_JOB_VERSION}"
                ),
            },
        })
        return

    if ENTRY_POINT not in blobs:
        raise BlenderError(
            "blender.execute requires the shipped op script blob "
            f"'{ENTRY_POINT}': the client sends every file of blender_ops/ in blobs"
        )

    try:
        binary = resolve_blender_binary()
    except BlenderError as e:
        raise BlenderError(f"blender is not available on this worker: {e}") from e

    cancel_event = asyncio.Event()
    if request_id:
        cancel_flags[request_id] = cancel_event

    scratch = Path(tempfile.mkdtemp(prefix="nodetool-blender-"))
    settled = False

    async def send_terminal(payload: dict[str, Any]) -> None:
        nonlocal settled
        if settled:
            return
        settled = True
        await send_result(request_id, payload)

    run_start = time.time()
    progress_sends: list[asyncio.Task[None]] = []
    try:
        _stage_request(scratch, job, inputs, blobs)

        def on_frame(frame: int) -> None:
            total = _progress_total(job, frame)
            progress_sends.append(
                asyncio.get_running_loop().create_task(
                    send_event(request_id, {"event": "progress", "frame": frame, "total": total})
                )
            )

        try:
            _, stderr = await _run_blender(
                binary.path, scratch, timeout_s, cancel_event, on_frame
            )
        except TimeoutError:
            await send_terminal({
                "ok": False,
                "error": {
                    "code": "timeout",
                    "message": (
                        f"Blender render timed out after {timeout_s:g}s. "
                        "Lower the samples, use EEVEE, or raise the timeout."
                    ),
                },
            })
            return
        except asyncio.CancelledError:
            await send_terminal({
                "ok": False,
                "error": {
                    "code": "cancelled",
                    "message": f'Blender execution "{request_id}" was cancelled.',
                },
            })
            return

        # Progress frames were fire-and-forget during the run; flush them so
        # the terminal result is always last on the wire.
        await asyncio.gather(*progress_sends, return_exceptions=True)
        progress_sends.clear()

        try:
            raw = (scratch / "result.json").read_text(encoding="utf-8")
            parsed = json.loads(raw)
        except (OSError, ValueError):
            parsed = None
        if not isinstance(parsed, dict):
            crash = _read_crash_log(scratch, run_start)
            await send_terminal({
                "ok": False,
                "error": {
                    "code": "bad_result",
                    "message": _bad_result_message(stderr, crash),
                },
            })
            return

        if not parsed.get("ok"):
            error = parsed.get("error") if isinstance(parsed.get("error"), dict) else {}
            code = error.get("code")
            message = error.get("message")
            await send_terminal({
                "ok": False,
                "error": {
                    "code": code if isinstance(code, str) and code else "bad_result",
                    "message": message if isinstance(message, str) else "Blender op failed.",
                },
            })
            return

        if (
            not isinstance(parsed.get("produced"), list)
            or not all(isinstance(n, str) for n in parsed["produced"])
            or not isinstance(parsed.get("stats"), dict)
            or not isinstance(parsed["stats"].get("blender_version"), str)
            or not _is_number(parsed["stats"].get("render_seconds"))
        ):
            crash = _read_crash_log(scratch, run_start)
            await send_terminal({
                "ok": False,
                "error": {
                    "code": "bad_result",
                    "message": _bad_result_message(stderr, crash),
                },
            })
            return

        try:
            produced, stats, sizes, out_blobs = _collect_outputs(scratch, job, parsed)
        except BlenderError as e:
            await send_terminal({
                "ok": False,
                "error": {"code": "missing_output", "message": str(e)},
            })
            return

        await send_terminal({
            "ok": True,
            "produced": produced,
            "stats": stats,
            "sizes": sizes,
            "blobs": out_blobs,
        })
    finally:
        await asyncio.gather(*progress_sends, return_exceptions=True)
        if request_id:
            cancel_flags.pop(request_id, None)
        shutil.rmtree(scratch, ignore_errors=True)


async def handle_blender_message(
    msg_type: str,
    request_id: str | None,
    data: dict[str, Any],
    transport: Any,  # WorkerTransport (exposes async send_msg)
    cancel_flags: dict[str, asyncio.Event],
) -> None:
    """Handle a blender.* message via any transport exposing ``send_msg``."""

    async def send_result(rid: str | None, d: dict) -> None:
        await transport.send_msg({"type": "result", "request_id": rid, "data": d})

    async def send_error(rid: str | None, error: str, tb: str | None = None) -> None:
        # Omitted rather than null — the JS side's frame schema types
        # `traceback` as an optional string, and a null fails validation.
        payload: dict[str, Any] = {"error": error}
        if tb:
            payload["traceback"] = tb
        await transport.send_msg({"type": "error", "request_id": rid, "data": payload})

    async def send_event(rid: str | None, d: dict) -> None:
        # Frame progress gets its own frame type: its shape
        # ({event, frame, total}) is nothing like the generic progress
        # frames ({progress, total, message}), so overloading `progress`
        # would force every consumer to sniff the payload.
        await transport.send_msg({"type": "blender.event", "request_id": rid, "data": d})

    try:
        if msg_type == "blender.execute":
            await _handle_execute(data, request_id, cancel_flags, send_event, send_result)
        elif msg_type == "blender.status":
            info = get_blender_info()
            await send_result(request_id, info)
        else:
            await send_error(request_id, f"Unknown blender message type: {msg_type}")
    except Exception as e:
        await send_error(request_id, str(e), traceback.format_exc())
