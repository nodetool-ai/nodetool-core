"""Handle models.* bridge messages: HuggingFace cache management on the worker.

models.* are HuggingFace-scoped for now (the worker cache is HF). They run
against whatever HF_HOME the worker process sees — when this worker is a remote
pod with a persistent volume, that is the pod's cache.

``models.download`` accepts an optional ``token`` field. A rented worker holds
no HuggingFace credential, so the host may pass one per request; the worker uses
it for that download only and never persists, logs, or echoes it. Without the
field the worker resolves a token itself, as before.

Mirrors ``provider_handler.handle_provider_message``: transport-agnostic, it
only needs a transport exposing an async ``send_msg``, so the same code path
serves both the WebSocket and stdio workers.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import shlex
import signal
import traceback
from typing import Any, Callable

from nodetool.integrations.huggingface.async_downloader import async_hf_download
from nodetool.integrations.huggingface.huggingface_models import (
    delete_cached_hf_model,
    get_hf_token,
    read_cached_hf_models,
)

_MODEL_BACKEND_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_ADAPTER_STDERR_LIMIT = 32 * 1024


def _prepare_command_env(backend: str) -> str:
    return f"NODETOOL_MODEL_PREPARE_COMMAND_{backend.upper().replace('-', '_')}"


def get_model_prepare_backends() -> list[str]:
    """Return image-provided model preparation backends.

    A backend is enabled only by a worker-side command. The authenticated peer
    selects an advertised id; it can never supply or alter the executable.
    """
    prefix = "NODETOOL_MODEL_PREPARE_COMMAND_"
    backends: list[str] = []
    for name, value in os.environ.items():
        if not name.startswith(prefix) or not value.strip():
            continue
        backend = name[len(prefix) :].lower()
        if _MODEL_BACKEND_RE.fullmatch(backend):
            backends.append(backend)
    return sorted(set(backends))


def _prepare_progress_frame(data: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Normalize an adapter update to the existing download progress shape."""
    status = update.get("status")
    if status not in {"start", "progress", "completed", "error", "cancelled"}:
        raise ValueError(f"Model preparation adapter returned invalid status: {status!r}")

    frame = dict(update)
    frame.update(
        {
            "status": status,
            "repo_id": str(data.get("repo_id") or f"{data['backend']}:{data['model_type']}"),
            "path": None,
            "model_type": str(data["model_type"]),
            "downloaded_bytes": max(0, int(update.get("downloaded_bytes") or 0)),
            # Zero is the established wire representation for an unknown total.
            "total_bytes": max(0, int(update.get("total_bytes") or 0)),
            "downloaded_files": max(0, int(update.get("downloaded_files") or 0)),
            "total_files": max(0, int(update.get("total_files") or 0)),
        }
    )
    current = update.get("current_files")
    frame["current_files"] = (
        [str(path) for path in current if isinstance(path, str)]
        if isinstance(current, list)
        else []
    )
    return frame


async def _terminate_adapter(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        await process.wait()


async def _read_adapter_stderr(stream: asyncio.StreamReader | None) -> str:
    if stream is None:
        return ""
    kept = bytearray()
    while chunk := await stream.read(4096):
        kept.extend(chunk)
        if len(kept) > _ADAPTER_STDERR_LIMIT:
            del kept[: len(kept) - _ADAPTER_STDERR_LIMIT]
    return kept.decode("utf-8", "replace").strip()


async def _handle_prepare(
    data: dict[str, Any],
    request_id: str | None,
    cancel_flags: dict[str, asyncio.Event],
    send_progress: Callable,
    send_result: Callable,
) -> None:
    """Run an image-owned model adapter and relay its JSON-lines progress."""
    backend = data.get("backend")
    model_type = data.get("model_type")
    if not isinstance(backend, str) or not _MODEL_BACKEND_RE.fullmatch(backend):
        raise ValueError("models.prepare requires a valid backend")
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("models.prepare requires model_type")

    command_text = os.environ.get(_prepare_command_env(backend), "").strip()
    if not command_text:
        raise ValueError(f"Model preparation backend is unavailable: {backend}")
    command = shlex.split(command_text)
    if not command:
        raise ValueError(f"Model preparation backend has an empty command: {backend}")

    cancel_event = asyncio.Event()
    if request_id:
        cancel_flags[request_id] = cancel_event

    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    stderr_task = asyncio.create_task(_read_adapter_stderr(process.stderr))
    last_frame = _prepare_progress_frame(data, {"status": "start"})
    await send_progress(request_id, last_frame)

    try:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Model preparation adapter pipes are unavailable")
        payload = {
            "backend": backend,
            "model_type": model_type.strip(),
            "repo_id": last_frame["repo_id"],
        }
        token = _request_token(data)
        if token:
            payload["token"] = token
        process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
        await process.stdin.drain()
        process.stdin.close()

        while True:
            line_task = asyncio.create_task(process.stdout.readline())
            cancel_task = asyncio.create_task(cancel_event.wait())
            done, pending = await asyncio.wait(
                {line_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            if cancel_task in done and cancel_task.result():
                await _terminate_adapter(process)
                last_frame = _prepare_progress_frame(
                    data,
                    {
                        **last_frame,
                        "status": "cancelled",
                        "current_files": [],
                        "message": "Model preparation cancelled",
                    },
                )
                await send_progress(request_id, last_frame)
                await send_result(
                    request_id,
                    {"repo_id": last_frame["repo_id"], "status": "cancelled"},
                )
                return

            line = line_task.result()
            if not line:
                break
            if len(line) > 1024 * 1024:
                raise ValueError("Model preparation adapter emitted an oversized progress line")
            try:
                update = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Model preparation adapter emitted invalid JSON") from exc
            if not isinstance(update, dict):
                raise ValueError("Model preparation adapter progress must be a JSON object")
            last_frame = _prepare_progress_frame(data, update)
            await send_progress(request_id, last_frame)

        returncode = await process.wait()
        stderr = await stderr_task
        if returncode != 0:
            detail = stderr.splitlines()[-1] if stderr else f"exit status {returncode}"
            raise RuntimeError(f"Model preparation adapter failed: {detail}")
        if last_frame["status"] != "completed":
            raise RuntimeError("Model preparation adapter exited without completing")
        await send_result(
            request_id,
            {"repo_id": last_frame["repo_id"], "status": "completed"},
        )
    except BaseException:
        await _terminate_adapter(process)
        if not stderr_task.done():
            stderr_task.cancel()
        await asyncio.gather(stderr_task, return_exceptions=True)
        raise
    finally:
        if request_id:
            cancel_flags.pop(request_id, None)


async def _list_repo_files(repo_id: str, token: str | None = None):
    """Return ``[(filename, size_bytes)]`` for a repo via the Hub (network)."""
    from huggingface_hub import HfApi

    def _list():
        api = HfApi(token=token) if token else HfApi()
        info = api.model_info(repo_id, files_metadata=True)
        out: list[tuple[str, int]] = []
        for sib in info.siblings or []:
            out.append((sib.rfilename, int(getattr(sib, "size", 0) or 0)))
        return out

    return await asyncio.to_thread(_list)


def _request_token(data: dict[str, Any]) -> str | None:
    """Return the HF token the caller supplied with this request, if any.

    A rented worker is a bare container: it has no per-user secret store and
    nothing sets ``HF_TOKEN`` in its environment, so ``get_hf_token()`` resolves
    to None and every gated repo answers 401. The host therefore passes the
    credential with the request, and it lives only for the life of the call.

    An absent, non-string, or blank value means "no token supplied" — an empty
    Bearer header fails differently than sending no header at all.

    Why accepting a credential from the request is not a new exposure
    ----------------------------------------------------------------
    On a rented pod this bridge is internet-reachable — a
    ``wss://<pod>-7777.proxy.runpod.net`` URL fronted by nothing but the
    ``NODETOOL_WORKER_TOKEN`` bearer — so the question is fair. The answer is
    that the request channel is already fully trusted: a peer that can reach it
    holds that bearer, and with it can execute arbitrary nodes, write model
    files, and read this worker's environment (``worker/server.py`` warns about
    exactly this when the worker binds off-loopback without a token). A
    download token adds no capability that channel does not already grant, and
    it spends the caller's own HuggingFace quota, not the worker's.

    An earlier version rejected the field, with a test asserting "a malicious
    client token must be ignored". That protected nothing — it only stopped the
    host from reaching a gated repo — and the alternative it forced was worse:
    baking ``HF_TOKEN`` into the pod's environment for the life of the pod,
    where every node and every subprocess can read it. Here the credential
    lives for one call. Do not re-add that test, and do not read its absence as
    evidence that the bridge is hardened; the bridge's security is the bearer
    token and the network boundary, and neither changes here.
    """
    raw = data.get("token")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _matches(path: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, p) for p in patterns)


# Files per repo downloaded at once. Eight matches huggingface_hub's own
# snapshot_download default; the limit exists because a rented worker shares a
# NIC and an unbounded fan-out over a many-shard repo starves everything else
# on the box. Override with NODETOOL_HF_DOWNLOAD_CONCURRENCY.
def _download_concurrency() -> int:
    raw = os.environ.get("NODETOOL_HF_DOWNLOAD_CONCURRENCY")
    if not raw:
        return 8
    try:
        value = int(raw)
    except ValueError:
        return 8
    return max(1, min(value, 32))


async def _handle_download(
    data: dict[str, Any],
    request_id: str | None,
    cancel_flags: dict[str, asyncio.Event],
    send_progress: Callable,
    send_result: Callable,
) -> None:
    repo_id = data["repo_id"]
    allow = data.get("allow_patterns")
    ignore = data.get("ignore_patterns")
    single = data.get("path")
    model_type = data.get("model_type")

    cancel_event = asyncio.Event()
    if request_id:
        cancel_flags[request_id] = cancel_event

    # Per-request token first; fall back to the worker's own resolution.
    token = _request_token(data) or await get_hf_token()
    loop = asyncio.get_running_loop()

    def frame(
        status: str,
        downloaded_bytes: int,
        total_bytes: int,
        downloaded_files: int,
        total_files: int,
        current: list[str],
        error: str | None = None,
    ) -> dict:
        d: dict[str, Any] = {
            "status": status,
            "repo_id": repo_id,
            "path": single,
            "model_type": model_type,
            "downloaded_bytes": downloaded_bytes,
            "total_bytes": total_bytes,
            "downloaded_files": downloaded_files,
            "current_files": current,
            "total_files": total_files,
        }
        if error:
            d["error"] = error
        return d

    try:
        files = await _list_repo_files(repo_id, token)
        if single:
            files = [(f, s) for f, s in files if f == single]
        else:
            if allow:
                files = [(f, s) for f, s in files if _matches(f, allow)]
            if ignore:
                files = [(f, s) for f, s in files if not _matches(f, ignore)]

        # No matching files means the requested model/path does not exist in
        # the repo. Report it as an error instead of falsely completing — the
        # loop below would otherwise never run and we'd emit status "completed"
        # for a download that never happened.
        if not files:
            if single:
                raise ValueError(f"No file matching path {single!r} found in repo {repo_id}")
            raise ValueError(
                f"No files in repo {repo_id} matched the requested patterns "
                f"(allow_patterns={allow}, ignore_patterns={ignore})"
            )

        total_files = len(files)
        total_bytes = sum(s for _, s in files)
        done_files = 0

        await send_progress(request_id, frame("start", 0, total_bytes, 0, total_files, []))

        # Track in-flight progress sends so the terminal frames cannot race
        # ahead of the per-byte updates fired from the sync callback.
        progress_tasks: set[asyncio.Task] = set()

        async def drain() -> None:
            if progress_tasks:
                await asyncio.gather(*progress_tasks, return_exceptions=True)

        # Files transfer concurrently. Serially, a repo of many shards moved at
        # one connection's throughput: SmolVLM-Instruct took 1798 s for 29.5 GB
        # (16 MB/s) on an A40, while the host had far more bandwidth available.
        #
        # Progress can no longer use a running base — completions interleave, so
        # each file accumulates into its own slot and the reported total is the
        # sum. The old `file_base = done_bytes` arithmetic silently corrupts the
        # next file's base as soon as two are in flight.
        semaphore = asyncio.Semaphore(_download_concurrency())
        file_bytes: dict[str, int] = {name: 0 for name, _ in files}
        cancelled = False

        def total_done() -> int:
            return sum(file_bytes.values())

        async def download_one(filename: str, size: int | None) -> None:
            nonlocal done_files, cancelled
            async with semaphore:
                if cancel_event.is_set():
                    cancelled = True
                    return

                def on_bytes(
                    delta: int,
                    _file_total: int | None = None,
                    _filename: str = filename,
                    _fsize: int | None = size,
                ) -> None:
                    # async_hf_download reports per-chunk deltas on the streaming
                    # path and one cumulative value on the cached fast path;
                    # accumulating deltas is correct for both.
                    acc = file_bytes[_filename] + delta
                    file_bytes[_filename] = min(acc, _fsize) if _fsize else acc
                    task = loop.create_task(
                        send_progress(
                            request_id,
                            frame(
                                "progress",
                                total_done(),
                                total_bytes,
                                # Read at call time, not closure-definition
                                # time, so it reports how many files have
                                # completed now.
                                done_files,
                                total_files,
                                [_filename],
                            ),
                        )
                    )
                    progress_tasks.add(task)
                    task.add_done_callback(progress_tasks.discard)

                try:
                    await async_hf_download(
                        repo_id,
                        filename,
                        token=token,
                        progress_callback=on_bytes,
                        cancel_event=cancel_event,
                    )
                except asyncio.CancelledError:
                    # Cooperative app-level cancel from cancel_event, not this
                    # task being cancelled. CancelledError is a BaseException and
                    # would escape the caller's `except Exception`, leaving the
                    # bridge's downloadModel() promise hanging with no terminal
                    # frame.
                    cancelled = True
                    return

                # Snap to the exact size so a callback that under-reported (missing
                # size metadata) does not leave the total short of 100%.
                if size:
                    file_bytes[filename] = size
                done_files += 1

        await asyncio.gather(*(download_one(name, size) for name, size in files))

        if cancelled or cancel_event.is_set():
            await drain()
            await send_progress(
                request_id,
                frame(
                    "cancelled",
                    total_done(),
                    total_bytes,
                    done_files,
                    total_files,
                    [],
                ),
            )
            await send_result(request_id, {"repo_id": repo_id, "status": "cancelled"})
            return

        # Drain any in-flight progress frames before the terminal frames.
        await drain()

        if cancel_event.is_set():
            await send_result(request_id, {"repo_id": repo_id, "status": "cancelled"})
            return

        await send_progress(
            request_id,
            frame("completed", total_bytes, total_bytes, total_files, total_files, []),
        )
        await send_result(request_id, {"repo_id": repo_id, "status": "completed"})

    except Exception as e:
        await send_progress(
            request_id,
            frame("error", 0, 0, 0, 0, [], error=str(e)),
        )
        raise
    finally:
        if request_id:
            cancel_flags.pop(request_id, None)


def _handle_evict(data: dict[str, Any]) -> dict[str, Any]:
    """Drop loaded model weights (``models.evict``, bridge protocol v4).

    All three scoping fields are optional and they compose: ``node_ids`` and
    ``job_id`` both narrow *which* models are candidates (a ``job_id`` resolves
    to the nodes that job executed), while ``target_vram_gb`` bounds *how much*
    gets dropped. With no scope at all, everything eligible is evicted.

    Unlike the reactive threshold reclaim, this runs whenever the host asks —
    the host is the only side that knows the user switched workflows or that
    another process wants the GPU.
    """
    from nodetool.ml.core.model_manager import ModelManager
    from nodetool.worker.job_registry import JobRegistry

    raw_node_ids = data.get("node_ids")
    node_ids: list[str] | None = None
    if isinstance(raw_node_ids, (list, tuple)):
        node_ids = [n for n in raw_node_ids if isinstance(n, str) and n]

    job_id = data.get("job_id")
    if isinstance(job_id, str) and job_id:
        # An unknown job contributes no nodes. That must not silently widen the
        # request into "evict everything": a scope was asked for, so an empty
        # scope evicts nothing.
        node_ids = (node_ids or []) + JobRegistry.node_ids_for(job_id)

    raw_target = data.get("target_vram_gb")
    target = (
        float(raw_target)
        if isinstance(raw_target, (int, float)) and not isinstance(raw_target, bool) and raw_target > 0
        else None
    )

    evicted, freed_gb = ModelManager.evict_models(
        node_ids=list(dict.fromkeys(node_ids)) if node_ids is not None else None,
        target_vram_gb=target,
    )
    result: dict[str, Any] = {"evicted": evicted}
    if evicted:
        result["freed_vram_gb"] = freed_gb
    return result


async def handle_models_message(
    msg_type: str,
    request_id: str | None,
    data: dict[str, Any],
    transport: Any,  # WorkerTransport (exposes async send_msg)
    cancel_flags: dict[str, asyncio.Event],
) -> None:
    """Handle a models.* message via any transport exposing ``send_msg``."""

    async def send_result(rid: str | None, d: dict) -> None:
        await transport.send_msg({"type": "result", "request_id": rid, "data": d})

    async def send_error(rid: str | None, error: str, tb: str | None = None) -> None:
        # `traceback` is omitted rather than sent as null — the JS side's frame
        # schema types it as an optional string, so a null fails validation.
        payload: dict[str, Any] = {"error": error}
        if tb:
            payload["traceback"] = tb
        await transport.send_msg({"type": "error", "request_id": rid, "data": payload})

    async def send_progress(rid: str | None, d: dict) -> None:
        await transport.send_msg({"type": "progress", "request_id": rid, "data": d})

    try:
        if msg_type == "models.list_cached":
            models = await read_cached_hf_models()
            # We only enumerate cached repos here, so guarantee downloaded=True.
            payload = []
            for m in models:
                d = m.model_dump()
                d["downloaded"] = True
                payload.append(d)
            await send_result(request_id, {"models": payload})

        elif msg_type == "models.download":
            await _handle_download(data, request_id, cancel_flags, send_progress, send_result)

        elif msg_type == "models.prepare":
            await _handle_prepare(data, request_id, cancel_flags, send_progress, send_result)

        elif msg_type == "models.delete":
            deleted = await delete_cached_hf_model(data["repo_id"])
            await send_result(request_id, {"deleted": bool(deleted)})

        elif msg_type == "models.evict":
            await send_result(request_id, _handle_evict(data))

        else:
            await send_error(request_id, f"Unknown models message type: {msg_type}")

    except Exception as e:
        await send_error(request_id, str(e), traceback.format_exc())
