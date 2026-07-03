"""Handle models.* bridge messages: HuggingFace cache management on the worker.

models.* are HuggingFace-scoped for now (the worker cache is HF). They run
against whatever HF_HOME the worker process sees — when this worker is a remote
pod with a persistent volume, that is the pod's cache.

Mirrors ``provider_handler.handle_provider_message``: transport-agnostic, it
only needs a transport exposing an async ``send_msg``, so the same code path
serves both the WebSocket and stdio workers.
"""

from __future__ import annotations

import asyncio
import fnmatch
import traceback
from typing import Any, Callable

from nodetool.integrations.huggingface.async_downloader import async_hf_download
from nodetool.integrations.huggingface.huggingface_models import (
    delete_cached_hf_model,
    get_hf_token,
    read_cached_hf_models,
)


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


def _matches(path: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, p) for p in patterns)


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

    token = await get_hf_token()
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
        done_bytes = 0
        done_files = 0

        await send_progress(request_id, frame("start", 0, total_bytes, 0, total_files, []))

        # Track in-flight progress sends so the terminal frames cannot race
        # ahead of the per-byte updates fired from the sync callback.
        progress_tasks: set[asyncio.Task] = set()

        async def drain() -> None:
            if progress_tasks:
                await asyncio.gather(*progress_tasks, return_exceptions=True)

        for filename, _size in files:
            if cancel_event.is_set():
                await drain()
                await send_progress(
                    request_id,
                    frame(
                        "cancelled",
                        done_bytes,
                        total_bytes,
                        done_files,
                        total_files,
                        [],
                    ),
                )
                await send_result(request_id, {"repo_id": repo_id, "status": "cancelled"})
                return

            file_base = done_bytes
            # async_hf_download's progress_callback reports per-chunk DELTAS on
            # the streaming path (len(chunk)) and a single cumulative value on
            # the cached/complete fast paths. Accumulating deltas is correct for
            # both: the fast path adds the whole size once, the streaming path
            # sums to the same total. (Treating each delta as an absolute total
            # would make downloaded_bytes oscillate and corrupt the next file's
            # base.)
            file_acc = {"bytes": 0}

            def on_bytes(
                delta: int,
                _file_total=None,
                _filename=filename,
                _base=file_base,
                _fsize=_size,
                _acc=file_acc,
            ):
                nonlocal done_bytes
                _acc["bytes"] += delta
                progressed = min(_acc["bytes"], _fsize) if _fsize else _acc["bytes"]
                done_bytes = _base + progressed
                # fire-and-forget; ordering is preserved by the transport
                # write-lock. Schedule on the running loop captured above so
                # the sync callback works regardless of its calling thread.
                task = loop.create_task(
                    send_progress(
                        request_id,
                        frame(
                            "progress",
                            done_bytes,
                            total_bytes,
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
                # Cooperative app-level cancel: async_hf_download raises
                # asyncio.CancelledError when cancel_event is set mid-stream.
                # CancelledError is a BaseException, so it escapes the outer
                # `except Exception` — without this, no terminal frame is sent
                # and the bridge's downloadModel() promise hangs forever.
                # Convert it to the cancelled terminal sequence and swallow it
                # (this task itself is not being cancelled).
                await drain()
                await send_progress(
                    request_id,
                    frame(
                        "cancelled",
                        done_bytes,
                        total_bytes,
                        done_files,
                        total_files,
                        [],
                    ),
                )
                await send_result(request_id, {"repo_id": repo_id, "status": "cancelled"})
                return

            # Snap to the exact file size so the next file's base is correct
            # even if the callback under-reported (e.g. size metadata missing).
            if _size:
                done_bytes = file_base + _size
            done_files += 1

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
        await transport.send_msg({"type": "error", "request_id": rid, "data": {"error": error, "traceback": tb}})

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

        elif msg_type == "models.delete":
            deleted = await delete_cached_hf_model(data["repo_id"])
            await send_result(request_id, {"deleted": bool(deleted)})

        else:
            await send_error(request_id, f"Unknown models message type: {msg_type}")

    except Exception as e:
        await send_error(request_id, str(e), traceback.format_exc())
