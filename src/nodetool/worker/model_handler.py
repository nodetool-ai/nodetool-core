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
                _file_total: int | None = None,
                _filename: str = filename,
                _base: int = file_base,
                _fsize: int | None = _size,
                _acc: dict[str, int] = file_acc,
            ) -> None:
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
                            # Deliberately late-bound, unlike the per-file
                            # values above: this must report how many files
                            # have completed *now*, not when the callback was
                            # defined (which is always 0 for the current file).
                            done_files,  # noqa: B023
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
            # Not enumerate(): on_bytes closes over this and must observe the
            # count as it advances, and the early-return paths above report it
            # mid-loop.
            done_files += 1  # noqa: SIM113

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

        elif msg_type == "models.delete":
            deleted = await delete_cached_hf_model(data["repo_id"])
            await send_result(request_id, {"deleted": bool(deleted)})

        elif msg_type == "models.evict":
            await send_result(request_id, _handle_evict(data))

        else:
            await send_error(request_id, f"Unknown models message type: {msg_type}")

    except Exception as e:
        await send_error(request_id, str(e), traceback.format_exc())
