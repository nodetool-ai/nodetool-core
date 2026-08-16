"""Handle job.* bridge messages: the run boundary (bridge protocol v4).

Both messages answer with a normal ``result`` frame. The JS side awaits
``result``/``error`` and would otherwise hang — and because it deliberately
logs and swallows ``job.*`` failures (a ``job.end`` that fails against a worker
already tearing down must not turn a finished run into a failed one), a broken
handler here is *silent* from the JS side. That is why the worker's own state,
not the JS response, is what the tests assert on.

Mirrors ``model_handler.handle_models_message``: transport-agnostic, it only
needs a transport exposing an async ``send_msg``, so the same code path serves
both the WebSocket and stdio workers.
"""

from __future__ import annotations

import traceback
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.worker.job_registry import JobRegistry

log = get_logger(__name__)

# Labels the JS side sends on job.end. Carried through to logs and the result
# frame; deliberately NOT branched on — an abnormal end retires the same nodes
# as a clean one.
JOB_END_REASONS = ("completed", "failed", "cancelled", "abandoned")


async def handle_job_message(
    msg_type: str,
    request_id: str | None,
    data: dict[str, Any],
    transport: Any,  # WorkerTransport (exposes async send_msg)
) -> None:
    """Handle a ``job.start`` / ``job.end`` message."""

    async def send_result(payload: dict[str, Any]) -> None:
        await transport.send_msg({"type": "result", "request_id": request_id, "data": payload})

    async def send_error(error: str, tb: str | None = None) -> None:
        # `traceback` is omitted rather than sent as null: the frame schema the
        # JS side generates types it as an optional string, so a null fails
        # validation on a worker that is otherwise behaving correctly.
        data: dict[str, Any] = {"error": error}
        if tb:
            data["traceback"] = tb
        await transport.send_msg({"type": "error", "request_id": request_id, "data": data})

    try:
        job_id = data.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            await send_error(f"{msg_type} requires a non-empty string job_id")
            return

        workflow_id = data.get("workflow_id")
        user_id = data.get("user_id")

        if msg_type == "job.start":
            JobRegistry.start(
                job_id,
                workflow_id=workflow_id if isinstance(workflow_id, str) else None,
                user_id=user_id if isinstance(user_id, str) else None,
            )
            # The one reclaim pass per run. Threshold- and cooldown-gated, so
            # it is a no-op unless VRAM is actually under pressure — the point
            # is that it happens once at the boundary instead of once per node.
            from nodetool.ml.core.model_manager import ModelManager

            ModelManager.free_vram_if_needed(reason=f"Starting job {job_id}")
            await send_result({"job_id": job_id, "started": True})
            return

        if msg_type == "job.end":
            reason = data.get("reason")
            if reason is not None and reason not in JOB_END_REASONS:
                # Unknown labels are logged, not rejected: the label is
                # metadata, and refusing one would fail a boundary whose whole
                # job is to release memory.
                log.debug("job.end for %s carried unknown reason %r", job_id, reason)
            # Read before end() pops it: a real job that recorded no nodes
            # releases nothing, and must still be distinguishable from a job
            # the worker never saw.
            was_known = JobRegistry.has(job_id)
            released = JobRegistry.end(job_id, reason if isinstance(reason, str) else None)
            await send_result(
                {
                    "job_id": job_id,
                    "released_nodes": released,
                    # False for a job the worker never saw (all nodes failed
                    # upstream, a duplicate job.end, a reconnected client).
                    # Reported, not errored — the no-op is the contract.
                    "known": was_known,
                }
            )
            return

        await send_error(f"Unknown job message type: {msg_type}")
    except Exception as e:  # pragma: no cover — defensive
        await send_error(str(e), traceback.format_exc())
