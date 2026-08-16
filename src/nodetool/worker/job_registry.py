"""Run-boundary bookkeeping for the worker (bridge protocol v4).

The worker sees an unlabeled stream of single-node executions. Before v4 it had
no idea which run a given ``execute`` belonged to, so nothing ever fired at the
point where "these nodes are retired, their models are eligible" is true:
``ModelManager.release_nodes()`` had no caller and the model cache only ever
shrank reactively, under memory pressure.

This module is that missing bookkeeping. ``job.start`` opens a run,
``execute`` records ``job_id -> node_id`` as it goes, and ``job.end`` retires
the run's nodes so their models can be released.

Design notes
------------

**Idempotent and tolerant of unknown jobs.** ``end()`` for a job the worker
never saw — a run whose nodes all failed before reaching the worker, a
duplicate ``job.end``, a client that reconnected — is a no-op that returns an
empty release list, not an error. Same for a second ``end()`` on a job already
closed.

**``reason`` does not branch the release.** An abnormal end retires exactly the
same nodes as a clean one; the label is carried for logging and metrics only.
Branching on it would mean the leak simply moves to the failure path.

**The backstop for a ``job.end`` that never arrives.** The JS side sends
``job.end`` from the same ``finally`` that closes the bridge, so completion,
failure, cancellation and timeout all reach it — but a hard-killed client sends
nothing. Of the three candidate backstops:

- *Release on transport disconnect* — rejected. The WebSocket worker shares one
  :class:`~nodetool.worker.protocol.WorkerProtocolServer` across connections, so
  one client dropping would retire another client's live jobs. Correct only for
  the strictly 1:1 stdio transport, and a backstop that is right for one
  transport and wrong for the other is worse than none.
- *Rely on the existing reactive reclaim* — rejected as the only backstop. That
  is exactly the behaviour v4 exists to stop relying on: it trims under memory
  pressure, after the fact.
- *TTL sweep over jobs with no activity* — **chosen.** Transport-agnostic,
  needs no background timer (the sweep is driven opportunistically off
  ``start`` / ``note_execution`` / ``end``, all of which are already on the
  message path), and it cannot retire a job that is still executing because any
  activity refreshes the job's timestamp. A job idle longer than
  ``NODETOOL_WORKER_JOB_TTL_SECONDS`` (default 30 min) is treated as abandoned
  and released as if a ``job.end`` with ``reason="abandoned"`` had arrived.

The TTL is deliberately long: it is a leak backstop for a dead client, not a
scheduler. A live run whose node executions are minutes apart must never be
swept out from under itself, and models a currently-executing node holds are
protected regardless by ``ModelManager``'s execution scopes.

State is class-level, mirroring :class:`~nodetool.ml.core.model_manager.ModelManager`
— there is one worker process and one model cache in it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import ClassVar

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_JOB_TTL_SECONDS = 30 * 60


def _configured_ttl() -> float:
    """Seconds of inactivity after which a job is considered abandoned.

    Read per call (not cached at import) so tests and operators can retune it
    without reimporting the module. A non-numeric or non-positive value
    disables the sweep, which is the honest reading of "0 means off".
    """
    raw = os.environ.get("NODETOOL_WORKER_JOB_TTL_SECONDS")
    if raw is None:
        return float(DEFAULT_JOB_TTL_SECONDS)
    try:
        value = float(raw)
    except ValueError:
        log.warning(
            "Ignoring non-numeric NODETOOL_WORKER_JOB_TTL_SECONDS=%r; using %ss",
            raw,
            DEFAULT_JOB_TTL_SECONDS,
        )
        return float(DEFAULT_JOB_TTL_SECONDS)
    return value


@dataclass
class JobState:
    """One open run: who it belongs to, which nodes it has executed."""

    job_id: str
    workflow_id: str | None = None
    user_id: str | None = None
    node_ids: set[str] = field(default_factory=set)
    last_activity: float = 0.0


class JobRegistry:
    """Tracks open runs and the graph nodes each one has executed."""

    _jobs: ClassVar[dict[str, JobState]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        job_id: str,
        workflow_id: str | None = None,
        user_id: str | None = None,
    ) -> JobState:
        """Open (or refresh) a run boundary.

        Idempotent: a repeated ``job.start`` for the same id keeps the nodes
        already recorded rather than resetting them, so a client that retries
        the boundary cannot orphan the first half of its own run.
        """
        cls.sweep()
        state = cls._jobs.get(job_id)
        if state is None:
            state = JobState(job_id=job_id, workflow_id=workflow_id, user_id=user_id)
            cls._jobs[job_id] = state
            log.debug("Job %s opened (workflow=%s, user=%s)", job_id, workflow_id, user_id)
        else:
            # Fill in identity the first boundary could not name, but never
            # blank out what we already know.
            state.workflow_id = workflow_id or state.workflow_id
            state.user_id = user_id or state.user_id
        state.last_activity = time.monotonic()
        return state

    @classmethod
    def note_execution(cls, job_id: str | None, node_id: str | None) -> None:
        """Record that ``node_id`` executed as part of ``job_id``.

        Both arguments are optional on the wire, so both are tolerated as
        ``None`` here: an execute with no identity is the pre-v4 path and
        simply is not attributed to any run.

        A job the worker never saw a ``job.start`` for is created implicitly —
        the JS side treats ``job.start`` as optional (every ``execute`` carries
        its own ``job_id``), so requiring it would drop exactly the
        associations ``job.end`` needs.
        """
        if not job_id or not node_id:
            return
        cls.sweep()
        state = cls._jobs.get(job_id)
        if state is None:
            state = JobState(job_id=job_id)
            cls._jobs[job_id] = state
        state.node_ids.add(node_id)
        state.last_activity = time.monotonic()

    @classmethod
    def end(cls, job_id: str, reason: str | None = None) -> list[str]:
        """Close a run and release the models its nodes owned.

        Returns the node ids that were retired — empty for a job the worker
        never saw or already closed, which is a no-op by design.

        ``reason`` is logged, never branched on: an abnormal end retires the
        same nodes as a clean one.
        """
        cls.sweep(exclude=job_id)
        state = cls._jobs.pop(job_id, None)
        if state is None:
            log.debug("job.end for unknown job %s (reason=%s) — no-op", job_id, reason)
            return []

        node_ids = sorted(state.node_ids)
        log.info(
            "Job %s ended (reason=%s, workflow=%s): retiring %d node(s)",
            job_id,
            reason or "completed",
            state.workflow_id,
            len(node_ids),
        )
        cls._release(node_ids)
        return node_ids

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @classmethod
    def node_ids_for(cls, job_id: str) -> list[str]:
        """Node ids recorded for an open job (empty for an unknown job)."""
        state = cls._jobs.get(job_id)
        return sorted(state.node_ids) if state else []

    @classmethod
    def open_job_ids(cls) -> list[str]:
        return sorted(cls._jobs)

    @classmethod
    def has(cls, job_id: str) -> bool:
        """Whether the worker currently knows this job."""
        return job_id in cls._jobs

    @classmethod
    def clear(cls) -> None:
        """Drop all bookkeeping without releasing anything (tests, shutdown)."""
        cls._jobs.clear()

    # ------------------------------------------------------------------
    # TTL backstop
    # ------------------------------------------------------------------

    @classmethod
    def sweep(cls, *, exclude: str | None = None) -> list[str]:
        """Retire jobs idle beyond the TTL, as if ``job.end`` had arrived.

        Returns the swept job ids. ``exclude`` protects a job the caller is
        about to handle explicitly, so a sweep can never race its own caller.
        """
        ttl = _configured_ttl()
        if ttl <= 0:
            return []
        now = time.monotonic()
        expired = [
            job_id
            for job_id, state in cls._jobs.items()
            if job_id != exclude and (now - state.last_activity) > ttl
        ]
        for job_id in expired:
            state = cls._jobs.pop(job_id, None)
            if state is None:  # pragma: no cover — defensive
                continue
            log.warning(
                "Job %s had no activity for %.0fs and is presumed abandoned "
                "(no job.end arrived); releasing %d node(s)",
                job_id,
                now - state.last_activity,
                len(state.node_ids),
            )
            cls._release(sorted(state.node_ids))
        return expired

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _release(node_ids: list[str]) -> None:
        """Hand retired nodes to the model cache.

        ``release_nodes`` already refuses to evict a key another node still
        references or that an open execution scope pins, so this is safe to
        call while other nodes are mid-flight.
        """
        if not node_ids:
            return
        from nodetool.ml.core.model_manager import ModelManager

        ModelManager.release_nodes(node_ids)
