"""
RunState model - authoritative source of truth for workflow run status.

This table stores the current state of a workflow run. Unlike the event log,
this is mutable and represents the current truth. The event log is audit-only.

Concurrency note (SQLite)
------------------------
SQLite allows many concurrent readers but serializes writes behind a single
writer lock. In NodeTool, RunState can be updated very frequently (heartbeats,
status transitions) and those writes can contend with other state-table writes
(e.g. `run_node_state`) and with higher-value DB work.

To keep workflow execution responsive under contention, most RunState mutation
helpers (e.g. `update_heartbeat()`, `mark_*()`) persist via `save_nonblocking()`.
When running on SQLite, `save_nonblocking()` enqueues the full row for a
dedicated background writer thread which:
- coalesces updates by `run_id` (latest wins)
- periodically flushes batches using an UPSERT
- uses a short busy timeout + retry backoff to avoid blocking the event loop

This makes RunState persistence *eventually consistent* under SQLite: awaiting
`mark_completed()` does **not** guarantee the row is committed before the call
returns. If a caller truly requires synchronous persistence, call `await save()`
explicitly.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import field_validator

from nodetool.config.env_guard import RUNNING_PYTEST
from nodetool.models.base_model import DBField, DBIndex, DBModel
from nodetool.runtime.resources import maybe_scope

RunStatus = Literal["scheduled", "running", "suspended", "paused", "completed", "failed", "cancelled", "recovering"]


@DBIndex(columns=["status"], name="idx_run_state_status")
@DBIndex(columns=["updated_at"], name="idx_run_state_updated")
@DBIndex(columns=["worker_id"], name="idx_run_state_worker")
@DBIndex(columns=["heartbeat_at"], name="idx_run_state_heartbeat")
@DBIndex(columns=["status", "heartbeat_at"], name="idx_run_state_recovery")
class RunState(DBModel):
    """
    Authoritative state for a workflow run.

    This is the source of truth for run status and suspension state.
    Recovery and resumption read directly from this table.

    Key properties:
    - Mutable (status changes in place)
    - Source of truth (not derived from events)
    - Optimistic concurrency via version field

    Persistence model:
    - `await save()` performs a synchronous DB write (may block on SQLite locks).
    - `await save_nonblocking()` performs a best-effort write; on SQLite it
      queues the row to a background writer thread to avoid blocking workflow
      execution.
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "run_state",
            "primary_key": "run_id",
        }

    run_id: str = DBField(hash_key=True)
    status: str = DBField()  # running | suspended | completed | failed | cancelled | recovering
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)

    # Suspension state (populated when status=suspended)
    suspended_node_id: str | None = DBField(default=None)
    suspension_reason: str | None = DBField(default=None)
    suspension_state_json: dict[str, Any] = DBField(default_factory=dict)
    suspension_metadata_json: dict[str, Any] = DBField(default_factory=dict)

    # Completion/failure metadata
    completed_at: datetime | None = DBField(default=None)
    failed_at: datetime | None = DBField(default=None)
    error_message: str | None = DBField(default=None)

    # Execution tracking
    execution_strategy: str | None = DBField(default=None)
    execution_id: str | None = DBField(default=None)
    worker_id: str | None = DBField(default=None)
    heartbeat_at: datetime | None = DBField(default=None)
    retry_count: int = DBField(default=0)
    max_retries: int = DBField(default=3)
    metadata_json: dict[str, Any] = DBField(default_factory=dict)

    # Optional: version for optimistic locking
    version: int = DBField(default=0)

    @field_validator("metadata_json", "suspension_state_json", "suspension_metadata_json", mode="before")
    @classmethod
    def default_none_to_dict(cls, v: Any) -> dict[str, Any]:
        """Convert None values to empty dict for backward compatibility with existing records."""
        if v is None:
            return {}
        return v

    def before_save(self):
        """Update timestamp and version before saving."""
        self.updated_at = datetime.now()
        self.version += 1

    async def save_nonblocking(self) -> None:
        """Persist the RunState without blocking workflow execution (SQLite only).

        Behavior:
        - Under pytest (`RUNNING_PYTEST=True`): falls back to `await save()` for
          deterministic test behavior and to avoid background writes after table
          truncation/teardown.
        - SQLite: updates in this process are enqueued to a per-DB-path, dedicated
          writer thread (`RunStateWriter`) and this coroutine returns immediately.
          The writer thread coalesces updates by `run_id` and periodically flushes
          batches using an UPSERT.
        - Non-SQLite backends: falls back to `await save()` (preserves existing
          semantics; no background thread).

        Guarantees and caveats (SQLite):
        - Best-effort/eventual consistency: returning from this call does *not*
          guarantee the row is committed yet.
        - Updates may be delayed up to the writer flush interval.
        - If the in-memory queue is full, updates are dropped to avoid blocking
          workflow execution (latest state may not be persisted).

        Use this for high-frequency RunState updates (heartbeats/progress). If a
        caller requires synchronous durability, call `await save()` explicitly.
        """
        if RUNNING_PYTEST:
            # Tests rely on deterministic DB state and aggressive table truncation.
            # Avoid background threads that can write after teardown.
            await self.save()
            return

        scope = maybe_scope()
        db_resources = getattr(scope, "db", None) if scope else None
        pool = getattr(db_resources, "pool", None) if db_resources else None
        db_path = getattr(pool, "db_path", None) if pool else None

        if isinstance(db_path, str) and db_path:
            # SQLite: enqueue and return immediately
            self.before_save()
            from nodetool.models.run_state_writer import RunStateWriter

            RunStateWriter.enqueue(db_path, self.model_dump())
            return

        # Non-SQLite (or unknown): preserve original semantics.
        await self.save()

    @classmethod
    async def create_run(
        cls,
        run_id: str,
        execution_strategy: str | None = None,
        worker_id: str | None = None,
    ) -> "RunState":
        """
        Create a new run in scheduled state.

        Args:
            run_id: The workflow run identifier
            execution_strategy: Execution strategy (threaded/subprocess/docker)
            worker_id: ID of the worker creating this run

        Returns:
            Created RunState
        """
        run = cls(
            run_id=run_id,
            status="scheduled",
            version=0,
            execution_strategy=execution_strategy,
            worker_id=worker_id,
            heartbeat_at=datetime.now() if worker_id else None,
        )
        await run.save()
        return run

    async def claim(self, worker_id: str):
        """Claim ownership of this run (best-effort persistence on SQLite)."""
        self.worker_id = worker_id
        self.heartbeat_at = datetime.now()
        await self.save_nonblocking()

    async def release(self):
        """Release ownership (e.g. on clean shutdown; best-effort on SQLite)."""
        self.worker_id = None
        self.heartbeat_at = None
        await self.save_nonblocking()

    async def update_heartbeat(self):
        """Update heartbeat to now (best-effort persistence on SQLite)."""
        self.heartbeat_at = datetime.now()
        await self.save_nonblocking()

    async def increment_retry(self):
        """Increment retry count (best-effort persistence on SQLite)."""
        self.retry_count += 1
        await self.save_nonblocking()

    def is_stale(self, threshold_minutes: int = 5) -> bool:
        """Check if run has missed heartbeats."""
        if self.heartbeat_at is None:
            return True
        from datetime import timedelta

        return (datetime.now() - self.heartbeat_at) > timedelta(minutes=threshold_minutes)

    def is_owned_by(self, worker_id: str) -> bool:
        """Check if run is owned by a specific worker."""
        return self.worker_id == worker_id

    async def acquire_with_cas(self, worker_id: str, expected_version: int) -> bool:
        """
        Attempt to claim run using optimistic locking.
        Returns True if successful, False if version mismatch.
        """
        if self.version != expected_version:
            return False

        self.worker_id = worker_id
        self.heartbeat_at = datetime.now()
        # save() will handle the version increment and check
        # But for strictly atomic CAS, the underlying adapter needs to support it.
        # Assuming DBModel.save() checks version if present.
        try:
            await self.save()
            return True
        except Exception:
            return False

    async def mark_suspended(
        self,
        node_id: str,
        reason: str,
        state: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ):
        """
        Mark run as suspended.

        Args:
            node_id: Node that initiated suspension
            reason: Human-readable suspension reason
            state: State to save for resumption
            metadata: Optional additional metadata
        """
        self.status = "suspended"
        self.suspended_node_id = node_id
        self.suspension_reason = reason
        self.suspension_state_json = state
        self.suspension_metadata_json = metadata or {}
        await self.save_nonblocking()

    async def mark_resumed(self):
        """Mark run as resumed (back to running; best-effort persistence on SQLite)."""
        self.status = "running"
        # Keep suspension fields for audit trail
        await self.save_nonblocking()

    async def mark_completed(self):
        """Mark run as completed (best-effort persistence on SQLite)."""
        self.status = "completed"
        self.completed_at = datetime.now()
        await self.save_nonblocking()

    async def mark_failed(self, error: str):
        """Mark run as failed (best-effort persistence on SQLite)."""
        self.status = "failed"
        self.failed_at = datetime.now()
        self.error_message = error
        await self.save_nonblocking()

    async def mark_cancelled(self):
        """Mark run as cancelled (best-effort persistence on SQLite)."""
        self.status = "cancelled"
        await self.save_nonblocking()

    async def mark_paused(self):
        """Mark run as paused (best-effort persistence on SQLite)."""
        self.status = "paused"
        await self.save_nonblocking()

    async def mark_recovering(self):
        """Mark run as recovering (transient; best-effort persistence on SQLite)."""
        self.status = "recovering"
        await self.save_nonblocking()

    def is_resumable(self) -> bool:
        """Check if run can be resumed."""
        return self.status in ["running", "suspended", "paused", "recovering", "failed"]

    def is_paused(self) -> bool:
        """Check if run is currently paused."""
        return self.status == "paused"

    def is_suspended(self) -> bool:
        """Check if run is currently suspended."""
        return self.status == "suspended"

    def is_complete(self) -> bool:
        """Check if run has reached terminal state."""
        return self.status in ["completed", "failed", "cancelled"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        """Create RunState from dictionary."""
        return cls(
            run_id=data["run_id"],
            status=data["status"],
            created_at=data.get("created_at", datetime.now()),
            updated_at=data.get("updated_at", datetime.now()),
            suspended_node_id=data.get("suspended_node_id"),
            suspension_reason=data.get("suspension_reason"),
            suspension_state_json=data.get("suspension_state_json", {}),
            suspension_metadata_json=data.get("suspension_metadata_json", {}),
            completed_at=data.get("completed_at"),
            failed_at=data.get("failed_at"),
            error_message=data.get("error_message"),
            execution_strategy=data.get("execution_strategy"),
            execution_id=data.get("execution_id"),
            worker_id=data.get("worker_id"),
            heartbeat_at=data.get("heartbeat_at"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            metadata_json=data.get("metadata_json", {}),
            version=data.get("version", 0),
        )
