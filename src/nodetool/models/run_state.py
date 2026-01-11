"""
RunState model - authoritative source of truth for workflow run status.

This table stores the current state of a workflow run. Unlike the event log,
this is mutable and represents the current truth. The event log is audit-only.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import field_validator

from nodetool.models.base_model import DBField, DBIndex, DBModel

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
        """Claim ownership of this run."""
        self.worker_id = worker_id
        self.heartbeat_at = datetime.now()
        await self.save()

    async def release(self):
        """Release ownership (e.g. on clean shutdown)."""
        self.worker_id = None
        self.heartbeat_at = None
        await self.save()

    async def update_heartbeat(self):
        """Update heartbeat to now."""
        self.heartbeat_at = datetime.now()
        await self.save()

    async def increment_retry(self):
        """Increment retry count."""
        self.retry_count += 1
        await self.save()

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
        await self.save()

    async def mark_resumed(self):
        """Mark run as resumed (back to running)."""
        self.status = "running"
        # Keep suspension fields for audit trail
        await self.save()

    async def mark_completed(self):
        """Mark run as completed."""
        self.status = "completed"
        self.completed_at = datetime.now()
        await self.save()

    async def mark_failed(self, error: str):
        """Mark run as failed."""
        self.status = "failed"
        self.failed_at = datetime.now()
        self.error_message = error
        await self.save()

    async def mark_cancelled(self):
        """Mark run as cancelled."""
        self.status = "cancelled"
        await self.save()

    async def mark_paused(self):
        """Mark run as paused."""
        self.status = "paused"
        await self.save()

    async def mark_recovering(self):
        """Mark run as recovering (transient state during recovery)."""
        self.status = "recovering"
        await self.save()

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
    async def find_by_status(cls, status: str, limit: int = 100) -> list["RunState"]:
        """
        Find runs by status.

        Args:
            status: The status to filter by
            limit: Maximum number of runs to return

        Returns:
            List of runs with the specified status
        """
        from nodetool.models.condition_builder import Field

        condition = Field("status").equals(status)
        results, _ = await cls.query(condition, limit=limit)
        return results
