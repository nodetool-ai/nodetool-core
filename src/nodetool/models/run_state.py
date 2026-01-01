"""
RunState model - authoritative source of truth for workflow run status.

This table stores the current state of a workflow run. Unlike the event log,
this is mutable and represents the current truth. The event log is audit-only.
"""

from datetime import datetime
from typing import Any, Literal

from nodetool.models.base_model import DBField, DBIndex, DBModel

RunStatus = Literal["running", "suspended", "paused", "completed", "failed", "cancelled", "recovering"]


@DBIndex(columns=["status"], name="idx_run_state_status")
@DBIndex(columns=["updated_at"], name="idx_run_state_updated")
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

    # Optional: version for optimistic locking
    version: int = DBField(default=0)

    def before_save(self):
        """Update timestamp and version before saving."""
        self.updated_at = datetime.now()
        self.version += 1

    @classmethod
    async def create_run(cls, run_id: str) -> "RunState":
        """
        Create a new run in running state.

        Args:
            run_id: The workflow run identifier

        Returns:
            Created RunState
        """
        run = cls(
            run_id=run_id,
            status="running",
            version=0,
        )
        await run.save()
        return run

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
