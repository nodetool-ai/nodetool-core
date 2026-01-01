"""
RunNodeState model - authoritative source of truth for per-node execution state.

This table stores the current state of each node in a workflow run.
Unlike the event log, this is mutable and represents the current truth.
"""

from datetime import datetime
from typing import Any, Literal

from nodetool.models.base_model import DBField, DBIndex, DBModel

NodeStatus = Literal["idle", "scheduled", "running", "paused", "completed", "failed", "suspended"]


@DBIndex(columns=["run_id", "status"], name="idx_run_node_state_run_status")
@DBIndex(columns=["run_id", "node_id"], unique=True, name="idx_run_node_state_run_node")
class RunNodeState(DBModel):
    """
    Authoritative state for a single node in a workflow run.

    This is the source of truth for node execution state, attempts, and status.
    Recovery and scheduling read directly from this table.

    Key properties:
    - Mutable (status changes in place)
    - Source of truth (not derived from events)
    - One row per (run_id, node_id)
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "run_node_state",
            "primary_key": "id",  # Use simple PK, enforce uniqueness via index
        }

    # Simple primary key
    id: str = DBField(hash_key=True, default_factory=lambda: "")  # Will be set via before_save

    # Identifiers (unique together via index)
    run_id: str = DBField()
    node_id: str = DBField()

    # Current state
    status: str = DBField()  # idle | scheduled | running | completed | failed | suspended
    attempt: int = DBField(default=1)

    # Timestamps
    scheduled_at: datetime | None = DBField(default=None)
    started_at: datetime | None = DBField(default=None)
    completed_at: datetime | None = DBField(default=None)
    failed_at: datetime | None = DBField(default=None)
    suspended_at: datetime | None = DBField(default=None)
    updated_at: datetime = DBField(default_factory=datetime.now)

    # Failure information
    last_error: str | None = DBField(default=None)
    retryable: bool = DBField(default=False)

    # Suspension/resumption state
    suspension_reason: str | None = DBField(default=None)
    resume_state_json: dict[str, Any] = DBField(default_factory=dict)

    # Optional: outputs for completed nodes (may be large - consider external storage)
    outputs_json: dict[str, Any] = DBField(default_factory=dict)

    def before_save(self):
        """Update timestamp and set composite ID before saving."""
        self.updated_at = datetime.now()
        if not self.id:
            self.id = f"{self.run_id}::{self.node_id}"

    @classmethod
    async def get_node_state(cls, run_id: str, node_id: str) -> "RunNodeState | None":
        """
        Get state for a specific node.

        Args:
            run_id: The workflow run identifier
            node_id: The node identifier

        Returns:
            RunNodeState or None if not found
        """
        # Assuming adapter supports composite key get
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, Field, LogicalOperator

        condition = ConditionBuilder(
            ConditionGroup([Field("run_id").equals(run_id), Field("node_id").equals(node_id)], LogicalOperator.AND)
        )

        results, _ = await adapter.query(condition=condition, limit=1)
        if not results:
            return None
        return cls.from_dict(results[0])

    @classmethod
    async def get_or_create(cls, run_id: str, node_id: str) -> "RunNodeState":
        """
        Get existing node state or create idle state.

        Args:
            run_id: The workflow run identifier
            node_id: The node identifier

        Returns:
            RunNodeState instance
        """
        state = await cls.get_node_state(run_id, node_id)
        if state:
            return state

        state = cls(
            run_id=run_id,
            node_id=node_id,
            status="idle",
            attempt=1,
        )
        await state.save()
        return state

    async def mark_scheduled(self, attempt: int | None = None):
        """
        Mark node as scheduled.

        Args:
            attempt: Optional attempt number (defaults to current + 1 if already attempted)
        """
        self.status = "scheduled"
        if attempt is not None:
            self.attempt = attempt
        elif self.started_at is not None:
            # If previously started, increment attempt
            self.attempt += 1
        self.scheduled_at = datetime.now()
        await self.save()

    async def mark_running(self):
        """Mark node as running."""
        self.status = "running"
        self.started_at = datetime.now()
        await self.save()

    async def mark_completed(self, outputs: dict[str, Any] | None = None):
        """
        Mark node as completed.

        Args:
            outputs: Optional outputs to store (be careful with size)
        """
        self.status = "completed"
        self.completed_at = datetime.now()
        if outputs is not None:
            self.outputs_json = outputs
        await self.save()

    async def mark_failed(self, error: str, retryable: bool = False):
        """
        Mark node as failed.

        Args:
            error: Error message
            retryable: Whether this failure is retryable
        """
        self.status = "failed"
        self.failed_at = datetime.now()
        self.last_error = error
        self.retryable = retryable
        await self.save()

    async def mark_suspended(
        self,
        reason: str,
        state: dict[str, Any],
    ):
        """
        Mark node as suspended.

        Args:
            reason: Human-readable suspension reason
            state: State to save for resumption
        """
        self.status = "suspended"
        self.suspended_at = datetime.now()
        self.suspension_reason = reason
        self.resume_state_json = state
        await self.save()

    async def mark_resuming(self, state: dict[str, Any]):
        """
        Prepare node for resumption (transition back to running).

        Args:
            state: Resumed state
        """
        self.status = "running"
        self.resume_state_json = state
        self.started_at = datetime.now()
        await self.save()

    def is_incomplete(self) -> bool:
        """Check if node needs to be resumed."""
        return self.status in ["scheduled", "running"]

    def is_suspended(self) -> bool:
        """Check if node is suspended."""
        return self.status == "suspended"

    def is_retryable_failure(self) -> bool:
        """Check if node failed but can be retried."""
        return self.status == "failed" and self.retryable

    def is_paused(self) -> bool:
        """Check if node is paused."""
        return self.status == "paused"

    async def mark_paused(self):
        """Mark node as paused."""
        self.status = "paused"
        await self.save()

    @classmethod
    async def get_incomplete_nodes(cls, run_id: str) -> list["RunNodeState"]:
        """
        Get all incomplete nodes for a run.

        Args:
            run_id: The workflow run identifier

        Returns:
            List of RunNodeState for nodes that need resumption
        """
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, Field, LogicalOperator

        condition = ConditionBuilder(
            ConditionGroup(
                [Field("run_id").equals(run_id), Field("status").in_list(["scheduled", "running"])], LogicalOperator.AND
            )
        )

        results, _ = await adapter.query(condition=condition, limit=10000)
        return [cls.from_dict(row) for row in results]

    @classmethod
    async def get_suspended_nodes(cls, run_id: str) -> list["RunNodeState"]:
        """
        Get all suspended nodes for a run.

        Args:
            run_id: The workflow run identifier

        Returns:
            List of RunNodeState for suspended nodes
        """
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, Field, LogicalOperator

        condition = ConditionBuilder(
            ConditionGroup([Field("run_id").equals(run_id), Field("status").equals("suspended")], LogicalOperator.AND)
        )

        results, _ = await adapter.query(condition=condition, limit=10000)
        return [cls.from_dict(row) for row in results]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunNodeState":
        """Create RunNodeState from dictionary."""
        return cls(
            run_id=data["run_id"],
            node_id=data["node_id"],
            status=data.get("status", "idle"),
            attempt=data.get("attempt", 1),
            scheduled_at=data.get("scheduled_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            failed_at=data.get("failed_at"),
            suspended_at=data.get("suspended_at"),
            updated_at=data.get("updated_at", datetime.now()),
            last_error=data.get("last_error"),
            retryable=data.get("retryable", False),
            suspension_reason=data.get("suspension_reason"),
            resume_state_json=data.get("resume_state_json", {}),
            outputs_json=data.get("outputs_json", {}),
        )
