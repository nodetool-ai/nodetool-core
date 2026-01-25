"""
Job model - unified model for job definition and execution state.

This model combines the previous Job (static definition) and RunState (mutable execution state)
into a single source of truth for workflow runs.
"""

from datetime import datetime, timedelta
from typing import Any, Literal, Optional

from pydantic import field_validator

from nodetool.config.logging_config import get_logger
from nodetool.models.base_model import DBField, DBIndex, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field

log = get_logger(__name__)

# Status type for jobs - mirrors the previous RunState statuses
JobStatus = Literal["scheduled", "running", "suspended", "paused", "completed", "failed", "cancelled", "recovering"]


@DBIndex(columns=["status"], name="idx_job_status")
@DBIndex(columns=["updated_at"], name="idx_job_updated")
@DBIndex(columns=["worker_id"], name="idx_job_worker")
@DBIndex(columns=["heartbeat_at"], name="idx_job_heartbeat")
@DBIndex(columns=["status", "heartbeat_at"], name="idx_job_recovery")
class Job(DBModel):
    """
    Unified model for workflow job definition and execution state.

    This is the single source of truth for job status and state.
    Combines the previous Job and RunState models.

    Key properties:
    - Mutable (status changes in place)
    - Source of truth (not derived from events)
    - Optimistic concurrency via version field
    """

    @classmethod
    def get_table_schema(cls):
        return {"table_name": "nodetool_jobs"}

    # Original Job fields
    id: str = DBField()
    user_id: str = DBField(default="")
    job_type: str = DBField(default="")
    workflow_id: str = DBField(default="")
    started_at: datetime = DBField(default_factory=datetime.now)
    finished_at: datetime | None = DBField(default=None)
    graph: dict = DBField(default_factory=dict)
    params: dict = DBField(default_factory=dict)
    error: str | None = DBField(default=None)
    cost: float | None = DBField(default=None)
    logs: list[dict] | None = DBField(default=None)

    # Fields from RunState - execution state
    status: str = DBField(default="scheduled")  # scheduled | running | suspended | completed | failed | cancelled | recovering
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
    async def find(cls, user_id: str, job_id: str):
        job = await cls.get(job_id)
        return job if job and job.user_id == user_id else None

    @classmethod
    async def create(
        cls,
        workflow_id: str,
        user_id: str,
        execution_strategy: str | None = None,
        worker_id: str | None = None,
        id: str | None = None,
        **kwargs,
    ):  # type: ignore[override]
        """
        Create a new job in scheduled state.

        Args:
            workflow_id: The workflow identifier
            user_id: The user identifier
            execution_strategy: Execution strategy (threaded/subprocess/docker)
            worker_id: ID of the worker creating this job
            id: Optional job ID (auto-generated if not provided)

        Returns:
            Created Job
        """
        job = await super().create(
            id=id or create_time_ordered_uuid(),
            workflow_id=workflow_id,
            user_id=user_id,
            status="scheduled",
            version=0,
            execution_strategy=execution_strategy,
            worker_id=worker_id,
            heartbeat_at=datetime.now() if worker_id else None,
            **kwargs,
        )
        return job

    @classmethod
    async def paginate(
        cls,
        user_id: str,
        workflow_id: Optional[str] = None,
        limit: int = 10,
        start_key: Optional[str] = None,
        started_after: Optional[datetime] = None,
    ):
        if workflow_id:
            items, key = await cls.query(
                Field("workflow_id").equals(workflow_id).and_(Field("id").greater_than(start_key or "")),
                limit=limit,
                columns=["id", "user_id", "job_type", "workflow_id", "started_at", "finished_at", "error", "cost", "status"],
            )
            return items, key
        elif user_id:
            items, key = await cls.query(
                Field("user_id").equals(user_id).and_(Field("id").greater_than(start_key or "")),
                limit=limit,
                columns=["id", "user_id", "job_type", "workflow_id", "started_at", "finished_at", "error", "cost", "status"],
            )
            return items, key
        elif started_after:
            items, key = await cls.query(
                Field("started_at").greater_than(started_after).and_(Field("id").greater_than(start_key or "")),
                limit=limit,
                columns=["id", "user_id", "job_type", "workflow_id", "started_at", "finished_at", "error", "cost", "status"],
            )
            return items, key
        else:
            raise ValueError("Must provide either user_id or workflow_id or started_after")

    # Methods from RunState - execution state management

    async def claim(self, worker_id: str):
        """Claim ownership of this job."""
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
        """Check if job has missed heartbeats."""
        if self.heartbeat_at is None:
            return True
        return (datetime.now() - self.heartbeat_at) > timedelta(minutes=threshold_minutes)

    def is_owned_by(self, worker_id: str) -> bool:
        """Check if job is owned by a specific worker."""
        return self.worker_id == worker_id

    async def acquire_with_cas(self, worker_id: str, expected_version: int) -> bool:
        """
        Attempt to claim job using optimistic locking.
        Returns True if successful, False if version mismatch.
        """
        if self.version != expected_version:
            return False

        self.worker_id = worker_id
        self.heartbeat_at = datetime.now()
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
        Mark job as suspended.

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
        """Mark job as resumed (back to running)."""
        self.status = "running"
        # Keep suspension fields for audit trail
        await self.save()

    async def mark_completed(self):
        """Mark job as completed."""
        self.status = "completed"
        self.completed_at = datetime.now()
        await self.save()

    async def mark_failed(self, error: str):
        """Mark job as failed."""
        self.status = "failed"
        self.failed_at = datetime.now()
        self.error_message = error
        await self.save()

    async def mark_cancelled(self):
        """Mark job as cancelled."""
        self.status = "cancelled"
        await self.save()

    async def mark_paused(self):
        """Mark job as paused."""
        self.status = "paused"
        await self.save()

    async def mark_recovering(self):
        """Mark job as recovering (transient state during recovery)."""
        self.status = "recovering"
        await self.save()

    async def mark_running(self):
        """Mark job as running."""
        self.status = "running"
        await self.save()

    def is_resumable(self) -> bool:
        """Check if job can be resumed."""
        return self.status in ["running", "suspended", "paused", "recovering", "failed"]

    def is_paused(self) -> bool:
        """Check if job is currently paused."""
        return self.status == "paused"

    def is_suspended(self) -> bool:
        """Check if job is currently suspended."""
        return self.status == "suspended"

    def is_complete(self) -> bool:
        """Check if job has reached terminal state."""
        return self.status in ["completed", "failed", "cancelled"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        """Create Job from dictionary."""
        return cls(
            id=data.get("id", data.get("run_id", "")),  # Support both id and run_id for compatibility
            user_id=data.get("user_id", ""),
            job_type=data.get("job_type", ""),
            workflow_id=data.get("workflow_id", ""),
            started_at=data.get("started_at", datetime.now()),
            finished_at=data.get("finished_at"),
            graph=data.get("graph", {}),
            params=data.get("params", {}),
            error=data.get("error"),
            cost=data.get("cost"),
            logs=data.get("logs"),
            status=data.get("status", "scheduled"),
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
