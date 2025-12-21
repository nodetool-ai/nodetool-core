"""
RunEvent model for append-only event log.

**AUDIT-ONLY**: This event log is for observability and debugging purposes only.
It is NOT the source of truth for workflow execution.

IMPORTANT: All scheduling, recovery, and correctness decisions must be based on
the mutable state tables (run_state, run_node_state) NOT on events.

Event writes may fail without breaking workflow execution. Event sequencing
may have gaps or be out-of-order. This is intentional and acceptable for
an audit log.

Use cases for events:
- Audit trails and compliance
- Debugging and troubleshooting
- Timeline visualization
- Observability and monitoring

Do NOT use events for:
- Determining what nodes to execute next
- Recovery and resumption logic
- Attempt tracking
- State restoration
"""

from datetime import datetime
from typing import Any, Literal

from nodetool.models.base_model import DBField, DBIndex, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field

# Event type literals for type safety
EventType = Literal[
    "RunCreated",
    "RunCompleted",
    "RunFailed",
    "RunCancelled",
    "RunSuspended",
    "RunResumed",
    "NodeScheduled",
    "NodeStarted",
    "NodeCheckpointed",
    "NodeCompleted",
    "NodeFailed",
    "NodeSuspended",
    "NodeResumed",
    "TriggerRegistered",
    "TriggerInputReceived",
    "TriggerCursorAdvanced",
    "OutboxEnqueued",
    "OutboxSent",
]


@DBIndex(columns=["run_id", "seq"], unique=True, name="idx_run_events_run_seq")
@DBIndex(columns=["run_id", "node_id"], name="idx_run_events_run_node")
@DBIndex(columns=["run_id", "event_type"], name="idx_run_events_run_type")
class RunEvent(DBModel):
    """
    Represents a single event in a workflow run's execution history.
    
    **AUDIT-ONLY**: Events form an append-only log for observability and debugging.
    They are NOT the source of truth. The mutable state tables (run_state, run_node_state)
    are the authoritative source for workflow state.
    
    Key characteristics:
    - Events are immutable once written
    - Sequence numbers are best-effort monotonic per run
    - Event writes may fail without breaking workflow execution
    - Used for audit trails, debugging, and timeline visualization
    
    Do NOT use events for scheduling, recovery, or correctness decisions
    """

    @classmethod
    def get_table_schema(cls):
        return {"table_name": "run_events"}

    id: str = DBField(hash_key=True, default_factory=create_time_ordered_uuid)
    run_id: str = DBField()
    seq: int = DBField()
    event_type: str = DBField()
    event_time: datetime = DBField(default_factory=datetime.now)
    node_id: str | None = DBField(default=None)
    payload: dict[str, Any] = DBField(default_factory=dict)

    @classmethod
    async def create(
        cls,
        run_id: str,
        seq: int,
        event_type: EventType,
        payload: dict[str, Any],
        node_id: str | None = None,
    ):
        """
        Create and append a new event to the log.
        
        Args:
            run_id: The workflow run identifier
            seq: Monotonic sequence number for this run
            event_type: Type of event (see EventType)
            payload: Event-specific data
            node_id: Optional node identifier for node-specific events
            
        Returns:
            The created RunEvent
            
        Note:
            This operation is idempotent. If an event with the same (run_id, seq)
            already exists, the existing event is returned without error.
        """
        event = cls(
            id=create_time_ordered_uuid(),
            run_id=run_id,
            seq=seq,
            event_type=event_type,
            event_time=datetime.now(),
            node_id=node_id,
            payload=payload,
        )
        await event.save()
        return event

    @classmethod
    async def get_next_seq(cls, run_id: str) -> int:
        """
        Get the next sequence number for a run.
        
        Args:
            run_id: The workflow run identifier
            
        Returns:
            The next available sequence number (max(seq) + 1, or 0 if no events)
        """
        adapter = await cls.adapter()
        events, _ = await adapter.query(
            condition=Field("run_id").equals(run_id),
            order_by="seq",
            reverse=True,
            limit=1,
            columns=["seq"],
        )
        if not events:
            return 0
        return events[0]["seq"] + 1

    @classmethod
    async def append_event(
        cls,
        run_id: str,
        event_type: EventType,
        payload: dict[str, Any],
        node_id: str | None = None,
    ) -> "RunEvent":
        """
        Append a new event to the log with automatic sequence number.
        
        This is the primary API for adding events. It automatically assigns
        the next sequence number and ensures idempotency.
        
        Args:
            run_id: The workflow run identifier
            event_type: Type of event (see EventType)
            payload: Event-specific data
            node_id: Optional node identifier for node-specific events
            
        Returns:
            The created RunEvent
        """
        seq = await cls.get_next_seq(run_id)
        return await cls.create(run_id, seq, event_type, payload, node_id)

    @classmethod
    async def get_events(
        cls,
        run_id: str,
        seq_gt: int | None = None,
        seq_lte: int | None = None,
        event_type: str | None = None,
        node_id: str | None = None,
        limit: int = 1000,
    ) -> list["RunEvent"]:
        """
        Query events for a run with optional filters.
        
        Args:
            run_id: The workflow run identifier
            seq_gt: Return events with seq > this value
            seq_lte: Return events with seq <= this value
            event_type: Filter by event type
            node_id: Filter by node ID
            limit: Maximum number of events to return
            
        Returns:
            List of RunEvent objects ordered by sequence number
        """
        adapter = await cls.adapter()
        
        conditions = [Field("run_id").equals(run_id)]
        
        if seq_gt is not None:
            conditions.append(Field("seq").greater_than(seq_gt))
        
        if seq_lte is not None:
            conditions.append(Field("seq").less_than_or_equal(seq_lte))
        
        if event_type is not None:
            conditions.append(Field("event_type").equals(event_type))
        
        if node_id is not None:
            conditions.append(Field("node_id").equals(node_id))
        
        # Build composite condition
        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, LogicalOperator
        
        condition = ConditionBuilder(
            ConditionGroup(conditions, LogicalOperator.AND)
        )
        
        results, _ = await adapter.query(
            condition=condition,
            order_by="seq",
            limit=limit,
        )
        
        return [cls.from_dict(row) for row in results]

    @classmethod
    async def get_last_event(
        cls, run_id: str, event_type: str | None = None, node_id: str | None = None
    ) -> "RunEvent | None":
        """
        Get the most recent event for a run, optionally filtered.
        
        Args:
            run_id: The workflow run identifier
            event_type: Optional event type filter
            node_id: Optional node ID filter
            
        Returns:
            The most recent matching event, or None if no events found
        """
        events = await cls.get_events(
            run_id=run_id,
            event_type=event_type,
            node_id=node_id,
            limit=1,
        )
        return events[0] if events else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunEvent":
        """
        Create a RunEvent instance from a dictionary.
        
        Args:
            data: Dictionary containing event data
            
        Returns:
            RunEvent instance
        """
        return cls(
            id=data.get("id", create_time_ordered_uuid()),
            run_id=data["run_id"],
            seq=data["seq"],
            event_type=data["event_type"],
            event_time=data.get("event_time", datetime.now()),
            node_id=data.get("node_id"),
            payload=data.get("payload", {}),
        )
