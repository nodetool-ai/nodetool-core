"""
RunProjection model for materialized workflow state views.

**DEPRECATED**: This projection is derived from audit events and should NOT
be used for correctness decisions.

IMPORTANT: Use run_state and run_node_state tables for all scheduling, recovery,
and correctness logic. RunProjection may be kept as a cached view for query
performance, but it is not authoritative.

For correctness, use:
- RunState.get(run_id) for run-level state
- RunNodeState.get_all_for_run(run_id) for node states
- RunNodeState.get_incomplete_nodes(run_id) for incomplete nodes

This projection may be removed in a future version.
"""

from datetime import datetime
from typing import Any

from nodetool.models.base_model import DBField, DBIndex, DBModel
from nodetool.models.condition_builder import Field


@DBIndex(columns=["status"], name="idx_run_projections_status")
class RunProjection(DBModel):
    """
    Materialized view of workflow run state derived from events.
    
    **DEPRECATED**: Do not use for correctness decisions. Use run_state and
    run_node_state tables instead.
    
    This projection is derived from audit events and may be kept as a cached
    view for query performance, but it is not the source of truth.
    
    For correctness, use the authoritative state tables:
    - RunState for run-level status
    - RunNodeState for node-level status
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "run_projections",
            "primary_key": "run_id",
        }

    run_id: str = DBField(hash_key=True)
    status: str = DBField()  # running, completed, failed, cancelled
    last_event_seq: int = DBField()
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)
    
    # Projection state stored as JSON
    node_states: dict[str, Any] = DBField(default_factory=dict)
    trigger_cursors: dict[str, Any] = DBField(default_factory=dict)
    pending_messages: list[dict[str, Any]] = DBField(default_factory=list)
    metadata: dict[str, Any] = DBField(default_factory=dict)

    def before_save(self):
        """Update timestamp before saving."""
        self.updated_at = datetime.now()

    @classmethod
    async def get_or_create(cls, run_id: str) -> "RunProjection":
        """
        Get existing projection or create a new one.
        
        Args:
            run_id: The workflow run identifier
            
        Returns:
            RunProjection instance
        """
        projection = await cls.get(run_id)
        if projection:
            return projection
        
        projection = cls(
            run_id=run_id,
            status="running",
            last_event_seq=-1,
            node_states={},
            trigger_cursors={},
            pending_messages=[],
            metadata={},
        )
        await projection.save()
        return projection

    async def update_from_event(self, event: Any) -> None:
        """
        Update projection state based on a single event.
        
        This method is idempotent: applying the same event multiple times
        produces the same result.
        
        Args:
            event: RunEvent to apply to this projection
        """
        # Skip if already processed
        if event.seq <= self.last_event_seq:
            return
        
        event_type = event.event_type
        payload = event.payload
        
        # Update based on event type
        if event_type == "RunCreated":
            self.status = "running"
            self.metadata = payload.get("metadata", {})
            
        elif event_type == "RunCompleted":
            self.status = "completed"
            
        elif event_type == "RunFailed":
            self.status = "failed"
            self.metadata["error"] = payload.get("error")
            
        elif event_type == "RunCancelled":
            self.status = "cancelled"
            self.metadata["reason"] = payload.get("reason")
            
        elif event_type == "RunSuspended":
            self.status = "suspended"
            self.metadata["suspension_reason"] = payload.get("reason")
            self.metadata["suspended_node_id"] = payload.get("node_id")
            self.metadata["suspension_metadata"] = payload.get("metadata", {})
            
        elif event_type == "RunResumed":
            self.status = "running"
            self.metadata["resumed_node_id"] = payload.get("node_id")
            self.metadata["resumption_metadata"] = payload.get("metadata", {})
            
        elif event_type == "NodeScheduled":
            node_id = event.node_id
            if node_id:
                self.node_states[node_id] = {
                    "status": "scheduled",
                    "attempt": payload.get("attempt", 1),
                    "scheduled_at": event.event_time.isoformat(),
                }
                
        elif event_type == "NodeStarted":
            node_id = event.node_id
            if node_id and node_id in self.node_states:
                self.node_states[node_id]["status"] = "started"
                self.node_states[node_id]["started_at"] = event.event_time.isoformat()
                
        elif event_type == "NodeCompleted":
            node_id = event.node_id
            if node_id and node_id in self.node_states:
                self.node_states[node_id]["status"] = "completed"
                self.node_states[node_id]["completed_at"] = event.event_time.isoformat()
                self.node_states[node_id]["outputs"] = payload.get("outputs", {})
                
        elif event_type == "NodeFailed":
            node_id = event.node_id
            if node_id and node_id in self.node_states:
                self.node_states[node_id]["status"] = "failed"
                self.node_states[node_id]["failed_at"] = event.event_time.isoformat()
                self.node_states[node_id]["error"] = payload.get("error")
                self.node_states[node_id]["retryable"] = payload.get("retryable", False)
                
        elif event_type == "NodeCheckpointed":
            node_id = event.node_id
            if node_id and node_id in self.node_states:
                self.node_states[node_id]["checkpoint"] = payload.get("checkpoint_data")
                self.node_states[node_id]["checkpointed_at"] = event.event_time.isoformat()
                
        elif event_type == "NodeSuspended":
            node_id = event.node_id
            if node_id:
                if node_id not in self.node_states:
                    self.node_states[node_id] = {}
                self.node_states[node_id]["status"] = "suspended"
                self.node_states[node_id]["suspended_at"] = event.event_time.isoformat()
                self.node_states[node_id]["suspension_reason"] = payload.get("reason")
                self.node_states[node_id]["suspension_state"] = payload.get("state", {})
                self.node_states[node_id]["suspension_metadata"] = payload.get("metadata", {})
                
        elif event_type == "NodeResumed":
            node_id = event.node_id
            if node_id and node_id in self.node_states:
                self.node_states[node_id]["status"] = "running"
                self.node_states[node_id]["resumed_at"] = event.event_time.isoformat()
                self.node_states[node_id]["resumed_state"] = payload.get("state", {})
                
        elif event_type == "TriggerRegistered":
            node_id = event.node_id
            if node_id:
                self.trigger_cursors[node_id] = payload.get("cursor", "")
                
        elif event_type == "TriggerCursorAdvanced":
            node_id = event.node_id
            if node_id:
                self.trigger_cursors[node_id] = payload.get("cursor")
                
        elif event_type == "OutboxEnqueued":
            message = {
                "message_id": payload.get("message_id"),
                "node_id": event.node_id,
                "edge_id": payload.get("edge_id"),
                "enqueued_at": event.event_time.isoformat(),
            }
            self.pending_messages.append(message)
            
        elif event_type == "OutboxSent":
            # Remove from pending
            message_id = payload.get("message_id")
            self.pending_messages = [
                m for m in self.pending_messages if m.get("message_id") != message_id
            ]
        
        # Update sequence tracker
        self.last_event_seq = event.seq

    @classmethod
    async def rebuild_from_events(cls, run_id: str) -> "RunProjection":
        """
        Rebuild projection from scratch by replaying all events.
        
        This is used for recovery or when projection is corrupted/lost.
        
        Args:
            run_id: The workflow run identifier
            
        Returns:
            Rebuilt RunProjection
        """
        from nodetool.models.run_event import RunEvent
        
        # Get all events for this run
        events = await RunEvent.get_events(run_id=run_id, limit=100000)
        
        # Create fresh projection
        projection = cls(
            run_id=run_id,
            status="running",
            last_event_seq=-1,
            node_states={},
            trigger_cursors={},
            pending_messages=[],
            metadata={},
        )
        
        # Replay all events
        for event in events:
            await projection.update_from_event(event)
        
        # Save rebuilt projection
        await projection.save()
        return projection

    def get_node_state(self, node_id: str) -> dict[str, Any] | None:
        """
        Get state for a specific node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node state dict or None if not found
        """
        return self.node_states.get(node_id)

    def get_trigger_cursor(self, node_id: str) -> Any:
        """
        Get cursor for a trigger node.
        
        Args:
            node_id: Trigger node identifier
            
        Returns:
            Cursor value or None if not found
        """
        return self.trigger_cursors.get(node_id)

    def is_node_completed(self, node_id: str) -> bool:
        """
        Check if a node has completed successfully.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if node status is "completed"
        """
        state = self.get_node_state(node_id)
        return state is not None and state.get("status") == "completed"

    def get_incomplete_nodes(self) -> list[str]:
        """
        Get list of node IDs that are scheduled or started but not completed.
        
        Returns:
            List of node IDs that need to be resumed
        """
        incomplete = []
        for node_id, state in self.node_states.items():
            status = state.get("status")
            if status in ["scheduled", "started"]:
                incomplete.append(node_id)
        return incomplete

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunProjection":
        """
        Create RunProjection instance from dictionary.
        
        Args:
            data: Dictionary containing projection data
            
        Returns:
            RunProjection instance
        """
        return cls(
            run_id=data["run_id"],
            status=data.get("status", "running"),
            last_event_seq=data.get("last_event_seq", -1),
            created_at=data.get("created_at", datetime.now()),
            updated_at=data.get("updated_at", datetime.now()),
            node_states=data.get("node_states", {}),
            trigger_cursors=data.get("trigger_cursors", {}),
            pending_messages=data.get("pending_messages", []),
            metadata=data.get("metadata", {}),
        )
