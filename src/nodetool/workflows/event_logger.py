"""
Event logger for workflow execution.

This module provides utilities for logging workflow events to the event log
and updating projections in real-time during workflow execution.
"""

import asyncio
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.models.run_event import EventType, RunEvent
from nodetool.models.run_projection import RunProjection

log = get_logger(__name__)


class WorkflowEventLogger:
    """
    Logger for recording workflow execution events.

    This class provides a high-level API for logging events and updating
    projections during workflow execution. It batches events when possible
    and handles projection updates asynchronously.
    """

    def __init__(self, run_id: str, enable_projection_updates: bool = True):
        """
        Initialize event logger.

        Args:
            run_id: The workflow run identifier
            enable_projection_updates: Whether to update projections on event append
        """
        self.run_id = run_id
        self.enable_projection_updates = enable_projection_updates
        self._projection: RunProjection | None = None
        self._projection_lock = asyncio.Lock()

    async def get_projection(self) -> RunProjection:
        """
        Get or create projection for this run.

        Returns:
            RunProjection instance
        """
        if self._projection is None:
            async with self._projection_lock:
                if self._projection is None:
                    self._projection = await RunProjection.get_or_create(self.run_id)
        return self._projection

    async def log_event(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        node_id: str | None = None,
        update_projection: bool = True,
    ) -> RunEvent:
        """
        Log a single event to the event log.

        Args:
            event_type: Type of event to log
            payload: Event-specific data
            node_id: Optional node identifier
            update_projection: Whether to update projection after logging

        Returns:
            The created RunEvent
        """
        try:
            # Append event to log
            event = await RunEvent.append_event(
                run_id=self.run_id,
                event_type=event_type,
                payload=payload,
                node_id=node_id,
            )

            log.debug(f"Logged event {event_type} for run {self.run_id} (seq={event.seq}, node={node_id})")

            # Update projection if enabled
            if update_projection and self.enable_projection_updates:
                projection = await self.get_projection()
                await projection.update_from_event(event)
                await projection.save()

            return event

        except Exception as e:
            log.error(
                f"Error logging event {event_type} for run {self.run_id}: {e}",
                exc_info=True,
            )
            raise

    # Convenience methods for common event types

    async def log_run_created(self, graph: dict, params: dict, user_id: str = ""):
        """Log RunCreated event."""
        return await self.log_event(
            "RunCreated",
            payload={"graph": graph, "params": params, "user_id": user_id},
        )

    async def log_run_completed(self, outputs: dict, duration_ms: int):
        """Log RunCompleted event."""
        return await self.log_event("RunCompleted", payload={"outputs": outputs, "duration_ms": duration_ms})

    async def log_run_failed(self, error: str, node_id: str | None = None):
        """Log RunFailed event."""
        return await self.log_event("RunFailed", payload={"error": error, "node_id": node_id})

    async def log_run_cancelled(self, reason: str):
        """Log RunCancelled event."""
        return await self.log_event("RunCancelled", payload={"reason": reason})

    async def log_run_suspended(self, node_id: str, reason: str, metadata: dict):
        """Log RunSuspended event."""
        return await self.log_event(
            "RunSuspended",
            payload={"node_id": node_id, "reason": reason, "metadata": metadata},
        )

    async def log_run_resumed(self, node_id: str, metadata: dict):
        """Log RunResumed event."""
        return await self.log_event(
            "RunResumed",
            payload={"node_id": node_id, "metadata": metadata},
        )

    async def log_node_scheduled(self, node_id: str, node_type: str, attempt: int = 1):
        """Log NodeScheduled event."""
        return await self.log_event(
            "NodeScheduled",
            payload={"node_type": node_type, "attempt": attempt},
            node_id=node_id,
        )

    async def log_node_started(self, node_id: str, attempt: int, inputs: dict):
        """Log NodeStarted event."""
        return await self.log_event(
            "NodeStarted",
            payload={"attempt": attempt, "inputs": inputs},
            node_id=node_id,
        )

    async def log_node_completed(self, node_id: str, attempt: int, outputs: dict, duration_ms: int):
        """Log NodeCompleted event."""
        return await self.log_event(
            "NodeCompleted",
            payload={
                "attempt": attempt,
                "outputs": outputs,
                "duration_ms": duration_ms,
            },
            node_id=node_id,
        )

    async def log_node_failed(self, node_id: str, attempt: int, error: str, retryable: bool = False):
        """Log NodeFailed event."""
        return await self.log_event(
            "NodeFailed",
            payload={"attempt": attempt, "error": error, "retryable": retryable},
            node_id=node_id,
        )

    async def log_node_suspended(self, node_id: str, reason: str, state: dict, metadata: dict):
        """Log NodeSuspended event."""
        return await self.log_event(
            "NodeSuspended",
            payload={"reason": reason, "state": state, "metadata": metadata},
            node_id=node_id,
        )

    async def log_node_resumed(self, node_id: str, state: dict):
        """Log NodeResumed event."""
        return await self.log_event(
            "NodeResumed",
            payload={"state": state},
            node_id=node_id,
        )

    async def log_node_checkpointed(self, node_id: str, attempt: int, checkpoint_data: dict):
        """Log NodeCheckpointed event."""
        return await self.log_event(
            "NodeCheckpointed",
            payload={"attempt": attempt, "checkpoint_data": checkpoint_data},
            node_id=node_id,
        )

    async def log_trigger_registered(self, node_id: str, trigger_type: str, config: dict, cursor: str = ""):
        """Log TriggerRegistered event."""
        return await self.log_event(
            "TriggerRegistered",
            payload={"trigger_type": trigger_type, "config": config, "cursor": cursor},
            node_id=node_id,
        )

    async def log_trigger_input_received(self, node_id: str, input_id: str, data: dict, cursor: str | None = None):
        """Log TriggerInputReceived event."""
        payload = {"input_id": input_id, "data": data}
        if cursor is not None:
            payload["cursor"] = cursor
        return await self.log_event("TriggerInputReceived", payload=payload, node_id=node_id)

    async def log_trigger_cursor_advanced(self, node_id: str, cursor: str, processed_count: int):
        """Log TriggerCursorAdvanced event."""
        return await self.log_event(
            "TriggerCursorAdvanced",
            payload={"cursor": cursor, "processed_count": processed_count},
            node_id=node_id,
        )

    async def log_outbox_enqueued(self, node_id: str, edge_id: str, message_id: str, data: dict):
        """Log OutboxEnqueued event."""
        return await self.log_event(
            "OutboxEnqueued",
            payload={"edge_id": edge_id, "message_id": message_id, "data": data},
            node_id=node_id,
        )

    async def log_outbox_sent(self, node_id: str, edge_id: str, message_id: str):
        """Log OutboxSent event."""
        return await self.log_event(
            "OutboxSent",
            payload={"edge_id": edge_id, "message_id": message_id},
            node_id=node_id,
        )

    async def flush_projection(self):
        """
        Manually flush projection to database.

        This is useful at checkpoints or boundaries where you want to ensure
        projection is persisted.
        """
        if self._projection is not None:
            await self._projection.save()
            log.debug(f"Flushed projection for run {self.run_id}")
