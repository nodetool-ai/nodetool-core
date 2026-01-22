"""
Event logger for workflow execution.

This module provides utilities for logging workflow events to the event log.
Events are queued and written asynchronously to reduce database contention
and improve throughput.
"""

import asyncio
from contextlib import suppress
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.models.run_event import EventType, RunEvent

log = get_logger(__name__)


class WorkflowEventLogger:
    """
    Logger for recording workflow execution events.

    This class provides a high-level API for logging events during workflow
    execution. Events are queued and written asynchronously to reduce database
    contention and improve throughput.
    """

    BATCH_SIZE = 10
    FLUSH_INTERVAL = 0.1

    def __init__(self, run_id: str):
        """
        Initialize event logger.

        Args:
            run_id: The workflow run identifier
        """
        self.run_id = run_id
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._flush_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self):
        """Start the background event flushing task."""
        if self._is_running:
            return
        self._is_running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        log.debug(f"Started event logger flush task for run {self.run_id}")

    async def stop(self):
        """Stop the background event flushing task and flush remaining events."""
        self._is_running = False
        if self._flush_task:
            await self._flush_pending()
            self._flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._flush_task
            self._flush_task = None
            log.debug(f"Stopped event logger flush task for run {self.run_id}")

    async def _flush_loop(self):
        """Background loop that periodically flushes queued events."""
        while self._is_running:
            try:
                await asyncio.sleep(self.FLUSH_INTERVAL)
                await self._flush_pending()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in event flush loop: {e}", exc_info=True)

    async def _flush_pending(self):
        """Flush all pending events from the queue.

        Events are flushed with a short timeout to prevent blocking.
        Failed events are logged but not retried to avoid contention.
        """
        events_to_flush = []
        while not self._event_queue.empty() and len(events_to_flush) < self.BATCH_SIZE:
            try:
                event_data = self._event_queue.get_nowait()
                events_to_flush.append(event_data)
            except asyncio.QueueEmpty:
                break

        if not events_to_flush:
            return

        for event_data in events_to_flush:
            try:
                event_type = event_data["event_type"]
                payload = event_data["payload"]
                node_id = event_data.get("node_id")

                # Use a short timeout to prevent blocking on database locks
                try:
                    await asyncio.wait_for(
                        RunEvent.append_event(
                            run_id=self.run_id,
                            event_type=event_type,
                            payload=payload,
                            node_id=node_id,
                        ),
                        timeout=2.0,  # 2 second timeout for audit events
                    )
                    log.debug(f"Flushed event {event_type} for run {self.run_id}")
                except TimeoutError:
                    log.warning(f"Timeout flushing event {event_type} (non-fatal)")
            except Exception as e:
                # Log but don't retry - these are audit events, not critical
                log.warning(f"Error flushing event (non-fatal): {e}")

    async def log_event(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        node_id: str | None = None,
        blocking: bool = True,
    ) -> RunEvent | None:
        """
        Log a single event to the event log.

        Args:
            event_type: Type of event to log
            payload: Event-specific data
            node_id: Optional node identifier
            blocking: If True, wait for event to be written; if False, queue and return None

        Returns:
            The created RunEvent if blocking=True, or None if queued (non-blocking)
        """
        if blocking:
            return await RunEvent.append_event(
                run_id=self.run_id,
                event_type=event_type,
                payload=payload,
                node_id=node_id,
            )
        else:
            self._event_queue.put_nowait(
                {
                    "event_type": event_type,
                    "payload": payload,
                    "node_id": node_id,
                }
            )
            return None

    async def log_run_created(
        self,
        graph: dict[str, Any],
        params: dict[str, Any],
        user_id: str = "",
    ) -> RunEvent | None:
        """Log a RunCreated event."""
        return await self.log_event(
            event_type="RunCreated",
            payload={"graph": graph, "params": params, "user_id": user_id},
        )

    async def log_run_completed(
        self,
        outputs: dict[str, Any] | None = None,
        duration_ms: int = 0,
    ) -> RunEvent | None:
        """Log a RunCompleted event."""
        return await self.log_event(
            event_type="RunCompleted",
            payload={"outputs": outputs or {}, "duration_ms": duration_ms},
        )

    async def log_run_failed(self, error: str) -> RunEvent | None:
        """Log a RunFailed event."""
        return await self.log_event(
            event_type="RunFailed",
            payload={"error": error},
        )

    async def log_node_scheduled(
        self,
        node_id: str,
        node_type: str,
        attempt: int = 1,
    ) -> RunEvent | None:
        """Log a NodeScheduled event."""
        return await self.log_event(
            event_type="NodeScheduled",
            payload={"node_type": node_type, "attempt": attempt},
            node_id=node_id,
            blocking=False,
        )

    async def log_node_started(
        self,
        node_id: str,
        attempt: int = 1,
        inputs: dict[str, Any] | None = None,
    ) -> RunEvent | None:
        """Log a NodeStarted event."""
        return await self.log_event(
            event_type="NodeStarted",
            payload={"attempt": attempt, "inputs": inputs or {}},
            node_id=node_id,
            blocking=False,
        )

    async def log_node_completed(
        self,
        node_id: str,
        attempt: int = 1,
        outputs: dict[str, Any] | None = None,
        duration_ms: int = 0,
    ) -> RunEvent | None:
        """Log a NodeCompleted event."""
        return await self.log_event(
            event_type="NodeCompleted",
            payload={"attempt": attempt, "outputs": outputs or {}, "duration_ms": duration_ms},
            node_id=node_id,
            blocking=False,
        )

    async def log_node_failed(
        self,
        node_id: str,
        attempt: int = 1,
        error: str = "",
        retryable: bool = False,
    ) -> RunEvent | None:
        """Log a NodeFailed event."""
        return await self.log_event(
            event_type="NodeFailed",
            payload={"attempt": attempt, "error": error, "retryable": retryable},
            node_id=node_id,
            blocking=False,
        )

    async def log_run_cancelled(self, reason: str = "") -> RunEvent | None:
        """Log a RunCancelled event."""
        return await self.log_event(
            event_type="RunCancelled",
            payload={"reason": reason},
        )

    async def log_node_suspended(
        self,
        node_id: str,
        reason: str,
        attempt: int = 1,
        state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunEvent | None:
        """Log a NodeSuspended event."""
        return await self.log_event(
            event_type="NodeSuspended",
            payload={"reason": reason, "attempt": attempt, "state": state or {}, "metadata": metadata or {}},
            node_id=node_id,
        )

    async def log_run_suspended(
        self,
        reason: str,
        suspended_node_id: str,
    ) -> RunEvent | None:
        """Log a RunSuspended event."""
        return await self.log_event(
            event_type="RunSuspended",
            payload={"reason": reason, "suspended_node_id": suspended_node_id},
        )
