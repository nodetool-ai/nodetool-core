"""
State Manager - Queue-based single writer pattern for run_node_state updates.

This module eliminates SQLite write contention during parallel node execution by:
1. Queuing all state updates in memory
2. Processing updates in batches via a single writer task
3. Grouping multiple updates into single transactions

Architecture:
    Multiple NodeActors → Async Queue → Single Batch Writer → Database

Benefits:
- No more "database locked" errors
- Better performance (batched updates)
- Predictable behavior (ordered updates)
- Works with SQLite's write serialization

Usage:
    # Initialize once per workflow run
    manager = StateManager(run_id="job-123")
    await manager.start()

    # From node actors (non-blocking)
    await manager.update_node_state(
        node_id="node-1",
        status="running",
        started_at=datetime.now()
    )

    # Shutdown
    await manager.stop()
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.models.run_node_state import NodeStatus, RunNodeState

log = get_logger(__name__)


class StateUpdate:
    """Represents a single state update request."""

    def __init__(
        self,
        node_id: str,
        status: NodeStatus | None = None,
        attempt: int | None = None,
        scheduled_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        failed_at: datetime | None = None,
        suspended_at: datetime | None = None,
        last_error: str | None = None,
        retryable: bool | None = None,
        suspension_reason: str | None = None,
        resume_state_json: dict[str, Any] | None = None,
        outputs_json: dict[str, Any] | None = None,
    ):
        self.node_id = node_id
        self.status = status
        self.attempt = attempt
        self.scheduled_at = scheduled_at
        self.started_at = started_at
        self.completed_at = completed_at
        self.failed_at = failed_at
        self.suspended_at = suspended_at
        self.last_error = last_error
        self.retryable = retryable
        self.suspension_reason = suspension_reason
        self.resume_state_json = resume_state_json
        self.outputs_json = outputs_json
        self.timestamp = datetime.now()


class StateManager:
    """
    Queue-based state manager for run_node_state updates.

    Implements the single-writer pattern to eliminate SQLite write contention.
    All state updates are queued and processed by a single background task.

    Key features:
    - Non-blocking enqueue (immediate return to caller)
    - Batch processing (groups updates in single transactions)
    - Update coalescing (merges multiple updates for same node)
    - Error resilience (failed updates logged, don't crash workflow)
    - Graceful shutdown (flushes pending updates)

    Args:
        run_id: The workflow run identifier
        batch_size: Max updates per transaction (default: 10)
        batch_interval: Max seconds between flushes (default: 0.1)
    """

    def __init__(
        self,
        run_id: str,
        batch_size: int = 10,
        batch_interval: float = 0.1,
    ):
        self.run_id = run_id
        self.batch_size = batch_size
        self.batch_interval = batch_interval

        # Queue for incoming updates
        self.queue: asyncio.Queue[StateUpdate | None] = asyncio.Queue()

        # Background task
        self.writer_task: asyncio.Task | None = None

        # State cache (node_id -> RunNodeState)
        # Keeps latest known state to avoid redundant DB reads
        self.state_cache: dict[str, RunNodeState] = {}

        # Shutdown flag
        self._stopped = False

        # Stats
        self.stats = {
            "updates_queued": 0,
            "updates_processed": 0,
            "batches_written": 0,
            "errors": 0,
        }

    async def start(self):
        """Start the background writer task."""
        if self.writer_task is not None:
            log.warning(f"StateManager for run {self.run_id} already started")
            return

        log.info(f"Starting StateManager for run {self.run_id}")
        self.writer_task = asyncio.create_task(self._writer_loop())

    async def stop(self, timeout: float = 5.0):
        """
        Stop the background writer and flush pending updates.

        Args:
            timeout: Max seconds to wait for flush (default: 5.0)
        """
        if self.writer_task is None:
            return

        log.info(f"Stopping StateManager for run {self.run_id}, flushing pending updates...")

        # Signal shutdown
        self._stopped = True
        await self.queue.put(None)  # Sentinel value

        # Wait for writer to finish
        try:
            await asyncio.wait_for(self.writer_task, timeout=timeout)
        except TimeoutError:
            log.warning(f"StateManager writer task timed out after {timeout}s")
            self.writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.writer_task

        log.info(
            f"StateManager stopped. Stats: {self.stats['updates_processed']}/{self.stats['updates_queued']} updates processed, "
            f"{self.stats['batches_written']} batches written, {self.stats['errors']} errors"
        )

    async def update_node_state(
        self,
        node_id: str,
        status: NodeStatus | None = None,
        attempt: int | None = None,
        scheduled_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        failed_at: datetime | None = None,
        suspended_at: datetime | None = None,
        last_error: str | None = None,
        retryable: bool | None = None,
        suspension_reason: str | None = None,
        resume_state_json: dict[str, Any] | None = None,
        outputs_json: dict[str, Any] | None = None,
    ):
        """
        Queue a state update (non-blocking).

        This method immediately returns. The update will be processed
        asynchronously by the background writer task.

        Args:
            node_id: Node identifier
            status: New status (optional)
            attempt: Attempt number (optional)
            scheduled_at: Scheduled timestamp (optional)
            started_at: Started timestamp (optional)
            completed_at: Completed timestamp (optional)
            failed_at: Failed timestamp (optional)
            suspended_at: Suspended timestamp (optional)
            last_error: Error message (optional)
            retryable: Whether error is retryable (optional)
            suspension_reason: Suspension reason (optional)
            resume_state_json: Resumption state (optional)
            outputs_json: Node outputs (optional)
        """
        if self._stopped:
            log.warning(f"StateManager is stopped, ignoring update for node {node_id}")
            return

        update = StateUpdate(
            node_id=node_id,
            status=status,
            attempt=attempt,
            scheduled_at=scheduled_at,
            started_at=started_at,
            completed_at=completed_at,
            failed_at=failed_at,
            suspended_at=suspended_at,
            last_error=last_error,
            retryable=retryable,
            suspension_reason=suspension_reason,
            resume_state_json=resume_state_json,
            outputs_json=outputs_json,
        )

        await self.queue.put(update)
        self.stats["updates_queued"] += 1

    async def _writer_loop(self):
        """
        Background task that processes queued updates in batches.

        This is the single writer that eliminates database contention.
        It groups updates and writes them in transactions.
        """
        log.debug(f"StateManager writer loop started for run {self.run_id}")

        while True:
            # Collect batch of updates
            batch: list[StateUpdate] = []
            timeout_task = asyncio.create_task(asyncio.sleep(self.batch_interval))

            try:
                while len(batch) < self.batch_size:
                    # Wait for update or timeout
                    get_task = asyncio.create_task(self.queue.get())
                    done, _pending = await asyncio.wait(
                        [get_task, timeout_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if get_task in done:
                        update = await get_task

                        # Check for sentinel (shutdown signal)
                        if update is None:
                            log.debug("Received shutdown signal")
                            # Process remaining batch before exiting
                            if batch:
                                await self._process_batch(batch)
                            # Cancel timeout task
                            if not timeout_task.done():
                                timeout_task.cancel()
                            return

                        batch.append(update)

                        # Cancel timeout task if batch is full
                        if len(batch) >= self.batch_size:
                            if not timeout_task.done():
                                timeout_task.cancel()
                            break
                    else:
                        # Timeout reached, process what we have
                        get_task.cancel()
                        break

                # Process batch if non-empty
                if batch:
                    await self._process_batch(batch)

            except Exception as e:
                log.error(f"Error in StateManager writer loop: {e}", exc_info=True)
                self.stats["errors"] += 1
                # Continue processing (don't let one error crash the writer)

    async def _process_batch(self, batch: list[StateUpdate]):
        """
        Process a batch of state updates.

        This method:
        1. Coalesces updates (merges multiple updates for same node)
        2. Loads state from DB for nodes not in cache
        3. Applies updates in memory
        4. Saves all changes in a single pass

        Args:
            batch: List of state updates to process
        """
        try:
            # Coalesce updates by node_id (keep last update for each field)
            coalesced: dict[str, StateUpdate] = {}
            for update in batch:
                if update.node_id in coalesced:
                    # Merge updates (later values override earlier)
                    existing = coalesced[update.node_id]
                    if update.status is not None:
                        existing.status = update.status
                    if update.attempt is not None:
                        existing.attempt = update.attempt
                    if update.scheduled_at is not None:
                        existing.scheduled_at = update.scheduled_at
                    if update.started_at is not None:
                        existing.started_at = update.started_at
                    if update.completed_at is not None:
                        existing.completed_at = update.completed_at
                    if update.failed_at is not None:
                        existing.failed_at = update.failed_at
                    if update.suspended_at is not None:
                        existing.suspended_at = update.suspended_at
                    if update.last_error is not None:
                        existing.last_error = update.last_error
                    if update.retryable is not None:
                        existing.retryable = update.retryable
                    if update.suspension_reason is not None:
                        existing.suspension_reason = update.suspension_reason
                    if update.resume_state_json is not None:
                        existing.resume_state_json = update.resume_state_json
                    if update.outputs_json is not None:
                        existing.outputs_json = update.outputs_json
                else:
                    coalesced[update.node_id] = update

            # Load states for nodes not in cache
            for node_id in coalesced:
                if node_id not in self.state_cache:
                    # Try to load from DB
                    state = await RunNodeState.get_node_state(self.run_id, node_id)
                    if state is None:
                        # Create new state
                        state = RunNodeState(
                            run_id=self.run_id,
                            node_id=node_id,
                            status="idle",
                            attempt=1,
                        )
                    self.state_cache[node_id] = state

            # Apply updates
            for node_id, update in coalesced.items():
                state = self.state_cache[node_id]

                if update.status is not None:
                    state.status = update.status
                if update.attempt is not None:
                    state.attempt = update.attempt
                if update.scheduled_at is not None:
                    state.scheduled_at = update.scheduled_at
                if update.started_at is not None:
                    state.started_at = update.started_at
                if update.completed_at is not None:
                    state.completed_at = update.completed_at
                if update.failed_at is not None:
                    state.failed_at = update.failed_at
                if update.suspended_at is not None:
                    state.suspended_at = update.suspended_at
                if update.last_error is not None:
                    state.last_error = update.last_error
                if update.retryable is not None:
                    state.retryable = update.retryable
                if update.suspension_reason is not None:
                    state.suspension_reason = update.suspension_reason
                if update.resume_state_json is not None:
                    state.resume_state_json = update.resume_state_json
                if update.outputs_json is not None:
                    state.outputs_json = update.outputs_json

                # Update timestamp
                state.updated_at = datetime.now()

            # Save all states (ideally in a transaction, but save() is per-record)
            # TODO: If adapter supports batch operations, use that here
            for node_id in coalesced:
                state = self.state_cache[node_id]
                await state.save()

            # Update stats
            self.stats["updates_processed"] += len(batch)
            self.stats["batches_written"] += 1

            log.debug(
                f"Processed batch of {len(batch)} updates (coalesced to {len(coalesced)} nodes) for run {self.run_id}"
            )

        except Exception as e:
            log.error(f"Error processing state update batch: {e}", exc_info=True)
            self.stats["errors"] += 1

    async def get_node_state(self, node_id: str) -> RunNodeState | None:
        """
        Get current node state (from cache if available).

        Args:
            node_id: Node identifier

        Returns:
            RunNodeState or None if not found
        """
        if node_id in self.state_cache:
            return self.state_cache[node_id]

        # Load from DB
        state = await RunNodeState.get_node_state(self.run_id, node_id)
        if state:
            self.state_cache[node_id] = state
        return state
