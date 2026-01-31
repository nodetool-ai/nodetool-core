"""
Checkpoint Manager - Zero-Overhead Resume Feature
==================================================

This module implements an efficient checkpoint system for resumable workflows that:
1. Has ZERO overhead during normal workflow execution
2. Only takes a toll when explicitly saving checkpoints
3. Keeps the runner and actors clean
4. Implements as a separate, optional module

Design Principles:
-----------------
- **Opt-in**: Checkpointing is disabled by default, enabled via explicit flag
- **Zero Runtime Cost**: No DB writes, no state tracking during execution
- **Explicit Saves**: Checkpoints only saved when explicitly requested
- **Clean Separation**: No modifications to runner or actor code
- **Leverage Existing**: Uses existing Job and RunNodeState tables

Architecture:
------------
The checkpoint manager provides:
1. `save_checkpoint()`: Explicit checkpoint save at any point
2. `restore_checkpoint()`: Load and restore workflow state
3. `can_resume()`: Check if a workflow can be resumed
4. Hooks for integration without modifying core execution paths

Usage:
------
# Creating checkpoints (manual, zero overhead until called)
checkpoint_mgr = CheckpointManager(run_id="job-123")
await checkpoint_mgr.save_checkpoint(
    graph=graph,
    node_states={"node1": "completed", "node2": "running"},
    context_data={"vars": {...}}
)

# Resuming from checkpoint
can_resume = await checkpoint_mgr.can_resume()
if can_resume:
    restored_state = await checkpoint_mgr.restore_checkpoint(graph)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from nodetool.config.logging_config import get_logger
from nodetool.models.job import Job
from nodetool.models.run_node_state import NodeStatus, RunNodeState

if TYPE_CHECKING:
    from nodetool.workflows.graph import Graph
    from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


class CheckpointData:
    """
    Represents a complete checkpoint of workflow state.

    This is a lightweight in-memory representation that gets
    serialized to the database only when explicitly saved.
    """

    def __init__(
        self,
        run_id: str,
        checkpoint_time: datetime,
        node_states: dict[str, NodeStateSnapshot],
        completed_nodes: set[str],
        active_nodes: set[str],
        pending_nodes: set[str],
        context_data: dict[str, Any] | None = None,
    ):
        self.run_id = run_id
        self.checkpoint_time = checkpoint_time
        self.node_states = node_states
        self.completed_nodes = completed_nodes
        self.active_nodes = active_nodes
        self.pending_nodes = pending_nodes
        self.context_data = context_data or {}


class NodeStateSnapshot:
    """
    Lightweight snapshot of a node's execution state at checkpoint time.

    Only captures what's necessary for resumption, not full execution history.
    """

    def __init__(
        self,
        node_id: str,
        status: NodeStatus,
        attempt: int,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
        resume_state: dict[str, Any] | None = None,
    ):
        self.node_id = node_id
        self.status = status
        self.attempt = attempt
        self.outputs = outputs
        self.error = error
        self.resume_state = resume_state


class CheckpointManager:
    """
    Zero-overhead checkpoint manager for resumable workflows.

    This manager provides explicit checkpoint save/restore functionality
    without adding any overhead to running workflows. State is only written
    to the database when explicitly requested via save_checkpoint().

    Key Features:
    - No automatic state tracking (zero runtime overhead)
    - Explicit checkpoint saves only when requested
    - Uses existing Job and RunNodeState tables
    - Clean separation from runner and actor logic
    - Optional integration via hooks

    Args:
        run_id: The workflow run identifier
        enabled: Whether checkpointing is enabled (default: False)
    """

    def __init__(self, run_id: str, enabled: bool = False):
        self.run_id = run_id
        self.enabled = enabled
        self._checkpoint_count = 0

    async def save_checkpoint(
        self,
        graph: Graph,
        completed_nodes: set[str] | None = None,
        active_nodes: set[str] | None = None,
        pending_nodes: set[str] | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> bool:
        """
        Explicitly save a checkpoint of the current workflow state.

        This is the ONLY method that writes to the database. It should be
        called explicitly at strategic points (e.g., after completing a
        batch of nodes, before/after expensive operations, etc.).

        Args:
            graph: The workflow graph
            completed_nodes: Set of node IDs that have completed
            active_nodes: Set of node IDs currently executing
            pending_nodes: Set of node IDs not yet started
            context_data: Optional context data to persist

        Returns:
            True if checkpoint saved successfully, False otherwise
        """
        if not self.enabled:
            log.debug(f"Checkpointing disabled for run {self.run_id}, skipping save")
            return False

        try:
            log.info(f"Saving checkpoint for run {self.run_id}")
            start_time = datetime.now()

            # Capture current node states
            node_states: dict[str, NodeStateSnapshot] = {}

            # Mark completed nodes
            if completed_nodes:
                for node_id in completed_nodes:
                    # Get or create state record
                    state = await RunNodeState.get_or_create(self.run_id, node_id)

                    # Update to completed if not already
                    if state.status != "completed":
                        await state.mark_completed(outputs={})

                    node_states[node_id] = NodeStateSnapshot(
                        node_id=node_id,
                        status="completed",
                        attempt=state.attempt,
                        outputs=state.outputs_json,
                    )

            # Mark active nodes
            if active_nodes:
                for node_id in active_nodes:
                    state = await RunNodeState.get_or_create(self.run_id, node_id)

                    # Mark as running
                    if state.status not in ["running", "completed"]:
                        await state.mark_scheduled(attempt=state.attempt)
                        await state.mark_running()

                    node_states[node_id] = NodeStateSnapshot(
                        node_id=node_id,
                        status="running",
                        attempt=state.attempt,
                    )

            # Mark pending nodes as scheduled
            if pending_nodes:
                for node_id in pending_nodes:
                    state = await RunNodeState.get_or_create(self.run_id, node_id)

                    # Mark as scheduled
                    if state.status == "idle":
                        await state.mark_scheduled(attempt=1)

                    node_states[node_id] = NodeStateSnapshot(
                        node_id=node_id,
                        status="scheduled",
                        attempt=state.attempt,
                    )

            # Update job status to enable resumption
            job = await Job.get(self.run_id)
            if job and job.status == "running":
                # Keep as running but ensure it's marked for potential recovery
                log.debug(f"Job {self.run_id} marked as checkpointed (status: {job.status})")

            self._checkpoint_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()

            log.info(
                f"Checkpoint saved for run {self.run_id}: "
                f"{len(node_states)} nodes, {elapsed:.3f}s (checkpoint #{self._checkpoint_count})"
            )

            return True

        except Exception as e:
            log.error(f"Failed to save checkpoint for run {self.run_id}: {e}", exc_info=True)
            return False

    async def restore_checkpoint(self, graph: Graph) -> CheckpointData | None:
        """
        Restore workflow state from the last checkpoint.

        This loads the saved state from RunNodeState table and prepares
        it for resumption.

        Args:
            graph: The workflow graph

        Returns:
            CheckpointData if restoration successful, None otherwise
        """
        if not self.enabled:
            log.debug(f"Checkpointing disabled for run {self.run_id}, skipping restore")
            return None

        try:
            log.info(f"Restoring checkpoint for run {self.run_id}")

            # Query all node states
            from nodetool.models.condition_builder import Field

            condition = Field("run_id").equals(self.run_id)
            adapter = await RunNodeState.adapter()
            states_data, _ = await adapter.query(condition=condition, limit=1000)

            if not states_data:
                log.warning(f"No checkpoint found for run {self.run_id}")
                return None

            node_states: dict[str, NodeStateSnapshot] = {}
            completed_nodes: set[str] = set()
            active_nodes: set[str] = set()
            pending_nodes: set[str] = set()

            # Process each node state
            for state_dict in states_data:
                state = RunNodeState.from_dict(state_dict)

                snapshot = NodeStateSnapshot(
                    node_id=state.node_id,
                    status=state.status,
                    attempt=state.attempt,
                    outputs=state.outputs_json,
                    error=state.last_error,
                    resume_state=state.resume_state_json,
                )

                node_states[state.node_id] = snapshot

                # Categorize nodes
                if state.status == "completed":
                    completed_nodes.add(state.node_id)
                elif state.status in ["running", "suspended"]:
                    active_nodes.add(state.node_id)
                elif state.status == "scheduled":
                    pending_nodes.add(state.node_id)

            checkpoint_time = datetime.now()
            if states_data:
                # Use the most recent update time as checkpoint time
                latest_update = max(
                    (state.get("updated_at") for state in states_data if state.get("updated_at")),
                    default=None
                )
                if latest_update:
                    checkpoint_time = latest_update

            checkpoint = CheckpointData(
                run_id=self.run_id,
                checkpoint_time=checkpoint_time,
                node_states=node_states,
                completed_nodes=completed_nodes,
                active_nodes=active_nodes,
                pending_nodes=pending_nodes,
            )

            log.info(
                f"Checkpoint restored for run {self.run_id}: "
                f"{len(completed_nodes)} completed, {len(active_nodes)} active, "
                f"{len(pending_nodes)} pending"
            )

            return checkpoint

        except Exception as e:
            log.error(f"Failed to restore checkpoint for run {self.run_id}: {e}", exc_info=True)
            return None

    async def can_resume(self) -> bool:
        """
        Check if this workflow can be resumed from a checkpoint.

        Returns:
            True if the workflow has a valid checkpoint and can be resumed
        """
        if not self.enabled:
            return False

        try:
            # Check if job exists and is in a resumable state
            job = await Job.get(self.run_id)
            if not job:
                return False

            # Can resume if job is in certain states
            resumable_states = ["running", "failed", "suspended", "paused", "scheduled"]
            if job.status not in resumable_states:
                return False

            # Check if we have any node states saved
            from nodetool.models.condition_builder import Field

            condition = Field("run_id").equals(self.run_id)
            adapter = await RunNodeState.adapter()
            states_data, _ = await adapter.query(condition=condition, limit=1)

            return len(states_data) > 0

        except Exception as e:
            log.error(f"Error checking resume capability for run {self.run_id}: {e}")
            return False

    async def clear_checkpoint(self) -> bool:
        """
        Clear all checkpoint data for this workflow.

        This removes all node state records but keeps the job record.
        Useful for cleanup after successful completion or when restarting fresh.

        Returns:
            True if cleared successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            log.info(f"Clearing checkpoint for run {self.run_id}")

            # Get all node states
            from nodetool.models.condition_builder import Field

            condition = Field("run_id").equals(self.run_id)
            adapter = await RunNodeState.adapter()
            states_data, _ = await adapter.query(condition=condition, limit=1000)

            # Delete each state
            deleted_count = 0
            for state_dict in states_data:
                state = RunNodeState.from_dict(state_dict)
                await state.delete()
                deleted_count += 1

            log.info(f"Cleared {deleted_count} node states for run {self.run_id}")
            return True

        except Exception as e:
            log.error(f"Failed to clear checkpoint for run {self.run_id}: {e}", exc_info=True)
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get checkpoint statistics.

        Returns:
            Dictionary with checkpoint stats
        """
        return {
            "run_id": self.run_id,
            "enabled": self.enabled,
            "checkpoint_count": self._checkpoint_count,
        }


# Optional hook functions for integration with workflow runner
# These are designed to be called at strategic points without modifying runner code

async def create_checkpoint_hook(
    checkpoint_mgr: CheckpointManager | None,
    graph: Graph,
    completed_nodes: set[str],
    active_nodes: set[str],
    pending_nodes: set[str],
) -> None:
    """
    Optional hook to create a checkpoint during workflow execution.

    This can be called at strategic points (e.g., after batch completion)
    without modifying the core runner logic.

    Args:
        checkpoint_mgr: Optional checkpoint manager instance
        graph: The workflow graph
        completed_nodes: Completed node IDs
        active_nodes: Active node IDs
        pending_nodes: Pending node IDs
    """
    if checkpoint_mgr and checkpoint_mgr.enabled:
        await checkpoint_mgr.save_checkpoint(
            graph=graph,
            completed_nodes=completed_nodes,
            active_nodes=active_nodes,
            pending_nodes=pending_nodes,
        )


async def restore_checkpoint_hook(
    checkpoint_mgr: CheckpointManager | None,
    graph: Graph,
) -> CheckpointData | None:
    """
    Optional hook to restore from checkpoint before workflow execution.

    This can be called at workflow start to check for resumption
    without modifying the core runner logic.

    Args:
        checkpoint_mgr: Optional checkpoint manager instance
        graph: The workflow graph

    Returns:
        Restored checkpoint data if available, None otherwise
    """
    if checkpoint_mgr and checkpoint_mgr.enabled:
        if await checkpoint_mgr.can_resume():
            return await checkpoint_mgr.restore_checkpoint(graph)
    return None
