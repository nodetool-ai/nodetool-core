"""
Workflow recovery service for resumable workflows.

This module implements the recovery algorithm for resuming workflows after
crashes, restarts, or interruptions. It uses the event log and projections
to deterministically reconstruct execution state.
"""

import asyncio
import logging
import os
import socket
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.models.run_event import RunEvent
from nodetool.models.run_lease import RunLease
from nodetool.models.run_projection import RunProjection
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


class WorkflowRecoveryService:
    """
    Service for recovering and resuming workflow executions.
    
    This service implements the recovery algorithm that:
    1. Loads and validates projections from event log
    2. Determines which nodes need to be re-scheduled
    3. Ensures only one worker processes a run at a time
    4. Re-registers triggers with correct cursors
    """

    def __init__(self, worker_id: str | None = None):
        """
        Initialize recovery service.
        
        Args:
            worker_id: Unique identifier for this worker (default: hostname-pid)
        """
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self.lease_ttl = 60  # seconds
        self.lease_renewal_interval = 30  # seconds

    async def can_resume(self, run_id: str) -> bool:
        """
        Check if a run can be resumed.
        
        Args:
            run_id: The workflow run identifier
            
        Returns:
            True if the run exists and is in a resumable state
        """
        projection = await RunProjection.get(run_id)
        if not projection:
            return False
        
        # Only running, failed, or suspended runs can be resumed
        return projection.status in ["running", "failed", "suspended"]

    async def acquire_run_lease(self, run_id: str) -> RunLease | None:
        """
        Acquire exclusive lease on a run.
        
        Args:
            run_id: The workflow run identifier
            
        Returns:
            RunLease if acquired, None if already leased by another worker
        """
        return await RunLease.acquire(run_id, self.worker_id, self.lease_ttl)

    async def renew_lease_periodically(
        self, lease: RunLease, stop_event: asyncio.Event
    ):
        """
        Background task to renew lease periodically.
        
        Args:
            lease: The lease to renew
            stop_event: Event to signal when to stop renewing
        """
        try:
            while not stop_event.is_set():
                await asyncio.sleep(self.lease_renewal_interval)
                if not stop_event.is_set():
                    await lease.renew(self.lease_ttl)
                    log.debug(f"Renewed lease for run {lease.run_id}")
        except asyncio.CancelledError:
            log.debug(f"Lease renewal cancelled for run {lease.run_id}")
        except Exception as e:
            log.error(f"Error renewing lease for run {lease.run_id}: {e}")

    async def load_or_rebuild_projection(self, run_id: str) -> RunProjection:
        """
        Load projection from database or rebuild from event log.
        
        Args:
            run_id: The workflow run identifier
            
        Returns:
            RunProjection with current state
        """
        projection = await RunProjection.get(run_id)
        
        if not projection:
            log.info(f"No projection found for run {run_id}, rebuilding from events")
            projection = await RunProjection.rebuild_from_events(run_id)
        else:
            # Check for new events since projection was last updated
            events = await RunEvent.get_events(
                run_id=run_id, seq_gt=projection.last_event_seq, limit=1000
            )
            
            if events:
                log.info(
                    f"Found {len(events)} new events since last projection update, applying them"
                )
                for event in events:
                    await projection.update_from_event(event)
                await projection.save()
        
        return projection

    async def determine_resumption_points(
        self, projection: RunProjection, graph: Graph
    ) -> dict[str, dict[str, Any]]:
        """
        Determine which nodes need to be resumed and how.
        
        Args:
            projection: Current projection state
            graph: Workflow graph
            
        Returns:
            Dict mapping node_id to resumption plan
        """
        resumption_plan = {}
        
        for node_id, state in projection.node_states.items():
            status = state.get("status")
            
            if status == "started":
                # Node was running but didn't complete - re-schedule with new attempt
                resumption_plan[node_id] = {
                    "action": "reschedule",
                    "reason": "incomplete_execution",
                    "attempt": state.get("attempt", 1) + 1,
                }
                log.info(
                    f"Node {node_id} was started but not completed, will reschedule"
                )
                
            elif status == "scheduled":
                # Node was scheduled but never started - re-schedule same attempt
                resumption_plan[node_id] = {
                    "action": "reschedule",
                    "reason": "never_started",
                    "attempt": state.get("attempt", 1),
                }
                log.info(f"Node {node_id} was scheduled but never started, will reschedule")
                
            elif status == "failed" and state.get("retryable", False):
                # Node failed but is retryable - re-schedule with new attempt
                resumption_plan[node_id] = {
                    "action": "reschedule",
                    "reason": "retryable_failure",
                    "attempt": state.get("attempt", 1) + 1,
                }
                log.info(f"Node {node_id} failed but is retryable, will reschedule")
                
            elif status == "suspended":
                # Node is suspended - resume with saved state
                resumption_plan[node_id] = {
                    "action": "resume_suspended",
                    "reason": "resuming_suspended_node",
                    "attempt": state.get("attempt", 1),
                    "saved_state": state.get("suspension_state", {}),
                }
                log.info(
                    f"Node {node_id} is suspended, will resume with saved state "
                    f"(reason: {state.get('suspension_reason', 'unknown')})"
                )
        
        return resumption_plan

    async def schedule_resumption_events(
        self, run_id: str, resumption_plan: dict[str, dict[str, Any]], graph: Graph
    ):
        """
        Append events for nodes that need to resume.
        
        Args:
            run_id: The workflow run identifier
            resumption_plan: Plan from determine_resumption_points()
            graph: Workflow graph (needed for suspended node handling)
        """
        for node_id, plan in resumption_plan.items():
            if plan["action"] == "reschedule":
                await RunEvent.append_event(
                    run_id=run_id,
                    event_type="NodeScheduled",
                    node_id=node_id,
                    payload={
                        "attempt": plan["attempt"],
                        "reason": plan["reason"],
                    },
                )
                log.info(
                    f"Scheduled node {node_id} for attempt {plan['attempt']} (reason: {plan['reason']})"
                )
                
            elif plan["action"] == "resume_suspended":
                # Log NodeResumed event with saved state
                await RunEvent.append_event(
                    run_id=run_id,
                    event_type="NodeResumed",
                    node_id=node_id,
                    payload={
                        "state": plan["saved_state"],
                    },
                )
                
                # Also log RunResumed to mark the workflow as running again
                await RunEvent.append_event(
                    run_id=run_id,
                    event_type="RunResumed",
                    node_id=None,
                    payload={
                        "node_id": node_id,
                        "metadata": {"reason": plan["reason"]},
                    },
                )
                
                # Set the node to resuming mode with saved state
                node = graph.find_node(node_id)
                if node and hasattr(node, "_set_resuming_state"):
                    node._set_resuming_state(
                        saved_state=plan["saved_state"],
                        event_seq=-1,  # Will be updated from actual event
                    )
                    log.info(
                        f"Set node {node_id} to resuming mode with saved state "
                        f"(keys: {list(plan['saved_state'].keys())})"
                    )
                else:
                    log.warning(
                        f"Node {node_id} is not suspendable but has suspended state"
                    )
                
                log.info(f"Resuming suspended node {node_id}")

    async def reregister_triggers(
        self, projection: RunProjection, graph: Graph, context: ProcessingContext
    ):
        """
        Re-register trigger nodes with their last known cursors.
        
        Args:
            projection: Current projection state
            graph: Workflow graph
            context: Processing context
        """
        for node_id, cursor in projection.trigger_cursors.items():
            node = graph.find_node(node_id)
            if node is None:
                log.warning(f"Trigger node {node_id} not found in graph")
                continue
            
            # Check if node has resume_from_cursor method
            if not hasattr(node, "resume_from_cursor"):
                log.warning(
                    f"Node {node_id} does not support trigger resume (no resume_from_cursor method)"
                )
                continue
            
            try:
                await node.resume_from_cursor(cursor, context)
                log.info(f"Re-registered trigger {node_id} from cursor {cursor}")
            except Exception as e:
                log.error(f"Error re-registering trigger {node_id}: {e}")

    async def resume_workflow(
        self,
        run_id: str,
        graph: Graph,
        context: ProcessingContext,
    ) -> tuple[bool, str]:
        """
        Main entry point for resuming a workflow.
        
        This implements the complete recovery algorithm:
        1. Acquire lease
        2. Load/rebuild projection
        3. Determine resumption points
        4. Schedule resumption events
        5. Re-register triggers
        
        Args:
            run_id: The workflow run identifier
            graph: Workflow graph
            context: Processing context
            
        Returns:
            Tuple of (success, message)
        """
        # Check if run can be resumed
        if not await self.can_resume(run_id):
            return False, f"Run {run_id} cannot be resumed (not found or wrong state)"
        
        # Acquire lease
        lease = await self.acquire_run_lease(run_id)
        if not lease:
            return False, f"Run {run_id} is already being processed by another worker"
        
        log.info(f"Acquired lease for run {run_id}, starting recovery")
        
        # Start lease renewal task
        stop_renewal = asyncio.Event()
        renewal_task = asyncio.create_task(
            self.renew_lease_periodically(lease, stop_renewal)
        )
        
        try:
            # Load or rebuild projection
            projection = await self.load_or_rebuild_projection(run_id)
            
            # Determine what needs to be resumed
            resumption_plan = await self.determine_resumption_points(projection, graph)
            
            if not resumption_plan:
                log.info(f"No nodes need resumption for run {run_id}")
                return True, "No resumption needed"
            
            # Schedule resumption events
            await self.schedule_resumption_events(run_id, resumption_plan, graph)
            
            # Re-register triggers if any
            await self.reregister_triggers(projection, graph, context)
            
            log.info(
                f"Recovery complete for run {run_id}, scheduled {len(resumption_plan)} nodes"
            )
            return True, f"Resumed {len(resumption_plan)} nodes"
            
        except Exception as e:
            log.error(f"Error during recovery for run {run_id}: {e}", exc_info=True)
            return False, f"Recovery error: {str(e)}"
            
        finally:
            # Stop lease renewal and release lease
            stop_renewal.set()
            renewal_task.cancel()
            try:
                await renewal_task
            except asyncio.CancelledError:
                pass
            
            # Note: Don't release lease here if we're continuing execution
            # The WorkflowRunner should hold it during execution
            # await lease.release()
