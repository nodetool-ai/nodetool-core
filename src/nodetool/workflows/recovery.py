"""
Workflow Recovery Service - State Table Based
=============================================

This module implements the recovery algorithm for resuming workflows after
crashes, restarts, or interruptions using MUTABLE STATE TABLES as source of truth.

IMPORTANT: This is Phase 5 of the architectural refactor. Event log is NOT used
for correctness - all recovery decisions are based on run_state and run_node_state tables.
"""

import asyncio
import os
import socket
from datetime import datetime, timedelta
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.models.run_lease import RunLease
from nodetool.models.run_node_state import RunNodeState
from nodetool.models.run_state import RunState
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


class WorkflowRecoveryService:
    """Service for recovering and resuming workflow executions using state tables."""

    def __init__(self, worker_id: str | None = None):
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self.lease_ttl = 60
        self.lease_renewal_interval = 30

    async def can_resume(self, run_id: str) -> bool:
        """Check if a run can be resumed by reading run_state table."""
        run_state = await RunState.get(run_id)
        if not run_state:
            return False
        return run_state.status in ["running", "failed", "suspended", "recovering"]

    async def acquire_run_lease(self, run_id: str) -> RunLease | None:
        """Acquire exclusive lease on a run."""
        return await RunLease.acquire(run_id, self.worker_id, self.lease_ttl)

    async def renew_lease_periodically(self, lease: RunLease, stop_event: asyncio.Event):
        """Background task to renew lease periodically."""
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

    async def get_incomplete_nodes(self, run_id: str) -> list[RunNodeState]:
        """Get nodes that are incomplete (need to be resumed)."""
        return await RunNodeState.find({
            "run_id": run_id,
            "status": {"$in": ["scheduled", "running"]},
        })

    async def get_suspended_nodes(self, run_id: str) -> list[RunNodeState]:
        """Get nodes that are suspended (need state restoration)."""
        return await RunNodeState.find({
            "run_id": run_id,
            "status": "suspended",
        })

    async def determine_resumption_points(
        self, run_id: str, graph: Graph
    ) -> dict[str, dict[str, Any]]:
        """Determine which nodes need to be resumed and how."""
        resumption_plan = {}

        incomplete = await self.get_incomplete_nodes(run_id)
        for node_state in incomplete:
            if node_state.status == "scheduled":
                resumption_plan[node_state.node_id] = {
                    "action": "reschedule",
                    "reason": "never_started",
                    "attempt": node_state.attempt,
                    "increment_attempt": False,
                }
            elif node_state.status == "running":
                resumption_plan[node_state.node_id] = {
                    "action": "reschedule",
                    "reason": "incomplete_execution",
                    "attempt": node_state.attempt + 1,
                    "increment_attempt": True,
                }

        suspended = await self.get_suspended_nodes(run_id)
        for node_state in suspended:
            resumption_plan[node_state.node_id] = {
                "action": "resume",
                "reason": "suspended",
                "attempt": node_state.attempt,
                "resume_state": node_state.resume_state_json,
                "increment_attempt": False,
            }

        return resumption_plan

    async def restore_node_state(
        self, node_id: str, resume_state: dict[str, Any], graph: Graph
    ) -> bool:
        """Restore state to a suspended node in the graph."""
        node = None
        for n in graph.nodes:
            if n._id == node_id:
                node = n
                break

        if not node or not hasattr(node, '_set_resuming_state'):
            return False

        try:
            node._set_resuming_state(resume_state)
            return True
        except Exception as e:
            log.error(f"Failed to restore state to node {node_id}: {e}")
            return False

    async def resume_workflow(
        self, run_id: str, graph: Graph, context: ProcessingContext
    ) -> tuple[bool, str]:
        """Resume a workflow execution."""
        if not await self.can_resume(run_id):
            return False, f"Run {run_id} cannot be resumed"

        lease = await self.acquire_run_lease(run_id)
        if not lease:
            return False, f"Run {run_id} is already being processed"

        try:
            run_state = await RunState.get(run_id)
            if not run_state:
                return False, f"Run {run_id} not found"

            await run_state.mark_recovering()
            resumption_plan = await self.determine_resumption_points(run_id, graph)

            if not resumption_plan:
                return True, "No incomplete nodes found"

            for node_id, plan in resumption_plan.items():
                if plan["action"] == "resume" and plan.get("resume_state"):
                    await self.restore_node_state(node_id, plan["resume_state"], graph)

            for node_id, plan in resumption_plan.items():
                node_state = await RunNodeState.get_or_create(run_id, node_id)
                if plan.get("increment_attempt"):
                    node_state.attempt = plan["attempt"]
                await node_state.mark_scheduled(node_state.attempt)

            return True, f"Successfully resumed run {run_id}"

        except Exception as e:
            log.error(f"Error resuming workflow {run_id}: {e}", exc_info=True)
            return False, f"Error resuming workflow: {str(e)}"
        finally:
            await lease.release()

    async def find_stuck_runs(self, max_age_minutes: int = 10) -> list[str]:
        """Find runs that are stuck (running with expired lease)."""
        runs = await RunState.find({"status": "running"}, limit=1000)

        stuck_runs = []
        for run_state in runs:
            lease = await RunLease.get(run_state.run_id)
            if not lease or lease.expires_at < datetime.now():
                stuck_runs.append(run_state.run_id)

        return stuck_runs
