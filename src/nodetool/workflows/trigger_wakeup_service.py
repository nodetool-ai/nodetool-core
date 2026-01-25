"""
Trigger Wakeup Service - Durable Cross-Process Trigger Management
=================================================================

Provides durable trigger wakeup functionality without relying on in-memory state.
Uses the trigger_inputs and Job tables to coordinate trigger delivery across
multiple servers.

Key Features:
- Stores trigger inputs durably in trigger_inputs table
- Idempotent trigger delivery (by input_id)
- Cross-process coordination (no in-memory registry)
- Appends trigger inputs as inbox messages to wake nodes
- Works with suspended trigger workflows

Usage:
------
```python
# Add a trigger input (external event)
service = TriggerWakeupService()
await service.deliver_trigger_input(
    run_id="job-123",
    node_id="trigger-1",
    input_id="webhook-event-456",
    payload={"event": "user_action", "user_id": 789},
)

# Find suspended triggers awaiting inputs
suspended = await service.find_suspended_triggers()

# Wake up a suspended trigger workflow
await service.wake_up_trigger(run_id="job-123")
```

Architecture:
-------------
This implements Phase 4 of the architectural refactor:
- Trigger inputs stored in trigger_inputs table (source of truth)
- Duplicate input_ids are silently ignored
- Trigger inputs also appended as inbox messages
- No in-memory state (works across all servers)
- Recovery service handles actual resumption
"""

from datetime import datetime
from typing import Any, Optional

from nodetool.config.logging_config import get_logger
from nodetool.models.job import Job
from nodetool.models.trigger_input import TriggerInput
from nodetool.workflows.durable_inbox import DurableInbox

log = get_logger(__name__)


class TriggerWakeupService:
    """
    Durable trigger wakeup service for cross-process coordination.

    Stores trigger inputs durably and coordinates wakeup of suspended trigger workflows.
    No in-memory state - all coordination via database tables.
    """

    async def deliver_trigger_input(
        self,
        run_id: str,
        node_id: str,
        input_id: str,
        payload: Any,
        cursor: Optional[str] = None,
    ) -> bool:
        """
        Deliver a trigger input to a trigger node.

        Stores the input durably and appends it as an inbox message.
        If input_id already exists, this is a no-op (idempotent).

        Args:
            run_id: The workflow run ID
            node_id: The trigger node ID
            input_id: Unique input ID (for idempotency)
            payload: Trigger event payload
            cursor: Optional cursor value for ordered triggers

        Returns:
            True if input was newly created, False if it already existed
        """
        # Check if input already exists (idempotency)
        existing = await TriggerInput.get_by_input_id(input_id)
        if existing:
            log.debug(f"Trigger input {input_id} already exists (idempotent)")
            return False

        # Create trigger input record
        trigger_input = TriggerInput(
            run_id=run_id,
            node_id=node_id,
            input_id=input_id,
            payload_json=payload,
            cursor=cursor,
            processed=False,
            created_at=datetime.now(),
        )

        await trigger_input.save()
        log.info(f"Stored trigger input {input_id} for {run_id}/{node_id}" + (f" (cursor={cursor})" if cursor else ""))

        # Also append as inbox message to wake the node
        inbox = DurableInbox(run_id=run_id, node_id=node_id)
        await inbox.append(
            handle="trigger",
            payload=payload,
            message_id=f"trigger-{input_id}",
        )

        log.info(f"Appended trigger input {input_id} to inbox for {run_id}/{node_id}")

        return True

    async def get_pending_inputs(
        self,
        run_id: str,
        node_id: str,
        limit: int = 100,
    ) -> list[TriggerInput]:
        """
        Get pending (unprocessed) trigger inputs for a node.

        Args:
            run_id: The workflow run ID
            node_id: The trigger node ID
            limit: Maximum number of inputs to return

        Returns:
            List of pending trigger inputs in creation order
        """
        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, Field, LogicalOperator

        condition = ConditionBuilder(
            ConditionGroup(
                [
                    Field("run_id").equals(run_id),
                    Field("node_id").equals(node_id),
                    Field("processed").equals(False),
                ],
                LogicalOperator.AND,
            )
        )

        adapter = await TriggerInput.adapter()
        results, _ = await adapter.query(
            condition=condition,
            limit=limit,
        )

        return [TriggerInput.from_dict(row) for row in results]

    async def mark_processed(self, trigger_input: TriggerInput) -> None:
        """
        Mark a trigger input as processed.

        Args:
            trigger_input: The trigger input to mark as processed
        """
        trigger_input.processed = True
        trigger_input.processed_at = datetime.now()
        await trigger_input.save()

        log.debug(f"Marked trigger input {trigger_input.input_id} as processed")

    async def find_suspended_triggers(self) -> list[tuple[str, str]]:
        """
        Find suspended trigger workflows that have pending inputs.

        Returns:
            List of (run_id, node_id) tuples for suspended triggers with pending inputs
        """
        from nodetool.models.condition_builder import Field

        condition = Field("status").equals("suspended")
        adapter = await Job.adapter()
        suspended_runs, _ = await adapter.query(
            condition=condition,
            limit=1000,
        )

        results = []
        for job_data in suspended_runs:
            job = Job.from_dict(job_data)
            if job.suspended_node_id:
                pending = await self.get_pending_inputs(
                    run_id=job.id,
                    node_id=job.suspended_node_id,
                    limit=1,
                )

                if pending:
                    results.append((job.id, job.suspended_node_id))

        return results

    async def wake_up_trigger(
        self,
        run_id: str,
        recovery_service=None,
    ) -> bool:
        """
        Wake up a suspended trigger workflow.

        This is a convenience method that triggers the recovery service
        to resume the workflow. The actual resumption is handled by
        WorkflowRecoveryService.

        Args:
            run_id: The workflow run ID to wake up
            recovery_service: Optional recovery service instance
                            (if None, this just logs and returns)

        Returns:
            True if wake-up was initiated, False otherwise
        """
        # Check if run is actually suspended
        job = await Job.get(run_id)
        if not job:
            log.warning(f"Cannot wake trigger: run {run_id} not found")
            return False

        if job.status != "suspended":
            log.warning(f"Cannot wake trigger: run {run_id} is not suspended (status={job.status})")
            return False

        log.info(f"Waking up suspended trigger workflow {run_id}")

        # If recovery service is provided, initiate resumption
        if recovery_service:
            # This will be implemented in Phase 5 (recovery refactor)
            # For now, just log the intention
            log.info(f"Recovery service will resume {run_id}")
            # await recovery_service.resume_workflow(run_id, graph, context)
        else:
            log.info(f"No recovery service provided, trigger wake-up logged for {run_id}")

        return True

    async def cleanup_processed(
        self,
        run_id: str,
        node_id: str,
        older_than_hours: int = 24,
    ) -> int:
        """
        Clean up processed trigger inputs older than specified hours.

        Helps manage storage growth by removing old processed inputs.

        Args:
            run_id: The workflow run ID
            node_id: The trigger node ID
            older_than_hours: Delete processed inputs older than this many hours

        Returns:
            Number of inputs deleted
        """
        from datetime import timedelta

        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, Field, LogicalOperator

        cutoff = datetime.now() - timedelta(hours=older_than_hours)

        condition = ConditionBuilder(
            ConditionGroup(
                [
                    Field("run_id").equals(run_id),
                    Field("node_id").equals(node_id),
                    Field("processed").equals(True),
                    Field("processed_at").less_than(cutoff),
                ],
                LogicalOperator.AND,
            )
        )

        adapter = await TriggerInput.adapter()
        results, _ = await adapter.query(
            condition=condition,
            limit=1000,
        )

        inputs = [TriggerInput.from_dict(row) for row in results]

        count = 0
        for inp in inputs:
            await inp.delete()
            count += 1

        if count > 0:
            log.info(f"Cleaned up {count} processed trigger inputs for {run_id}/{node_id}")

        return count
