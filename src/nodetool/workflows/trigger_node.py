"""
Trigger Node Support for Suspendable Workflows
===============================================

This module provides trigger nodes that can suspend workflows during inactivity
and resume them when trigger events arrive. This consolidates trigger functionality
with the suspendable node mechanism for efficient resource usage.

Key Features:
- Trigger nodes extend SuspendableNode for state persistence
- Automatic suspension after inactivity timeout (default: 5 minutes)
- Wake-up mechanism when trigger events arrive
- Integration with event sourcing and recovery service
- Efficient resource usage (no indefinite running)

Usage:
------
```python
from nodetool.workflows.trigger_node import TriggerNode

class IntervalTrigger(TriggerNode):
    interval_seconds: int = 60

    async def process(self, context: ProcessingContext) -> dict:
        # Check if resuming from suspension
        if self.is_resuming():
            saved_state = await self.get_saved_state()
            last_trigger_time = saved_state.get('last_trigger_time')
            return {'resumed_at': datetime.now(), 'last_trigger': last_trigger_time}

        # Wait for trigger event with timeout
        try:
            event = await self.wait_for_trigger_event(
                timeout_seconds=self.get_inactivity_timeout()
            )
            return {'triggered_at': datetime.now(), 'event': event}
        except TriggerInactivityTimeout:
            # No activity for 5 minutes - suspend workflow
            await self.suspend_workflow(
                reason="Trigger inactivity timeout",
                state={
                    'last_trigger_time': datetime.now().isoformat(),
                    'interval_seconds': self.interval_seconds,
                },
            )
```

Workflow Flow:
--------------
1. Trigger workflow starts, enters trigger node
2. Node waits for trigger events (with timeout)
3. If event arrives: process and continue workflow
4. If timeout (5 min): suspend workflow with state
5. External trigger event arrives: wake up workflow
6. Recovery service resumes workflow at trigger node
7. Node retrieves saved state and continues
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

from nodetool.config.logging_config import get_logger
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.suspendable_node import SuspendableNode

log = get_logger(__name__)

DEFAULT_INACTIVITY_TIMEOUT = 300


class TriggerInactivityTimeout(Exception):
    """Exception raised when trigger node times out waiting for events."""

    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        super().__init__(f"No trigger activity for {timeout_seconds} seconds, suspending workflow")


class TriggerNode(SuspendableNode):
    """
    Base class for trigger nodes that can suspend workflows during inactivity.

    Trigger nodes wait for external events (intervals, webhooks, schedules, etc.)
    and suspend the workflow if no activity occurs within the timeout period.
    When a trigger event arrives for a suspended workflow, the recovery service
    resumes it automatically.

    Key features:
    - Extends SuspendableNode for state persistence
    - Auto-suspension after inactivity timeout
    - Wake-up on trigger event arrival
    - Efficient resource usage (no indefinite running)

    Subclasses should:
    1. Implement trigger-specific logic in process()
    2. Call wait_for_trigger_event() with timeout
    3. Handle TriggerInactivityTimeout to suspend
    4. Check is_resuming() to detect wake-ups
    """

    _inactivity_timeout_seconds: int = DEFAULT_INACTIVITY_TIMEOUT
    _last_activity_time: datetime | None = None
    _trigger_event_queue: asyncio.Queue | None = None
    _is_trigger_node: bool = True

    def is_trigger_node(self) -> bool:
        """
        Indicate that this is a trigger node.

        Returns:
            True - this node is a trigger node
        """
        return True

    def get_inactivity_timeout(self) -> int:
        """
        Get the inactivity timeout in seconds.

        Returns:
            Number of seconds before auto-suspension (default: 300 = 5 minutes)
        """
        return self._inactivity_timeout_seconds

    def set_inactivity_timeout(self, seconds: int) -> None:
        """
        Set the inactivity timeout.

        Args:
            seconds: Number of seconds of inactivity before suspension
        """
        if seconds < 1:
            raise ValueError("Inactivity timeout must be at least 1 second")
        self._inactivity_timeout_seconds = seconds

    def _update_activity_time(self) -> None:
        """Update the last activity timestamp."""
        self._last_activity_time = datetime.now()

    def get_last_activity_time(self) -> datetime | None:
        """
        Get the last activity timestamp.

        Returns:
            Last activity time or None if no activity yet
        """
        return self._last_activity_time

    def get_inactivity_duration(self) -> timedelta | None:
        """
        Get the duration since last activity.

        Returns:
            Time since last activity, or None if no activity yet
        """
        if self._last_activity_time is None:
            return None
        return datetime.now() - self._last_activity_time

    async def wait_for_trigger_event(
        self,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        Wait for a trigger event with timeout.

        This method blocks until:
        1. A trigger event arrives (returns event data)
        2. Timeout occurs (raises TriggerInactivityTimeout)

        Args:
            timeout_seconds: Timeout in seconds (default: use inactivity timeout)

        Returns:
            Dictionary containing trigger event data

        Raises:
            TriggerInactivityTimeout: If timeout occurs without events

        Example:
            try:
                event = await self.wait_for_trigger_event(timeout_seconds=300)
                return {'event_data': event}
            except TriggerInactivityTimeout:
                await self.suspend_workflow(
                    reason="No trigger activity",
                    state={'last_check': datetime.now().isoformat()},
                )
        """
        if timeout_seconds is None:
            timeout_seconds = self.get_inactivity_timeout()

        if self._trigger_event_queue is None:
            self._trigger_event_queue = asyncio.Queue()

        self._update_activity_time()

        try:
            event = await asyncio.wait_for(
                self._trigger_event_queue.get(),
                timeout=timeout_seconds,
            )

            self._update_activity_time()

            log.debug(f"Trigger node {self._id} received event after waiting {self.get_inactivity_duration()}")

            return event

        except TimeoutError:
            log.info(f"Trigger node {self._id} timed out after {timeout_seconds}s of inactivity")
            raise TriggerInactivityTimeout(timeout_seconds) from None

    async def send_trigger_event(self, event_data: dict[str, Any]) -> None:
        """
        Send a trigger event to this node (called externally).

        This method is called by external systems (webhooks, schedulers, etc.)
        to deliver trigger events to the node. If the workflow is suspended,
        this will wake it up.

        Args:
            event_data: Dictionary containing trigger event data

        Example:
            # External webhook handler
            trigger_node = get_trigger_node(node_id)
            await trigger_node.send_trigger_event({
                'webhook_id': 'abc123',
                'payload': request.json(),
                'timestamp': datetime.now().isoformat(),
            })
        """
        if self._trigger_event_queue is None:
            self._trigger_event_queue = asyncio.Queue()

        await self._trigger_event_queue.put(event_data)

        self._update_activity_time()

        log.info(f"Trigger node {self._id} received external event: {list(event_data.keys())}")

    async def should_suspend_for_inactivity(self) -> bool:
        """
        Check if node should suspend due to inactivity.

        Returns:
            True if inactivity timeout has been exceeded
        """
        duration = self.get_inactivity_duration()
        if duration is None:
            return False

        timeout = timedelta(seconds=self.get_inactivity_timeout())
        return duration >= timeout

    async def process_trigger_resumption(self, context: ProcessingContext) -> dict[str, Any]:
        """
        Handle resumption from suspended state (called by subclasses).

        This is a helper method that subclasses can call when resuming
        to get the saved trigger state.

        Args:
            context: Processing context

        Returns:
            Dictionary containing resumption information

        Example:
            async def process(self, context):
                if self.is_resuming():
                    return await self.process_trigger_resumption(context)

                # Normal trigger logic...
        """
        saved_state = await self.get_saved_state()

        log.info(
            f"Trigger node {self._id} resuming from suspension "
            f"(suspended at: {saved_state.get('suspended_at', 'unknown')})"
        )

        return {
            "status": "resumed",
            "trigger_node_id": self._id,
            "saved_state": saved_state,
            "resumed_at": datetime.now().isoformat(),
        }

    async def suspend_for_inactivity(
        self,
        additional_state: dict[str, Any] | None = None,
    ) -> None:
        """
        Suspend workflow due to trigger inactivity.

        This is a convenience method that calls suspend_workflow() with
        trigger-specific metadata.

        Args:
            additional_state: Optional additional state to save

        Example:
            # In process() method
            try:
                event = await self.wait_for_trigger_event()
                # Process event...
            except TriggerInactivityTimeout:
                await self.suspend_for_inactivity({
                    'interval_seconds': self.interval,
                })
        """
        state = {
            "suspended_at": datetime.now().isoformat(),
            "last_activity": self._last_activity_time.isoformat() if self._last_activity_time else None,
            "inactivity_timeout_seconds": self.get_inactivity_timeout(),
            "trigger_node_type": self.__class__.__name__,
        }

        if additional_state:
            state.update(additional_state)

        await self.suspend_workflow(
            reason=f"Trigger inactivity timeout ({self.get_inactivity_timeout()}s)",
            state=state,
            metadata={
                "trigger_node": True,
                "inactivity_suspension": True,
            },
        )


class TriggerWakeupService:
    """
    Service for waking up suspended trigger workflows when events arrive.

    This service:
    - Tracks suspended trigger workflows
    - Receives trigger events from external sources
    - Resumes workflows when trigger events arrive
    - Integrates with WorkflowRecoveryService
    """

    _instance: Optional["TriggerWakeupService"] = None
    _suspended_triggers: dict[str, dict]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._suspended_triggers = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "TriggerWakeupService":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = TriggerWakeupService()
        return cls._instance

    def register_suspended_trigger(
        self,
        workflow_id: str,
        node_id: str,
        trigger_metadata: dict[str, Any],
    ) -> None:
        """
        Register a suspended trigger workflow for wake-up.

        Args:
            workflow_id: The workflow identifier
            node_id: The trigger node identifier
            trigger_metadata: Metadata about the trigger
        """
        key = f"{workflow_id}:{node_id}"
        self._suspended_triggers[key] = {
            "workflow_id": workflow_id,
            "node_id": node_id,
            "metadata": trigger_metadata,
            "suspended_at": datetime.now(),
        }

        log.info(f"Registered suspended trigger: workflow={workflow_id}, node={node_id}")

    def unregister_suspended_trigger(
        self,
        workflow_id: str,
        node_id: str,
    ) -> None:
        """
        Unregister a suspended trigger workflow.

        Args:
            workflow_id: The workflow identifier
            node_id: The trigger node identifier
        """
        key = f"{workflow_id}:{node_id}"
        if key in self._suspended_triggers:
            del self._suspended_triggers[key]
            log.info(f"Unregistered suspended trigger: workflow={workflow_id}, node={node_id}")

    async def wake_up_trigger_workflow(
        self,
        workflow_id: str,
        node_id: str,
        trigger_event: dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Wake up a suspended trigger workflow.

        This method resumes a suspended workflow that was waiting for trigger events.

        Args:
            workflow_id: The workflow identifier
            node_id: The trigger node identifier
            trigger_event: The trigger event that woke up the workflow

        Returns:
            Tuple of (success, message)

        Example:
            service = TriggerWakeupService.get_instance()
            success, msg = await service.wake_up_trigger_workflow(
                workflow_id="wf-123",
                node_id="trigger-node-1",
                trigger_event={'webhook_id': 'abc', 'data': {...}},
            )
        """
        from nodetool.models.workflow import Workflow
        from nodetool.workflows.graph import Graph
        from nodetool.workflows.processing_context import ProcessingContext
        from nodetool.workflows.recovery import WorkflowRecoveryService

        log.info(f"Waking up trigger workflow: workflow={workflow_id}, node={node_id}")

        try:
            workflow = await Workflow.get(workflow_id)
            if not workflow:
                return False, f"Workflow {workflow_id} not found"

            graph = Graph.from_dict(workflow.graph)
            context = ProcessingContext()

            recovery = WorkflowRecoveryService()
            success, message = await recovery.resume_workflow(
                run_id=workflow_id,
                graph=graph,
                context=context,
            )

            if success:
                self.unregister_suspended_trigger(workflow_id, node_id)
                log.info(f"Successfully woke up trigger workflow: workflow={workflow_id}")
            else:
                log.error(f"Failed to wake up trigger workflow: workflow={workflow_id}, reason={message}")

            return success, message

        except Exception as e:
            log.error(
                f"Error waking up trigger workflow: workflow={workflow_id}, error={e}",
                exc_info=True,
            )
            return False, f"Error: {str(e)}"

    def list_suspended_triggers(self) -> dict[str, dict]:
        """
        List all suspended trigger workflows.

        Returns:
            Dictionary mapping trigger keys to metadata
        """
        return dict(self._suspended_triggers)
