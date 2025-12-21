"""
Suspendable Node Support for Resumable Workflows
=================================================

This module provides support for suspendable nodes that can pause workflow execution,
save their state to the event log, and resume later. This enables long-running workflows
with human-in-the-loop operations, external API callbacks, or scheduled resumption.

Key Features:
- Automatic state persistence using event log
- Workflow suspension at suspendable nodes
- State restoration on workflow resumption
- Integration with existing event sourcing infrastructure

Usage:
------
```python
from nodetool.workflows.suspendable_node import SuspendableNode

class WaitForApproval(SuspendableNode):
    request_data: dict = {}
    
    async def process(self, context: ProcessingContext) -> dict:
        # Check if we're resuming from suspension
        if self.is_resuming():
            # Get saved state
            saved_state = await self.get_saved_state()
            return saved_state['approval_result']
        
        # First execution - suspend and wait for approval
        await self.suspend_workflow(
            reason="Waiting for approval",
            state={'request_data': self.request_data}
        )
        
        # This line is reached only after resumption
        return await self.get_saved_state()
```

Workflow Suspension Flow:
-------------------------
1. Node calls `suspend_workflow()` during execution
2. NodeSuspended event logged with node state
3. Workflow execution pauses (runner exits cleanly)
4. External system updates state (approval granted, data received, etc.)
5. Resume initiated via WorkflowRecoveryService
6. Node detects resumption, retrieves state, continues processing
"""

from typing import Any, Optional

from nodetool.config.logging_config import get_logger
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


class SuspendableNode(BaseNode):
    """
    Base class for nodes that can suspend workflow execution.
    
    Suspendable nodes can pause the entire workflow, save their state,
    and resume execution later. This is useful for:
    - Human-in-the-loop operations (approvals, feedback)
    - Waiting for external events (webhooks, callbacks)
    - Long-running tasks with checkpoints
    - Scheduled workflow resumption
    
    Subclasses should:
    1. Call `suspend_workflow()` when they need to suspend
    2. Check `is_resuming()` to detect resumption
    3. Use `get_saved_state()` to retrieve state after resumption
    4. Use `update_suspended_state()` to modify state while suspended
    """
    
    # Private attributes for suspension state
    _is_resuming_from_suspension: bool = False
    _saved_suspension_state: Optional[dict[str, Any]] = None
    _suspension_event_seq: Optional[int] = None
    
    def is_suspendable(self) -> bool:
        """
        Indicate that this node can suspend workflow execution.
        
        Returns:
            True - this node supports suspension
        """
        return True
    
    def is_resuming(self) -> bool:
        """
        Check if this node is resuming from a previous suspension.
        
        Returns:
            True if workflow is resuming at this node, False on first execution
        """
        return self._is_resuming_from_suspension
    
    async def get_saved_state(self) -> dict[str, Any]:
        """
        Get the state that was saved when the workflow was suspended.
        
        Returns:
            Dictionary of saved state, or empty dict if not resuming
            
        Raises:
            ValueError: If called when not resuming from suspension
        """
        if not self._is_resuming_from_suspension:
            raise ValueError("get_saved_state() can only be called when resuming from suspension")
        
        if self._saved_suspension_state is None:
            log.warning(f"No saved state found for suspended node {self._id}")
            return {}
        
        return self._saved_suspension_state
    
    async def suspend_workflow(
        self,
        reason: str,
        state: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Suspend workflow execution at this node.
        
        This method:
        1. Logs a NodeSuspended event with the provided state
        2. Updates the workflow projection to mark run as suspended
        3. Raises a WorkflowSuspendedException to cleanly exit execution
        
        Args:
            reason: Human-readable reason for suspension
            state: State to save (will be available on resumption)
            metadata: Optional metadata about the suspension
            
        Raises:
            WorkflowSuspendedException: Always raised to stop execution
            
        Example:
            await self.suspend_workflow(
                reason="Waiting for approval",
                state={'request_id': req_id, 'submitted_at': now()},
                metadata={'approver': 'admin@example.com'}
            )
        """
        log.info(f"Suspending workflow at node {self._id}: {reason}")
        
        # Get the event logger from context (if available)
        # Note: This will be integrated with the workflow runner
        try:
            from nodetool.workflows.processing_context import ProcessingContext
            # The context should provide access to the event logger
            # This is a placeholder - actual integration happens in workflow_runner
            
            # For now, log the intention
            log.info(
                f"Node {self._id} requesting suspension: {reason}\n"
                f"State keys: {list(state.keys())}\n"
                f"Metadata: {metadata}"
            )
            
        except Exception as e:
            log.error(f"Error preparing suspension: {e}")
            raise
        
        # Raise exception to stop workflow execution cleanly
        raise WorkflowSuspendedException(
            node_id=self._id,
            reason=reason,
            state=state,
            metadata=metadata or {}
        )
    
    async def update_suspended_state(
        self,
        state_updates: dict[str, Any],
        context: Optional[ProcessingContext] = None
    ) -> None:
        """
        Update the state of a suspended workflow (called externally).
        
        This method can be called by external systems to update the state
        while the workflow is suspended (e.g., approval granted, data received).
        
        Args:
            state_updates: Dictionary of state updates to merge with saved state
            context: Optional processing context
            
        Note:
            This is typically called via an API endpoint, not by the node itself
        """
        if not self._is_resuming_from_suspension:
            log.warning("update_suspended_state called on non-suspended node")
        
        # Merge updates with existing state
        if self._saved_suspension_state is None:
            self._saved_suspension_state = {}
        
        self._saved_suspension_state.update(state_updates)
        
        log.info(f"Updated suspended state for node {self._id}: {list(state_updates.keys())}")
    
    def _set_resuming_state(
        self,
        saved_state: dict[str, Any],
        event_seq: int
    ) -> None:
        """
        Internal method to set resumption state (called by workflow runner).
        
        Args:
            saved_state: The state that was saved at suspension
            event_seq: The sequence number of the suspension event
        """
        self._is_resuming_from_suspension = True
        self._saved_suspension_state = saved_state
        self._suspension_event_seq = event_seq
        
        log.debug(
            f"Node {self._id} set to resuming mode "
            f"(event_seq={event_seq}, state_keys={list(saved_state.keys())})"
        )


class WorkflowSuspendedException(Exception):
    """
    Exception raised when a node suspends workflow execution.
    
    This is a control flow exception used to cleanly exit workflow execution
    when a suspendable node requests suspension. The workflow runner catches
    this exception and performs the necessary cleanup and state persistence.
    
    Attributes:
        node_id: ID of the node that requested suspension
        reason: Human-readable reason for suspension
        state: State to persist for resumption
        metadata: Additional metadata about the suspension
    """
    
    def __init__(
        self,
        node_id: str,
        reason: str,
        state: dict[str, Any],
        metadata: dict[str, Any]
    ):
        self.node_id = node_id
        self.reason = reason
        self.state = state
        self.metadata = metadata
        
        super().__init__(
            f"Workflow suspended at node {node_id}: {reason}"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception details to dictionary for logging."""
        return {
            'node_id': self.node_id,
            'reason': self.reason,
            'state': self.state,
            'metadata': self.metadata,
        }
