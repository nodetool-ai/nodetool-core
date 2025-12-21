"""
Checkpoint Manager for Workflow State Persistence.

This module provides functionality to save and restore the complete state of
a running workflow, enabling pause/resume and recovery after crashes.

Key responsibilities:
- Save workflow execution state to database
- Restore workflow state from database
- Handle serialization of complex state (inboxes, actors, etc.)
- Coordinate checkpointing with workflow execution
"""

import asyncio
from datetime import datetime
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.models.workflow_execution_state import (
    IndexedEdgeState,
    IndexedInputQueueState,
    IndexedNodeExecutionState,
    IndexedWorkflowExecutionState,
)
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner

log = get_logger(__name__)


class CheckpointManager:
    """
    Manages checkpointing and restoration of workflow execution state.
    
    This class handles the persistence layer for resumable workflows,
    coordinating between the in-memory execution state (WorkflowRunner,
    NodeActors, NodeInboxes) and the database models.
    """

    def __init__(self, workflow_execution_id: str | None = None):
        """
        Initialize a checkpoint manager.
        
        Args:
            workflow_execution_id: Optional ID of existing workflow execution to manage.
                                  If None, a new execution will be created on first checkpoint.
        """
        self.workflow_execution_id = workflow_execution_id
        self._checkpoint_lock = asyncio.Lock()
        
    async def create_execution_state(
        self,
        job_id: str,
        workflow_id: str,
        user_id: str,
        graph: dict[str, Any],
        params: dict[str, Any],
        device: str,
        disable_caching: bool,
        buffer_limit: int | None,
    ) -> str:
        """
        Create a new workflow execution state record in the database.
        
        Returns:
            The ID of the created workflow execution state.
        """
        execution_state = IndexedWorkflowExecutionState(
            job_id=job_id,
            workflow_id=workflow_id,
            user_id=user_id,
            graph=graph,
            params=params,
            device=device,
            disable_caching=disable_caching,
            buffer_limit=buffer_limit,
            status="running",
        )
        await execution_state.save()
        self.workflow_execution_id = execution_state.id
        log.info(
            f"Created workflow execution state: {execution_state.id} for job {job_id}"
        )
        return execution_state.id

    async def save_checkpoint(
        self,
        runner: WorkflowRunner,
        context: ProcessingContext,
        reason: str = "periodic",
    ) -> None:
        """
        Save a complete checkpoint of the workflow execution state.
        
        This captures:
        - Overall workflow status and metadata
        - Per-node execution states (status, properties, outputs)
        - Edge/inbox buffer states
        - Input queue state for streaming inputs
        
        Args:
            runner: The WorkflowRunner instance to checkpoint
            context: The ProcessingContext for the workflow
            reason: Reason for checkpoint (e.g., 'periodic', 'pause', 'error')
        """
        async with self._checkpoint_lock:
            try:
                if self.workflow_execution_id is None:
                    log.warning("Cannot save checkpoint: workflow_execution_id not set")
                    return

                log.debug(
                    f"Saving checkpoint for workflow execution {self.workflow_execution_id} "
                    f"(reason: {reason})"
                )

                # Update the main execution state
                await self._update_execution_state(runner, context)
                
                # Save node states
                await self._save_node_states(runner, context)
                
                # Save edge/inbox states
                await self._save_edge_states(runner, context)
                
                # Save input queue state
                await self._save_input_queue_state(runner)
                
                # Update checkpoint metadata
                execution_state = await IndexedWorkflowExecutionState.get(
                    self.workflow_execution_id
                )
                if execution_state:
                    await execution_state.update(
                        checkpoint_count=execution_state.checkpoint_count + 1,
                        last_checkpoint_at=datetime.now(),
                        updated_at=datetime.now(),
                    )

                log.info(
                    f"Checkpoint saved successfully for workflow execution "
                    f"{self.workflow_execution_id}"
                )

            except Exception as e:
                log.error(
                    f"Failed to save checkpoint for workflow execution "
                    f"{self.workflow_execution_id}: {e}",
                    exc_info=True,
                )

    async def update_status(self, runner: WorkflowRunner, context: ProcessingContext) -> None:
        """
        Update the workflow execution state status in the database.
        
        Args:
            runner: The WorkflowRunner instance
            context: The processing context
        """
        await self._update_execution_state(runner, context)

    async def _update_execution_state(
        self, runner: WorkflowRunner, context: ProcessingContext
    ) -> None:
        """Update the main workflow execution state record."""
        execution_state = await IndexedWorkflowExecutionState.get(
            self.workflow_execution_id
        )
        if execution_state:
            await execution_state.update(
                status=runner.status,
                updated_at=datetime.now(),
            )

    async def _save_node_states(
        self, runner: WorkflowRunner, context: ProcessingContext
    ) -> None:
        """Save the execution state for all nodes in the workflow."""
        if context.graph is None:
            return

        for node in context.graph.nodes:
            try:
                # Determine node status based on runner state
                if node._id in runner.active_processing_node_ids:
                    status = "running"
                else:
                    # Check if node has completed by looking at outputs
                    status = "pending"  # Default to pending
                    # TODO: Improve status detection based on actual node completion

                # Serialize node properties
                properties = {}
                try:
                    # Get all properties with their current values
                    for prop in node.properties():
                        prop_name = prop.name
                        if hasattr(node, prop_name):
                            value = getattr(node, prop_name)
                            # Only save serializable values
                            if self._is_serializable(value):
                                properties[prop_name] = value
                except Exception as e:
                    log.debug(
                        f"Failed to serialize properties for node {node._id}: {e}"
                    )

                # Check if state already exists
                from nodetool.models.condition_builder import Field
                
                existing_states, _ = await IndexedNodeExecutionState.query(
                    Field("workflow_execution_id").equals(self.workflow_execution_id)
                    .and_(Field("node_id").equals(node._id)),
                    limit=1,
                )
                
                if existing_states and len(existing_states) > 0:
                    # Update existing state
                    state_dict = existing_states[0]
                    state = await IndexedNodeExecutionState.get(state_dict["id"])
                    if state:
                        await state.update(
                            status=status,
                            properties=properties,
                            updated_at=datetime.now(),
                        )
                else:
                    # Create new state
                    state = IndexedNodeExecutionState(
                        workflow_execution_id=self.workflow_execution_id,
                        node_id=node._id,
                        node_type=node.get_node_type(),
                        node_name=node.get_title(),
                        status=status,
                        properties=properties,
                    )
                    await state.save()

            except Exception as e:
                log.error(
                    f"Failed to save state for node {node._id}: {e}", exc_info=True
                )

    async def _save_edge_states(
        self, runner: WorkflowRunner, context: ProcessingContext
    ) -> None:
        """Save the state of all edges and their inboxes."""
        if context.graph is None:
            return

        for edge in context.graph.edges:
            try:
                # Get inbox for target node
                inbox = runner.node_inboxes.get(edge.target)
                if inbox is None:
                    continue

                # Serialize buffered values for this handle
                buffered_values = []
                try:
                    buf = inbox._buffers.get(edge.targetHandle, [])
                    for value in buf:
                        if self._is_serializable(value):
                            buffered_values.append(value)
                except Exception as e:
                    log.debug(
                        f"Failed to serialize buffer for edge {edge.id}: {e}"
                    )

                # Get open upstream count
                open_count = inbox._open_counts.get(edge.targetHandle, 0)
                
                # Check if edge is streaming
                is_streaming = runner.edge_streams(edge)
                
                # Get message counter
                message_count = runner._edge_counters.get(edge.id or "", 0)

                # Check if state already exists
                from nodetool.models.condition_builder import Field
                
                existing_states, _ = await IndexedEdgeState.query(
                    Field("workflow_execution_id").equals(self.workflow_execution_id)
                    .and_(Field("edge_id").equals(edge.id or "")),
                    limit=1,
                )
                
                if existing_states and len(existing_states) > 0:
                    # Update existing state
                    state_dict = existing_states[0]
                    state = await IndexedEdgeState.get(state_dict["id"])
                    if state:
                        await state.update(
                            buffered_values=buffered_values,
                            open_upstream_count=open_count,
                            is_streaming=is_streaming,
                            message_count=message_count,
                            updated_at=datetime.now(),
                        )
                else:
                    # Create new state
                    state = IndexedEdgeState(
                        workflow_execution_id=self.workflow_execution_id,
                        edge_id=edge.id or "",
                        source_node_id=edge.source,
                        source_handle=edge.sourceHandle,
                        target_node_id=edge.target,
                        target_handle=edge.targetHandle,
                        buffered_values=buffered_values,
                        open_upstream_count=open_count,
                        is_streaming=is_streaming,
                        message_count=message_count,
                    )
                    await state.save()

            except Exception as e:
                log.error(
                    f"Failed to save state for edge {edge.id}: {e}", exc_info=True
                )

    async def _save_input_queue_state(self, runner: WorkflowRunner) -> None:
        """Save the state of the input queue for streaming inputs."""
        try:
            # For now, we don't persist pending events in the input queue
            # as they are typically ephemeral. In a production system, you
            # might want to persist them for full resumability.
            pending_events = []
            
            # Track state of each input (whether EOS has been sent)
            input_states = {}
            
            # Check if state already exists
            from nodetool.models.condition_builder import Field
            
            existing_states, _ = await IndexedInputQueueState.query(
                Field("workflow_execution_id").equals(self.workflow_execution_id),
                limit=1,
            )
            
            if existing_states and len(existing_states) > 0:
                # Update existing state
                state_dict = existing_states[0]
                state = await IndexedInputQueueState.get(state_dict["id"])
                if state:
                    await state.update(
                        pending_events=pending_events,
                        input_states=input_states,
                        updated_at=datetime.now(),
                    )
            else:
                # Create new state
                state = IndexedInputQueueState(
                    workflow_execution_id=self.workflow_execution_id,
                    pending_events=pending_events,
                    input_states=input_states,
                )
                await state.save()

        except Exception as e:
            log.error(
                f"Failed to save input queue state: {e}", exc_info=True
            )

    def _is_serializable(self, value: Any) -> bool:
        """
        Check if a value is serializable to JSON.
        
        Returns True for basic types (str, int, float, bool, None, dict, list).
        Returns False for complex objects that can't be directly serialized.
        """
        if value is None:
            return True
        if isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, dict):
            return all(
                self._is_serializable(k) and self._is_serializable(v)
                for k, v in value.items()
            )
        if isinstance(value, list):
            return all(self._is_serializable(item) for item in value)
        # For asset refs and other complex types, we'd need special handling
        # For now, we skip them to avoid errors
        return False

    async def mark_paused(self) -> None:
        """Mark the workflow execution as paused."""
        if self.workflow_execution_id:
            execution_state = await IndexedWorkflowExecutionState.get(
                self.workflow_execution_id
            )
            if execution_state:
                await execution_state.update(
                    status="paused",
                    paused_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                log.info(
                    f"Workflow execution {self.workflow_execution_id} marked as paused"
                )

    async def mark_completed(self) -> None:
        """Mark the workflow execution as completed."""
        if self.workflow_execution_id:
            execution_state = await IndexedWorkflowExecutionState.get(
                self.workflow_execution_id
            )
            if execution_state:
                await execution_state.update(
                    status="completed",
                    updated_at=datetime.now(),
                )
                log.info(
                    f"Workflow execution {self.workflow_execution_id} marked as completed"
                )

    async def mark_error(self, error: str) -> None:
        """Mark the workflow execution as having an error."""
        if self.workflow_execution_id:
            execution_state = await IndexedWorkflowExecutionState.get(
                self.workflow_execution_id
            )
            if execution_state:
                await execution_state.update(
                    status="error",
                    error=error[:1000],  # Limit error message length
                    updated_at=datetime.now(),
                )
                log.info(
                    f"Workflow execution {self.workflow_execution_id} marked as error"
                )

    @staticmethod
    async def load_execution_state(
        workflow_execution_id: str,
    ) -> tuple[
        IndexedWorkflowExecutionState | None,
        list[IndexedNodeExecutionState],
        list[IndexedEdgeState],
        IndexedInputQueueState | None,
    ]:
        """
        Load the complete workflow execution state from database.
        
        Returns:
            Tuple of (execution_state, node_states, edge_states, input_queue_state)
        """
        # Load main execution state
        execution_state = await IndexedWorkflowExecutionState.get(workflow_execution_id)
        if execution_state is None:
            log.warning(
                f"Workflow execution state not found: {workflow_execution_id}"
            )
            return None, [], [], None

        # Load node states
        from nodetool.models.condition_builder import Field
        
        node_states_data, _ = await IndexedNodeExecutionState.query(
            Field("workflow_execution_id").equals(workflow_execution_id)
        )
        node_states = [
            await IndexedNodeExecutionState.get(state["id"])
            for state in node_states_data
        ]
        node_states = [state for state in node_states if state is not None]

        # Load edge states
        edge_states_data, _ = await IndexedEdgeState.query(
            Field("workflow_execution_id").equals(workflow_execution_id)
        )
        edge_states = [
            await IndexedEdgeState.get(state["id"]) for state in edge_states_data
        ]
        edge_states = [state for state in edge_states if state is not None]

        # Load input queue state
        input_queue_states_data, _ = await IndexedInputQueueState.query(
            Field("workflow_execution_id").equals(workflow_execution_id),
            limit=1,
        )
        input_queue_state = None
        if input_queue_states_data:
            input_queue_state = await IndexedInputQueueState.get(
                input_queue_states_data[0]["id"]
            )

        log.info(
            f"Loaded execution state: {len(node_states)} nodes, "
            f"{len(edge_states)} edges for workflow {workflow_execution_id}"
        )

        return execution_state, node_states, edge_states, input_queue_state

    @staticmethod
    async def get_execution_state_by_job_id(job_id: str) -> IndexedWorkflowExecutionState | None:
        """
        Get workflow execution state by job ID.
        
        Returns:
            The workflow execution state, or None if not found.
        """
        from nodetool.models.condition_builder import Field
        
        states_data, _ = await IndexedWorkflowExecutionState.query(
            Field("job_id").equals(job_id),
            limit=1,
        )
        if states_data:
            return await IndexedWorkflowExecutionState.get(states_data[0]["id"])
        return None
