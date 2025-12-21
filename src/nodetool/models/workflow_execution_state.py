"""
Database models for persisting workflow execution state to enable resumable workflows.

This module defines the models needed to save and restore the complete state of
a running workflow, including:
- Overall workflow execution metadata
- Per-node execution state (status, properties, outputs)
- Edge/inbox buffer states (pending values, EOS markers)
- Input queue state for streaming inputs

These models enable workflows to be paused, stopped (including crashes), and
resumed from the exact point of interruption.
"""

from datetime import datetime
from typing import Any

from nodetool.models.base_model import DBField, DBIndex, DBModel, create_time_ordered_uuid


class WorkflowExecutionState(DBModel):
    """
    Stores the overall state of a workflow execution for resumability.
    
    This tracks the high-level execution state including which nodes have
    completed, which are in progress, and the current status of the workflow.
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "workflow_execution_states",
            "primary_key": "id",
        }

    id: str = DBField(default_factory=create_time_ordered_uuid)
    job_id: str = DBField()  # Foreign key to Job
    workflow_id: str = DBField(default="")
    user_id: str = DBField(default="")
    
    # Execution status: 'running', 'paused', 'completed', 'error', 'cancelled'
    status: str = DBField(default="running")
    
    # Graph definition (snapshot at execution start)
    graph: dict[str, Any] = DBField(default_factory=dict)
    
    # Current execution parameters
    params: dict[str, Any] = DBField(default_factory=dict)
    
    # Workflow runner configuration
    device: str = DBField(default="cpu")
    disable_caching: bool = DBField(default=False)
    buffer_limit: int | None = DBField(default=3)
    
    # Timestamps
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)
    paused_at: datetime | None = DBField(default=None)
    
    # Checkpoint metadata
    checkpoint_count: int = DBField(default=0)
    last_checkpoint_at: datetime | None = DBField(default=None)
    
    # Error information
    error: str | None = DBField(default=None)


@DBIndex(columns=["job_id"], unique=True, name="idx_workflow_exec_state_job_id")
@DBIndex(columns=["user_id", "created_at"], name="idx_workflow_exec_state_user_created")
@DBIndex(columns=["status", "updated_at"], name="idx_workflow_exec_state_status_updated")
class IndexedWorkflowExecutionState(WorkflowExecutionState):
    """WorkflowExecutionState with indexes for efficient querying."""
    pass


class NodeExecutionState(DBModel):
    """
    Stores the execution state of a single node within a workflow.
    
    This captures everything needed to resume a node's execution:
    - Current status (pending, running, completed, error)
    - Node properties and their current values
    - Output values produced so far
    - Whether the node has been initialized
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "node_execution_states",
            "primary_key": "id",
        }

    id: str = DBField(default_factory=create_time_ordered_uuid)
    workflow_execution_id: str = DBField()  # Foreign key to WorkflowExecutionState
    node_id: str = DBField()  # Node ID from the graph
    
    # Node identification
    node_type: str = DBField(default="")
    node_name: str = DBField(default="")
    
    # Execution status: 'pending', 'running', 'completed', 'error', 'waiting'
    status: str = DBField(default="pending")
    
    # Node properties (current values)
    properties: dict[str, Any] = DBField(default_factory=dict)
    
    # Output values produced (mapping output handle -> value)
    outputs: dict[str, Any] = DBField(default_factory=dict)
    
    # Cached result if available
    cached_result: dict[str, Any] | None = DBField(default=None)
    
    # Whether node has been initialized
    initialized: bool = DBField(default=False)
    
    # Whether node has been finalized
    finalized: bool = DBField(default=False)
    
    # Timestamps
    started_at: datetime | None = DBField(default=None)
    completed_at: datetime | None = DBField(default=None)
    updated_at: datetime = DBField(default_factory=datetime.now)
    
    # Error information
    error: str | None = DBField(default=None)


@DBIndex(columns=["workflow_execution_id", "node_id"], unique=True, name="idx_node_exec_state_workflow_node")
@DBIndex(columns=["workflow_execution_id", "status"], name="idx_node_exec_state_workflow_status")
class IndexedNodeExecutionState(NodeExecutionState):
    """NodeExecutionState with indexes for efficient querying."""
    pass


class EdgeState(DBModel):
    """
    Stores the state of an edge/inbox in the workflow execution.
    
    This captures the buffered values waiting to be consumed by downstream
    nodes and the EOS (end-of-stream) markers for each input handle.
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "edge_states",
            "primary_key": "id",
        }

    id: str = DBField(default_factory=create_time_ordered_uuid)
    workflow_execution_id: str = DBField()  # Foreign key to WorkflowExecutionState
    
    # Edge identification
    edge_id: str = DBField(default="")
    source_node_id: str = DBField()
    source_handle: str = DBField()
    target_node_id: str = DBField()
    target_handle: str = DBField()
    
    # Buffer state: list of values waiting to be consumed
    buffered_values: list[Any] = DBField(default_factory=list)
    
    # Number of upstream sources that are still open (not EOS)
    open_upstream_count: int = DBField(default=0)
    
    # Whether this edge is marked as streaming
    is_streaming: bool = DBField(default=False)
    
    # Message counter for this edge
    message_count: int = DBField(default=0)
    
    # Timestamps
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)


@DBIndex(columns=["workflow_execution_id", "target_node_id", "target_handle"], name="idx_edge_state_workflow_target")
@DBIndex(columns=["workflow_execution_id", "edge_id"], unique=True, name="idx_edge_state_workflow_edge")
class IndexedEdgeState(EdgeState):
    """EdgeState with indexes for efficient querying."""
    pass


class InputQueueState(DBModel):
    """
    Stores the state of the input queue for streaming InputNodes.
    
    This captures pending input events that haven't been dispatched yet,
    enabling streaming workflows to resume correctly.
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "input_queue_states",
            "primary_key": "id",
        }

    id: str = DBField(default_factory=create_time_ordered_uuid)
    workflow_execution_id: str = DBField()  # Foreign key to WorkflowExecutionState
    
    # Pending input events (list of event dicts)
    pending_events: list[dict[str, Any]] = DBField(default_factory=list)
    
    # State of each streaming input: input_name -> {'eos': bool, 'last_handle': str}
    input_states: dict[str, dict[str, Any]] = DBField(default_factory=dict)
    
    # Timestamps
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)


@DBIndex(columns=["workflow_execution_id"], unique=True, name="idx_input_queue_state_workflow")
class IndexedInputQueueState(InputQueueState):
    """InputQueueState with indexes for efficient querying."""
    pass
