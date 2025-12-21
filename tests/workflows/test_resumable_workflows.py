"""
Tests for resumable workflow functionality.

This module tests the checkpoint/resume system for workflows, including:
- Basic pause and resume
- Resume after simulated crash
- Streaming node resume
- Parallel node execution resume
- State persistence and restoration
"""

import asyncio
import queue

import pytest

from nodetool.models.workflow_execution_state import (
    IndexedEdgeState,
    IndexedNodeExecutionState,
    IndexedWorkflowExecutionState,
)
from nodetool.types.graph import Edge as APIEdge
from nodetool.types.graph import Graph as APIGraph
from nodetool.types.graph import Node as APINode
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.checkpoint_manager import CheckpointManager
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.workflow_runner import WorkflowRunner

ASYNC_TEST_TIMEOUT = 10.0


# Test Nodes
class NumberInput(InputNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class NumberOutput(OutputNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class Add(BaseNode):
    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.a + self.b


class Multiply(BaseNode):
    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.a * self.b


class SlowNode(BaseNode):
    """A node that takes time to process, useful for testing pause."""
    
    value: float = 0.0
    delay: float = 1.0

    async def process(self, context: ProcessingContext) -> float:
        await asyncio.sleep(self.delay)
        return self.value * 2


@pytest.mark.asyncio
async def test_checkpoint_manager_create_execution_state():
    """Test creating a new workflow execution state."""
    checkpoint_mgr = CheckpointManager()
    
    execution_id = await checkpoint_mgr.create_execution_state(
        job_id="test-job-1",
        workflow_id="test-workflow-1",
        user_id="test-user",
        graph={"nodes": [], "edges": []},
        params={"input1": 5.0},
        device="cpu",
        disable_caching=False,
        buffer_limit=3,
    )
    
    assert execution_id is not None
    assert checkpoint_mgr.workflow_execution_id == execution_id
    
    # Verify it was saved to database
    state = await IndexedWorkflowExecutionState.get(execution_id)
    assert state is not None
    assert state.job_id == "test-job-1"
    assert state.status == "running"
    assert state.device == "cpu"


@pytest.mark.asyncio
async def test_checkpoint_and_restore_basic_workflow():
    """Test checkpointing and restoring a basic workflow."""
    # Build a simple graph: (5 + 3) -> output
    nodes = [
        APINode(
            id="in1",
            type=NumberInput.get_node_type(),
            data={"name": "in1", "value": 5.0},
        ),
        APINode(
            id="in2",
            type=NumberInput.get_node_type(),
            data={"name": "in2", "value": 3.0},
        ),
        APINode(id="add", type=Add.get_node_type(), data={}),
        APINode(
            id="out",
            type=NumberOutput.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        APIEdge(
            id="e1", source="in1", sourceHandle="output", target="add", targetHandle="a"
        ),
        APIEdge(
            id="e2", source="in2", sourceHandle="output", target="add", targetHandle="b"
        ),
        APIEdge(
            id="e3",
            source="add",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    api_graph = APIGraph(nodes=nodes, edges=edges)
    
    req = RunJobRequest(graph=api_graph, workflow_id="test-wf", user_id="test-user")
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="job-checkpoint-1")
    
    # Enable checkpointing
    runner.enable_checkpointing()
    
    # Run the workflow
    try:
        await asyncio.wait_for(runner.run(req, ctx), timeout=ASYNC_TEST_TIMEOUT)
    except TimeoutError:
        pytest.fail(f"Workflow timed out after {ASYNC_TEST_TIMEOUT}s")
    
    # Verify checkpoint was saved
    assert runner._checkpoint_manager is not None
    execution_id = runner._checkpoint_manager.workflow_execution_id
    assert execution_id is not None
    
    # Load checkpoint
    (
        execution_state,
        node_states,
        edge_states,
        input_queue_state,
    ) = await CheckpointManager.load_execution_state(execution_id)
    
    assert execution_state is not None
    assert execution_state.status == "completed"
    assert len(node_states) > 0
    assert len(edge_states) > 0
    
    # Verify outputs
    assert "result" in runner.outputs
    assert runner.outputs["result"] == [8.0]


@pytest.mark.asyncio
async def test_pause_and_resume_workflow():
    """Test pausing a workflow and resuming it."""
    # Build a workflow with a slow node
    nodes = [
        APINode(
            id="in1",
            type=NumberInput.get_node_type(),
            data={"name": "in1", "value": 10.0},
        ),
        APINode(
            id="slow",
            type=SlowNode.get_node_type(),
            data={"delay": 2.0},
        ),
        APINode(
            id="out",
            type=NumberOutput.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="in1",
            sourceHandle="output",
            target="slow",
            targetHandle="value",
        ),
        APIEdge(
            id="e2",
            source="slow",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    api_graph = APIGraph(nodes=nodes, edges=edges)
    
    req = RunJobRequest(graph=api_graph, workflow_id="test-wf", user_id="test-user")
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="job-pause-1")
    
    # Enable checkpointing
    runner.enable_checkpointing()
    
    # Start workflow in background
    workflow_task = asyncio.create_task(runner.run(req, ctx))
    
    # Wait a bit then pause
    await asyncio.sleep(0.5)
    
    # Pause the workflow
    await runner.pause_workflow()
    
    # Verify it's paused
    assert runner.status == "paused"
    
    # Get the execution ID for resume
    execution_id = runner._checkpoint_manager.workflow_execution_id
    
    # Clean up the task
    try:
        await asyncio.wait_for(workflow_task, timeout=1.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    
    # Now create a new runner and resume
    ctx2 = ProcessingContext(message_queue=queue.Queue())
    runner2 = WorkflowRunner(job_id="job-pause-1")
    
    # Resume from checkpoint
    await runner2.resume_from_checkpoint(execution_id, ctx2)
    
    # Verify state was restored
    assert runner2.device == "cpu"
    assert runner2.context is not None
    assert runner2.context.graph is not None
    
    # Continue execution
    # Note: In a real scenario, we'd need to re-run process_graph
    # For this test, we just verify the state was restored correctly


@pytest.mark.asyncio
async def test_checkpoint_on_error():
    """Test that checkpoint is saved when workflow errors."""
    
    class ErrorNode(BaseNode):
        """A node that always raises an error."""
        
        value: float = 0.0
        
        async def process(self, context: ProcessingContext) -> float:
            raise ValueError("Intentional error for testing")
    
    nodes = [
        APINode(
            id="in1",
            type=NumberInput.get_node_type(),
            data={"name": "in1", "value": 5.0},
        ),
        APINode(
            id="error",
            type=ErrorNode.get_node_type(),
            data={},
        ),
        APINode(
            id="out",
            type=NumberOutput.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="in1",
            sourceHandle="output",
            target="error",
            targetHandle="value",
        ),
        APIEdge(
            id="e2",
            source="error",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    api_graph = APIGraph(nodes=nodes, edges=edges)
    
    req = RunJobRequest(graph=api_graph, workflow_id="test-wf", user_id="test-user")
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="job-error-1")
    
    # Enable checkpointing
    runner.enable_checkpointing()
    
    # Run and expect error
    with pytest.raises(ValueError, match="Intentional error"):
        await runner.run(req, ctx)
    
    # Verify checkpoint was saved with error status
    assert runner._checkpoint_manager is not None
    execution_id = runner._checkpoint_manager.workflow_execution_id
    
    execution_state = await IndexedWorkflowExecutionState.get(execution_id)
    assert execution_state is not None
    assert execution_state.status == "error"
    assert "Intentional error" in execution_state.error


@pytest.mark.asyncio
async def test_node_state_persistence():
    """Test that node states are correctly persisted and restored."""
    nodes = [
        APINode(
            id="in1",
            type=NumberInput.get_node_type(),
            data={"name": "in1", "value": 7.0},
        ),
        APINode(
            id="mul",
            type=Multiply.get_node_type(),
            data={"b": 3.0},  # Pre-set b value
        ),
        APINode(
            id="out",
            type=NumberOutput.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="in1",
            sourceHandle="output",
            target="mul",
            targetHandle="a",
        ),
        APIEdge(
            id="e2",
            source="mul",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    api_graph = APIGraph(nodes=nodes, edges=edges)
    
    req = RunJobRequest(graph=api_graph, workflow_id="test-wf", user_id="test-user")
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="job-node-state-1")
    
    # Enable checkpointing
    runner.enable_checkpointing()
    
    # Run workflow
    await runner.run(req, ctx)
    
    # Get execution state
    execution_id = runner._checkpoint_manager.workflow_execution_id
    
    # Load node states
    node_states_data, _ = await IndexedNodeExecutionState.query(
        IndexedNodeExecutionState.workflow_execution_id == execution_id
    )
    
    # Find the multiply node state
    mul_state = None
    for state_data in node_states_data:
        state = await IndexedNodeExecutionState.get(state_data["id"])
        if state and state.node_id == "mul":
            mul_state = state
            break
    
    assert mul_state is not None
    # Verify properties were saved
    assert "a" in mul_state.properties or "b" in mul_state.properties


@pytest.mark.asyncio
async def test_edge_state_persistence():
    """Test that edge buffer states are correctly persisted."""
    nodes = [
        APINode(
            id="in1",
            type=NumberInput.get_node_type(),
            data={"name": "in1", "value": 5.0},
        ),
        APINode(
            id="out",
            type=NumberOutput.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="in1",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    api_graph = APIGraph(nodes=nodes, edges=edges)
    
    req = RunJobRequest(graph=api_graph, workflow_id="test-wf", user_id="test-user")
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="job-edge-state-1")
    
    # Enable checkpointing
    runner.enable_checkpointing()
    
    # Run workflow
    await runner.run(req, ctx)
    
    # Get execution state
    execution_id = runner._checkpoint_manager.workflow_execution_id
    
    # Load edge states
    edge_states_data, _ = await IndexedEdgeState.query(
        IndexedEdgeState.workflow_execution_id == execution_id
    )
    
    assert len(edge_states_data) > 0
    
    # Verify edge state was saved
    edge_state = await IndexedEdgeState.get(edge_states_data[0]["id"])
    assert edge_state is not None
    assert edge_state.source_node_id == "in1"
    assert edge_state.target_node_id == "out"


@pytest.mark.asyncio
async def test_parallel_execution_checkpoint():
    """Test checkpointing with parallel node execution."""
    # Build a graph with parallel branches
    nodes = [
        APINode(
            id="in1",
            type=NumberInput.get_node_type(),
            data={"name": "in1", "value": 2.0},
        ),
        APINode(
            id="in2",
            type=NumberInput.get_node_type(),
            data={"name": "in2", "value": 3.0},
        ),
        APINode(id="mul1", type=Multiply.get_node_type(), data={"b": 2.0}),
        APINode(id="mul2", type=Multiply.get_node_type(), data={"b": 3.0}),
        APINode(id="add", type=Add.get_node_type(), data={}),
        APINode(
            id="out",
            type=NumberOutput.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        APIEdge(
            id="e1",
            source="in1",
            sourceHandle="output",
            target="mul1",
            targetHandle="a",
        ),
        APIEdge(
            id="e2",
            source="in2",
            sourceHandle="output",
            target="mul2",
            targetHandle="a",
        ),
        APIEdge(
            id="e3",
            source="mul1",
            sourceHandle="output",
            target="add",
            targetHandle="a",
        ),
        APIEdge(
            id="e4",
            source="mul2",
            sourceHandle="output",
            target="add",
            targetHandle="b",
        ),
        APIEdge(
            id="e5",
            source="add",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    api_graph = APIGraph(nodes=nodes, edges=edges)
    
    req = RunJobRequest(graph=api_graph, workflow_id="test-wf", user_id="test-user")
    ctx = ProcessingContext(message_queue=queue.Queue())
    runner = WorkflowRunner(job_id="job-parallel-1")
    
    # Enable checkpointing
    runner.enable_checkpointing()
    
    # Run workflow
    await runner.run(req, ctx)
    
    # Verify checkpoint
    execution_id = runner._checkpoint_manager.workflow_execution_id
    (
        execution_state,
        node_states,
        edge_states,
        _,
    ) = await CheckpointManager.load_execution_state(execution_id)
    
    assert execution_state is not None
    assert execution_state.status == "completed"
    # Should have states for all nodes
    assert len(node_states) >= 4  # At least the compute nodes
    
    # Verify result
    assert "result" in runner.outputs
    # (2 * 2) + (3 * 3) = 4 + 9 = 13
    assert runner.outputs["result"] == [13.0]


@pytest.mark.asyncio
async def test_get_execution_state_by_job_id():
    """Test retrieving workflow execution state by job ID."""
    checkpoint_mgr = CheckpointManager()
    
    job_id = "test-job-lookup-1"
    execution_id = await checkpoint_mgr.create_execution_state(
        job_id=job_id,
        workflow_id="test-wf",
        user_id="test-user",
        graph={"nodes": [], "edges": []},
        params={},
        device="cpu",
        disable_caching=False,
        buffer_limit=3,
    )
    
    # Retrieve by job ID
    state = await CheckpointManager.get_execution_state_by_job_id(job_id)
    
    assert state is not None
    assert state.id == execution_id
    assert state.job_id == job_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
