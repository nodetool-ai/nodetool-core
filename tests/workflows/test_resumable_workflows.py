"""
Tests for resumable workflows event logging and recovery.

This module tests the event log, projection, and recovery mechanisms.
"""

import asyncio
import queue
from datetime import datetime

import pytest

from nodetool.models.run_event import RunEvent
from nodetool.models.run_lease import RunLease
from nodetool.models.run_projection import RunProjection
from nodetool.types.graph import Edge as APIEdge
from nodetool.types.graph import Graph as APIGraph
from nodetool.types.graph import Node as APINode
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.event_logger import WorkflowEventLogger
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.recovery import WorkflowRecoveryService
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.workflow_runner import WorkflowRunner


class NumberInput(InputNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class NumberOutput(OutputNode):
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class Multiply(BaseNode):
    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.a * self.b


@pytest.mark.asyncio
async def test_run_event_creation():
    """Test creating and appending events."""
    run_id = "test-run-1"
    
    # Create first event
    event1 = await RunEvent.append_event(
        run_id=run_id,
        event_type="RunCreated",
        payload={"graph": {}, "params": {}},
    )
    assert event1.seq == 0
    assert event1.event_type == "RunCreated"
    assert event1.run_id == run_id
    
    # Create second event
    event2 = await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeScheduled",
        payload={"node_type": "Multiply", "attempt": 1},
        node_id="node1",
    )
    assert event2.seq == 1
    assert event2.node_id == "node1"
    
    # Query events
    events = await RunEvent.get_events(run_id=run_id)
    assert len(events) == 2
    assert events[0].seq == 0
    assert events[1].seq == 1


@pytest.mark.asyncio
async def test_run_projection_update():
    """Test projection updates from events."""
    run_id = "test-run-2"
    
    # Create events
    await RunEvent.append_event(
        run_id=run_id,
        event_type="RunCreated",
        payload={"graph": {}, "params": {}},
    )
    
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeScheduled",
        payload={"node_type": "Multiply", "attempt": 1},
        node_id="node1",
    )
    
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeStarted",
        payload={"attempt": 1, "inputs": {}},
        node_id="node1",
    )
    
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeCompleted",
        payload={"attempt": 1, "outputs": {"output": 10}, "duration_ms": 100},
        node_id="node1",
    )
    
    # Rebuild projection
    projection = await RunProjection.rebuild_from_events(run_id)
    
    assert projection.status == "running"
    assert projection.last_event_seq == 3
    assert "node1" in projection.node_states
    assert projection.node_states["node1"]["status"] == "completed"
    assert projection.node_states["node1"]["outputs"] == {"output": 10}


@pytest.mark.asyncio
async def test_projection_idempotency():
    """Test that replaying events produces same projection."""
    run_id = "test-run-3"
    
    # Create events
    await RunEvent.append_event(
        run_id=run_id,
        event_type="RunCreated",
        payload={"graph": {}, "params": {}},
    )
    
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeScheduled",
        payload={"node_type": "Multiply", "attempt": 1},
        node_id="node1",
    )
    
    # Build projection twice
    projection1 = await RunProjection.rebuild_from_events(run_id)
    projection2 = await RunProjection.rebuild_from_events(run_id)
    
    # Should be identical
    assert projection1.status == projection2.status
    assert projection1.last_event_seq == projection2.last_event_seq
    assert projection1.node_states == projection2.node_states


@pytest.mark.asyncio
async def test_run_lease_acquisition():
    """Test lease acquisition and expiry."""
    run_id = "test-run-4"
    worker1 = "worker-1"
    worker2 = "worker-2"
    
    # Worker 1 acquires lease
    lease1 = await RunLease.acquire(run_id, worker1, ttl_seconds=1)
    assert lease1 is not None
    assert lease1.worker_id == worker1
    
    # Worker 2 cannot acquire while lease is held
    lease2 = await RunLease.acquire(run_id, worker2, ttl_seconds=1)
    assert lease2 is None
    
    # Wait for lease to expire
    await asyncio.sleep(1.5)
    
    # Worker 2 can now acquire expired lease
    lease3 = await RunLease.acquire(run_id, worker2, ttl_seconds=1)
    assert lease3 is not None
    assert lease3.worker_id == worker2


@pytest.mark.asyncio
async def test_event_logger_convenience_methods():
    """Test WorkflowEventLogger convenience methods."""
    run_id = "test-run-5"
    logger = WorkflowEventLogger(run_id)
    
    # Log various events
    await logger.log_run_created(graph={}, params={}, user_id="test-user")
    await logger.log_node_scheduled("node1", "Multiply", attempt=1)
    await logger.log_node_started("node1", attempt=1, inputs={})
    await logger.log_node_completed("node1", attempt=1, outputs={}, duration_ms=100)
    await logger.log_run_completed(outputs={}, duration_ms=1000)
    
    # Verify events were logged
    events = await RunEvent.get_events(run_id=run_id)
    assert len(events) == 5
    assert events[0].event_type == "RunCreated"
    assert events[1].event_type == "NodeScheduled"
    assert events[4].event_type == "RunCompleted"


@pytest.mark.asyncio
async def test_workflow_runner_logs_events():
    """Test that WorkflowRunner logs events during execution."""
    # Build a simple graph
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
        APINode(id="mul", type=Multiply.get_node_type(), data={}),
        APINode(
            id="out",
            type=NumberOutput.get_node_type(),
            data={"name": "result"},
        ),
    ]
    edges = [
        APIEdge(
            id="e1", source="in1", sourceHandle="output", target="mul", targetHandle="a"
        ),
        APIEdge(
            id="e2", source="in2", sourceHandle="output", target="mul", targetHandle="b"
        ),
        APIEdge(
            id="e3",
            source="mul",
            sourceHandle="output",
            target="out",
            targetHandle="value",
        ),
    ]
    api_graph = APIGraph(nodes=nodes, edges=edges)
    
    req = RunJobRequest(graph=api_graph)
    ctx = ProcessingContext(message_queue=queue.Queue())
    
    job_id = "test-job-events"
    runner = WorkflowRunner(job_id=job_id, enable_event_logging=True)
    
    # Run workflow
    await asyncio.wait_for(runner.run(req, ctx), timeout=5.0)
    
    # Verify events were logged
    events = await RunEvent.get_events(run_id=job_id)
    assert len(events) > 0
    
    # Check for run events
    run_created = next((e for e in events if e.event_type == "RunCreated"), None)
    assert run_created is not None
    
    run_completed = next((e for e in events if e.event_type == "RunCompleted"), None)
    assert run_completed is not None
    
    # Check for node events
    node_scheduled = [e for e in events if e.event_type == "NodeScheduled"]
    assert len(node_scheduled) > 0
    
    node_completed = [e for e in events if e.event_type == "NodeCompleted"]
    assert len(node_completed) > 0


@pytest.mark.asyncio
async def test_recovery_service_determine_resumption():
    """Test recovery service identifies incomplete nodes."""
    run_id = "test-run-6"
    
    # Simulate a run with incomplete nodes
    await RunEvent.append_event(
        run_id=run_id,
        event_type="RunCreated",
        payload={"graph": {}, "params": {}},
    )
    
    # Node 1: scheduled but never started
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeScheduled",
        payload={"node_type": "Multiply", "attempt": 1},
        node_id="node1",
    )
    
    # Node 2: started but not completed (simulates crash)
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeScheduled",
        payload={"node_type": "Add", "attempt": 1},
        node_id="node2",
    )
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeStarted",
        payload={"attempt": 1, "inputs": {}},
        node_id="node2",
    )
    
    # Node 3: completed successfully
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeScheduled",
        payload={"node_type": "Divide", "attempt": 1},
        node_id="node3",
    )
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeStarted",
        payload={"attempt": 1, "inputs": {}},
        node_id="node3",
    )
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeCompleted",
        payload={"attempt": 1, "outputs": {}, "duration_ms": 100},
        node_id="node3",
    )
    
    # Build projection and determine resumption
    projection = await RunProjection.rebuild_from_events(run_id)
    recovery = WorkflowRecoveryService()
    
    from nodetool.workflows.graph import Graph
    graph = Graph(nodes=[], edges=[])  # Empty graph for this test
    
    resumption_plan = await recovery.determine_resumption_points(projection, graph)
    
    # Should resume node1 (never started) and node2 (incomplete)
    assert "node1" in resumption_plan
    assert resumption_plan["node1"]["action"] == "reschedule"
    assert resumption_plan["node1"]["reason"] == "never_started"
    
    assert "node2" in resumption_plan
    assert resumption_plan["node2"]["action"] == "reschedule"
    assert resumption_plan["node2"]["reason"] == "incomplete_execution"
    assert resumption_plan["node2"]["attempt"] == 2  # Should retry with new attempt
    
    # Node 3 should not be in resumption plan (completed)
    assert "node3" not in resumption_plan


@pytest.mark.asyncio
async def test_incomplete_nodes_detection():
    """Test detection of incomplete nodes in projection."""
    run_id = "test-run-7"
    
    # Create projection with mixed node states
    projection = RunProjection(
        run_id=run_id,
        status="running",
        last_event_seq=5,
        node_states={
            "node1": {"status": "completed"},
            "node2": {"status": "started"},
            "node3": {"status": "scheduled"},
            "node4": {"status": "completed"},
        },
    )
    
    incomplete = projection.get_incomplete_nodes()
    
    assert len(incomplete) == 2
    assert "node2" in incomplete
    assert "node3" in incomplete
    assert "node1" not in incomplete
    assert "node4" not in incomplete


@pytest.mark.asyncio
async def test_event_query_filters():
    """Test event querying with filters."""
    run_id = "test-run-8"
    
    # Create diverse events
    await RunEvent.append_event(
        run_id=run_id,
        event_type="RunCreated",
        payload={},
    )
    
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeScheduled",
        payload={},
        node_id="node1",
    )
    
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeScheduled",
        payload={},
        node_id="node2",
    )
    
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeCompleted",
        payload={},
        node_id="node1",
    )
    
    # Query by event type
    scheduled_events = await RunEvent.get_events(
        run_id=run_id,
        event_type="NodeScheduled",
    )
    assert len(scheduled_events) == 2
    
    # Query by node_id
    node1_events = await RunEvent.get_events(
        run_id=run_id,
        node_id="node1",
    )
    assert len(node1_events) == 2
    
    # Query with seq range
    events_after_0 = await RunEvent.get_events(
        run_id=run_id,
        seq_gt=0,
    )
    assert len(events_after_0) == 3  # Should not include seq=0
