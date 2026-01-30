"""
Tests for resumable workflows event logging and recovery.

This module tests the event log and recovery mechanisms.
"""

import asyncio
import queue

import pytest

from nodetool.models.run_event import RunEvent
from nodetool.models.run_lease import RunLease
from nodetool.types.api_graph import Edge as APIEdge
from nodetool.types.api_graph import Graph as APIGraph
from nodetool.types.api_graph import Node as APINode
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.event_logger import WorkflowEventLogger
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.recovery import WorkflowRecoveryService
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.workflow_runner import WorkflowRunner

pytestmark = pytest.mark.xdist_group(name="database")


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

    # Start the logger to enable background flushing of non-blocking events
    await logger.start()

    try:
        # Log various events
        await logger.log_run_created(graph={}, params={}, user_id="test-user")
        await logger.log_node_scheduled("node1", "Multiply", attempt=1)
        await logger.log_node_started("node1", attempt=1, inputs={})
        await logger.log_node_completed("node1", attempt=1, outputs={}, duration_ms=100)
        await logger.log_run_completed(outputs={}, duration_ms=1000)
    finally:
        # Stop the logger to flush any remaining events
        await logger.stop()

    # Verify events were logged
    # Note: Blocking events (RunCreated, RunCompleted) get seq numbers immediately,
    # while non-blocking events (NodeScheduled, NodeStarted, NodeCompleted) are flushed
    # after logger.stop(), so they get later seq numbers.
    events = await RunEvent.get_events(run_id=run_id)
    assert len(events) == 5

    # Check that all event types are present
    event_types = {e.event_type for e in events}
    assert event_types == {"RunCreated", "NodeScheduled", "NodeStarted", "NodeCompleted", "RunCompleted"}

    # First event should be RunCreated (blocking, seq 0)
    assert events[0].event_type == "RunCreated"


@pytest.mark.asyncio
async def test_recovery_service_determine_resumption():
    """Test recovery service identifies incomplete nodes."""
    from nodetool.models.job import Job
    from nodetool.models.run_node_state import RunNodeState

    run_id = "test-run-6"

    # Create job with scheduled status
    job = await Job.create(workflow_id="test-workflow", user_id="test-user", execution_strategy="threaded")
    run_id = job.id

    # Node 1: scheduled but never started
    node1 = await RunNodeState.get_or_create(run_id, "node1")
    await node1.mark_scheduled(attempt=1)

    # Node 2: started but not completed (simulates crash)
    node2 = await RunNodeState.get_or_create(run_id, "node2")
    await node2.mark_scheduled(attempt=1)
    await node2.mark_running()

    # Node 3: completed successfully
    node3 = await RunNodeState.get_or_create(run_id, "node3")
    await node3.mark_scheduled(attempt=1)
    await node3.mark_running()
    await node3.mark_completed({"output": 42})

    recovery = WorkflowRecoveryService()

    from nodetool.workflows.graph import Graph

    graph = Graph(nodes=[], edges=[])  # Empty graph for this test

    resumption_plan = await recovery.determine_resumption_points(run_id, graph)

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
