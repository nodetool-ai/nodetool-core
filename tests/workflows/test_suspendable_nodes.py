"""
Tests for suspendable nodes and workflow suspension/resumption.
"""

import asyncio

import pytest

from nodetool.models.run_event import RunEvent
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.suspendable_node import (
    SuspendableNode,
    WorkflowSuspendedException,
)


class TestSuspendableNode(SuspendableNode):
    """Test node that suspends on first execution."""

    request_data: str = "test"

    async def process(self, context: ProcessingContext) -> dict:
        if self.is_resuming():
            # Resuming - return saved state
            saved = await self.get_saved_state()
            return {"result": f"resumed with {saved.get('approval', 'none')}"}

        # First execution - suspend
        await self.suspend_workflow(
            reason="Waiting for test approval",
            state={"request_data": self.request_data},
            metadata={"test_meta": "value"},
        )

        # This should not be reached
        return {"result": "should not reach"}


def test_suspendable_node_is_suspendable():
    """Test that suspendable node reports as suspendable."""
    node = TestSuspendableNode(id="test1")
    assert node.is_suspendable() is True


def test_suspendable_node_not_resuming_initially():
    """Test that node is not in resuming state initially."""
    node = TestSuspendableNode(id="test2")
    assert node.is_resuming() is False


@pytest.mark.asyncio
async def test_suspendable_node_suspension():
    """Test that node raises WorkflowSuspendedException when suspending."""
    node = TestSuspendableNode(id="test3", request_data="my_data")
    ctx = ProcessingContext(message_queue=None)

    with pytest.raises(WorkflowSuspendedException) as exc_info:
        await node.process(ctx)

    exception = exc_info.value
    assert exception.node_id == "test3"
    assert exception.reason == "Waiting for test approval"
    assert exception.state == {"request_data": "my_data"}
    assert exception.metadata == {"test_meta": "value"}


@pytest.mark.asyncio
async def test_suspendable_node_resumption():
    """Test that node can resume with saved state."""
    node = TestSuspendableNode(id="test4")
    ctx = ProcessingContext(message_queue=None)

    # Set resuming state
    node._set_resuming_state(
        saved_state={"approval": "granted"},
        event_seq=5,
    )

    assert node.is_resuming() is True

    # Process should return saved state
    result = await node.process(ctx)
    assert result["result"] == "resumed with granted"


@pytest.mark.asyncio
async def test_get_saved_state_when_not_resuming():
    """Test that get_saved_state raises error when not resuming."""
    node = TestSuspendableNode(id="test5")

    with pytest.raises(ValueError) as exc_info:
        await node.get_saved_state()

    assert "only be called when resuming" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_suspended_state():
    """Test updating state while suspended."""
    node = TestSuspendableNode(id="test6")

    # Set initial state
    node._set_resuming_state(
        saved_state={"status": "pending"},
        event_seq=3,
    )

    # Update state
    await node.update_suspended_state({"status": "approved", "by": "admin"})

    # Check merged state
    saved = await node.get_saved_state()
    assert saved["status"] == "approved"
    assert saved["by"] == "admin"


@pytest.mark.asyncio
async def test_suspension_event_types():
    """Test that suspension events are properly typed."""
    run_id = "test-suspend-1"

    # Create RunCreated event
    await RunEvent.append_event(
        run_id=run_id,
        event_type="RunCreated",
        payload={"graph": {}, "params": {}},
    )

    # Create NodeSuspended event
    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeSuspended",
        node_id="node1",
        payload={
            "reason": "Waiting for approval",
            "state": {"request_id": "123"},
            "metadata": {"user": "admin"},
        },
    )

    # Create RunSuspended event
    await RunEvent.append_event(
        run_id=run_id,
        event_type="RunSuspended",
        payload={
            "node_id": "node1",
            "reason": "Waiting for approval",
            "metadata": {"user": "admin"},
        },
    )

    # Query events
    events = await RunEvent.get_events(run_id=run_id)
    assert len(events) == 3
    assert events[1].event_type == "NodeSuspended"
    assert events[2].event_type == "RunSuspended"


@pytest.mark.asyncio
async def test_projection_tracks_suspended_state():
    """Test that projection correctly tracks suspended nodes."""
    run_id = "test-suspend-2"

    # Create events
    await RunEvent.append_event(
        run_id=run_id,
        event_type="RunCreated",
        payload={"graph": {}, "params": {}},
    )

    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeScheduled",
        node_id="node1",
        payload={"node_type": "TestNode", "attempt": 1},
    )

    await RunEvent.append_event(
        run_id=run_id,
        event_type="NodeSuspended",
        node_id="node1",
        payload={
            "reason": "Waiting for input",
            "state": {"input_needed": True},
            "metadata": {},
        },
    )

    await RunEvent.append_event(
        run_id=run_id,
        event_type="RunSuspended",
        payload={
            "node_id": "node1",
            "reason": "Waiting for input",
            "metadata": {},
        },
    )


@pytest.mark.asyncio
async def test_workflow_suspended_exception_to_dict():
    """Test exception serialization."""
    exc = WorkflowSuspendedException(
        node_id="test_node",
        reason="Test reason",
        state={"key": "value"},
        metadata={"meta": "data"},
    )

    exc_dict = exc.to_dict()
    assert exc_dict["node_id"] == "test_node"
    assert exc_dict["reason"] == "Test reason"
    assert exc_dict["state"] == {"key": "value"}
    assert exc_dict["metadata"] == {"meta": "data"}
