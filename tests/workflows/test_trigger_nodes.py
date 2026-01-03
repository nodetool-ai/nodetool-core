"""
Tests for trigger nodes with suspension/resumption capability.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.suspendable_node import WorkflowSuspendedException
from nodetool.workflows.trigger_node import (
    DEFAULT_INACTIVITY_TIMEOUT,
    TriggerInactivityTimeout,
    TriggerNode,
    TriggerWakeupService,
)


class TestIntervalTrigger(TriggerNode):
    """Test trigger node that waits for interval events."""

    interval_seconds: int = 60

    async def process(self, context: ProcessingContext) -> dict:
        if self.is_resuming():
            return await self.process_trigger_resumption(context)

        try:
            event = await self.wait_for_trigger_event(timeout_seconds=self.get_inactivity_timeout())
            return {"triggered_at": datetime.now(), "event": event}
        except TriggerInactivityTimeout:
            await self.suspend_for_inactivity(
                {
                    "interval_seconds": self.interval_seconds,
                }
            )


def test_trigger_node_is_trigger():
    """Test that trigger node reports as trigger node."""
    node = TestIntervalTrigger(id="test1")
    assert node.is_trigger_node() is True


def test_trigger_node_is_suspendable():
    """Test that trigger node is also suspendable."""
    node = TestIntervalTrigger(id="test2")
    assert node.is_suspendable() is True


def test_trigger_node_default_timeout():
    """Test default inactivity timeout."""
    node = TestIntervalTrigger(id="test3")
    assert node.get_inactivity_timeout() == DEFAULT_INACTIVITY_TIMEOUT


def test_trigger_node_set_timeout():
    """Test setting custom inactivity timeout."""
    node = TestIntervalTrigger(id="test4")
    node.set_inactivity_timeout(180)
    assert node.get_inactivity_timeout() == 180


def test_trigger_node_set_invalid_timeout():
    """Test that invalid timeout raises error."""
    node = TestIntervalTrigger(id="test5")
    with pytest.raises(ValueError):
        node.set_inactivity_timeout(0)


def test_trigger_node_activity_tracking():
    """Test activity time tracking."""
    node = TestIntervalTrigger(id="test6")

    # Initially no activity
    assert node.get_last_activity_time() is None
    assert node.get_inactivity_duration() is None

    # Update activity
    node._update_activity_time()
    assert node.get_last_activity_time() is not None
    assert node.get_inactivity_duration() is not None
    assert node.get_inactivity_duration().total_seconds() < 1


@pytest.mark.asyncio
async def test_trigger_node_wait_timeout():
    """Test that wait_for_trigger_event times out correctly."""
    node = TestIntervalTrigger(id="test7")
    node.set_inactivity_timeout(1)  # 1 second timeout

    with pytest.raises(TriggerInactivityTimeout) as exc_info:
        await node.wait_for_trigger_event(timeout_seconds=1)

    assert exc_info.value.timeout_seconds == 1


@pytest.mark.asyncio
async def test_trigger_node_send_and_receive_event():
    """Test sending and receiving trigger events."""
    node = TestIntervalTrigger(id="test8")

    # Send event to node
    event_data = {"type": "interval", "value": 123}
    await node.send_trigger_event(event_data)

    # Wait for event (should return immediately)
    received = await node.wait_for_trigger_event(timeout_seconds=5)
    assert received == event_data


@pytest.mark.asyncio
async def test_trigger_node_suspension_on_timeout():
    """Test that node suspends when timing out."""
    node = TestIntervalTrigger(id="test9", interval_seconds=30)
    node.set_inactivity_timeout(1)
    ctx = ProcessingContext(message_queue=None)

    with pytest.raises(WorkflowSuspendedException) as exc_info:
        await node.process(ctx)

    exception = exc_info.value
    assert exception.node_id == "test9"
    assert "inactivity timeout" in exception.reason.lower()
    assert exception.state["interval_seconds"] == 30
    assert exception.metadata.get("trigger_node") is True
    assert exception.metadata.get("inactivity_suspension") is True


@pytest.mark.asyncio
async def test_trigger_node_resumption():
    """Test that trigger node can resume from suspension."""
    node = TestIntervalTrigger(id="test10")
    ctx = ProcessingContext(message_queue=None)

    # Set resuming state
    node._set_resuming_state(
        saved_state={
            "suspended_at": datetime.now().isoformat(),
            "interval_seconds": 60,
        },
        event_seq=10,
    )

    assert node.is_resuming() is True

    # Process should return resumption info
    result = await node.process(ctx)
    assert result["status"] == "resumed"
    assert result["trigger_node_id"] == "test10"


@pytest.mark.asyncio
async def test_trigger_node_should_suspend_for_inactivity():
    """Test inactivity detection logic."""
    node = TestIntervalTrigger(id="test11")
    node.set_inactivity_timeout(2)  # 2 seconds

    # Initially no activity
    assert await node.should_suspend_for_inactivity() is False

    # Update activity and check immediately
    node._update_activity_time()
    assert await node.should_suspend_for_inactivity() is False

    # Wait for timeout
    await asyncio.sleep(2.1)
    assert await node.should_suspend_for_inactivity() is True


@pytest.mark.asyncio
async def test_trigger_node_process_trigger_resumption():
    """Test trigger resumption helper."""
    node = TestIntervalTrigger(id="test12")
    ctx = ProcessingContext(message_queue=None)

    saved_state = {
        "suspended_at": "2025-12-26T10:00:00",
        "trigger_data": "test_value",
    }

    node._set_resuming_state(saved_state, event_seq=5)

    result = await node.process_trigger_resumption(ctx)

    assert result["status"] == "resumed"
    assert result["trigger_node_id"] == "test12"
    assert result["saved_state"] == saved_state
    assert "resumed_at" in result


@pytest.mark.asyncio
async def test_trigger_node_suspend_for_inactivity():
    """Test convenience method for inactivity suspension."""
    node = TestIntervalTrigger(id="test13")

    node._update_activity_time()

    with pytest.raises(WorkflowSuspendedException) as exc_info:
        await node.suspend_for_inactivity({"custom_field": "value"})

    exception = exc_info.value
    assert "inactivity timeout" in exception.reason.lower()
    assert exception.state["custom_field"] == "value"
    assert exception.state["trigger_node_type"] == "TestIntervalTrigger"
    assert exception.metadata["trigger_node"] is True


def test_trigger_wakeup_service_singleton():
    """Test that TriggerWakeupService is a singleton."""
    service1 = TriggerWakeupService.get_instance()
    service2 = TriggerWakeupService.get_instance()
    assert service1 is service2


def test_trigger_wakeup_service_register():
    """Test registering suspended trigger."""
    service = TriggerWakeupService.get_instance()

    service.register_suspended_trigger(
        workflow_id="wf-1",
        node_id="node-1",
        trigger_metadata={"type": "interval"},
    )

    triggers = service.list_suspended_triggers()
    assert "wf-1:node-1" in triggers
    assert triggers["wf-1:node-1"]["workflow_id"] == "wf-1"
    assert triggers["wf-1:node-1"]["node_id"] == "node-1"

    # Cleanup
    service.unregister_suspended_trigger("wf-1", "node-1")


def test_trigger_wakeup_service_unregister():
    """Test unregistering suspended trigger."""
    service = TriggerWakeupService.get_instance()

    service.register_suspended_trigger(
        workflow_id="wf-2",
        node_id="node-2",
        trigger_metadata={},
    )

    service.unregister_suspended_trigger("wf-2", "node-2")

    triggers = service.list_suspended_triggers()
    assert "wf-2:node-2" not in triggers


def test_trigger_wakeup_service_list():
    """Test listing suspended triggers."""
    service = TriggerWakeupService.get_instance()

    # Clear any existing
    for key in list(service.list_suspended_triggers().keys()):
        parts = key.split(":")
        service.unregister_suspended_trigger(parts[0], parts[1])

    # Register multiple
    service.register_suspended_trigger("wf-3", "node-3", {})
    service.register_suspended_trigger("wf-4", "node-4", {})

    triggers = service.list_suspended_triggers()
    assert len(triggers) >= 2
    assert "wf-3:node-3" in triggers
    assert "wf-4:node-4" in triggers

    # Cleanup
    service.unregister_suspended_trigger("wf-3", "node-3")
    service.unregister_suspended_trigger("wf-4", "node-4")


@pytest.mark.asyncio
async def test_trigger_inactivity_timeout_exception():
    """Test TriggerInactivityTimeout exception."""
    exc = TriggerInactivityTimeout(timeout_seconds=300)
    assert exc.timeout_seconds == 300
    assert "300" in str(exc)
    assert "suspending" in str(exc).lower()
