"""
Tests for the base TriggerNode class and trigger infrastructure.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import pytest

from nodetool.nodes.triggers.base import TriggerEvent, TriggerNode
from nodetool.workflows.processing_context import ProcessingContext


class SimpleTrigger(TriggerNode):
    """A simple trigger for testing that emits a fixed number of events."""

    events_to_emit: int = 3
    delay_between_events: float = 0.01

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._events_emitted = 0

    async def setup_trigger(self, context: ProcessingContext) -> None:
        self._events_emitted = 0

    async def wait_for_event(self, context: ProcessingContext) -> TriggerEvent | None:
        if self._events_emitted >= self.events_to_emit:
            return None

        # Check if we should stop
        if not self._is_running:
            return None

        await asyncio.sleep(self.delay_between_events)

        # Check again after sleep in case stop() was called
        if not self._is_running:
            return None

        self._events_emitted += 1
        return {
            "data": {"count": self._events_emitted},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "test",
            "event_type": "test_event",
        }

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        pass


class QueueBasedTrigger(TriggerNode):
    """A trigger that uses the event queue for testing."""

    async def setup_trigger(self, context: ProcessingContext) -> None:
        pass

    async def wait_for_event(self, context: ProcessingContext) -> TriggerEvent | None:
        return await self.get_event_from_queue(timeout=1.0)

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        pass


@pytest.mark.asyncio
async def test_trigger_node_is_streaming():
    """Test that trigger nodes are marked as streaming output."""
    assert SimpleTrigger.is_streaming_output() is True


@pytest.mark.asyncio
async def test_trigger_node_not_cacheable():
    """Test that trigger nodes are not cacheable."""
    assert SimpleTrigger.is_cacheable() is False


@pytest.mark.asyncio
async def test_trigger_emits_events():
    """Test that a trigger emits events correctly."""
    trigger = SimpleTrigger(id="test-trigger", events_to_emit=3)
    context = ProcessingContext()

    # Initialize and run
    await trigger.initialize(context)

    events = []
    async for result in trigger.gen_process(context):
        events.append(result["event"])

    await trigger.finalize(context)

    assert len(events) == 3
    assert all(e["event_type"] == "test_event" for e in events)
    assert [e["data"]["count"] for e in events] == [1, 2, 3]


@pytest.mark.asyncio
async def test_trigger_max_events_limit():
    """Test that max_events limit stops the trigger."""
    trigger = SimpleTrigger(
        id="test-trigger",
        events_to_emit=10,  # Would emit 10
        max_events=3,  # But limited to 3
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []
    async for result in trigger.gen_process(context):
        events.append(result["event"])

    await trigger.finalize(context)

    assert len(events) == 3


@pytest.mark.asyncio
async def test_trigger_stop():
    """Test that a trigger can be stopped."""
    trigger = SimpleTrigger(
        id="test-trigger",
        events_to_emit=100,  # Many events
        delay_between_events=0.1,  # Slow enough to stop
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])
            if len(events) >= 2:
                trigger.stop()

    await asyncio.wait_for(collect_events(), timeout=2.0)
    await trigger.finalize(context)

    # Should have collected ~2-3 events before stopping
    assert 2 <= len(events) <= 4


@pytest.mark.asyncio
async def test_trigger_push_event():
    """Test that events can be pushed to a trigger."""
    trigger = QueueBasedTrigger(id="test-trigger")
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])
            # Stop after collecting all events
            if len(events) >= 3:
                trigger.stop()

    async def push_events():
        # Small delay to ensure collector is waiting
        await asyncio.sleep(0.02)
        for i in range(3):
            trigger.push_event({
                "data": {"value": i},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "test",
                "event_type": "pushed",
            })
            await asyncio.sleep(0.01)

    # Run both concurrently
    await asyncio.gather(push_events(), collect_events())

    await trigger.finalize(context)

    # Should have received all 3 pushed events
    assert len(events) == 3
    assert [e["data"]["value"] for e in events] == [0, 1, 2]


@pytest.mark.asyncio
async def test_trigger_event_structure():
    """Test that trigger events have the correct structure."""
    trigger = SimpleTrigger(id="test-trigger", events_to_emit=1)
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []
    async for result in trigger.gen_process(context):
        events.append(result["event"])

    await trigger.finalize(context)

    event = events[0]
    assert "data" in event
    assert "timestamp" in event
    assert "source" in event
    assert "event_type" in event

    # Timestamp should be valid ISO format
    datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))


@pytest.mark.asyncio
async def test_trigger_cancellation():
    """Test that a trigger handles cancellation gracefully."""
    trigger = SimpleTrigger(
        id="test-trigger",
        events_to_emit=100,
        delay_between_events=0.1,
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])

    task = asyncio.create_task(collect_events())

    # Let it start and collect at least one event
    await asyncio.sleep(0.15)

    # Cancel
    task.cancel()

    # Wait for task to complete (cancellation is handled internally)
    try:
        await task
    except asyncio.CancelledError:
        pass

    await trigger.finalize(context)

    # Should have collected at least one event before cancel
    assert len(events) >= 1
