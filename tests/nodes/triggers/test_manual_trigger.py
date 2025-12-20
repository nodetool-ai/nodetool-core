"""
Tests for the ManualTrigger node.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from nodetool.nodes.triggers.manual import ManualTrigger
from nodetool.workflows.processing_context import ProcessingContext


@pytest.mark.asyncio
async def test_manual_trigger_receives_pushed_events():
    """Test that manual trigger receives pushed events."""
    trigger = ManualTrigger(id="test-trigger", name="test_manual")
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])
            if len(events) >= 3:
                trigger.stop()

    async def push_events():
        await asyncio.sleep(0.02)
        for i in range(3):
            trigger.push_data({"value": i})
            await asyncio.sleep(0.01)

    await asyncio.gather(push_events(), collect_events())
    await trigger.finalize(context)

    assert len(events) == 3
    assert [e["data"]["value"] for e in events] == [0, 1, 2]
    assert all(e["source"] == "test_manual" for e in events)


@pytest.mark.asyncio
async def test_manual_trigger_timeout():
    """Test that manual trigger times out when no events arrive."""
    trigger = ManualTrigger(
        id="test-trigger",
        name="test_manual",
        timeout_seconds=0.1,
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    # Don't push any events
    events = []
    async for result in trigger.gen_process(context):
        events.append(result["event"])

    await trigger.finalize(context)

    # Should have no events due to timeout
    assert len(events) == 0


@pytest.mark.asyncio
async def test_manual_trigger_push_data_method():
    """Test the push_data convenience method."""
    trigger = ManualTrigger(id="test-trigger", name="test_manual")
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])
            trigger.stop()

    async def push_event():
        await asyncio.sleep(0.02)
        # Use push_data with custom event type
        trigger.push_data({"message": "hello"}, event_type="custom")

    await asyncio.gather(push_event(), collect_events())
    await trigger.finalize(context)

    assert len(events) == 1
    assert events[0]["data"]["message"] == "hello"
    assert events[0]["event_type"] == "custom"


@pytest.mark.asyncio
async def test_manual_trigger_async_push():
    """Test that events can be pushed asynchronously."""
    trigger = ManualTrigger(id="test-trigger", name="test_manual")
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def push_events():
        await asyncio.sleep(0.05)
        trigger.push_data({"seq": 1})
        await asyncio.sleep(0.05)
        trigger.push_data({"seq": 2})
        await asyncio.sleep(0.05)
        trigger.stop()

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])

    # Run both concurrently
    await asyncio.gather(
        push_events(),
        collect_events(),
    )

    await trigger.finalize(context)

    assert len(events) == 2
    assert [e["data"]["seq"] for e in events] == [1, 2]


@pytest.mark.asyncio
async def test_manual_trigger_no_timeout():
    """Test manual trigger with no timeout waits indefinitely."""
    trigger = ManualTrigger(
        id="test-trigger",
        name="test_manual",
        timeout_seconds=None,  # No timeout
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])

    task = asyncio.create_task(collect_events())

    # Wait briefly then push and stop
    await asyncio.sleep(0.05)
    trigger.push_data({"value": 1})
    await asyncio.sleep(0.02)
    trigger.stop()

    await asyncio.wait_for(task, timeout=1.0)
    await trigger.finalize(context)

    assert len(events) == 1
    assert events[0]["data"]["value"] == 1
