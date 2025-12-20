"""
Tests for the IntervalTrigger node.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from nodetool.nodes.triggers.interval import IntervalTrigger
from nodetool.workflows.processing_context import ProcessingContext


@pytest.mark.asyncio
async def test_interval_trigger_emits_ticks():
    """Test that interval trigger emits ticks at regular intervals."""
    trigger = IntervalTrigger(
        id="test-trigger",
        interval_seconds=0.05,
        max_events=3,
        emit_on_start=True,
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []
    start = datetime.now(timezone.utc)

    async for result in trigger.gen_process(context):
        events.append(result["event"])

    await trigger.finalize(context)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    assert len(events) == 3
    assert all(e["event_type"] == "tick" for e in events)
    assert events[0]["data"]["tick"] == 1
    assert events[1]["data"]["tick"] == 2
    assert events[2]["data"]["tick"] == 3

    # Should have taken roughly 0.1s (2 intervals after first immediate emit)
    assert 0.05 <= elapsed <= 0.5


@pytest.mark.asyncio
async def test_interval_trigger_no_emit_on_start():
    """Test interval trigger without emit_on_start."""
    trigger = IntervalTrigger(
        id="test-trigger",
        interval_seconds=0.05,
        max_events=2,
        emit_on_start=False,
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []
    start = datetime.now(timezone.utc)

    async for result in trigger.gen_process(context):
        events.append(result["event"])

    await trigger.finalize(context)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    # With emit_on_start=False, first tick comes after interval_seconds
    assert len(events) == 2
    # Should have waited at least 1 interval (first tick comes after one interval)
    assert elapsed >= 0.05


@pytest.mark.asyncio
async def test_interval_trigger_initial_delay():
    """Test interval trigger with initial delay."""
    trigger = IntervalTrigger(
        id="test-trigger",
        interval_seconds=0.02,
        initial_delay_seconds=0.05,
        max_events=1,
        emit_on_start=True,
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    start = datetime.now(timezone.utc)

    events = []
    async for result in trigger.gen_process(context):
        events.append(result["event"])

    await trigger.finalize(context)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    # Should have waited for initial delay
    assert elapsed >= 0.05
    assert len(events) == 1


@pytest.mark.asyncio
async def test_interval_trigger_event_data():
    """Test that interval trigger events contain correct data."""
    trigger = IntervalTrigger(
        id="test-trigger",
        interval_seconds=0.05,
        max_events=1,
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []
    async for result in trigger.gen_process(context):
        events.append(result["event"])

    await trigger.finalize(context)

    event = events[0]
    assert "tick" in event["data"]
    assert "elapsed_seconds" in event["data"]
    assert "interval_seconds" in event["data"]
    assert event["data"]["interval_seconds"] == 0.05
    assert event["source"] == "interval"
    assert event["event_type"] == "tick"


@pytest.mark.asyncio
async def test_interval_trigger_stop():
    """Test that interval trigger can be stopped."""
    trigger = IntervalTrigger(
        id="test-trigger",
        interval_seconds=0.05,
        max_events=100,  # High limit
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

    # Should have stopped after ~2 events
    assert 2 <= len(events) <= 4
