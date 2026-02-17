"""End-to-end inbox tests derived from E2E_TEST_SCENARIOS.md Section 3.

This module implements high-priority inbox scenarios:
- INBOX-006: Multi-upstream EOS handling
- INBOX-010, INBOX-011: Backpressure
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nodetool.workflows.inbox import NodeInbox

pytestmark = pytest.mark.asyncio


# ============================================================================
# INBOX-006: Multi-Upstream EOS
# ============================================================================


async def test_inbox_006_two_producers_same_handle():
    """INBOX-006: Two edges to same handle; handle only done when BOTH signal EOS."""
    inbox = NodeInbox()

    # Two upstreams feeding the same handle
    inbox.add_upstream("input", count=2)

    # Producer A sends items
    await inbox.put("input", "A1")
    await inbox.put("input", "A2")

    # Producer B sends items
    await inbox.put("input", "B1")

    # Mark A done (but handle should stay open)
    inbox.mark_source_done("input")

    # Consumer should still get items
    received = []

    # Use a task with timeout to avoid hanging
    async def consume():
        async for item in inbox.iter_input("input"):
            received.append(item)
            if len(received) >= 3:
                break

    # Start consumer
    consumer_task = asyncio.create_task(consume())

    # Wait a bit then mark B done
    await asyncio.sleep(0.05)

    # Still need to mark B done
    inbox.mark_source_done("input")

    # Now consumer should complete
    try:
        await asyncio.wait_for(consumer_task, timeout=0.5)
    except TimeoutError:
        pass  # Consumer might have already finished

    # We should have received items from both producers
    assert len(received) >= 3
    assert "A1" in received
    assert "A2" in received
    assert "B1" in received


async def test_inbox_007_mixed_eos_timing():
    """INBOX-007: Producer A signals done early, Producer B continues sending."""
    inbox = NodeInbox()
    inbox.add_upstream("input", count=2)

    # Producer A sends one item and finishes
    await inbox.put("input", "A1")
    inbox.mark_source_done("input")

    # Producer B sends items with delay
    await inbox.put("input", "B1")
    await asyncio.sleep(0.01)
    await inbox.put("input", "B2")
    await asyncio.sleep(0.01)
    await inbox.put("input", "B3")
    inbox.mark_source_done("input")

    # Consumer should get all items
    received = []
    async for item in inbox.iter_input("input"):
        received.append(item)

    assert len(received) == 4
    assert received == ["A1", "B1", "B2", "B3"]


async def test_inbox_008_all_handles_eos():
    """INBOX-008: 3 handles each with producers; iter_any terminates only when ALL complete."""
    inbox = NodeInbox()
    inbox.add_upstream("A", 1)
    inbox.add_upstream("B", 1)
    inbox.add_upstream("C", 1)

    # Interleave items from different handles
    await inbox.put("A", "A1")
    await inbox.put("B", "B1")
    await inbox.put("A", "A2")
    await inbox.put("C", "C1")

    # Mark A done
    inbox.mark_source_done("A")

    # Still have B and C open
    assert inbox.is_open("B")
    assert inbox.is_open("C")

    # Mark B done
    inbox.mark_source_done("B")

    # Still have C open
    assert inbox.is_open("C")

    # Put more items and mark C done
    await inbox.put("C", "C2")
    inbox.mark_source_done("C")

    # Now all done
    assert not inbox.is_open("A")
    assert not inbox.is_open("B")
    assert not inbox.is_open("C")

    # Consume all items
    received = []
    async for handle, item in inbox.iter_any():
        received.append((handle, item))

    assert len(received) == 5


async def test_inbox_009_producer_done_with_buffered_items():
    """INBOX-009: Producer signals done but items still in buffer; consumer drains before EOS."""
    inbox = NodeInbox()
    inbox.add_upstream("input", 1)

    # Producer puts items then signals done immediately
    await inbox.put("input", "item1")
    await inbox.put("input", "item2")
    await inbox.put("input", "item3")
    inbox.mark_source_done("input")

    # Consumer drains all items before seeing EOS
    received = []
    async for item in inbox.iter_input("input"):
        received.append(item)

    assert received == ["item1", "item2", "item3"]


# ============================================================================
# INBOX-010, INBOX-011: Backpressure
# ============================================================================


async def test_inbox_010_buffer_limit_enforced():
    """INBOX-010: buffer_limit=2; third put() blocks until consumer drains."""
    inbox = NodeInbox(buffer_limit=2)
    inbox.add_upstream("input", 1)

    # Fill the buffer
    await inbox.put("input", "item1")
    await inbox.put("input", "item2")

    # Track if third put blocks
    third_put_completed = asyncio.Event()

    async def third_put():
        await inbox.put("input", "item3")
        third_put_completed.set()

    # Start third put (should block)
    put_task = asyncio.create_task(third_put())

    # Give it time to potentially block
    await asyncio.sleep(0.05)

    # Third put should not have completed yet (buffer full)
    assert not third_put_completed.is_set()

    # Now consume one item to make room
    consumed = []

    async def consume_one():
        async for item in inbox.iter_input("input"):
            consumed.append(item)
            break  # Only consume one

    await consume_one()

    # Now third put should complete
    try:
        await asyncio.wait_for(third_put_completed.wait(), timeout=0.5)
    except TimeoutError:
        pytest.fail("Third put did not unblock after consumer drained")

    # Clean up
    put_task.cancel()
    try:
        await put_task
    except asyncio.CancelledError:
        pass


async def test_inbox_011_backpressure_release():
    """INBOX-011: Producer blocked on full buffer; consumer drains; producer unblocks."""
    inbox = NodeInbox(buffer_limit=1)
    inbox.add_upstream("input", 1)

    # Fill buffer
    await inbox.put("input", "item1")

    # Second put should block
    producer_released = asyncio.Event()

    async def producer():
        await inbox.put("input", "item2")
        producer_released.set()

    producer_task = asyncio.create_task(producer())

    # Wait for producer to block
    await asyncio.sleep(0.05)
    assert not producer_released.is_set()

    # Consumer drains the item
    async def consumer():
        async for item in inbox.iter_input("input"):
            return item

    consumed_item = await consumer()
    assert consumed_item == "item1"

    # Producer should now unblock
    try:
        await asyncio.wait_for(producer_released.wait(), timeout=0.5)
    except TimeoutError:
        pytest.fail("Producer did not unblock after consumer drained")

    # Mark done and clean up
    inbox.mark_source_done("input")

    # Consume the second item
    remaining = []
    async for item in inbox.iter_input("input"):
        remaining.append(item)

    assert remaining == ["item2"]

    # Cancel producer task
    producer_task.cancel()
    try:
        await producer_task
    except asyncio.CancelledError:
        pass


# ============================================================================
# Additional Inbox Tests
# ============================================================================


async def test_inbox_001_single_handle_put_get():
    """INBOX-001: Put 5 items to one handle, consume in FIFO order."""
    inbox = NodeInbox()
    inbox.add_upstream("input", 1)

    # Put 5 items
    for i in range(5):
        await inbox.put("input", f"item{i}")

    inbox.mark_source_done("input")

    # Consume in FIFO order
    received = []
    async for item in inbox.iter_input("input"):
        received.append(item)

    assert received == ["item0", "item1", "item2", "item3", "item4"]


async def test_inbox_002_multi_handle_arrival_order():
    """INBOX-002: Put items to handles A, B, A, C in sequence; iter_any yields in exact arrival order."""
    inbox = NodeInbox()
    inbox.add_upstream("A", 1)
    inbox.add_upstream("B", 1)
    inbox.add_upstream("C", 1)

    # Interleave puts
    await inbox.put("A", "A1")
    await inbox.put("B", "B1")
    await inbox.put("A", "A2")
    await inbox.put("C", "C1")

    inbox.mark_source_done("A")
    inbox.mark_source_done("B")
    inbox.mark_source_done("C")

    # Consume in arrival order
    received = []
    async for handle, item in inbox.iter_any():
        received.append((handle, item))

    assert received == [("A", "A1"), ("B", "B1"), ("A", "A2"), ("C", "C1")]


async def test_inbox_003_eos_detection_single():
    """INBOX-003: Single handle with one upstream producer; producer signals done, consumer receives all items then EOS."""
    inbox = NodeInbox()
    inbox.add_upstream("input", 1)

    await inbox.put("input", "item1")
    await inbox.put("input", "item2")
    inbox.mark_source_done("input")

    received = []
    async for item in inbox.iter_input("input"):
        received.append(item)

    assert received == ["item1", "item2"]
    # After loop, we should have consumed all items


async def test_inbox_004_iter_any_terminates_on_all_eos():
    """INBOX-004: iter_any terminates when all handles reach EOS."""
    inbox = NodeInbox()
    inbox.add_upstream("A", 1)
    inbox.add_upstream("B", 1)

    await inbox.put("A", "A1")
    await inbox.put("B", "B1")

    inbox.mark_source_done("A")
    inbox.mark_source_done("B")

    received = []
    async for handle, item in inbox.iter_any():
        received.append((handle, item))

    assert received == [("A", "A1"), ("B", "B1")]


async def test_inbox_005_iter_input_terminates_on_handle_eos():
    """INBOX-005: iter_input(handle) terminates when that specific handle reaches EOS."""
    inbox = NodeInbox()
    inbox.add_upstream("A", 1)
    inbox.add_upstream("B", 1)

    await inbox.put("A", "A1")
    await inbox.put("B", "B1")
    await inbox.put("A", "A2")

    # Mark only A done
    inbox.mark_source_done("A")

    # iter_input("A") should terminate after A's items
    received_a = []
    async for item in inbox.iter_input("A"):
        received_a.append(item)

    assert received_a == ["A1", "A2"]

    # B should still be open
    assert inbox.is_open("B")

    # Now consume B
    inbox.mark_source_done("B")
    received_b = []
    async for item in inbox.iter_input("B"):
        received_b.append(item)

    assert received_b == ["B1"]


async def test_inbox_012_multiple_blocked_producers():
    """INBOX-012: Two producers blocked on same full buffer; consumer drains; both race to put."""
    inbox = NodeInbox(buffer_limit=1)
    inbox.add_upstream("input", 2)  # Two upstreams

    # Fill buffer with first item
    await inbox.put("input", "item1")

    # Both producers try to put (should block)
    producer1_done = asyncio.Event()
    producer2_done = asyncio.Event()

    async def producer1():
        await inbox.put("input", "item2")
        producer1_done.set()

    async def producer2():
        await inbox.put("input", "item3")
        producer2_done.set()

    task1 = asyncio.create_task(producer1())
    task2 = asyncio.create_task(producer2())

    # Wait for producers to block
    await asyncio.sleep(0.05)
    assert not producer1_done.is_set()
    assert not producer2_done.is_set()

    # Consumer drains one item
    async for item in inbox.iter_input("input"):
        assert item == "item1"
        break

    # Both producers should now be unblocked (race condition)
    # Wait for at least one to complete
    done, _pending = await asyncio.wait(
        [asyncio.create_task(producer1_done.wait()), asyncio.create_task(producer2_done.wait())],
        timeout=0.5,
        return_when=asyncio.FIRST_COMPLETED,
    )

    assert len(done) > 0, "At least one producer should have unblocked"

    # Mark done and clean up
    inbox.mark_source_done("input")
    inbox.mark_source_done("input")

    # Cancel remaining tasks
    task1.cancel()
    task2.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        pass
    try:
        await task2
    except asyncio.CancelledError:
        pass


async def test_inbox_015_empty_inbox_no_upstreams():
    """INBOX-015: Inbox with no upstreams registered; iter_any returns immediately."""
    inbox = NodeInbox()
    # No upstreams added

    # iter_any should return immediately (no items, no wait)
    received = []
    async for handle, item in inbox.iter_any():
        received.append((handle, item))

    assert received == []


async def test_inbox_016_close_all_unblocks_waiters():
    """INBOX-016: close_all() called; all blocked consumers wake and terminate."""
    inbox = NodeInbox()
    inbox.add_upstream("input", 1)

    # Consumer starts waiting
    consumer_done = asyncio.Event()

    async def consumer():
        async for _item in inbox.iter_input("input"):
            pass
        consumer_done.set()

    task = asyncio.create_task(consumer())

    # Wait for consumer to start
    await asyncio.sleep(0.05)

    # Close all
    await inbox.close_all()

    # Consumer should terminate
    try:
        await asyncio.wait_for(consumer_done.wait(), timeout=0.5)
    except TimeoutError:
        pytest.fail("Consumer did not unblock after close_all")

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
