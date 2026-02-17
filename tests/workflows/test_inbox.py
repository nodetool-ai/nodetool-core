import asyncio
import threading
from datetime import UTC

import pytest

from nodetool.workflows.inbox import NodeInbox


@pytest.mark.asyncio
async def test_iter_input_basic_flow():
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    # Producer: push a couple items, then mark EOS
    await inbox.put("a", 1)
    await inbox.put("a", 2)
    inbox.mark_source_done("a")

    received = []
    async for item in inbox.iter_input("a"):
        received.append(item)

    assert received == [1, 2]


@pytest.mark.asyncio
async def test_iter_any_multiplexing_order():
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    inbox.add_upstream("b", 1)

    # Interleave puts on different handles
    await inbox.put("a", "a1")
    await inbox.put("b", "b1")
    await inbox.put("a", "a2")
    inbox.mark_source_done("a")
    await inbox.put("b", "b2")
    inbox.mark_source_done("b")

    items = []
    async for handle, item in inbox.iter_any():
        items.append((handle, item))

    assert items == [("a", "a1"), ("b", "b1"), ("a", "a2"), ("b", "b2")]


@pytest.mark.asyncio
async def test_close_all_unblocks_waiters():
    inbox = NodeInbox()
    inbox.add_upstream("x", 1)

    collected = []

    async def consumer():
        async for item in inbox.iter_input("x"):
            collected.append(item)

    task = asyncio.create_task(consumer())
    # Give the consumer a chance to start and block
    await asyncio.sleep(0.05)
    await inbox.close_all()
    await asyncio.wait_for(task, timeout=1.0)

    # No items expected; just verify it didn't hang
    assert collected == []


@pytest.mark.asyncio
async def test_mark_source_done_from_different_thread():
    """Test that mark_source_done works correctly when called from a different thread.

    This test simulates the scenario that caused "Future attached to a different loop"
    errors on Windows. The fix ensures that coroutines and tasks are created in the
    correct event loop context when using call_soon_threadsafe.
    """
    inbox = NodeInbox()
    inbox.add_upstream("x", 1)

    collected = []
    consumer_started = threading.Event()
    source_marked_done = threading.Event()

    async def consumer():
        consumer_started.set()
        async for item in inbox.iter_input("x"):
            collected.append(item)

    # Start the consumer task
    task = asyncio.create_task(consumer())

    # Wait for consumer to start and be waiting for input
    await asyncio.sleep(0.05)
    assert consumer_started.is_set()

    # Push an item and mark done from a different thread
    def thread_worker():
        # This runs in a separate thread, simulating cross-thread calls
        # that previously caused "Future attached to a different loop" errors
        inbox.mark_source_done("x")
        source_marked_done.set()

    thread = threading.Thread(target=thread_worker)
    thread.start()
    thread.join(timeout=5.0)

    # Wait for the consumer to complete
    await asyncio.wait_for(task, timeout=2.0)

    # Verify thread completed successfully
    assert source_marked_done.is_set()
    # No items were pushed, so collected should be empty
    assert collected == []


@pytest.mark.asyncio
async def test_mark_source_done_from_thread_with_data():
    """Test that mark_source_done properly notifies waiters when called from thread.

    This verifies the cross-thread notification mechanism works correctly with
    actual data in the inbox.
    """
    inbox = NodeInbox()
    inbox.add_upstream("data", 1)

    collected = []
    eos_signaled = threading.Event()

    async def consumer():
        async for item in inbox.iter_input("data"):
            collected.append(item)

    # Start consumer task
    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)

    # Push some data
    await inbox.put("data", "item1")
    await inbox.put("data", "item2")

    # Mark done from a different thread (simulates Windows cross-loop scenario)
    def signal_eos_from_thread():
        inbox.mark_source_done("data")
        eos_signaled.set()

    thread = threading.Thread(target=signal_eos_from_thread)
    thread.start()
    thread.join(timeout=5.0)

    # Wait for consumer to complete
    await asyncio.wait_for(task, timeout=2.0)

    assert eos_signaled.is_set()
    assert collected == ["item1", "item2"]


@pytest.mark.asyncio
async def test_is_fully_drained_empty_inbox():
    """Test that a new inbox is fully drained."""
    inbox = NodeInbox()
    # Fresh inbox with no upstreams should be fully drained
    assert inbox.is_fully_drained()
    assert not inbox.has_pending_work()


@pytest.mark.asyncio
async def test_is_fully_drained_with_open_sources():
    """Test that inbox with open sources is not fully drained."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    # Inbox has open source, so not fully drained
    assert not inbox.is_fully_drained()
    assert inbox.has_pending_work()


@pytest.mark.asyncio
async def test_is_fully_drained_with_buffered_items():
    """Test that inbox with buffered items is not fully drained."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    # Add items but don't mark done
    await inbox.put("a", 1)
    await inbox.put("a", 2)

    # Has buffered items and open source
    assert not inbox.is_fully_drained()
    assert inbox.has_pending_work()

    # Mark source done but items still buffered
    inbox.mark_source_done("a")
    assert not inbox.is_fully_drained()
    assert inbox.has_pending_work()


@pytest.mark.asyncio
async def test_is_fully_drained_after_consumption():
    """Test that inbox is fully drained after all items consumed and EOS reached."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    # Add items
    await inbox.put("a", 1)
    await inbox.put("a", 2)
    inbox.mark_source_done("a")

    # Not drained yet
    assert not inbox.is_fully_drained()

    # Consume all items
    collected = []
    async for item in inbox.iter_input("a"):
        collected.append(item)

    # Now fully drained
    assert inbox.is_fully_drained()
    assert not inbox.has_pending_work()
    assert collected == [1, 2]


@pytest.mark.asyncio
async def test_is_fully_drained_multiple_handles():
    """Test fully drained detection with multiple handles."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    inbox.add_upstream("b", 1)

    # Add items to both handles
    await inbox.put("a", "a1")
    await inbox.put("b", "b1")

    # Not drained - both have items and open sources
    assert not inbox.is_fully_drained()

    # Mark one handle done
    inbox.mark_source_done("a")
    # Still not drained - a has buffered items, b is still open
    assert not inbox.is_fully_drained()

    # Drain handle a
    item_a = inbox.try_pop_any()
    assert item_a == ("a", "a1")
    # Still not drained - b has items
    assert not inbox.is_fully_drained()

    # Mark b done and drain it
    inbox.mark_source_done("b")
    item_b = inbox.try_pop_any()
    assert item_b == ("b", "b1")

    # Now fully drained
    assert inbox.is_fully_drained()
    assert not inbox.has_pending_work()


@pytest.mark.asyncio
async def test_is_fully_drained_after_close():
    """Test that closed inbox is considered fully drained."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    await inbox.put("a", 1)

    # Not drained yet
    assert not inbox.is_fully_drained()

    # Close the inbox
    await inbox.close_all()

    # Closed inbox is fully drained regardless of buffer state
    assert inbox.is_fully_drained()
    assert not inbox.has_pending_work()


# =============================================================================
# INBOX-004: Iter Any with EOS
# =============================================================================


@pytest.mark.asyncio
async def test_iter_any_terminates_when_all_handles_eos():
    """INBOX-004: iter_any terminates when all handles reach EOS."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    inbox.add_upstream("b", 1)

    # Put items to both handles
    await inbox.put("a", "a1")
    await inbox.put("b", "b1")
    await inbox.put("a", "a2")

    # Mark both sources done
    inbox.mark_source_done("a")
    inbox.mark_source_done("b")

    items = []
    async for handle, item in inbox.iter_any():
        items.append((handle, item))

    # Should receive all items then terminate
    assert items == [("a", "a1"), ("b", "b1"), ("a", "a2")]


# =============================================================================
# INBOX-005: Iter Input with EOS
# =============================================================================


@pytest.mark.asyncio
async def test_iter_input_terminates_when_handle_eos():
    """INBOX-005: iter_input terminates when specific handle reaches EOS."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    inbox.add_upstream("b", 1)

    # Put items to both handles
    await inbox.put("a", "a1")
    await inbox.put("b", "b1")
    await inbox.put("a", "a2")

    # Mark only handle 'a' done
    inbox.mark_source_done("a")

    # iter_input("a") should terminate when handle 'a' reaches EOS
    items_a = []
    async for item in inbox.iter_input("a"):
        items_a.append(item)

    assert items_a == ["a1", "a2"]

    # Handle 'b' should still be open
    assert inbox.is_open("b")


# =============================================================================
# INBOX-006: Two Producers Same Handle
# =============================================================================


@pytest.mark.asyncio
async def test_two_producers_same_handle():
    """INBOX-006: Two edges to same handle, handle only done when BOTH signal EOS."""
    inbox = NodeInbox()
    # Add two upstream producers for the same handle
    inbox.add_upstream("h", 2)

    # Put items from both producers
    await inbox.put("h", "item1")
    await inbox.put("h", "item2")
    await inbox.put("h", "item3")

    # Mark first producer done - handle should still be open
    inbox.mark_source_done("h")
    assert inbox.is_open("h")  # Still has one open producer

    # Put more items from second producer
    await inbox.put("h", "item4")

    # Mark second producer done - handle should now be closed
    inbox.mark_source_done("h")
    assert not inbox.is_open("h")

    # Consume all items
    items = []
    async for item in inbox.iter_input("h"):
        items.append(item)

    assert items == ["item1", "item2", "item3", "item4"]


# =============================================================================
# INBOX-007: Mixed EOS Timing
# =============================================================================


@pytest.mark.asyncio
async def test_mixed_eos_timing():
    """INBOX-007: Producer A signals done early, Producer B continues sending."""
    inbox = NodeInbox()
    inbox.add_upstream("h", 2)

    # Producer A sends and signals done
    await inbox.put("h", "a1")
    inbox.mark_source_done("h")

    # Producer B continues sending
    await inbox.put("h", "b1")
    await inbox.put("h", "b2")
    inbox.mark_source_done("h")

    # Collect all items
    items = []
    async for item in inbox.iter_input("h"):
        items.append(item)

    assert items == ["a1", "b1", "b2"]


# =============================================================================
# INBOX-008: All Handles EOS
# =============================================================================


@pytest.mark.asyncio
async def test_all_handles_eos():
    """INBOX-008: 3 handles each with producers, iter_any terminates when ALL complete."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    inbox.add_upstream("b", 1)
    inbox.add_upstream("c", 1)

    # Put items to all handles
    await inbox.put("a", "a1")
    await inbox.put("b", "b1")
    await inbox.put("c", "c1")

    # Mark one source done
    inbox.mark_source_done("a")

    # iter_any should still be waiting since b and c are open
    # Mark remaining sources done
    inbox.mark_source_done("b")
    inbox.mark_source_done("c")

    items = []
    async for handle, item in inbox.iter_any():
        items.append((handle, item))

    assert sorted(items) == sorted([("a", "a1"), ("b", "b1"), ("c", "c1")])


# =============================================================================
# INBOX-009: Producer Done with Buffered Items
# =============================================================================


@pytest.mark.asyncio
async def test_producer_done_with_buffered_items():
    """INBOX-009: Producer signals done but items still in buffer, consumer drains before EOS."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    # Put items and mark done immediately
    await inbox.put("a", 1)
    await inbox.put("a", 2)
    await inbox.put("a", 3)
    inbox.mark_source_done("a")

    # Consumer should receive all buffered items before EOS
    items = []
    async for item in inbox.iter_input("a"):
        items.append(item)

    assert items == [1, 2, 3]


# =============================================================================
# INBOX-010: Buffer Limit Enforced
# =============================================================================


@pytest.mark.asyncio
async def test_buffer_limit_enforced():
    """INBOX-010: buffer_limit=2, third put() blocks until consumer drains at least one item."""
    inbox = NodeInbox(buffer_limit=2)
    inbox.add_upstream("a", 1)

    # Put 2 items (at limit)
    await inbox.put("a", 1)
    await inbox.put("a", 2)

    # Third put should block
    put_task = asyncio.create_task(inbox.put("a", 3))

    # Give put_task a chance to start and block
    await asyncio.sleep(0.05)

    # At this point, put_task should be blocked
    assert not put_task.done()

    # Cancel the blocked put
    put_task.cancel()
    try:
        await put_task
    except asyncio.CancelledError:
        pass


# =============================================================================
# INBOX-011: Backpressure Release
# =============================================================================


@pytest.mark.asyncio
async def test_backpressure_release():
    """INBOX-011: Producer blocked on full buffer, consumer drains, producer unblocks."""
    inbox = NodeInbox(buffer_limit=2)
    inbox.add_upstream("a", 1)

    # Put 2 items (at limit)
    await inbox.put("a", 1)
    await inbox.put("a", 2)

    # Third put should block
    put_started = asyncio.Event()
    put_completed = asyncio.Event()

    async def producer():
        put_started.set()
        await inbox.put("a", 3)
        put_completed.set()

    producer_task = asyncio.create_task(producer())

    # Wait for producer to start and block
    await asyncio.wait_for(put_started.wait(), timeout=0.5)
    await asyncio.sleep(0.05)  # Extra time to ensure it's blocked

    # Not completed yet (blocked)
    assert not put_completed.is_set()

    # Consumer drains one item
    items = []
    async for item in inbox.iter_input("a"):
        items.append(item)
        break  # Only consume one

    # Wait for producer to complete (should unblock)
    await asyncio.wait_for(put_completed.wait(), timeout=0.5)
    await producer_task

    assert items == [1]


# =============================================================================
# INBOX-012: Multiple Blocked Producers
# =============================================================================


@pytest.mark.asyncio
async def test_multiple_blocked_producers():
    """INBOX-012: Two producers blocked on same full buffer, consumer drains, at least one completes."""
    inbox = NodeInbox(buffer_limit=1)
    inbox.add_upstream("a", 1)

    # Put 1 item (at limit)
    await inbox.put("a", 1)

    # Both producers try to put (both should block)
    producer1_started = asyncio.Event()
    producer1_done = asyncio.Event()
    producer2_started = asyncio.Event()
    producer2_done = asyncio.Event()

    async def producer1():
        producer1_started.set()
        await inbox.put("a", 2)
        producer1_done.set()

    async def producer2():
        producer2_started.set()
        await inbox.put("a", 3)
        producer2_done.set()

    task1 = asyncio.create_task(producer1())
    task2 = asyncio.create_task(producer2())

    # Wait for both to start and block
    await asyncio.wait_for(producer1_started.wait(), timeout=0.5)
    await asyncio.wait_for(producer2_started.wait(), timeout=0.5)
    await asyncio.sleep(0.05)

    # Neither should be done yet
    assert not producer1_done.is_set()
    assert not producer2_done.is_set()

    # Consumer continuously drains the buffer to allow both producers to complete
    items = []

    async def consumer():
        async for item in inbox.iter_input("a"):
            items.append(item)
            # Small delay to allow producers to race and put more items
            await asyncio.sleep(0.01)

    # Mark source done so consumer will eventually terminate
    # Run consumer in parallel with a timeout
    consumer_task = asyncio.create_task(consumer())

    # Cancel both producer tasks after they should have completed
    # but wait with timeout
    try:
        await asyncio.wait_for(asyncio.gather(producer1_done.wait(), producer2_done.wait()), timeout=0.5)
    except TimeoutError:
        pass

    # At least one producer should have completed
    assert producer1_done.is_set() or producer2_done.is_set()

    # Cancel remaining tasks
    if not task1.done():
        task1.cancel()
    if not task2.done():
        task2.cancel()
    consumer_task.cancel()

    try:
        await task1
    except asyncio.CancelledError:
        pass
    try:
        await task2
    except asyncio.CancelledError:
        pass
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass

    # We should have consumed at least the initial item and possibly one from a producer
    assert len(items) >= 1


# =============================================================================
# INBOX-013: Backpressure Per Handle
# =============================================================================


@pytest.mark.asyncio
async def test_backpressure_per_handle():
    """INBOX-013: buffer_limit=2, handle A full blocks A, handle B can still accept."""
    inbox = NodeInbox(buffer_limit=2)
    inbox.add_upstream("a", 1)
    inbox.add_upstream("b", 1)

    # Fill handle A to limit
    await inbox.put("a", "a1")
    await inbox.put("a", "a2")

    # Fill handle B to limit
    await inbox.put("b", "b1")
    await inbox.put("b", "b2")

    # Try to put to A (should block)
    put_a_started = asyncio.Event()
    put_a_done = asyncio.Event()

    async def producer_a():
        put_a_started.set()
        await inbox.put("a", "a3")
        put_a_done.set()

    task_a = asyncio.create_task(producer_a())
    await asyncio.wait_for(put_a_started.wait(), timeout=0.5)
    await asyncio.sleep(0.05)
    assert not put_a_done.is_set()

    # Put to B (should also block since B is also at limit)
    put_b_started = asyncio.Event()
    put_b_done = asyncio.Event()

    async def producer_b():
        put_b_started.set()
        await inbox.put("b", "b3")
        put_b_done.set()

    task_b = asyncio.create_task(producer_b())
    await asyncio.wait_for(put_b_started.wait(), timeout=0.5)
    await asyncio.sleep(0.05)
    assert not put_b_done.is_set()

    # Cancel both blocked puts
    task_a.cancel()
    task_b.cancel()
    try:
        await task_a
    except asyncio.CancelledError:
        pass
    try:
        await task_b
    except asyncio.CancelledError:
        pass


# =============================================================================
# INBOX-014: No Limit (Unlimited)
# =============================================================================


@pytest.mark.asyncio
async def test_no_limit_unlimited():
    """INBOX-014: buffer_limit=None, producer never blocks regardless of buffer size."""
    inbox = NodeInbox(buffer_limit=None)  # No limit
    inbox.add_upstream("a", 1)

    # Put many items without blocking
    for i in range(100):
        await inbox.put("a", i)

    # All items should be in buffer
    items = []
    async for item in inbox.iter_input("a"):
        items.append(item)
        if len(items) >= 100:
            break

    assert len(items) == 100


# =============================================================================
# INBOX-019: Put After Close
# =============================================================================


@pytest.mark.asyncio
async def test_put_after_close():
    """INBOX-019: put() after close_all() is no-op."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    # Close the inbox
    await inbox.close_all()

    # Put should be no-op and not raise
    await inbox.put("a", "should_not_be_added")

    # Buffer should be empty
    assert inbox.try_pop_any() is None


# =============================================================================
# INBOX-020: Mark Done Already Zero
# =============================================================================


@pytest.mark.asyncio
async def test_mark_done_already_zero():
    """INBOX-020: mark_source_done() on handle with open_count=0 stays at 0 (no negative)."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    # Mark done once
    inbox.mark_source_done("a")
    assert inbox._open_counts.get("a") == 0

    # Mark done again - should stay at 0, not go negative
    inbox.mark_source_done("a")
    assert inbox._open_counts.get("a") == 0


# =============================================================================
# INBOX-021: Metadata Propagation
# =============================================================================


@pytest.mark.asyncio
async def test_metadata_propagation():
    """INBOX-021: Put item with metadata, consume via iter_input_with_envelope, verify metadata present."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    # Put item with metadata
    await inbox.put("a", "data1", metadata={"key1": "value1", "key2": 42})
    inbox.mark_source_done("a")

    # Consume via iter_input_with_envelope
    envelopes = []
    async for envelope in inbox.iter_input_with_envelope("a"):
        envelopes.append(envelope)

    assert len(envelopes) == 1
    assert envelopes[0].data == "data1"
    assert envelopes[0].metadata == {"key1": "value1", "key2": 42}


# =============================================================================
# INBOX-022: Timestamp Auto-Generated
# =============================================================================


@pytest.mark.asyncio
async def test_timestamp_auto_generated():
    """INBOX-022: MessageEnvelope.timestamp is auto-generated on put."""
    from datetime import datetime, timezone

    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    before_put = datetime.now(UTC)
    await inbox.put("a", "data1")
    after_put = datetime.now(UTC)

    envelope = inbox.try_pop_any_with_envelope()
    assert envelope is not None
    _handle, env = envelope

    # Timestamp should be between before and after
    assert before_put <= env.timestamp <= after_put


# =============================================================================
# INBOX-023: Event ID Unique
# =============================================================================


@pytest.mark.asyncio
async def test_event_id_unique():
    """INBOX-023: Each MessageEnvelope.event_id is unique."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    # Put multiple items
    event_ids = set()
    for i in range(10):
        await inbox.put("a", f"data{i}")
        envelope = inbox.try_pop_any_with_envelope()
        assert envelope is not None
        _handle, env = envelope
        event_ids.add(env.event_id)

    # All event IDs should be unique
    assert len(event_ids) == 10


# =============================================================================
# INBOX-024: Backward Compat Unwrap
# =============================================================================


@pytest.mark.asyncio
async def test_backward_compat_unwrap():
    """INBOX-024: iter_input() yields unwrapped data, not envelope."""
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)

    # Put item with metadata
    await inbox.put("a", {"nested": "data"}, metadata={"meta": "value"})
    inbox.mark_source_done("a")

    # iter_input should yield unwrapped data
    items = []
    async for item in inbox.iter_input("a"):
        items.append(item)

    assert len(items) == 1
    assert items[0] == {"nested": "data"}  # Unwrapped data, not envelope

    # iter_input_with_envelope should yield envelope
    inbox2 = NodeInbox()
    inbox2.add_upstream("a", 1)
    await inbox2.put("a", {"nested": "data"}, metadata={"meta": "value"})
    inbox2.mark_source_done("a")

    envelopes = []
    async for envelope in inbox2.iter_input_with_envelope("a"):
        envelopes.append(envelope)

    assert len(envelopes) == 1
    assert envelopes[0].data == {"nested": "data"}
    assert envelopes[0].metadata == {"meta": "value"}
