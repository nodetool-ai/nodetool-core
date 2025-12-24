import asyncio
import threading

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
