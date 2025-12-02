import asyncio

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
