"""Regression tests for Channel.close() deadlock and NodeInbox arrival-queue sync."""

import asyncio

import pytest

from nodetool.workflows.channel import Channel, ChannelManager
from nodetool.workflows.inbox import NodeInbox


async def _publish_until_blocked(channel: Channel, count: int = 5) -> None:
    """Publish until backpressure blocks, leaving the subscriber queue full."""
    for i in range(count):
        try:
            await asyncio.wait_for(channel.publish(i), timeout=0.3)
        except TimeoutError:
            return
    raise AssertionError("expected publisher to block on a full subscriber queue")


@pytest.mark.asyncio
async def test_close_does_not_block_on_full_subscriber_queue():
    channel: Channel[int] = Channel("full", buffer_limit=2)

    async def stalled_subscriber():
        async for _item in channel.subscribe("sub"):
            await asyncio.sleep(3600)

    task = asyncio.create_task(stalled_subscriber())
    await asyncio.sleep(0.05)
    await _publish_until_blocked(channel)
    await asyncio.sleep(0.05)

    try:
        await asyncio.wait_for(channel.close(), timeout=1.0)

        # The lock must not be held after close returns.
        await asyncio.wait_for(channel._lock.acquire(), timeout=0.5)
        channel._lock.release()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_close_wakes_and_unregisters_stalled_subscriber():
    channel: Channel[int] = Channel("wake", buffer_limit=2)
    gate = asyncio.Event()
    finished = asyncio.Event()

    async def subscriber():
        try:
            async for _item in channel.subscribe("sub"):
                await gate.wait()
        finally:
            finished.set()

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.05)
    await _publish_until_blocked(channel)
    await asyncio.sleep(0.05)

    await asyncio.wait_for(channel.close(), timeout=1.0)
    gate.set()

    await asyncio.wait_for(finished.wait(), timeout=1.0)
    await asyncio.wait_for(task, timeout=1.0)
    assert channel.get_stats().subscriber_count == 0


@pytest.mark.asyncio
async def test_close_all_not_blocked_by_wedged_channel():
    manager = ChannelManager()
    wedged = await manager.create_channel("wedged", buffer_limit=1)
    await manager.create_channel("healthy", buffer_limit=10)

    async def stalled_subscriber():
        async for _item in wedged.subscribe("sub"):
            await asyncio.sleep(3600)

    task = asyncio.create_task(stalled_subscriber())
    await asyncio.sleep(0.05)
    await _publish_until_blocked(wedged)
    await asyncio.sleep(0.05)

    try:
        await asyncio.wait_for(manager.close_all(), timeout=1.0)
        assert manager.list_channels() == []
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_iter_input_keeps_arrival_queue_in_sync():
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    for i in range(3):
        await inbox.put("a", i)
    inbox.mark_source_done("a")

    received = [item async for item in inbox.iter_input("a")]

    assert received == [0, 1, 2]
    assert list(inbox._arrival) == []


@pytest.mark.asyncio
async def test_iter_input_with_envelope_keeps_arrival_queue_in_sync():
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    await inbox.put("a", "x", metadata={"k": "v"})
    inbox.mark_source_done("a")

    envelopes = [env async for env in inbox.iter_input_with_envelope("a")]

    assert [env.data for env in envelopes] == ["x"]
    assert envelopes[0].metadata == {"k": "v"}
    assert list(inbox._arrival) == []


@pytest.mark.asyncio
async def test_try_pop_any_sees_data_after_other_handle_drained():
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    inbox.add_upstream("b", 1)

    await inbox.put("a", "A1")
    inbox.mark_source_done("a")
    assert [item async for item in inbox.iter_input("a")] == ["A1"]

    await inbox.put("b", "B1")

    assert inbox.try_pop_any() == ("b", "B1")
    assert inbox.try_pop_any() is None


@pytest.mark.asyncio
async def test_iter_any_still_sees_remaining_handle_after_partial_drain():
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    inbox.add_upstream("b", 1)

    await inbox.put("a", "A1")
    await inbox.put("b", "B1")
    inbox.mark_source_done("a")
    assert [item async for item in inbox.iter_input("a")] == ["A1"]
    inbox.mark_source_done("b")

    assert [pair async for pair in inbox.iter_any()] == [("b", "B1")]
