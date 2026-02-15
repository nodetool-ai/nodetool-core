"""Tests for AsyncChannel utility."""
import asyncio

import pytest

from nodetool.concurrency import (
    AsyncChannel,
    ChannelClosedError,
    create_channel,
    fan_in,
    fan_out,
)


@pytest.mark.asyncio
async def test_channel_send_and_receive():
    """Test basic send and receive operations."""
    channel = AsyncChannel[str]()

    await channel.send("hello")
    await channel.send("world")

    assert await channel.receive() == "hello"
    assert await channel.receive() == "world"


@pytest.mark.asyncio
async def test_channel_send_nowait():
    """Test send_nowait operation."""
    channel = AsyncChannel[str]()

    channel.send_nowait("hello")
    channel.send_nowait("world")

    assert await channel.receive() == "hello"
    assert await channel.receive() == "world"


@pytest.mark.asyncio
async def test_channel_receive_nowait():
    """Test receive_nowait operation."""
    channel = AsyncChannel[str]()

    await channel.send("hello")

    assert channel.receive_nowait() == "hello"


@pytest.mark.asyncio
async def test_channel_receive_nowait_empty():
    """Test receive_nowait on empty channel raises QueueEmpty."""
    import asyncio

    channel = AsyncChannel[str]()

    with pytest.raises(asyncio.QueueEmpty):
        channel.receive_nowait()


@pytest.mark.asyncio
async def test_channel_close():
    """Test channel closing."""
    channel = AsyncChannel[str]()

    await channel.send("hello")
    channel.close()

    # Should still be able to receive existing items
    assert await channel.receive() == "hello"

    # But cannot send more
    with pytest.raises(ChannelClosedError):
        await channel.send("world")


@pytest.mark.asyncio
async def test_channel_closed_property():
    """Test closed property."""
    channel = AsyncChannel[str]()

    assert not channel.closed
    channel.close()
    assert channel.closed


@pytest.mark.asyncio
async def test_channel_empty_property():
    """Test empty property."""
    channel = AsyncChannel[str]()

    assert channel.empty
    await channel.send("hello")
    assert not channel.empty
    await channel.receive()
    assert channel.empty


@pytest.mark.asyncio
async def test_channel_full_property():
    """Test full property with bounded channel."""
    channel = AsyncChannel[str](max_size=2)

    assert not channel.full
    await channel.send("hello")
    await channel.send("world")
    assert channel.full


@pytest.mark.asyncio
async def test_channel_qsize():
    """Test qsize property."""
    channel = AsyncChannel[str]()

    assert channel.qsize == 0
    await channel.send("hello")
    assert channel.qsize == 1
    await channel.send("world")
    assert channel.qsize == 2
    await channel.receive()
    assert channel.qsize == 1


@pytest.mark.asyncio
async def test_channel_iteration():
    """Test async iteration over channel."""
    channel = AsyncChannel[int]()

    async def producer():
        for i in range(5):
            await channel.send(i)
        channel.close()

    async def consumer():
        results = []
        async for item in channel:
            results.append(item)
        return results

    # Run producer and consumer concurrently
    producer_task = asyncio.create_task(producer())
    results = await consumer()
    await producer_task

    assert results == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_channel_context_manager():
    """Test using channel as context manager."""
    async with AsyncChannel[str]() as channel:
        await channel.send("hello")
        await channel.send("world")

        assert await channel.receive() == "hello"
        assert await channel.receive() == "world"

    # Channel should be closed after exiting context
    assert channel.closed


@pytest.mark.asyncio
async def test_channel_receive_or_wait_with_timeout():
    """Test receive_or_wait with timeout."""
    channel = AsyncChannel[str]()

    # Timeout on empty channel
    result = await channel.receive_or_wait(timeout=0.1)
    assert result is None

    # Success when item is available
    async def send_delayed():
        await asyncio.sleep(0.05)
        await channel.send("hello")

    send_task = asyncio.create_task(send_delayed())
    result = await channel.receive_or_wait(timeout=1.0)
    assert result == "hello"
    await send_task


@pytest.mark.asyncio
async def test_create_channel():
    """Test create_channel convenience function."""
    channel = await create_channel(max_size=10)

    assert isinstance(channel, AsyncChannel)
    await channel.send("test")
    assert await channel.receive() == "test"


@pytest.mark.asyncio
async def test_fan_in():
    """Test fan_in combining multiple channels."""
    ch1 = AsyncChannel[str]()
    ch2 = AsyncChannel[str]()

    merged = await fan_in(ch1, ch2)

    # Send to both channels
    await ch1.send("from_ch1_1")
    await ch2.send("from_ch2_1")
    await ch1.send("from_ch1_2")

    # Close both channels
    ch1.close()
    ch2.close()

    # Receive from merged (order may vary)
    results = set()
    async for item in merged:
        results.add(item)

    assert results == {"from_ch1_1", "from_ch2_1", "from_ch1_2"}


@pytest.mark.asyncio
async def test_fan_out():
    """Test fan_out distributing to multiple channels."""
    source = AsyncChannel[str]()
    out1 = AsyncChannel[str]()
    out2 = AsyncChannel[str]()

    # Start fan_out in background
    fan_task = asyncio.create_task(fan_out(source, out1, out2))

    # Send items
    await source.send("hello")
    await source.send("world")
    source.close()

    # Wait for fan_out to complete
    await fan_task

    # Both output channels should have the same items
    results1 = []
    results2 = []

    async def collect1():
        async for item in out1:
            results1.append(item)

    async def collect2():
        async for item in out2:
            results2.append(item)

    await asyncio.gather(collect1(), collect2())

    assert results1 == ["hello", "world"]
    assert results2 == ["hello", "world"]


@pytest.mark.asyncio
async def test_channel_backpressure():
    """Test that bounded channel provides backpressure."""
    channel = AsyncChannel[int](max_size=2)

    # Fill the channel
    await channel.send(1)
    await channel.send(2)
    assert channel.full

    # This send should block until there's space
    async def slow_receiver():
        await asyncio.sleep(0.1)
        assert await channel.receive() == 1
        await asyncio.sleep(0.1)
        assert await channel.receive() == 2
        return 3

    # Start receiver and sender concurrently
    receiver_task = asyncio.create_task(slow_receiver())
    await asyncio.sleep(0.05)  # Let receiver start waiting

    # This should complete when receiver makes space
    await channel.send(3)

    result = await receiver_task
    assert result == 3

    # Clean up
    assert await channel.receive() == 3


@pytest.mark.asyncio
async def test_channel_with_task_group():
    """Test using channel with asyncio.gather for producer-consumer pattern."""
    channel = AsyncChannel[str]()

    async def producer():
        items = ["apple", "banana", "cherry"]
        for item in items:
            await channel.send(item)
        channel.close()

    async def consumer():
        results = []
        async for item in channel:
            results.append(item)
        return results

    results = await asyncio.gather(producer(), consumer())
    assert results[1] == ["apple", "banana", "cherry"]


@pytest.mark.asyncio
async def test_multiple_consumers():
    """Test multiple consumers reading from the same channel."""
    channel = AsyncChannel[int]()

    async def consumer(name: str):
        results = []
        async for item in channel:
            results.append((name, item))
        return results

    # Start multiple consumers
    consumer1 = asyncio.create_task(consumer("c1"))
    consumer2 = asyncio.create_task(consumer("c2"))

    # Produce items
    for i in range(5):
        await channel.send(i)
    channel.close()

    # Wait for consumers
    results1 = await consumer1
    results2 = await consumer2

    # All items should be distributed between consumers
    all_results = results1 + results2
    assert len(all_results) == 5
    assert {item for _, item in all_results} == {0, 1, 2, 3, 4}
