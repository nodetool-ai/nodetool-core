"""Tests for streaming channels functionality.

These tests verify the Channel and ChannelManager classes used for
named, many-to-many, graph-independent communication via ProcessingContext.
"""

import asyncio

import pytest

from nodetool.workflows.channel import Channel, ChannelManager, ChannelStats
from nodetool.workflows.processing_context import ProcessingContext


class TestChannel:
    """Tests for the Channel class."""

    @pytest.mark.asyncio
    async def test_basic_pub_sub(self):
        """Test 1: Basic Pub/Sub - 1 Publisher, 1 Subscriber."""
        channel = Channel("test", buffer_limit=100)
        received = []

        async def subscriber():
            async for item in channel.subscribe("sub1"):
                received.append(item)

        # Start subscriber task
        sub_task = asyncio.create_task(subscriber())
        # Give subscriber time to register
        await asyncio.sleep(0.01)

        # Publish messages
        await channel.publish("msg1")
        await channel.publish("msg2")
        await channel.publish("msg3")

        # Close channel to signal end
        await channel.close()

        # Wait for subscriber to finish
        await asyncio.wait_for(sub_task, timeout=1.0)

        assert received == ["msg1", "msg2", "msg3"]

    @pytest.mark.asyncio
    async def test_fan_out(self):
        """Test 2: Fan-out - 1 Publisher, 3 Subscribers. All get all messages."""
        channel = Channel("broadcast", buffer_limit=100)
        received_by_sub1 = []
        received_by_sub2 = []
        received_by_sub3 = []

        async def subscriber(sub_id: str, storage: list):
            async for item in channel.subscribe(sub_id):
                storage.append(item)

        # Start all subscribers
        sub1_task = asyncio.create_task(subscriber("sub1", received_by_sub1))
        sub2_task = asyncio.create_task(subscriber("sub2", received_by_sub2))
        sub3_task = asyncio.create_task(subscriber("sub3", received_by_sub3))

        # Give subscribers time to register
        await asyncio.sleep(0.01)

        # Verify all 3 subscribers registered
        stats = channel.get_stats()
        assert stats.subscriber_count == 3

        # Publish messages
        await channel.publish("A")
        await channel.publish("B")
        await channel.publish("C")

        # Close channel
        await channel.close()

        # Wait for all subscribers
        await asyncio.gather(sub1_task, sub2_task, sub3_task)

        # All subscribers should receive all messages
        expected = ["A", "B", "C"]
        assert received_by_sub1 == expected
        assert received_by_sub2 == expected
        assert received_by_sub3 == expected

    @pytest.mark.asyncio
    async def test_late_joiner(self):
        """Test 3: Late Joiner - Subscriber misses messages before joining."""
        channel = Channel("late-test", buffer_limit=100)
        received = []

        # Publish message A before any subscriber
        await channel.publish("A")

        async def subscriber():
            async for item in channel.subscribe("late-sub"):
                received.append(item)

        # Now start subscriber
        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.01)

        # Publish message B after subscriber joined
        await channel.publish("B")

        # Close channel
        await channel.close()

        await asyncio.wait_for(sub_task, timeout=1.0)

        # Late joiner should only get B, not A
        assert received == ["B"]

    @pytest.mark.asyncio
    async def test_isolation_slow_reader(self):
        """Test 4: Isolation - Slow reader doesn't block fast reader's processing.

        Note: Due to "Block on Slowest" backpressure, the publisher may block
        if the slow reader's queue fills up, but the fast reader's processing
        of already-received items is not blocked by the slow reader's processing.
        """
        channel = Channel("isolation-test", buffer_limit=100)
        fast_received = []
        slow_received = []
        fast_timestamps = []
        slow_timestamps = []

        async def fast_subscriber():
            async for item in channel.subscribe("fast"):
                fast_timestamps.append(asyncio.get_event_loop().time())
                fast_received.append(item)

        async def slow_subscriber():
            async for item in channel.subscribe("slow"):
                slow_timestamps.append(asyncio.get_event_loop().time())
                slow_received.append(item)
                await asyncio.sleep(0.1)  # Simulate slow processing

        fast_task = asyncio.create_task(fast_subscriber())
        slow_task = asyncio.create_task(slow_subscriber())
        await asyncio.sleep(0.01)

        # Publish several messages
        for i in range(5):
            await channel.publish(f"msg{i}")

        await channel.close()

        await asyncio.gather(fast_task, slow_task)

        # Both should receive all messages
        assert fast_received == ["msg0", "msg1", "msg2", "msg3", "msg4"]
        assert slow_received == ["msg0", "msg1", "msg2", "msg3", "msg4"]

        # Fast reader should process messages much quicker
        # (all messages received before slow reader finishes)
        if fast_timestamps and slow_timestamps:
            # Fast reader's last timestamp should be much earlier than slow reader's last
            assert fast_timestamps[-1] < slow_timestamps[-1]

    @pytest.mark.asyncio
    async def test_cleanup_breaks_loop(self):
        """Test 5: Cleanup - close() breaks the async for loops of subscribers."""
        channel = Channel("cleanup-test", buffer_limit=100)
        received = []
        loop_exited = asyncio.Event()

        async def subscriber():
            try:
                async for item in channel.subscribe("sub"):
                    received.append(item)
            finally:
                loop_exited.set()

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.01)

        # Publish one message
        await channel.publish("before-close")

        # Close channel - should break the loop
        await channel.close()

        # Wait for subscriber task to complete
        await asyncio.wait_for(sub_task, timeout=1.0)

        # Verify loop exited cleanly
        assert loop_exited.is_set()
        assert received == ["before-close"]

    @pytest.mark.asyncio
    async def test_closed_channel_rejects_publish(self):
        """Test that publishing to a closed channel raises an error."""
        channel = Channel("closed-test", buffer_limit=100)
        await channel.close()

        with pytest.raises(RuntimeError, match="closed"):
            await channel.publish("should-fail")

    @pytest.mark.asyncio
    async def test_closed_channel_subscribe_returns_immediately(self):
        """Test that subscribing to a closed channel returns immediately."""
        channel = Channel("closed-sub-test", buffer_limit=100)
        await channel.close()

        received = []
        async for item in channel.subscribe("sub"):
            received.append(item)

        assert received == []

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test ChannelStats reporting."""
        channel = Channel("stats-test", buffer_limit=50)

        stats = channel.get_stats()
        assert stats.name == "stats-test"
        assert stats.subscriber_count == 0
        assert stats.is_closed is False

        # Add a subscriber
        async def subscriber():
            async for _ in channel.subscribe("sub1"):
                pass

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.01)

        stats = channel.get_stats()
        assert stats.subscriber_count == 1

        await channel.close()
        await sub_task

        stats = channel.get_stats()
        assert stats.is_closed is True
        assert stats.subscriber_count == 0


class TestChannelManager:
    """Tests for the ChannelManager class."""

    @pytest.mark.asyncio
    async def test_create_channel(self):
        """Test creating a channel through the manager."""
        manager = ChannelManager()

        channel = await manager.create_channel("logs", buffer_limit=50)

        assert channel.name == "logs"
        assert manager.get_channel("logs") is channel
        assert "logs" in manager.list_channels()

        await manager.close_all()

    @pytest.mark.asyncio
    async def test_create_channel_duplicate_raises(self):
        """Test that creating duplicate channel raises error."""
        manager = ChannelManager()

        await manager.create_channel("unique")

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_channel("unique")

        await manager.close_all()

    @pytest.mark.asyncio
    async def test_create_channel_replace(self):
        """Test replacing an existing channel."""
        manager = ChannelManager()

        channel1 = await manager.create_channel("replaceable")
        channel2 = await manager.create_channel("replaceable", replace=True)

        assert channel2 is not channel1
        assert manager.get_channel("replaceable") is channel2

        await manager.close_all()

    @pytest.mark.asyncio
    async def test_get_or_create_channel(self):
        """Test get_or_create_channel creates or retrieves."""
        manager = ChannelManager()

        # First call creates
        channel1 = await manager.get_or_create_channel("auto")
        assert manager.get_channel("auto") is channel1

        # Second call retrieves
        channel2 = await manager.get_or_create_channel("auto")
        assert channel2 is channel1

        await manager.close_all()

    @pytest.mark.asyncio
    async def test_publish_subscribe_helpers(self):
        """Test publish/subscribe helper methods."""
        manager = ChannelManager()
        received = []

        async def subscriber():
            async for item in manager.subscribe("events", "consumer-1"):
                received.append(item)

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.01)

        await manager.publish("events", "event1")
        await manager.publish("events", "event2")

        await manager.close_all()
        await sub_task

        assert received == ["event1", "event2"]

    @pytest.mark.asyncio
    async def test_close_channel(self):
        """Test closing a specific channel."""
        manager = ChannelManager()

        await manager.create_channel("temp")
        assert "temp" in manager.list_channels()

        await manager.close_channel("temp")
        assert "temp" not in manager.list_channels()

        await manager.close_all()

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all channels."""
        manager = ChannelManager()

        await manager.create_channel("ch1")
        await manager.create_channel("ch2")
        await manager.create_channel("ch3")

        assert len(manager.list_channels()) == 3

        await manager.close_all()

        assert len(manager.list_channels()) == 0

    @pytest.mark.asyncio
    async def test_get_all_stats(self):
        """Test getting stats for all channels."""
        manager = ChannelManager()

        await manager.create_channel("alpha")
        await manager.create_channel("beta")

        stats = manager.get_all_stats()
        names = {s.name for s in stats}

        assert "alpha" in names
        assert "beta" in names

        await manager.close_all()


class TestProcessingContextChannels:
    """Tests for channels integration with ProcessingContext."""

    @pytest.mark.asyncio
    async def test_context_has_channel_manager(self):
        """Test that ProcessingContext has a ChannelManager."""
        context = ProcessingContext()

        assert hasattr(context, "channels")
        assert isinstance(context.channels, ChannelManager)

    @pytest.mark.asyncio
    async def test_context_cleanup_closes_channels(self):
        """Test that context.cleanup() closes all channels."""
        context = ProcessingContext()

        # Create a channel
        await context.channels.create_channel("test-cleanup")
        assert "test-cleanup" in context.channels.list_channels()

        # Cleanup
        await context.cleanup()

        # Channel should be closed
        assert len(context.channels.list_channels()) == 0

    @pytest.mark.asyncio
    async def test_context_channel_usage(self):
        """Test typical channel usage through context."""
        context = ProcessingContext()
        received = []

        async def node_subscriber():
            async for item in context.channels.subscribe("progress", "node-1"):
                received.append(item)

        sub_task = asyncio.create_task(node_subscriber())
        await asyncio.sleep(0.01)

        # Simulate progress updates from another node
        await context.channels.publish("progress", {"percent": 25})
        await context.channels.publish("progress", {"percent": 50})
        await context.channels.publish("progress", {"percent": 100})

        await context.cleanup()
        await sub_task

        assert len(received) == 3
        assert received[0]["percent"] == 25
        assert received[-1]["percent"] == 100


class TestChannelBackpressure:
    """Tests for backpressure behavior."""

    @pytest.mark.asyncio
    async def test_backpressure_blocks_publisher(self):
        """Test that publisher blocks when subscriber queue is full."""
        # Small buffer to trigger backpressure quickly
        channel = Channel("backpressure", buffer_limit=2)
        publish_events = []

        async def slow_subscriber():
            async for _item in channel.subscribe("slow"):
                # Very slow processing
                await asyncio.sleep(0.1)

        async def publisher():
            for i in range(5):
                publish_events.append(f"start_{i}")
                await channel.publish(f"msg{i}")
                publish_events.append(f"done_{i}")

        # Start slow subscriber first
        sub_task = asyncio.create_task(slow_subscriber())
        await asyncio.sleep(0.01)

        # Start publisher
        pub_task = asyncio.create_task(publisher())

        # Wait a bit for some messages to be published
        await asyncio.sleep(0.05)

        # Not all messages should be published yet due to backpressure
        done_count = sum(1 for e in publish_events if e.startswith("done_"))

        # With buffer of 2 and slow consumer, should be blocked
        # (exact number depends on timing, but shouldn't be all 5)
        assert done_count <= 3

        # Close channel to unblock everything
        await channel.close()
        await asyncio.gather(sub_task, pub_task, return_exceptions=True)


class TestMemoryStability:
    """Tests for memory stability requirements."""

    @pytest.mark.asyncio
    async def test_no_memory_leak_without_subscribers(self):
        """Test that publishing to channel with no subscribers doesn't accumulate."""
        channel = Channel("no-sub", buffer_limit=100)

        # Publish many messages with no subscribers
        for i in range(1000):
            await channel.publish(f"msg{i}")

        stats = channel.get_stats()
        # No subscribers means no queues to accumulate
        assert stats.subscriber_count == 0

        await channel.close()

    @pytest.mark.asyncio
    async def test_subscriber_cleanup_on_exit(self):
        """Test that subscriber queues are cleaned up when subscriber exits."""
        channel = Channel("cleanup-check", buffer_limit=100)

        async def short_lived_subscriber():
            count = 0
            async for _ in channel.subscribe("temp"):
                count += 1
                if count >= 2:
                    return  # Exit early

        # Start subscriber
        sub_task = asyncio.create_task(short_lived_subscriber())
        await asyncio.sleep(0.01)

        assert channel.get_stats().subscriber_count == 1

        # Publish enough to trigger early exit
        await channel.publish("1")
        await channel.publish("2")
        await channel.publish("3")

        await asyncio.wait_for(sub_task, timeout=1.0)

        # Allow a small delay for the finally block to execute after return
        await asyncio.sleep(0.01)

        # After subscriber exits, it should be cleaned up
        assert channel.get_stats().subscriber_count == 0

        await channel.close()
