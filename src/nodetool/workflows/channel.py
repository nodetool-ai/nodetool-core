"""Streaming channels for named, many-to-many, graph-independent communication.

This module provides streaming channels exposed via the ProcessingContext for
broadcasting progress updates, aggregating logs from parallel nodes, and
dynamic cross-node coordination without modifying the static graph topology.

Architecture:
- **Queue-per-Subscriber**: Each subscriber gets its own asyncio.Queue for isolation.
- **Broadcast Pattern**: Publishing iterates over all subscriber queues and puts a copy.
- **Backpressure**: "Block on Slowest" - if any subscriber's queue is full, publisher blocks.

Features:
- Named channels for dynamic communication paths
- Many-to-many publisher/subscriber relationships
- FIFO message ordering within each channel
- Clean shutdown with graceful subscriber termination
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


# Internal sentinel object to signal channel closure to subscribers
_STOP_SIGNAL = object()


@dataclass
class ChannelStats:
    """Statistics about a channel's current state."""

    name: str
    subscriber_count: int
    is_closed: bool


class Channel:
    """A named broadcast channel with queue-per-subscriber semantics.

    Each subscriber gets its own asyncio.Queue for isolation. Publishing broadcasts
    to all active subscribers. When any subscriber's queue is full, the publisher
    blocks (backpressure).

    Args:
        name: The channel name for identification.
        buffer_limit: Maximum items per subscriber queue (default: 100).

    Example:
        channel = Channel("logs", buffer_limit=50)

        # Publisher
        await channel.publish({"level": "info", "message": "Hello"})

        # Subscriber
        async for item in channel.subscribe("subscriber-1"):
            print(item)
    """

    def __init__(self, name: str, buffer_limit: int = 100):
        self.name = name
        self._buffer_limit = buffer_limit
        self._subscribers: dict[str, asyncio.Queue[Any]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def publish(self, item: Any) -> None:
        """Publish an item to all active subscribers.

        If the channel is closed, raises RuntimeError. If any subscriber's queue
        is full, this method blocks until space is available (backpressure).

        Args:
            item: The item to broadcast to all subscribers.

        Raises:
            RuntimeError: If the channel is closed.
        """
        if self._closed:
            raise RuntimeError(f"Channel {self.name} is closed")

        # Snapshot subscribers under lock to avoid modification during iteration
        async with self._lock:
            queues = list(self._subscribers.values())

        # Broadcast to all subscribers
        # Sequential await enforces "Block on Slowest" backpressure
        for q in queues:
            await q.put(item)

    async def subscribe(self, subscriber_id: str) -> AsyncIterator[Any]:
        """Subscribe to this channel and yield items as they arrive.

        Creates an ephemeral queue for this subscriber. Yields items until the
        channel is closed or the subscriber exits.

        Args:
            subscriber_id: Unique identifier for this subscriber.

        Yields:
            Items published to this channel.
        """
        if self._closed:
            return

        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._buffer_limit)

        async with self._lock:
            self._subscribers[subscriber_id] = queue

        try:
            while True:
                item = await queue.get()
                if item is _STOP_SIGNAL:
                    break
                yield item
        finally:
            async with self._lock:
                self._subscribers.pop(subscriber_id, None)

    async def close(self) -> None:
        """Close the channel and signal all subscribers to stop.

        After calling close, no more items can be published. All active
        subscribers will receive a stop signal and their iterators will
        terminate cleanly.
        """
        self._closed = True
        async with self._lock:
            for q in self._subscribers.values():
                try:
                    await q.put(_STOP_SIGNAL)
                except Exception:
                    # Queue might be full or closed; ignore
                    pass

    def get_stats(self) -> ChannelStats:
        """Get current statistics for this channel.

        Returns:
            ChannelStats with name, subscriber count, and closed state.
        """
        return ChannelStats(
            name=self.name,
            subscriber_count=len(self._subscribers),
            is_closed=self._closed,
        )


class ChannelManager:
    """Registry and factory for named streaming channels.

    Provides methods to create, retrieve, and manage channels. Exposes
    convenience methods for publish/subscribe that auto-create channels.

    Example:
        manager = ChannelManager()

        # Create channel explicitly
        channel = await manager.create_channel("logs")

        # Or use helpers (auto-creates if needed)
        await manager.publish("logs", {"msg": "Hello"})

        async for item in manager.subscribe("logs", "my-subscriber"):
            process(item)

        # Cleanup
        await manager.close_all()
    """

    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}
        self._lock = asyncio.Lock()

    def get_channel(self, name: str) -> Channel | None:
        """Get an existing channel by name.

        Args:
            name: The channel name.

        Returns:
            The Channel if it exists, otherwise None.
        """
        return self._channels.get(name)

    async def create_channel(
        self, name: str, buffer_limit: int = 100, replace: bool = False
    ) -> Channel:
        """Create a new channel.

        Args:
            name: The channel name.
            buffer_limit: Maximum items per subscriber queue.
            replace: If True, replace existing channel. If False, raise error.

        Returns:
            The newly created Channel.

        Raises:
            ValueError: If channel exists and replace is False.
        """
        async with self._lock:
            if name in self._channels and not replace:
                raise ValueError(f"Channel '{name}' already exists")

            # Close existing channel if replacing
            if name in self._channels:
                old_channel = self._channels[name]
                await old_channel.close()

            channel = Channel(name, buffer_limit=buffer_limit)
            self._channels[name] = channel
            return channel

    async def get_or_create_channel(
        self, name: str, buffer_limit: int = 100
    ) -> Channel:
        """Get an existing channel or create a new one.

        Args:
            name: The channel name.
            buffer_limit: Maximum items per subscriber queue (used if creating).

        Returns:
            The existing or newly created Channel.
        """
        async with self._lock:
            if name not in self._channels:
                self._channels[name] = Channel(name, buffer_limit=buffer_limit)
            return self._channels[name]

    async def publish(self, name: str, item: Any, buffer_limit: int = 100) -> None:
        """Publish an item to a channel, creating it if necessary.

        Args:
            name: The channel name.
            item: The item to publish.
            buffer_limit: Buffer limit if creating the channel.
        """
        channel = await self.get_or_create_channel(name, buffer_limit=buffer_limit)
        await channel.publish(item)

    async def subscribe(
        self, name: str, subscriber_id: str, buffer_limit: int = 100
    ) -> AsyncIterator[Any]:
        """Subscribe to a channel, creating it if necessary.

        Args:
            name: The channel name.
            subscriber_id: Unique identifier for this subscriber.
            buffer_limit: Buffer limit if creating the channel.

        Yields:
            Items published to the channel.
        """
        channel = await self.get_or_create_channel(name, buffer_limit=buffer_limit)
        async for item in channel.subscribe(subscriber_id):
            yield item

    async def close_channel(self, name: str) -> None:
        """Close a specific channel.

        Args:
            name: The channel name to close.
        """
        async with self._lock:
            channel = self._channels.get(name)
            if channel:
                await channel.close()
                del self._channels[name]

    async def close_all(self) -> None:
        """Close all channels and clear the registry.

        This should be called during context cleanup to ensure all
        subscriber iterators terminate cleanly.
        """
        async with self._lock:
            for channel in self._channels.values():
                try:
                    await channel.close()
                except Exception as e:
                    log.debug(f"Error closing channel {channel.name}: {e}")
            self._channels.clear()

    def list_channels(self) -> list[str]:
        """List all channel names.

        Returns:
            List of channel names.
        """
        return list(self._channels.keys())

    def get_all_stats(self) -> list[ChannelStats]:
        """Get stats for all channels.

        Returns:
            List of ChannelStats for all channels.
        """
        return [channel.get_stats() for channel in self._channels.values()]
