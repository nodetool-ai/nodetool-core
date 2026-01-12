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
- **Type-safe channels**: Generic `Channel[T]` enforces type consistency between publishers and subscribers
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Generic, TypeVar, overload

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

# Type variable for typed channels
T = TypeVar("T")


# Internal sentinel object to signal channel closure to subscribers
_STOP_SIGNAL = object()


@dataclass
class ChannelStats:
    """Statistics about a channel's current state."""

    name: str
    subscriber_count: int
    is_closed: bool
    message_type: type | None = None


class Channel(Generic[T]):
    """A named broadcast channel with queue-per-subscriber semantics and type safety.

    Each subscriber gets its own asyncio.Queue for isolation. Publishing broadcasts
    to all active subscribers. When any subscriber's queue is full, the publisher
    blocks (backpressure).

    The channel is generic over the message type T, ensuring type consistency
    between publishers and subscribers at static analysis time.

    Args:
        name: The channel name for identification.
        buffer_limit: Maximum items per subscriber queue (default: 100).
        message_type: Optional type for runtime type checking of published messages.

    Example:
        # Typed channel for string messages
        channel: Channel[str] = Channel("logs", buffer_limit=50, message_type=str)

        # Publisher - type checked at runtime
        await channel.publish("Hello, World!")

        # Subscriber - yields str items
        async for item in channel.subscribe("subscriber-1"):
            print(item)  # item is str

        # Typed channel for dict messages
        channel_dict: Channel[dict] = Channel("events", message_type=dict)
        await channel_dict.publish({"level": "info", "message": "Hello"})
    """

    def __init__(
        self, name: str, buffer_limit: int = 100, message_type: type[T] | None = None
    ):
        self.name = name
        self._buffer_limit = buffer_limit
        self._subscribers: dict[str, asyncio.Queue[T | object]] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._message_type = message_type

    @property
    def message_type(self) -> type[T] | None:
        """The message type for this channel, if specified."""
        return self._message_type

    async def publish(self, item: T) -> None:
        """Publish an item to all active subscribers.

        If the channel is closed, raises RuntimeError. If any subscriber's queue
        is full, this method blocks until space is available (backpressure).

        If the channel was created with a message_type, the item is validated
        against that type at runtime.

        Args:
            item: The item to broadcast to all subscribers.

        Raises:
            RuntimeError: If the channel is closed.
            TypeError: If the item does not match the channel's message_type.
        """
        if self._closed:
            raise RuntimeError(f"Channel {self.name} is closed")

        # Runtime type checking if message_type is specified
        if self._message_type is not None and not isinstance(item, self._message_type):
            raise TypeError(
                f"Channel '{self.name}' expects messages of type {self._message_type.__name__}, "
                f"got {type(item).__name__}"
            )

        # Snapshot subscribers under lock to avoid modification during iteration
        async with self._lock:
            queues = list(self._subscribers.values())

        # Broadcast to all subscribers
        # Sequential await enforces "Block on Slowest" backpressure
        for q in queues:
            await q.put(item)

    async def subscribe(self, subscriber_id: str) -> AsyncIterator[T]:
        """Subscribe to this channel and yield items as they arrive.

        Creates an ephemeral queue for this subscriber. Yields items until the
        channel is closed or the subscriber exits.

        Args:
            subscriber_id: Unique identifier for this subscriber.

        Yields:
            Items published to this channel, typed as T.
        """
        if self._closed:
            return

        queue: asyncio.Queue[T | object] = asyncio.Queue(maxsize=self._buffer_limit)

        async with self._lock:
            self._subscribers[subscriber_id] = queue

        try:
            while True:
                item = await queue.get()
                if item is _STOP_SIGNAL:
                    break
                yield item  # type: ignore[misc]
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
                except asyncio.QueueFull:
                    # Queue is full; subscriber will see closed flag on next iteration
                    log.debug(
                        f"Queue full when closing channel {self.name}, subscriber will exit on next poll"
                    )
                except RuntimeError as e:
                    # Queue might be closed or event loop issues
                    log.debug(f"RuntimeError closing channel {self.name}: {e}")

    def get_stats(self) -> ChannelStats:
        """Get current statistics for this channel.

        Returns:
            ChannelStats with name, subscriber count, closed state, and message type.
        """
        return ChannelStats(
            name=self.name,
            subscriber_count=len(self._subscribers),
            is_closed=self._closed,
            message_type=self._message_type,
        )


class ChannelManager:
    """Registry and factory for named streaming channels with type safety.

    Provides methods to create, retrieve, and manage typed channels. Exposes
    convenience methods for publish/subscribe that auto-create channels.

    Type Safety:
    - Channels can be created with an explicit message type for runtime validation.
    - When a channel is created with a type, all publish operations validate
      that the message matches the expected type.
    - Subscribing to a typed channel returns an iterator yielding the correct type.

    Example:
        manager = ChannelManager()

        # Create typed channel explicitly
        channel: Channel[str] = await manager.create_channel("logs", message_type=str)

        # Or use typed helpers (auto-creates if needed)
        await manager.publish_typed("events", {"msg": "Hello"}, message_type=dict)

        async for item in manager.subscribe_typed("events", "my-subscriber", message_type=dict):
            process(item)  # item is dict

        # Untyped usage still works for backwards compatibility
        await manager.publish("logs", {"msg": "Hello"})

        # Cleanup
        await manager.close_all()
    """

    def __init__(self) -> None:
        self._channels: dict[str, Channel[Any]] = {}
        self._channel_types: dict[str, type] = {}
        self._lock = asyncio.Lock()

    def get_channel(self, name: str) -> Channel[Any] | None:
        """Get an existing channel by name.

        Args:
            name: The channel name.

        Returns:
            The Channel if it exists, otherwise None.
        """
        return self._channels.get(name)

    @overload
    async def create_channel(
        self,
        name: str,
        buffer_limit: int = 100,
        replace: bool = False,
        *,
        message_type: type[T],
    ) -> Channel[T]: ...

    @overload
    async def create_channel(
        self,
        name: str,
        buffer_limit: int = 100,
        replace: bool = False,
        *,
        message_type: None = None,
    ) -> Channel[Any]: ...

    async def create_channel(
        self,
        name: str,
        buffer_limit: int = 100,
        replace: bool = False,
        *,
        message_type: type[T] | None = None,
    ) -> Channel[T] | Channel[Any]:
        """Create a new channel with optional type enforcement.

        Args:
            name: The channel name.
            buffer_limit: Maximum items per subscriber queue.
            replace: If True, replace existing channel. If False, raise error.
            message_type: Optional type for runtime message validation.

        Returns:
            The newly created Channel, typed as Channel[T] if message_type is provided.

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

            channel: Channel[Any] = Channel(
                name, buffer_limit=buffer_limit, message_type=message_type
            )
            self._channels[name] = channel
            if message_type is not None:
                self._channel_types[name] = message_type
            elif name in self._channel_types:
                del self._channel_types[name]
            return channel

    @overload
    async def get_or_create_channel(
        self, name: str, buffer_limit: int = 100, *, message_type: type[T]
    ) -> Channel[T]: ...

    @overload
    async def get_or_create_channel(
        self, name: str, buffer_limit: int = 100, *, message_type: None = None
    ) -> Channel[Any]: ...

    async def get_or_create_channel(
        self, name: str, buffer_limit: int = 100, *, message_type: type[T] | None = None
    ) -> Channel[T] | Channel[Any]:
        """Get an existing channel or create a new one with optional type enforcement.

        Args:
            name: The channel name.
            buffer_limit: Maximum items per subscriber queue (used if creating).
            message_type: Optional type for runtime message validation (used if creating).

        Returns:
            The existing or newly created Channel.

        Raises:
            TypeError: If channel exists with a different message_type than requested.
        """
        async with self._lock:
            if name not in self._channels:
                channel: Channel[Any] = Channel(
                    name, buffer_limit=buffer_limit, message_type=message_type
                )
                self._channels[name] = channel
                if message_type is not None:
                    self._channel_types[name] = message_type
            else:
                # Validate type consistency if both existing and requested have types
                existing_type = self._channel_types.get(name)
                if (
                    message_type is not None
                    and existing_type is not None
                    and message_type != existing_type
                ):
                    raise TypeError(
                        f"Channel '{name}' has type {existing_type.__name__}, "
                        f"but {message_type.__name__} was requested"
                    )
            return self._channels[name]

    def get_channel_type(self, name: str) -> type | None:
        """Get the registered message type for a channel.

        Args:
            name: The channel name.

        Returns:
            The message type if set, otherwise None.
        """
        return self._channel_types.get(name)

    async def publish(self, name: str, item: Any, buffer_limit: int = 100) -> None:
        """Publish an item to a channel, creating it if necessary.

        This is the untyped publish method for backwards compatibility.
        For type-safe publishing, use publish_typed().

        Args:
            name: The channel name.
            item: The item to publish.
            buffer_limit: Buffer limit if creating the channel.
        """
        channel = await self.get_or_create_channel(name, buffer_limit=buffer_limit)
        await channel.publish(item)

    async def publish_typed(
        self, name: str, item: T, *, message_type: type[T], buffer_limit: int = 100
    ) -> None:
        """Publish a typed item to a channel, creating it if necessary.

        This method ensures type consistency: if the channel doesn't exist,
        it creates it with the specified message_type. If it exists, it validates
        that the channel's type matches.

        Args:
            name: The channel name.
            item: The item to publish.
            message_type: The type of the message (used for validation and channel creation).
            buffer_limit: Buffer limit if creating the channel.

        Raises:
            TypeError: If the channel exists with a different type or item doesn't match type.
        """
        channel = await self.get_or_create_channel(
            name, buffer_limit=buffer_limit, message_type=message_type
        )
        await channel.publish(item)

    async def subscribe(
        self, name: str, subscriber_id: str, buffer_limit: int = 100
    ) -> AsyncIterator[Any]:
        """Subscribe to a channel, creating it if necessary.

        This is the untyped subscribe method for backwards compatibility.
        For type-safe subscribing, use subscribe_typed().

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

    async def subscribe_typed(
        self,
        name: str,
        subscriber_id: str,
        *,
        message_type: type[T],
        buffer_limit: int = 100,
    ) -> AsyncIterator[T]:
        """Subscribe to a typed channel, creating it if necessary.

        This method ensures type consistency: if the channel doesn't exist,
        it creates it with the specified message_type. If it exists, it validates
        that the channel's type matches.

        Args:
            name: The channel name.
            subscriber_id: Unique identifier for this subscriber.
            message_type: The expected type of messages (used for validation and channel creation).
            buffer_limit: Buffer limit if creating the channel.

        Yields:
            Items published to the channel, typed as T.

        Raises:
            TypeError: If the channel exists with a different type.
        """
        channel = await self.get_or_create_channel(
            name, buffer_limit=buffer_limit, message_type=message_type
        )
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
                self._channel_types.pop(name, None)

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
                    log.debug(
                        f"Error closing channel {channel.name}: {type(e).__name__}: {e}"
                    )
            self._channels.clear()
            self._channel_types.clear()

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
