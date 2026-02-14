"""
Async Channel for producer-consumer patterns.

Provides a typed wrapper around asyncio.Queue with additional convenience methods
for send/receive operations and graceful closing.
"""
import asyncio
from collections.abc import AsyncIterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ChannelClosedError(Exception):
    """Raised when trying to send to a closed channel."""

    pass


class AsyncChannel(Generic[T]):
    """
    An asynchronous channel for producer-consumer communication.

    Provides a cleaner API than raw asyncio.Queue with methods like send(),
    receive(), and close(). Supports iteration over received values and
    graceful shutdown.

    This is ideal for workflows where one part of the code produces values
    that another part consumes, with built-in backpressure and cancellation
    support.

    Example:
        channel = AsyncChannel[str](max_size=10)

        # Producer
        async def producer():
            for i in range(5):
                await channel.send(f"message_{i}")
            await channel.close()

        # Consumer
        async def consumer():
            async for message in channel:
                print(f"Received: {message}")

        # Run both concurrently
        await asyncio.gather(producer(), consumer())

    The channel can be used with timeouts:
        try:
            message = await channel.receive(timeout=1.0)
        except TimeoutError:
            print("No message received within timeout")
    """

    def __init__(self, max_size: int = 0) -> None:
        """
        Initialize the channel.

        Args:
            max_size: Maximum number of items in the channel.
                     0 means unlimited (default).
        """
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self._closed: bool = False

    async def send(self, item: T) -> None:
        """
        Send an item to the channel.

        Blocks if the channel is full until space is available.
        Raises ChannelClosedError if the channel is closed.

        Args:
            item: The item to send.

        Raises:
            ChannelClosedError: If the channel is closed.
        """
        if self._closed:
            raise ChannelClosedError("Cannot send to closed channel")
        await self._queue.put(item)

    def send_nowait(self, item: T) -> None:
        """
        Send an item to the channel without blocking.

        Raises QueueFull if the channel is full and ChannelClosedError
        if the channel is closed.

        Args:
            item: The item to send.

        Raises:
            ChannelClosedError: If the channel is closed.
            asyncio.QueueFull: If the channel is full.
        """
        if self._closed:
            raise ChannelClosedError("Cannot send to closed channel")
        self._queue.put_nowait(item)

    async def receive(self) -> T:
        """
        Receive an item from the channel.

        Blocks if the channel is empty until an item is available.
        Returns None if the channel is closed and empty.

        Returns:
            The received item, or None if channel is closed and empty.

        Raises:
            ChannelClosedError: If the channel is closed and empty.
        """
        try:
            return await self._queue.get()
        except RuntimeError:
            # Queue was closed/destroyed
            if self._closed and self._queue.empty():
                raise ChannelClosedError("Channel is closed and empty") from None
            raise

    def receive_nowait(self) -> T:
        """
        Receive an item from the channel without blocking.

        Raises QueueEmpty if the channel is empty.

        Returns:
            The received item.

        Raises:
            asyncio.QueueEmpty: If the channel is empty.
        """
        return self._queue.get_nowait()

    async def receive_or_wait(
        self, timeout: float | None = None
    ) -> T | None:
        """
        Receive an item from the channel with optional timeout.

        Args:
            timeout: Maximum time to wait in seconds.
                     None means wait indefinitely.

        Returns:
            The received item, or None if timeout is reached.

        Raises:
            ChannelClosedError: If the channel is closed and empty.
        """
        if timeout is None:
            return await self.receive()

        try:
            return await asyncio.wait_for(self.receive(), timeout=timeout)
        except TimeoutError:
            return None

    def close(self) -> None:
        """
        Close the channel.

        No more items can be sent after closing, but pending items
        can still be received. The channel can be iterated over to
        drain remaining items.
        """
        self._closed = True

    @property
    def closed(self) -> bool:
        """Check if the channel is closed."""
        return self._closed

    @property
    def empty(self) -> bool:
        """Check if the channel is empty."""
        return self._queue.empty()

    @property
    def full(self) -> bool:
        """Check if the channel is full."""
        return self._queue.full()

    @property
    def qsize(self) -> int:
        """Get the approximate number of items in the channel."""
        return self._queue.qsize()

    def __aiter__(self) -> "AsyncChannelIterator[T]":
        """
        Create an async iterator for receiving items from the channel.

        Example:
            async for item in channel:
                print(item)

        The iterator stops when the channel is closed and empty.
        """
        return AsyncChannelIterator(self)

    async def __aenter__(self) -> "AsyncChannel[T]":
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes the channel."""
        self.close()


class AsyncChannelIterator(Generic[T]):
    """
    Async iterator for AsyncChannel.

    Automatically handles channel closing and yields items
    until the channel is closed and empty.
    """

    def __init__(self, channel: AsyncChannel[T]) -> None:
        self._channel = channel

    def __aiter__(self) -> "AsyncChannelIterator[T]":
        return self

    async def __anext__(self) -> T:
        """Get the next item from the channel."""
        while True:
            if self._channel.closed and self._channel.empty:
                raise StopAsyncIteration

            try:
                # Use a small timeout to allow checking closed status
                item = await asyncio.wait_for(self._channel.receive(), timeout=0.1)
                return item
            except TimeoutError:
                # Check if we should stop
                if self._channel.closed and self._channel.empty:
                    raise StopAsyncIteration from None
                # Continue waiting via the loop


async def create_channel(
    max_size: int = 0,
) -> AsyncChannel[Any]:
    """
    Convenience function to create a new channel.

    Args:
        max_size: Maximum number of items in the channel (0 = unlimited).

    Returns:
        A new AsyncChannel instance.

    Example:
        channel = await create_channel(max_size=100)
    """
    return AsyncChannel(max_size=max_size)


async def fan_in(
    *channels: AsyncChannel[T], output_max_size: int = 0
) -> AsyncChannel[T]:
    """
    Fan-in multiple channels into a single output channel.

    All items from input channels are forwarded to the output channel.
    The output channel closes when all input channels are closed.

    Args:
        *channels: Input channels to fan in.
        output_max_size: Max size of output channel (0 = unlimited).

    Returns:
        A new channel containing items from all input channels.

    Example:
        ch1 = AsyncChannel[int]()
        ch2 = AsyncChannel[int]()

        merged = await fan_in(ch1, ch2)

        # Items from both ch1 and ch2 appear in merged
    """
    output = AsyncChannel[T](max_size=output_max_size)

    async def forwarder(input_channel: AsyncChannel[T]) -> None:
        async for item in input_channel:
            await output.send(item)

    tasks = [asyncio.create_task(forwarder(ch)) for ch in channels]

    async def wait_for_all() -> None:
        await asyncio.gather(*tasks)
        output.close()

    # Task runs in background to close output when all inputs are done
    asyncio.create_task(wait_for_all())  # noqa: RUF006
    return output


async def fan_out(
    channel: AsyncChannel[T],
    *output_channels: AsyncChannel[T],
) -> None:
    """
    Fan-out items from one channel to multiple output channels.

    Each item from the input channel is sent to all output channels.
    Stops when the input channel is closed and empty.
    Closes all output channels when done.

    Args:
        channel: Input channel.
        *output_channels: Output channels to send items to.

    Example:
        source = AsyncChannel[str]()
        out1 = AsyncChannel[str]()
        out2 = AsyncChannel[str]()

        # Run fan_out in background
        asyncio.create_task(fan_out(source, out1, out2))

        await source.send("hello")
        # Both out1 and out2 receive "hello"
    """
    try:
        async for item in channel:
            # Send to all output channels concurrently to avoid deadlock
            # when one channel is full or slow
            await asyncio.gather(*[out_ch.send(item) for out_ch in output_channels])
    finally:
        # Close all output channels when input is exhausted
        for out_ch in output_channels:
            out_ch.close()


__all__ = [
    "AsyncChannel",
    "AsyncChannelIterator",
    "ChannelClosedError",
    "create_channel",
    "fan_in",
    "fan_out",
]
