"""Variable channel utilities for streaming variables in ProcessingContext.

This module provides the ``VariableChannel`` primitive used to create channel-based
streaming for workflow variables. This follows the same architectural pattern as
``NodeInbox`` but is designed for variables that can be read/written by any node
in the workflow.

Features:
  - Per-variable FIFO buffers for streaming values
  - Async iteration helpers that block until data arrives or EOS is reached
  - Support for both one-shot values and streaming updates
  - Thread-safe operations for cross-task communication

Key Differences from NodeInbox:
  - Variables are global to the workflow, not scoped to a specific node
  - Any node can write to any variable
  - Variables support both streaming (multiple values) and scalar (single value) modes
  - Variables persist the latest value for late subscribers

Notes:
  - A scalar variable is a channel with capacity 1 that overwrites on each write
  - A streaming variable accumulates values until drained
  - ``iter_variable()`` yields all values as they arrive
  - ``get_variable()`` returns the latest value or waits for one to arrive
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, AsyncIterator

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class VariableChannel:
    """Per-variable channel with FIFO buffer and streaming support.

    Methods are safe to call from multiple producer tasks. Consumers use the
    async iterators :meth:`iter_values` to receive values as they arrive, or
    :meth:`get_latest` to get the most recent value.

    Args:
        name: The name of the variable this channel represents.
        buffer_limit: Maximum number of items allowed in the buffer.
                     When the buffer reaches this limit, oldest values are dropped
                     (for streaming mode) or the value is overwritten (for scalar mode).
                     None means unlimited (streaming accumulation).
        scalar_mode: If True, the channel holds only the latest value (capacity 1).
                    If False, the channel accumulates values in a FIFO queue.
    """

    def __init__(
        self,
        name: str,
        buffer_limit: int | None = None,
        scalar_mode: bool = True,
    ) -> None:
        self._name = name
        self._buffer: deque[Any] = deque()
        self._buffer_limit = buffer_limit
        self._scalar_mode = scalar_mode
        self._latest_value: Any = None
        self._has_value: bool = False
        self._closed: bool = False
        # Condition to coordinate producers/consumers
        self._cond: asyncio.Condition = asyncio.Condition()
        # Track if there are active producers (for EOS detection)
        self._open_producers: int = 0
        # Event loop where this channel was created
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    @property
    def name(self) -> str:
        """Return the variable name for this channel."""
        return self._name

    @property
    def has_value(self) -> bool:
        """Return True if the channel has received at least one value."""
        return self._has_value

    @property
    def latest_value(self) -> Any:
        """Return the latest value written to this channel, or None if no value."""
        return self._latest_value

    def add_producer(self, count: int = 1) -> None:
        """Register producers that will write to this channel.

        Args:
            count: Number of producers to register.
        """
        if count > 0:
            self._open_producers += count

    def _notify_waiters_threadsafe(self) -> None:
        """Schedule notification of waiters in a thread-safe manner."""
        if self._loop is None:
            return

        def _schedule_notify() -> None:
            async def _notify() -> None:
                async with self._cond:
                    self._cond.notify_all()

            self._loop.create_task(_notify())

        self._loop.call_soon_threadsafe(_schedule_notify)

    async def put(self, value: Any) -> None:
        """Write a value to the channel.

        In scalar mode, this replaces the current value.
        In streaming mode, this appends to the buffer.

        Args:
            value: The value to write.
        """
        if self._closed:
            return

        self._latest_value = value
        self._has_value = True

        if self._scalar_mode:
            # Scalar mode: replace the buffer with just this value
            self._buffer.clear()
            self._buffer.append(value)
        else:
            # Streaming mode: append to buffer
            self._buffer.append(value)
            # Apply buffer limit if set (drop oldest)
            if self._buffer_limit is not None:
                while len(self._buffer) > self._buffer_limit:
                    self._buffer.popleft()

        # Notify waiters
        async with self._cond:
            self._cond.notify_all()

    def put_sync(self, value: Any) -> None:
        """Synchronously write a value to the channel.

        This is a non-blocking version for use when no event loop is running
        or when called from synchronous code.

        Args:
            value: The value to write.
        """
        if self._closed:
            return

        self._latest_value = value
        self._has_value = True

        if self._scalar_mode:
            self._buffer.clear()
            self._buffer.append(value)
        else:
            self._buffer.append(value)
            if self._buffer_limit is not None:
                while len(self._buffer) > self._buffer_limit:
                    self._buffer.popleft()

        # Try to notify waiters if we have a loop
        if self._loop is not None:
            self._notify_waiters_threadsafe()

    def mark_producer_done(self) -> None:
        """Mark one producer as finished.

        When all producers are done and the buffer is empty, iteration stops.
        """
        if self._open_producers > 0:
            self._open_producers -= 1
        self._notify_waiters_threadsafe()

    async def get(self, default: Any = None, timeout: float | None = None) -> Any:
        """Get the latest value, waiting if necessary.

        If no value has been written yet, this method blocks until a value
        arrives or the timeout expires.

        Args:
            default: Value to return if no value is available and timeout expires.
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Returns:
            The latest value, or default if timeout expires without a value.
        """
        if self._has_value:
            return self._latest_value

        try:
            async with asyncio.timeout(timeout) if timeout else asyncio.nullcontext():
                async with self._cond:
                    while not self._has_value and not self._closed:
                        await self._cond.wait()
                    return self._latest_value if self._has_value else default
        except TimeoutError:
            return default

    def get_nowait(self, default: Any = None) -> Any:
        """Get the latest value without waiting.

        Args:
            default: Value to return if no value is available.

        Returns:
            The latest value, or default if no value has been written.
        """
        return self._latest_value if self._has_value else default

    async def iter_values(self) -> AsyncIterator[Any]:
        """Iterate all values in the channel until closed or all producers done.

        In scalar mode, this yields each time the value changes.
        In streaming mode, this yields all accumulated values in FIFO order.

        Yields:
            Values from the channel as they arrive.
        """
        while True:
            # Drain any available values
            while self._buffer:
                yield self._buffer.popleft()
                # Notify any blocked producers
                async with self._cond:
                    self._cond.notify_all()

            # Check termination conditions
            if self._closed or (self._open_producers == 0 and not self._buffer):
                return

            # Wait for more data or producer completion
            async with self._cond:
                await self._cond.wait()

    def has_buffered(self) -> bool:
        """Return True if the channel has buffered values waiting.

        Returns:
            True if there are values in the buffer.
        """
        return len(self._buffer) > 0

    def is_open(self) -> bool:
        """Return True if there are still active producers.

        Returns:
            True if at least one producer is still active.
        """
        return self._open_producers > 0 and not self._closed

    async def close(self) -> None:
        """Close the channel and wake any blocked consumers."""
        self._closed = True
        async with self._cond:
            self._cond.notify_all()


class VariableChannelManager:
    """Manages multiple variable channels for a workflow.

    Provides a unified interface for creating, accessing, and managing
    variable channels within a ProcessingContext.

    Args:
        default_scalar_mode: Whether new channels default to scalar mode.
    """

    def __init__(self, default_scalar_mode: bool = True) -> None:
        self._channels: dict[str, VariableChannel] = {}
        self._default_scalar_mode = default_scalar_mode

    def get_channel(
        self,
        name: str,
        create: bool = True,
        scalar_mode: bool | None = None,
    ) -> VariableChannel | None:
        """Get or create a variable channel.

        Args:
            name: The variable name.
            create: If True, create the channel if it doesn't exist.
            scalar_mode: Override scalar mode for new channels. None uses default.

        Returns:
            The variable channel, or None if it doesn't exist and create is False.
        """
        if name not in self._channels:
            if not create:
                return None
            mode = scalar_mode if scalar_mode is not None else self._default_scalar_mode
            self._channels[name] = VariableChannel(name, scalar_mode=mode)
        return self._channels[name]

    def has_channel(self, name: str) -> bool:
        """Check if a channel exists.

        Args:
            name: The variable name.

        Returns:
            True if the channel exists.
        """
        return name in self._channels

    async def set_variable(self, name: str, value: Any, scalar_mode: bool = True) -> None:
        """Set a variable value, creating the channel if necessary.

        Args:
            name: The variable name.
            value: The value to set.
            scalar_mode: Whether this is a scalar (single value) variable.
        """
        channel = self.get_channel(name, create=True, scalar_mode=scalar_mode)
        if channel:
            await channel.put(value)

    def set_variable_sync(self, name: str, value: Any, scalar_mode: bool = True) -> None:
        """Synchronously set a variable value.

        Args:
            name: The variable name.
            value: The value to set.
            scalar_mode: Whether this is a scalar variable.
        """
        channel = self.get_channel(name, create=True, scalar_mode=scalar_mode)
        if channel:
            channel.put_sync(value)

    async def get_variable(
        self,
        name: str,
        default: Any = None,
        timeout: float | None = None,
    ) -> Any:
        """Get a variable value, waiting if necessary.

        Args:
            name: The variable name.
            default: Value to return if variable doesn't exist or timeout expires.
            timeout: Maximum time to wait for a value.

        Returns:
            The variable value, or default.
        """
        channel = self.get_channel(name, create=False)
        if channel is None:
            return default
        return await channel.get(default=default, timeout=timeout)

    def get_variable_nowait(self, name: str, default: Any = None) -> Any:
        """Get a variable value without waiting.

        Args:
            name: The variable name.
            default: Value to return if variable doesn't exist.

        Returns:
            The variable value, or default.
        """
        channel = self.get_channel(name, create=False)
        if channel is None:
            return default
        return channel.get_nowait(default=default)

    async def iter_variable(self, name: str) -> AsyncIterator[Any]:
        """Iterate values for a variable until the channel closes.

        Args:
            name: The variable name.

        Yields:
            Values from the variable channel.
        """
        channel = self.get_channel(name, create=True, scalar_mode=False)
        if channel:
            async for value in channel.iter_values():
                yield value

    def list_variables(self) -> list[str]:
        """List all variable names with channels.

        Returns:
            List of variable names.
        """
        return list(self._channels.keys())

    def get_all_values(self) -> dict[str, Any]:
        """Get all current variable values as a dict.

        Returns:
            Dict mapping variable names to their latest values.
        """
        return {
            name: channel.latest_value
            for name, channel in self._channels.items()
            if channel.has_value
        }

    async def close_all(self) -> None:
        """Close all channels."""
        for channel in self._channels.values():
            await channel.close()
        self._channels.clear()
