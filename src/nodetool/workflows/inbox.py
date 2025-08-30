"""Node input inbox utilities for streaming inputs.

This module provides the ``NodeInbox`` primitive used by the workflow runner to
deliver input values to nodes that opt into input streaming.

Features:
  - Per-handle FIFO buffers
  - Global arrival queue for cross-handle arrival order
  - Upstream source counting to determine end-of-stream (EOS) per handle
  - Async iteration helpers that block until data arrives or EOS is reached

Notes:
  - Everything is a stream; a scalar input is delivered as a one-item stream.
  - ``iter_any()`` preserves cross-handle arrival order and does not align streams.
  - Backpressure is currently unbounded; policies can be added later.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, AsyncIterator


class NodeInbox:
    """Per-node input inbox with per-handle buffers and EOS tracking.

    Methods are safe to call from multiple producer tasks. Consumers use the
    async iterators :meth:`iter_input` and :meth:`iter_any` to receive values
    as they arrive.
    """

    def __init__(self) -> None:
        # Per-handle FIFO buffers
        self._buffers: dict[str, deque[Any]] = {}
        # Number of still-open upstream producers per handle
        self._open_counts: dict[str, int] = {}
        # Global arrival order as a queue of handle names
        self._arrival: deque[str] = deque()
        # Condition to coordinate producers/consumers
        self._cond: asyncio.Condition = asyncio.Condition()
        self._closed: bool = False

    def add_upstream(self, handle: str, count: int = 1) -> None:
        """Declare upstream producers for a given handle.

        Args:
            handle: Input handle name.
            count: Number of upstream producers to add for this handle.
        """
        if count <= 0:
            return
        self._open_counts[handle] = self._open_counts.get(handle, 0) + count
        # Ensure buffer exists
        self._buffers.setdefault(handle, deque())

    def put(self, handle: str, item: Any) -> None:
        """Enqueue an item for a handle and notify any waiters.

        Args:
            handle: Input handle name.
            item: The item to append to the handle's buffer.
        """
        if self._closed:
            return
        self._buffers.setdefault(handle, deque()).append(item)
        # Record arrival for multiplexed consumption preserving cross-handle order
        self._arrival.append(handle)

        # Notify waiters in the running loop
        async def _notify() -> None:
            async with self._cond:
                self._cond.notify_all()

        asyncio.create_task(_notify())

    def mark_source_done(self, handle: str) -> None:
        """Mark one upstream source for a handle as completed and notify waiters.

        Args:
            handle: Input handle name to decrement open-count for.
        """
        current = self._open_counts.get(handle, 0)
        new_val = current - 1
        if new_val < 0:
            new_val = 0
        self._open_counts[handle] = new_val

        async def _notify() -> None:
            async with self._cond:
                self._cond.notify_all()

        asyncio.create_task(_notify())

    def has_any(self) -> bool:
        """Return True if any handle currently has buffered items.

        Returns:
            True if any per-handle buffer is non-empty.
        """
        return any(len(buf) > 0 for buf in self._buffers.values())

    # Removed per-handle streaming metadata; all inputs are treated uniformly.

    async def iter_input(self, handle: str) -> AsyncIterator[Any]:
        """Yield items for a specific handle until its EOS is reached.

        Args:
            handle: Input handle name to read from.

        Yields:
            Items from the per-handle buffer in FIFO order.

        Terminates when the buffer is empty and the open-count for the handle
        reaches zero, or if the inbox is closed.
        """
        # Ensure structures exist for the handle
        self._buffers.setdefault(handle, deque())
        self._open_counts.setdefault(handle, 0)

        while True:
            # Drain any available items without waiting
            while self._buffers[handle]:
                yield self._buffers[handle].popleft()
            # If no producers remain and buffer is empty -> EOS
            if self._closed or self._open_counts.get(handle, 0) == 0:
                return
            # Otherwise wait for more data or a producer to finish
            async with self._cond:
                await self._cond.wait()

    async def iter_any(self) -> AsyncIterator[tuple[str, Any]]:
        """Yield ``(handle, item)`` across all handles in arrival order.

        Yields:
            Tuples of ``(handle, item)`` in cross-handle arrival order.

        Terminates when all handles with declared upstreams have reached EOS
        and all buffers are drained, or if the inbox is closed.
        """
        while True:
            # Emit quickly if something is available in arrival queue
            if self._arrival:
                handle = self._arrival.popleft()
                buf = self._buffers.get(handle)
                if buf:
                    # It is possible that another consumer popped from the handle-specific
                    # buffer; guard against empty deque.
                    if buf:
                        yield handle, buf.popleft()
                        continue
                # If buffer was empty, just continue the loop to recompute state
                continue

            # Check termination: no arrivals, no buffered items, all sources done
            any_buffered = any(len(buf) > 0 for buf in self._buffers.values())
            any_open = any(v > 0 for v in self._open_counts.values())
            if self._closed or (not any_buffered and not any_open):
                return

            # Wait for new arrivals or EOS updates
            async with self._cond:
                await self._cond.wait()

    async def close_all(self) -> None:
        """Close the inbox and wake any blocked consumers."""
        self._closed = True
        async with self._cond:
            self._cond.notify_all()

    # Helpers for NodeInputs
    def has_buffered(self, handle: str) -> bool:
        """Return True if a handle currently has buffered items.

        Args:
            handle: Input handle name to check.

        Returns:
            True if at least one item is buffered for the handle.
        """
        buf = self._buffers.get(handle)
        return bool(buf and len(buf) > 0)

    def is_open(self, handle: str) -> bool:
        """Return True if the handle has open upstream producers.

        Args:
            handle: Input handle name to check.

        Returns:
            True if the upstream open-count for the handle is greater than zero.
        """
        return self._open_counts.get(handle, 0) > 0

    # Non-blocking pop of any available item in arrival order
    def try_pop_any(self) -> tuple[str, Any] | None:
        """Pop one buffered arrival in cross-handle order without blocking.

        Returns:
            A tuple of ``(handle, item)`` if available, otherwise ``None``.
        """
        if not self._arrival:
            return None
        handle = self._arrival.popleft()
        buf = self._buffers.get(handle)
        if buf and len(buf) > 0:
            return handle, buf.popleft()
        return None
