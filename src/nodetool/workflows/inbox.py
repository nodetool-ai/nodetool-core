"""Node input inbox utilities for streaming inputs.

This module provides the ``NodeInbox`` primitive used by the workflow runner to
deliver input values to nodes that opt into input streaming.

Context for upcoming streaming refactor
=======================================

Only nodes that return ``True`` from ``is_streaming_input()`` are expected to
drain the inbox manually via ``iter_input`` / ``iter_any``. Nodes that leave the
flag at ``False`` will continue to rely on the actor for batching (according to
``sync_mode``) even if they produce streaming outputs. The inbox API itself does
not change - the documentation here simply records the planned division of
responsibilities so future edits remain consistent.

Features:
  - Per-handle FIFO buffers
  - Global arrival queue for cross-handle arrival order
  - Upstream source counting to determine end-of-stream (EOS) per handle
  - Async iteration helpers that block until data arrives or EOS is reached
  - Configurable backpressure to prevent fast producers from overwhelming consumers

Notes:
  - Everything is a stream; a scalar input is delivered as a one-item stream.
  - ``iter_any()`` preserves cross-handle arrival order and does not align streams.
  - Backpressure is implemented via a configurable buffer limit per handle.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from typing import Any, AsyncIterator

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class NodeInbox:
    """Per-node input inbox with per-handle buffers and EOS tracking.

    Methods are safe to call from multiple producer tasks. Consumers use the
    async iterators :meth:`iter_input` and :meth:`iter_any` to receive values
    as they arrive.

    Args:
        buffer_limit: Maximum number of items allowed in each per-handle buffer.
                     When a buffer reaches this limit, producers will block until
                     space becomes available. None means unlimited (default).
    """

    def __init__(self, buffer_limit: int | None = None) -> None:
        # Per-handle FIFO buffers
        self._buffers: dict[str, deque[Any]] = {}
        # Number of still-open upstream producers per handle
        self._open_counts: dict[str, int] = {}
        # Global arrival order as a queue of handle names
        self._arrival: deque[str] = deque()
        # Condition to coordinate producers/consumers
        self._cond: asyncio.Condition = asyncio.Condition()
        # Event loop where this inbox/condition were created (runner loop)
        self._loop = asyncio.get_running_loop()
        self._closed: bool = False
        # Buffer limit for backpressure (None = unlimited)
        self._buffer_limit: int | None = buffer_limit
        assert self._loop is not None, "Event loop is not running"

    def _notify_waiters_threadsafe(self) -> None:
        """Schedule notification of waiters in a thread-safe manner.

        This method is safe to call from any thread. It schedules the
        notification to run on the inbox's event loop, ensuring that
        the coroutine and task are created in the correct loop context.
        This avoids "Future attached to a different loop" errors on Windows.
        """

        def _schedule_notify() -> None:
            """Create and schedule the notification task on the target loop.

            This runs in the target loop's thread via call_soon_threadsafe.
            """

            async def _notify() -> None:
                async with self._cond:
                    self._cond.notify_all()

            self._loop.create_task(_notify())

        self._loop.call_soon_threadsafe(_schedule_notify)

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

    async def put(self, handle: str, item: Any) -> None:
        """Enqueue an item for a handle and notify any waiters.

        If buffer_limit is set and the handle's buffer is full, this method
        will block until space becomes available.

        Args:
            handle: Input handle name.
            item: The item to append to the handle's buffer.
        """
        if self._closed:
            return

        # Wait if buffer is full (backpressure)
        if self._buffer_limit is not None:
            async with self._cond:
                while not self._closed and len(self._buffers.get(handle, [])) >= self._buffer_limit:
                    # Block producer until consumer drains the buffer
                    await self._cond.wait()

        if self._closed:
            return

        self._buffers.setdefault(handle, deque()).append(item)
        # Record arrival for multiplexed consumption preserving cross-handle order
        self._arrival.append(handle)
        # log.debug(
        #     f"Inbox[{id(self)}] put: handle={handle} size={len(self._buffers.get(handle, []))}"
        # )

        # Notify waiters
        async with self._cond:
            self._cond.notify_all()

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
        # log.debug(
        #     f"Inbox[{id(self)}] eos: handle={handle} open={self._open_counts.get(handle,0)}"
        # )

        self._notify_waiters_threadsafe()

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
                item = self._buffers[handle].popleft()
                # Notify producers that space is available (backpressure release)
                async with self._cond:
                    self._cond.notify_all()
                yield item
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
                    item = buf.popleft()
                    # Notify producers that space is available (backpressure release)
                    async with self._cond:
                        self._cond.notify_all()
                    yield handle, item
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

    def is_fully_drained(self) -> bool:
        """Return True if inbox is fully drained: no buffered items and all sources done.

        This is the definitive check for whether a node's inbox has completed
        all its work - both buffered messages have been consumed AND all upstream
        sources have signaled end-of-stream.

        Returns:
            True if the inbox is completely drained and finished.
        """
        if self._closed:
            return True
        # Check if any buffers have items
        any_buffered = any(len(buf) > 0 for buf in self._buffers.values())
        # Check if any sources are still open
        any_open = any(v > 0 for v in self._open_counts.values())
        return not any_buffered and not any_open

    def has_pending_work(self) -> bool:
        """Return True if inbox has pending work: buffered items or open sources.

        This is the inverse of is_fully_drained(), useful for checking if there's
        still work to be done.

        Returns:
            True if there are buffered items or open upstream sources.
        """
        return not self.is_fully_drained()

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
            item = buf.popleft()
            # Notify producers that space is available (backpressure release)
            # Note: This is synchronous, so we schedule notification without blocking
            if self._loop is not None:
                with suppress(Exception):
                    self._notify_waiters_threadsafe()
            return handle, item
        return None
