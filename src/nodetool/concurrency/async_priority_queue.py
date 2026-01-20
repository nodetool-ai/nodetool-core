import asyncio
import time
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class PriorityItem(Generic[T]):
    priority: int
    item: T
    creation_time: float

    def __lt__(self, other: "PriorityItem[T]") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.creation_time < other.creation_time


class AsyncPriorityQueue(Generic[T]):
    """
    An asynchronous priority queue with configurable max size and timeout support.

    Items are dequeued in priority order (lower values = higher priority).
    When priorities are equal, items are ordered by creation time (FIFO).

    This implementation uses a heap for O(log n) enqueue and dequeue operations,
    making it efficient for workflows that need priority-based task scheduling.

    Example:
        queue = AsyncPriorityQueue(max_size=100)

        # Put items with different priorities (0 = highest priority)
        await queue.put(0, "urgent_task")
        await queue.put(2, "normal_task")
        await queue.put(1, "important_task")

        # Items are returned in priority order
        item = await queue.get()  # Returns "urgent_task"
        item = await queue.get()  # Returns "important_task"

        # With timeout
        try:
            item = await queue.get(timeout=5.0)
        except TimeoutError:
            print("Queue is empty")
    """

    def __init__(self, max_size: int | None = None):
        """
        Initialize the priority queue.

        Args:
            max_size (int | None): Maximum number of items the queue can hold.
                                   If None, the queue has unlimited capacity.

        Raises:
            ValueError: If max_size is not a positive integer.
        """
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be a positive integer or None")

        self._max_size = max_size
        self._heap: list[PriorityItem[T]] = []
        self._get_waiters: list[asyncio.Future[Any]] = []

    @property
    def max_size(self) -> int | None:
        """Return the maximum size of the queue, or None if unlimited."""
        return self._max_size

    @property
    def qsize(self) -> int:
        """Return the current number of items in the queue."""
        return len(self._heap)

    @property
    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self._heap) == 0

    @property
    def full(self) -> bool:
        """Return True if the queue is at maximum capacity."""
        if self._max_size is None:
            return False
        return len(self._heap) >= self._max_size

    async def put(self, priority: int, item: T) -> None:
        """
        Add an item to the queue with the given priority.

        Lower priority values indicate higher priority (e.g., 0 = highest).

        Args:
            priority (int): The priority of the item. Lower values = higher priority.
            item (T): The item to add to the queue.

        Raises:
            asyncio.QueueFull: If the queue is at maximum capacity.
        """
        if self.full:
            raise asyncio.QueueFull()

        priority_item = PriorityItem(
            priority=priority,
            item=item,
            creation_time=time.monotonic(),
        )
        heappush(self._heap, priority_item)

        if self._get_waiters:
            waiter = self._get_waiters.pop(0)
            waiter.set_result(item)

    async def get(self, timeout: float | None = None) -> T:
        """
        Remove and return the highest priority item from the queue.

        Items with the same priority are returned in FIFO order based on
        when they were added.

        Args:
            timeout (float | None): Maximum time to wait in seconds. If None,
                                    wait indefinitely. If <= 0, raise immediately
                                    if queue is empty.

        Returns:
            T: The highest priority item from the queue.

        Raises:
            TimeoutError: If the timeout expired before an item was available.
            asyncio.QueueEmpty: If the queue is empty and no timeout was specified.
        """
        if self.empty:
            if timeout is None:
                raise asyncio.QueueEmpty()

            if timeout <= 0:
                raise asyncio.QueueEmpty()

            loop = asyncio.get_event_loop()
            waiter = loop.create_future()
            self._get_waiters.append(waiter)

            try:
                await asyncio.wait_for(waiter, timeout=timeout)
            except TimeoutError:
                if waiter in self._get_waiters:
                    self._get_waiters.remove(waiter)
                raise

        return heappop(self._heap).item

    def get_nowait(self) -> T:
        """
        Remove and return the highest priority item without waiting.

        Returns:
            T: The highest priority item from the queue.

        Raises:
            asyncio.QueueEmpty: If the queue is empty.
        """
        if self.empty:
            raise asyncio.QueueEmpty()
        return heappop(self._heap).item

    def put_nowait(self, priority: int, item: T) -> None:
        """
        Add an item to the queue without waiting.

        Args:
            priority (int): The priority of the item. Lower values = higher priority.
            item (T): The item to add to the queue.

        Raises:
            asyncio.QueueFull: If the queue is at maximum capacity.
        """
        if self.full:
            raise asyncio.QueueFull()

        priority_item = PriorityItem(
            priority=priority,
            item=item,
            creation_time=time.monotonic(),
        )
        heappush(self._heap, priority_item)

        if self._get_waiters:
            waiter = self._get_waiters.pop(0)
            waiter.set_result(item)

    def peek(self) -> T | None:
        """
        Return the highest priority item without removing it.

        Returns:
            T | None: The highest priority item, or None if queue is empty.
        """
        if self.empty:
            return None
        return self._heap[0].item

    def clear(self) -> list[T]:
        """
        Remove and return all items from the queue.

        Returns:
            list[T]: All items that were in the queue, in priority order.
        """
        items = [item.item for item in self._heap]
        self._heap.clear()
        return items

    def __aiter__(self):
        """Make the queue async iterable."""
        return self

    async def __anext__(self) -> T:
        """Return the next item from the queue."""
        try:
            return await self.get(timeout=1.0)
        except TimeoutError:
            raise StopAsyncIteration from None
