"""Async generator orchestration utilities.

This module provides the AsyncGeneratorOrchestrator class for orchestrating
multiple async generators with advanced consumption patterns like round-robin,
priority-based yielding, and selective consumption.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, AsyncIterable, Callable
from typing import TypeVar

T = TypeVar("T")


class AsyncGeneratorOrchestrator:
    """
    Orchestrate multiple async generators with advanced consumption patterns.

    Provides patterns like round-robin, priority-based yielding, and selective
    consumption for scenarios where you need to consume from multiple generators
    in specific ways.

    This is particularly useful for:
    - Aggregating data from multiple APIs in a controlled manner
    - Processing logs from multiple sources with priority
    - Implementing fair scheduling among multiple data streams
    - Multi-source data ingestion with filtering

    Example:
        >>> async def gen1():
        ...     for i in range(5):
        ...         yield f"gen1-{i}"
        >>> async def gen2():
        ...     for i in range(3):
        ...         yield f"gen2-{i}"
        >>> orchestrator = AsyncGeneratorOrchestrator(gen1(), gen2())
        >>> # Round-robin consumption
        >>> async for item in orchestrator.round_robin():
        ...     print(item)
        gen1-0
        gen2-0
        gen1-1
        gen2-1
        gen1-2
        gen2-2
        gen1-3
        gen1-4
    """

    def __init__(self, *generators: AsyncIterable[T]):
        """
        Initialize the orchestrator with async generators.

        Args:
            *generators: Variable number of async generators to orchestrate.

        Raises:
            ValueError: If no generators are provided.
        """
        if not generators:
            raise ValueError("At least one generator must be provided")

        self._generators = generators

    async def round_robin(self) -> AsyncIterator[T]:
        """
        Yield from each generator in round-robin fashion.

        Cycles through generators, yielding one available item from each
        in turn. When a generator is exhausted, it's removed from the cycle.
        Continues until all generators are exhausted.

        Yields:
            Items from generators in round-robin order.

        Example:
            >>> async def numbers():
            ...     for i in range(3):
            ...         yield i
            >>> async def letters():
            ...     for c in ['a', 'b']:
            ...         yield c
            >>> orchestrator = AsyncGeneratorOrchestrator(numbers(), letters())
            >>> result = [item async for item in orchestrator.round_robin()]
            >>> result
            [0, 'a', 1, 'b', 2]
        """
        # Create iterators from generators
        iterators: list[AsyncIterator[T]] = [aiter(g) for g in self._generators]  # type: ignore[arg-type]
        active = list(range(len(iterators)))

        while active:
            # Iterate over active indices in order
            for idx in list(active):  # Use list() to allow modification during iteration
                try:
                    # Use asyncio.wait_for to avoid blocking indefinitely
                    item = await asyncio.wait_for(iterators[idx].__anext__(), timeout=0.1)
                    yield item
                except TimeoutError:
                    # No item available within timeout, continue to next generator
                    continue
                except StopAsyncIteration:
                    # Generator exhausted, remove from active list
                    active.remove(idx)

    async def priority_round_robin(
        self, priorities: list[int]
    ) -> AsyncIterator[T]:
        """
        Yield based on priority - higher priority generators get more items.

        Uses weighted round-robin where priority determines how many items
        to yield from each generator per cycle. A priority of 2 means that
        generator yields 2 items for every 1 item from a priority-1 generator.

        Args:
            priorities: List of priorities, one per generator.
                       Must be same length as generators. Higher values = higher priority.

        Yields:
            Items from generators according to priority weights.

        Raises:
            ValueError: If priorities length doesn't match generators length
                       or if any priority is less than 1.

        Example:
            >>> async def high():
            ...     for i in range(10):
            ...         yield f"high-{i}"
            >>> async def low():
            ...     for i in range(5):
            ...         yield f"low-{i}"
            >>> orchestrator = AsyncGeneratorOrchestrator(high(), low())
            >>> # Priority 3 for high, 1 for low (3:1 ratio)
            >>> async for item in orchestrator.priority_round_robin([3, 1]):
            ...     print(item)
            high-0
            high-1
            high-2
            low-0
            high-3
            ...
        """
        if len(priorities) != len(self._generators):
            raise ValueError(
                f"Priorities length ({len(priorities)}) must match "
                f"generators count ({len(self._generators)})"
            )

        if any(p < 1 for p in priorities):
            raise ValueError("All priorities must be at least 1")

        # Create iterators from generators
        iterators: list[AsyncIterator[T]] = [aiter(g) for g in self._generators]  # type: ignore[arg-type]
        active = list(range(len(iterators)))

        while active:
            for idx in list(active):
                try:
                    # Yield up to priority items from this generator
                    for _ in range(priorities[idx]):
                        try:
                            item = await asyncio.wait_for(
                                iterators[idx].__anext__(), timeout=0.1
                            )
                            yield item
                        except TimeoutError:
                            # No more items available from this generator in this cycle
                            break
                        except StopAsyncIteration:
                            # Generator exhausted, remove from active list
                            active.remove(idx)
                            break
                except StopAsyncIteration:
                    # Already handled above, but catch for safety
                    if idx in active:
                        active.remove(idx)

    async def selective_consume(
        self,
        condition: Callable[[T, int], bool] | None = None,
        include_generator_index: bool = False,
    ) -> AsyncIterator[T | tuple[T, int]]:
        """
        Consume items based on a condition function.

        Only yields items that meet the condition. Can optionally include
        the generator index with each item.

        Args:
            condition: Function taking (item, generator_index) returning bool.
                      If None, all items are yielded (similar to async_merge).
            include_generator_index: If True, yield (item, generator_index) tuples.
                                    If False, yield only items.

        Yields:
            Items that meet the condition, or (item, generator_index) tuples
            if include_generator_index is True.

        Example:
            >>> async def errors():
            ...     yield "error1"
            ...     yield "warning"
            ...     yield "error2"
            >>> async def info():
            ...     yield "info1"
            ...     yield "info2"
            >>> orchestrator = AsyncGeneratorOrchestrator(errors(), info())
            >>> # Only yield items containing "error"
            >>> async for item in orchestrator.selective_consume(
            ...     condition=lambda x, i: "error" in x
            ... ):
            ...     print(item)
            error1
            error2
        """
        if condition is None:

            def _default_condition(item: T, idx: int) -> bool:
                """Default condition that accepts all items."""
                return True

            condition = _default_condition

        # Create iterators from generators
        iterators: list[AsyncIterator[T]] = [aiter(g) for g in self._generators]  # type: ignore[arg-type]

        for gen_idx, iterator in enumerate(iterators):
            try:
                async for item in iterator:
                    if condition(item, gen_idx):
                        if include_generator_index:
                            yield (item, gen_idx)
                        else:
                            yield item
            except StopAsyncIteration:
                # Generator exhausted naturally
                continue

    async def fair_merge(self, timeout: float = 0.1) -> AsyncIterator[T]:
        """
        Merge generators fairly, ensuring no single generator dominates.

        Similar to round_robin but with a configurable timeout to prevent
        slow generators from blocking faster ones indefinitely.

        Args:
            timeout: Maximum time to wait for each generator per cycle.

        Yields:
            Items from all generators in a fair, interleaved manner.

        Example:
            >>> async def fast():
            ...     for i in range(100):
            ...         yield i
            >>> async def slow():
            ...     for i in range(5):
            ...         await asyncio.sleep(0.01)
            ...         yield f"slow-{i}"
            >>> orchestrator = AsyncGeneratorOrchestrator(fast(), slow())
            >>> # Fair merge won't let fast generator dominate
            >>> async for item in orchestrator.fair_merge():
            ...     print(item)
        """
        # Create iterators from generators
        iterators: list[AsyncIterator[T]] = [aiter(g) for g in self._generators]  # type: ignore[arg-type]
        active = list(range(len(iterators)))

        while active:
            for idx in list(active):
                try:
                    item = await asyncio.wait_for(iterators[idx].__anext__(), timeout=timeout)
                    yield item
                except TimeoutError:
                    # This generator is slow, move to next
                    continue
                except StopAsyncIteration:
                    # Generator exhausted
                    active.remove(idx)

    async def race(self) -> AsyncIterator[T]:
        """
        Yield items as they become available, without fair scheduling.

        This is a "race" mode where the first available item from any generator
        is yielded immediately. Useful when you want the fastest possible throughput
        without fairness guarantees.

        This implementation uses a queue-based approach to efficiently collect
        items from all generators as they become available.

        Yields:
            Items from whichever generator produces them first.

        Example:
            >>> async def source1():
            ...     await asyncio.sleep(0.1)
            ...     yield "delayed"
            >>> async def source2():
            ...     yield "immediate"
            >>> orchestrator = AsyncGeneratorOrchestrator(source1(), source2())
            >>> async for item in orchestrator.race():
            ...     print(item)
            immediate
            delayed
        """
        # Create iterators from generators
        iterators: list[AsyncIterator[T]] = [aiter(g) for g in self._generators]  # type: ignore[arg-type]
        queue: asyncio.Queue[T | None] = asyncio.Queue()
        active_count = len(iterators)
        tasks: list[asyncio.Task[None]] = []

        async def producer(iterator_idx: int) -> None:
            """Producer task that feeds items from one generator into the queue."""
            nonlocal active_count
            try:
                async for item in iterators[iterator_idx]:
                    await queue.put(item)
            except StopAsyncIteration:
                pass
            except Exception:
                pass
            finally:
                active_count -= 1

        # Start producer tasks for each generator
        for i in range(len(iterators)):
            task = asyncio.create_task(producer(i))
            tasks.append(task)

        # Consume items from the queue until all producers are done
        try:
            while active_count > 0 or not queue.empty():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.1)
                    if item is not None:
                        yield item
                except TimeoutError:
                    # Check if all producers are done
                    if active_count == 0:
                        break
        finally:
            # Cancel any remaining producer tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for tasks to finish cleanup
            await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["AsyncGeneratorOrchestrator"]
