import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


async def batched_async_iterable(
    items: list[T],
    batch_size: int,
) -> AsyncIterator[list[T]]:
    """
    Yield batches of items from a list asynchronously.

    This is useful for processing large datasets in chunks where you want
    to control memory usage and parallelize batch processing.

    Args:
        items: List of items to batch.
        batch_size: Maximum number of items per batch. Must be > 0.

    Yields:
        Lists of items, each up to batch_size in length.

    Raises:
        ValueError: If batch_size is not a positive integer.

    Example:
        items = [1, 2, 3, 4, 5]
        async for batch in batched_async_iterable(items, 2):
            await process_batch(batch)  # [1, 2], then [3, 4], then [5]
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if not items:
        return

    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


async def process_in_batches(
    items: list[T],
    processor: Callable[[list[T]], Awaitable[U]],
    batch_size: int,
    max_concurrent: int = 1,
) -> list[U]:
    """
    Process items in batches with optional concurrency control.

    This function divides items into batches and processes them using
    the provided processor function. Batches can be processed sequentially
    or with controlled concurrency.

    Args:
        items: List of items to process.
        processor: Async function to call with each batch.
        batch_size: Maximum number of items per batch.
        max_concurrent: Maximum concurrent batch processing (default: 1).

    Returns:
        List of results from processing each batch, in order.

    Raises:
        ValueError: If batch_size or max_concurrent is invalid.

    Example:
        async def process_batch(items):
            # items is a list of URLs to fetch
            results = await fetch_all(items)
            return results

        all_results = await process_in_batches(
            items=urls,
            processor=process_batch,
            batch_size=10,
            max_concurrent=2,
        )
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be a positive integer")

    if not items:
        return []

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    if max_concurrent == 1:
        results = []
        for batch in batches:
            result = await processor(batch)
            results.append(result)
        return results

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(batch: list[T]) -> U:
        async with semaphore:
            return await processor(batch)

    return await asyncio.gather(*(process_with_semaphore(batch) for batch in batches))


__all__ = ["batched_async_iterable", "process_in_batches"]
