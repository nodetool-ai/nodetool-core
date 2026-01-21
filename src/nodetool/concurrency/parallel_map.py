import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


async def parallel_map(
    items: list[T],
    mapper: Callable[[T], Awaitable[U]],
    max_concurrent: int = 10,
) -> list[U]:
    """
    Apply an async mapper function to items in parallel with controlled concurrency.

    This is useful when you want to process individual items concurrently rather than
    in batches. The results are returned in the same order as the input items.

    Args:
        items: List of items to process.
        mapper: Async function to apply to each item.
        max_concurrent: Maximum number of concurrent operations (default: 10).

    Returns:
        List of results in the same order as input items.

    Raises:
        ValueError: If max_concurrent is not a positive integer.

    Example:
        async def fetch_url(url: str) -> Response:
            return await fetch(url)

        responses = await parallel_map(
            items=["https://a.com", "https://b.com", "https://c.com"],
            mapper=fetch_url,
            max_concurrent=5,
        )
        # Returns [Response(...), Response(...), Response(...)]
    """
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be a positive integer")

    if not items:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)

    async def map_with_semaphore(item: T) -> U:
        async with semaphore:
            return await mapper(item)

    return await asyncio.gather(*(map_with_semaphore(item) for item in items))


__all__ = ["parallel_map"]
