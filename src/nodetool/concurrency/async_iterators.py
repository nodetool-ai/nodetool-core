from collections.abc import AsyncIterator
from typing import AsyncIterable, TypeVar

T = TypeVar("T")


class AsyncByteStream:
    """
    An asynchronous iterator that iterates over a byte sequence in chunks.

    Args:
        data (bytes): The byte sequence to iterate over.
        chunk_size (int, optional): The size of each chunk. Defaults to 1024.

    Attributes:
        data (bytes): The byte sequence to iterate over.
        chunk_size (int): The size of each chunk.
        index (int): The current index in the byte sequence.

    Yields:
        bytes: The next chunk of bytes from the byte sequence.

    """

    def __init__(self, data: bytes, chunk_size: int = 1024):
        self.data = data
        self.chunk_size = chunk_size
        self.index = 0

    def __aiter__(self) -> "AsyncByteStream":
        return self

    async def __anext__(self) -> bytes:
        if self.index >= len(self.data):
            raise StopAsyncIteration
        chunk = self.data[self.index : self.index + self.chunk_size]
        self.index += self.chunk_size
        return chunk


async def async_take(aiterable: AsyncIterable[T], n: int) -> list[T]:
    """
    Take the first n items from an async iterable.

    Args:
        aiterable: The async iterable to take items from.
        n: The number of items to take.

    Returns:
        A list containing the first n items from the iterable.

    Example:
        >>> async def gen():
        ...     for i in range(10):
        ...         yield i
        >>> result = await async_take(gen(), 3)
        >>> result
        [0, 1, 2]
    """
    result: list[T] = []
    async for item in aiterable:
        if len(result) >= n:
            break
        result.append(item)
    return result


async def async_slice(aiterable: AsyncIterable[T], start: int, stop: int | None = None) -> list[T]:
    """
    Slice an async iterable, similar to list slicing.

    Args:
        aiterable: The async iterable to slice.
        start: The starting index (inclusive).
        stop: The stopping index (exclusive). If None, takes all items from start.

    Returns:
        A list containing items from start to stop.

    Example:
        >>> async def gen():
        ...     for i in range(10):
        ...         yield i
        >>> result = await async_slice(gen(), 2, 5)
        >>> result
        [2, 3, 4]
    """
    result: list[T] = []
    idx = 0
    async for item in aiterable:
        if idx >= start:
            if stop is not None and idx >= stop:
                break
            result.append(item)
        idx += 1
    return result


async def async_first(aiterable: AsyncIterable[T], default: T | None = None) -> T | None:
    """
    Get the first item from an async iterable.

    Args:
        aiterable: The async iterable to get the first item from.
        default: The default value to return if the iterable is empty.

    Returns:
        The first item, or default if the iterable is empty.

    Example:
        >>> async def gen():
        ...     yield 1
        ...     yield 2
        >>> result = await async_first(gen())
        >>> result
        1
    """
    async for item in aiterable:
        return item
    return default


async def async_list(aiterable: AsyncIterable[T]) -> list[T]:
    """
    Consume an async iterable into a list.

    Args:
        aiterable: The async iterable to consume.

    Returns:
        A list containing all items from the iterable.

    Example:
        >>> async def gen():
        ...     for i in range(3):
        ...         yield i
        >>> result = await async_list(gen())
        >>> result
        [0, 1, 2]
    """
    result: list[T] = []
    async for item in aiterable:
        result.append(item)
    return result


async def async_merge(*iterables: AsyncIterable[T]) -> AsyncIterator[T]:
    """
    Merge multiple async iterables into a single async iterator.

    Yields items from each iterable in sequence, moving to the next
    iterable when the current one is exhausted.

    Args:
        *iterables: Variable number of async iterables to merge.

    Yields:
        Items from each iterable in sequence.

    Example:
        >>> async def gen1():
        ...     yield 1
        ...     yield 2
        >>> async def gen2():
        ...     yield 3
        ...     yield 4
        >>> result = await async_list(async_merge(gen1(), gen2()))
        >>> result
        [1, 2, 3, 4]
    """
    for aiterable in iterables:
        async for item in aiterable:
            yield item
