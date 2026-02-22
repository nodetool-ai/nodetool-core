import asyncio
from collections.abc import AsyncIterator, Callable
from typing import AsyncIterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


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


async def async_filter(
    predicate: Callable[[T], bool] | Callable[[T], object], aiterable: AsyncIterable[T]
) -> AsyncIterator[T]:
    """
    Filter items from an async iterable using a predicate function.

    Yields items from the iterable for which the predicate returns True.
    The predicate can be a sync function or an async function.

    Args:
        predicate: A function that takes an item and returns bool.
                   Can be sync or async.
        aiterable: The async iterable to filter.

    Yields:
        Items from the iterable for which predicate(item) is True.

    Example:
        >>> async def gen():
        ...     for i in range(10):
        ...         yield i
        >>> # With sync predicate
        >>> result = await async_list(async_filter(lambda x: x % 2 == 0, gen()))
        >>> result
        [0, 2, 4, 6, 8]
        >>> # With async predicate
        >>> async def is_even(x):
        ...     return x % 2 == 0
        >>> result = await async_list(async_filter(is_even, gen()))
        >>> result
        [0, 2, 4, 6, 8]
    """
    async for item in aiterable:
        result = predicate(item)
        if asyncio.iscoroutine(result):
            result = await result  # type: ignore[misc]
        if result:
            yield item


async def async_map(
    func: Callable[[T], R] | Callable[[T], object], aiterable: AsyncIterable[T]
) -> AsyncIterator[R]:
    """
    Map a function over an async iterable, transforming each item.

    Applies the function to each item in the iterable and yields the results.
    The function can be sync or async.

    Args:
        func: A function that takes an item and returns a transformed value.
              Can be sync or async.
        aiterable: The async iterable to map over.

    Yields:
        Transformed items from applying func to each item in the iterable.

    Example:
        >>> async def gen():
        ...     for i in range(5):
        ...         yield i
        >>> # With sync function
        >>> result = await async_list(async_map(lambda x: x * 2, gen()))
        >>> result
        [0, 2, 4, 6, 8]
        >>> # With async function
        >>> async def double(x):
        ...     return x * 2
        >>> result = await async_list(async_map(double, gen()))
        >>> result
        [0, 2, 4, 6, 8]
    """
    async for item in aiterable:
        result = func(item)
        if asyncio.iscoroutine(result):
            result = await result  # type: ignore[misc]
        yield result  # type: ignore[misc]


async def async_reduce(
    func: Callable[[R, T], R] | Callable[[R, T], object],
    aiterable: AsyncIterable[T],
    initial: R,
) -> R:
    """
    Reduce an async iterable to a single value using a reduction function.

    Applies the reduction function cumulatively to the items of the iterable,
    from left to right, starting with the initial value. The reduction function
    can be sync or async.

    Args:
        func: A reduction function that takes (accumulator, item) and returns
              the new accumulator value. Can be sync or async.
        aiterable: The async iterable to reduce.
        initial: The initial accumulator value.

    Returns:
        The final accumulated value after reducing all items.

    Raises:
        TypeError: If the iterable is empty (always returns initial value).

    Example:
        >>> async def gen():
        ...     for i in range(5):
        ...         yield i
        >>> # Sum all numbers
        >>> result = await async_reduce(lambda acc, x: acc + x, gen(), 0)
        >>> result
        10
        >>> # Multiply all numbers
        >>> result = await async_reduce(lambda acc, x: acc * x, gen(), 1)
        >>> result
        0
        >>> # Build a list
        >>> async def append(acc, x):
        ...     acc.append(x)
        ...     return acc
        >>> result = await async_reduce(append, gen(), [])
        >>> result
        [0, 1, 2, 3, 4]
    """
    accumulator: R = initial
    async for item in aiterable:
        result = func(accumulator, item)
        if asyncio.iscoroutine(result):
            result = await result  # type: ignore[misc]
        accumulator = result  # type: ignore[misc]
    return accumulator


async def async_flat_map(
    func: Callable[[T], AsyncIterable[R]] | Callable[[T], object],
    aiterable: AsyncIterable[T],
) -> AsyncIterator[R]:
    """
    Map a function over an async iterable and flatten the result.

    Applies the function to each item in the iterable (which must return
    an async iterable), then flattens all the results into a single async
    iterator. The function can be sync or async.

    This is useful for one-to-many transformations where you want to transform
    each element into multiple elements and then flatten them.

    Args:
        func: A function that takes an item and returns an async iterable.
              Can be sync or async.
        aiterable: The async iterable to map over.

    Yields:
        Flattened items from applying func to each item in the iterable.

    Example:
        >>> async def gen():
        ...     for i in range(3):
        ...         yield i
        >>> # Split each number into range
        >>> async def split(x):
        ...     for j in range(x):
        ...         yield j
        >>> result = await async_list(async_flat_map(split, gen()))
        >>> result
        [0, 0, 1, 0, 1, 2]
        >>> # With sync function
        >>> result = await async_list(async_flat_map(lambda x: (i for i in range(x)), gen()))
        >>> result
        [0, 0, 1, 0, 1, 2]
    """
    async for item in aiterable:
        result = func(item)
        if asyncio.iscoroutine(result):
            result = await result  # type: ignore[misc]
        async for inner_item in result:  # type: ignore[misc]
            yield inner_item


async def async_partition(
    predicate: Callable[[T], bool] | Callable[[T], object],
    aiterable: AsyncIterable[T],
) -> tuple[list[T], list[T]]:
    """
    Partition an async iterable into two lists based on a predicate.

    Items that satisfy the predicate (return truthy value) go into the first
    list, items that don't satisfy the predicate go into the second list.
    The predicate can be a sync function or an async function.

    This is useful when you need to split items into pass/fail groups,
    valid/invalid groups, or any binary classification.

    Args:
        predicate: A function that takes an item and returns bool.
                   Can be sync or async.
        aiterable: The async iterable to partition.

    Returns:
        A tuple of two lists: (items_matching, items_not_matching)

    Example:
        >>> async def gen():
        ...     for i in range(10):
        ...         yield i
        >>> # Partition into even and odd numbers
        >>> evens, odds = await async_partition(lambda x: x % 2 == 0, gen())
        >>> evens
        [0, 2, 4, 6, 8]
        >>> odds
        [1, 3, 5, 7, 9]
        >>> # With async predicate
        >>> async def is_even(x):
        ...     return x % 2 == 0
        >>> evens, odds = await async_partition(is_even, gen())
        >>> evens
        [0, 2, 4, 6, 8]
    """
    matching: list[T] = []
    not_matching: list[T] = []

    async for item in aiterable:
        result = predicate(item)
        if asyncio.iscoroutine(result):
            result = await result  # type: ignore[misc]
        if result:
            matching.append(item)
        else:
            not_matching.append(item)

    return matching, not_matching


async def async_chunked(
    aiterable: AsyncIterable[T], chunk_size: int
) -> AsyncIterator[list[T]]:
    """
    Chunk an async iterable into fixed-size lists.

    Yields lists of up to chunk_size items from the iterable. The last
    chunk may contain fewer items if the total number of items is not
    evenly divisible by chunk_size.

    This is useful for batch processing, rate limiting, or when you need
    to process items in groups rather than individually.

    Args:
        aiterable: The async iterable to chunk.
        chunk_size: Maximum number of items per chunk. Must be > 0.

    Yields:
        Lists of items, each up to chunk_size in length.

    Raises:
        ValueError: If chunk_size is not a positive integer.

    Example:
        >>> async def gen():
        ...     for i in range(10):
        ...         yield i
        >>> # Chunk into groups of 3
        >>> chunks = await async_list(async_chunked(gen(), 3))
        >>> chunks
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> # Chunk into pairs
        >>> chunks = await async_list(async_chunked(gen(), 2))
        >>> chunks
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    chunk: list[T] = []
    async for item in aiterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk
