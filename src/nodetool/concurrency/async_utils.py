import asyncio
from typing import Optional


class AsyncSemaphore:
    """
    An async semaphore with timeout support for rate limiting and concurrency control.

    This provides a more convenient interface than asyncio.Semaphore by adding:
    - Built-in timeout support via the `acquire` method
    - Context manager support for automatic release
    - Clear separation between async acquire and sync wait

    Example:
        sem = AsyncSemaphore(max_tasks=5)

        # Using async context manager
        async with sem:
            await do_concurrent_work()

        # Using acquire with timeout
        if await sem.acquire(timeout=10.0):
            try:
                await do_concurrent_work()
            finally:
                sem.release()
    """

    def __init__(self, max_tasks: int):
        """
        Initialize the semaphore with a maximum number of concurrent tasks.

        Args:
            max_tasks (int): Maximum number of tasks that can acquire this semaphore concurrently.
                             Must be greater than 0.

        Raises:
            ValueError: If max_tasks is not a positive integer.
        """
        if max_tasks <= 0:
            raise ValueError("max_tasks must be a positive integer")
        self._semaphore = asyncio.Semaphore(max_tasks)
        self._max_tasks = max_tasks

    @property
    def max_tasks(self) -> int:
        """Return the maximum number of concurrent tasks allowed."""
        return self._max_tasks

    @property
    def available(self) -> int:
        """Return the number of available slots for new tasks."""
        try:
            return self._semaphore._value  # type: ignore
        except AttributeError:
            return 0

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the semaphore with an optional timeout.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds. If None (default),
                                      wait indefinitely. If <= 0, attempt to acquire
                                      without waiting.

        Returns:
            bool: True if the semaphore was acquired, False if the timeout expired.
        """
        if timeout is None:
            await self._semaphore.acquire()
            return True

        if timeout <= 0:
            if self._semaphore.locked():
                return False
            await self._semaphore.acquire()
            return True

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    def release(self) -> None:
        """Release the semaphore, allowing another task to acquire it."""
        self._semaphore.release()

    async def __aenter__(self) -> "AsyncSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def locked(self) -> bool:
        """Return True if the semaphore cannot be acquired immediately."""
        return self._semaphore.locked()


async def gather_with_concurrency(
    coro_list: list,
    max_concurrent: int,
) -> list:
    """
    Run coroutines with a maximum concurrency limit.

    This function wraps asyncio.gather to limit the number of concurrent tasks
    running at any given time. Tasks are started in order, but may complete
    out of order depending on execution time.

    Args:
        coro_list (list): List of coroutine functions or awaitables to execute.
        max_concurrent (int): Maximum number of coroutines to run concurrently.
                             Must be greater than 0.

    Returns:
        list: List of results in the same order as the input coroutines.

    Raises:
        ValueError: If max_concurrent is not a positive integer.

    Example:
        async def fetch_url(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()

        urls = [...]  # list of URLs
        results = await gather_with_concurrency(
            [fetch_url(url) for url in urls],
            max_concurrent=10
        )
    """
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be a positive integer")

    if not coro_list:
        return []

    sem = AsyncSemaphore(max_concurrent)
    results: list = [None] * len(coro_list)  # type: ignore[var-annotated]

    async def run_with_sem(index: int, coro):
        async with sem:
            results[index] = await coro

    await asyncio.gather(*(run_with_sem(i, coro) for i, coro in enumerate(coro_list)))
    return results
