import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

R = TypeVar("R")


@dataclass
class PoolStats:
    """Statistics for a worker pool."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    queued_tasks: int = 0
    active_workers: int = 0


@dataclass
class PoolConfig:
    """Configuration for an AsyncPool."""

    max_workers: int = 4
    max_queue_size: int = 0
    timeout: float | None = None


class AsyncPoolClosedError(Exception):
    """Raised when operations are attempted on a closed pool."""

    pass


class AsyncPoolFullError(Exception):
    """Raised when the queue is full and no timeout was specified."""

    pass


class AsyncPool(Generic[R]):
    """
    A bounded async worker pool for managing concurrent task execution.

    The pool maintains a fixed number of worker tasks that process jobs from
    a queue. This provides controlled parallelism unlike AsyncTaskGroup which
    spawns unlimited tasks. Jobs are processed in FIFO order by default.

    Example:
        pool = AsyncPool(max_workers=4)

        async def process_item(item: int) -> int:
            await asyncio.sleep(0.1)
            return item * 2

        async with pool:
            futures = []
            for i in range(10):
                future = pool.submit(process_item(i))
                futures.append(future)

            results = await pool.gather_results(futures)

        print(results)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 0,
        timeout: float | None = None,
    ):
        """
        Initialize the async pool.

        Args:
            max_workers: Maximum number of concurrent workers.
            max_queue_size: Maximum number of pending jobs (0 for unlimited).
            timeout: Default timeout for job execution in seconds.

        Raises:
            ValueError: If max_workers < 1 or max_queue_size < 0.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be non-negative")

        self._config = PoolConfig(
            max_workers=max_workers,
            max_queue_size=max_queue_size,
            timeout=timeout,
        )
        self._workers: set[asyncio.Task[Any]] = set()
        self._queue: asyncio.Queue[Any]
        maxsize = max_queue_size if max_queue_size > 0 else 0
        if maxsize > 0:
            self._queue = asyncio.Queue(maxsize=maxsize)
        else:
            self._queue = asyncio.Queue()
        self._closed = False
        self._started = False
        self._stats = PoolStats()
        self._next_job_id = 0
        self._lock = asyncio.Lock()

    @property
    def config(self) -> PoolConfig:
        """Return the pool configuration."""
        return self._config

    @property
    def stats(self) -> PoolStats:
        """Return current pool statistics."""
        return PoolStats(
            total_tasks=self._stats.total_tasks,
            completed_tasks=self._stats.completed_tasks,
            failed_tasks=self._stats.failed_tasks,
            cancelled_tasks=self._stats.cancelled_tasks,
            queued_tasks=self._queue.qsize(),
            active_workers=len(self._workers),
        )

    @property
    def active_workers(self) -> int:
        """Return the number of currently active workers."""
        return len(self._workers)

    @property
    def queued_tasks(self) -> int:
        """Return the number of tasks waiting in the queue."""
        return self._queue.qsize()

    @property
    def is_closed(self) -> bool:
        """Return whether the pool is closed."""
        return self._closed

    @property
    def is_started(self) -> bool:
        """Return whether the pool has been started."""
        return self._started

    async def __aenter__(self) -> "AsyncPool[R]":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._workers:
            await self.drain()
        await self.close()

    def submit(
        self,
        coro: Callable[..., Coroutine[Any, Any, R]],
        *args: Any,
        **kwargs: Any,
    ) -> asyncio.Future[R]:
        """
        Submit a job to the pool.

        Args:
            coro: Callable that returns a coroutine.
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.

        Returns:
            A Future that will contain the result when complete.

        Raises:
            AsyncPoolClosedError: If the pool is closed or not started.
            AsyncPoolFullError: If the queue is full and no timeout.
        """
        if self._closed or not self._started:
            raise AsyncPoolClosedError("Cannot submit to a closed or not-started pool")

        future: asyncio.Future[R] = asyncio.get_event_loop().create_future()
        job_id = self._next_job_id
        self._next_job_id += 1

        try:
            self._queue.put_nowait((job_id, coro, args, kwargs, future))
        except asyncio.QueueFull:
            raise AsyncPoolFullError("Pool queue is full") from None

        return future

    async def submit_with_timeout(
        self,
        coro: Callable[..., Coroutine[Any, Any, R]],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> R:
        """
        Submit a job with a timeout.

        Args:
            coro: Callable that returns a coroutine.
            *args: Positional arguments for the callable.
            timeout: Timeout in seconds (uses default if None).
            **kwargs: Keyword arguments for the callable.

        Returns:
            The result of the coroutine.

        Raises:
            AsyncPoolClosedError: If the pool is closed.
            asyncio.TimeoutError: If the timeout expires.
        """
        future = self.submit(coro, *args, **kwargs)

        timeout_value = timeout if timeout is not None else self._config.timeout

        try:
            return await asyncio.wait_for(future, timeout=timeout_value)
        except TimeoutError:
            future.cancel()
            raise

    async def start(self) -> None:
        """Start the worker pool."""
        if self._closed:
            raise AsyncPoolClosedError("Cannot start a closed pool")

        if self._started:
            return

        self._started = True

        async def worker() -> None:
            while True:
                try:
                    _job_id, coro, args, kwargs, future = await self._queue.get()

                    if coro is None:
                        self._queue.task_done()
                        break

                    async with self._lock:
                        self._stats.total_tasks += 1

                    try:
                        if self._config.timeout:
                            result = await asyncio.wait_for(
                                coro(*args, **kwargs),
                                timeout=self._config.timeout,
                            )
                        else:
                            result = await coro(*args, **kwargs)

                        if not future.done():
                            future.set_result(result)

                        async with self._lock:
                            self._stats.completed_tasks += 1

                    except asyncio.CancelledError:
                        if not future.done():
                            future.cancel()
                        async with self._lock:
                            self._stats.cancelled_tasks += 1

                    except BaseException as e:
                        if not future.done():
                            future.set_exception(e)
                        async with self._lock:
                            self._stats.failed_tasks += 1

                    finally:
                        self._queue.task_done()

                except asyncio.CancelledError:
                    break

                except BaseException:
                    pass

        for _ in range(self._config.max_workers):
            task = asyncio.create_task(worker())
            self._workers.add(task)
            task.add_done_callback(self._workers.discard)

    async def close(self) -> None:
        """Close the pool and wait for all pending jobs to complete."""
        if self._closed:
            return

        self._closed = True

        for _ in range(len(self._workers)):
            await self._queue.put((0, None, (), {}, asyncio.get_event_loop().create_future()))

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()

    async def drain(self) -> None:
        """Wait for all queued jobs to complete without accepting new ones."""
        await self._queue.join()

    async def gather_results(
        self,
        futures: list[asyncio.Future[R]],
        *,
        raise_on_error: bool = True,
        timeout: float | None = None,
    ) -> list[R]:
        """
        Wait for multiple futures to complete and return their results.

        Args:
            futures: List of futures to wait for.
            raise_on_error: Raise exception if any future fails.
            timeout: Maximum time to wait.

        Returns:
            List of results in the same order as the input futures.

        Raises:
            Exception: If any future raised an exception and raise_on_error is True.
        """
        if not futures:
            return []

        timeout_value = timeout if timeout is not None else self._config.timeout

        _, pending = await asyncio.wait(
            futures,
            timeout=timeout_value,
            return_when=asyncio.ALL_COMPLETED if raise_on_error else asyncio.FIRST_EXCEPTION,
        )

        if pending:
            for f in pending:
                f.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        if raise_on_error:
            for f in futures:
                try:
                    exc = f.exception()
                except asyncio.CancelledError:
                    continue
                if exc is not None:
                    raise exc

        results = []
        for f in futures:
            try:
                results.append(f.result())
            except BaseException as e:
                if raise_on_error:
                    raise
                results.append(e)
        return results

    async def map(
        self,
        coro: Callable[..., Coroutine[Any, Any, R]],
        items: list[Any],
        *,
        timeout: float | None = None,
    ) -> list[R]:
        """
        Apply a coroutine to each item in a list concurrently.

        Args:
            coro: Async function to apply to each item.
            items: List of items to process.
            timeout: Timeout for each individual operation.

        Returns:
            List of results in the same order as input items.

        Example:
            async def double(x: int) -> int:
                return x * 2

            pool = AsyncPool(max_workers=4)
            results = await pool.map(double, [1, 2, 3, 4])
            # [2, 4, 6, 8]
        """
        futures = []
        for item in items:
            future = self.submit(coro, item)
            futures.append(future)

        results = await self.gather_results(futures)
        return results


__all__ = ["AsyncPool", "AsyncPoolClosedError", "AsyncPoolFullError", "PoolConfig", "PoolStats"]
