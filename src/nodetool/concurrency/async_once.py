import asyncio
from typing import Any, Coroutine, Generic, TypeVar

T = TypeVar("T")


class AsyncOnceError(Exception):
    """Raised when an attempt is made to call an AsyncOnce-protected function incorrectly."""

    pass


class AsyncOnce(Generic[T]):
    """
    A one-time execution guard for async functions.

    Ensures that an async function is only executed once, even when called
    concurrently from multiple coroutines. Additional calls will await
    the first call's completion and return the same result or raise the
    same exception.

    This is useful for:
    - Lazy initialization that may be triggered from multiple places
    - One-time setup or teardown operations
    - Singleton patterns in async code

    Example:
        once = AsyncOnce()

        async def initialize():
            # This will only run once, even if called concurrently
            return await setup_database()

        # First call triggers initialization
        result1 = await once.run(initialize)

        # Second call returns the same result without re-running
        result2 = await once.run(initialize)
        assert result1 == result2
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[T] | None = None
        self._result: T | None = None
        self._exception: BaseException | None = None
        self._initialized = False

    def run(self, coro: Coroutine[Any, Any, T]) -> asyncio.Future[T]:
        """
        Run the coroutine if it hasn't been run before.

        If this is the first call, the coroutine is started and its result
        or exception is cached. All concurrent and subsequent calls will
        return the same result or raise the same exception without re-running.

        Args:
            coro: The coroutine to execute once.

        Returns:
            A Future that resolves to the result of the coroutine.

        Raises:
            AsyncOnceError: If the wrapped coroutine raises an exception.
        """
        if not self._initialized:
            self._initialize(coro)
        return self._wait()

    def _initialize(self, coro: Coroutine[Any, Any, T]) -> None:
        self._initialized = True
        self._task = asyncio.create_task(coro)

    def _set_result(self, future: asyncio.Future[Any]) -> None:
        try:
            self._result = future.result()  # type: ignore[assignment]
        except BaseException as e:
            self._exception = e

    def _wait(self) -> asyncio.Future[T]:
        assert self._task is not None
        future: asyncio.Future[T] = asyncio.wrap_future(self._task)
        future.add_done_callback(self._set_result)
        return future

    @property
    def done(self) -> bool:
        """Return True if the one-time operation has completed."""
        return self._task is not None and self._task.done()  # type: ignore[union-attr]

    @property
    def result(self) -> T | None:
        """Return the cached result if available."""
        if self.done and self._exception is None:
            return self._result
        return None

    @property
    def exception(self) -> BaseException | None:
        """Return the cached exception if one was raised."""
        return self._exception
