"""
Async cancellation scope for structured cancellation handling.

Provides a context manager that allows controlled cancellation of async operations
with cleanup callbacks and cancellation status tracking.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable


class CancellationError(Exception):
    """Raised when an operation is cancelled through a CancellationScope."""

    def __init__(self, message: str = "Operation was cancelled"):
        self.message = message
        super().__init__(self.message)


class AsyncCancellationScope:
    """
    A context manager for handling cancellation of async operations.

    Provides structured cancellation handling with cleanup callbacks and
    cancellation status tracking. This is useful for coordinating cancellation
    across multiple concurrent tasks.

    Example:
        async def worker(scope: AsyncCancellationScope, task_id: int):
            try:
                while not scope.is_cancelled():
                    # Do work
                    await asyncio.sleep(0.1)
            except CancellationError:
                print(f"Task {task_id} cleaned up")

        scope = AsyncCancellationScope()

        async with scope:
            # Launch workers
            tasks = [
                asyncio.create_task(worker(scope, i))
                for i in range(5)
            ]

            # Cancel all workers after some time
            await asyncio.sleep(1.0)
            scope.cancel()

            # Wait for cleanup
            await asyncio.gather(*tasks, return_exceptions=True)
    """

    def __init__(self) -> None:
        """Initialize the cancellation scope."""
        self._cancelled = False
        self._cleanup_callbacks: list[Callable[[], Any]] = []
        self._lock = asyncio.Lock()
        self._cleanup_tasks: list[asyncio.Task[Any]] = []

    def cancel(self) -> None:
        """
        Cancel the scope, signalling all operations to stop.

        This method is thread-safe and can be called from anywhere.
        """
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """
        Check if the scope has been cancelled.

        Returns:
            True if the scope has been cancelled, False otherwise.
        """
        return self._cancelled

    def add_cleanup_callback(
        self, callback: Callable[[], Any]
    ) -> Callable[[], None]:
        """
        Add a cleanup callback to be called when the scope exits.

        Args:
            callback: A callable that will be invoked on scope exit.

        Returns:
            A function that removes the callback when called.

        Example:
            def cleanup():
                print("Cleaning up resources")

            remove = scope.add_cleanup_callback(cleanup)
            # ... do work ...
            remove()  # Remove callback if needed
        """
        self._cleanup_callbacks.append(callback)

        def remove() -> None:
            """Remove the cleanup callback."""
            if callback in self._cleanup_callbacks:
                self._cleanup_callbacks.remove(callback)

        return remove

    def raise_if_cancelled(self) -> None:
        """
        Raise CancellationError if the scope has been cancelled.

        This is useful for cooperative cancellation - tasks can periodically
        check if they should stop.

        Raises:
            CancellationError: If the scope has been cancelled.

        Example:
            async def long_running_task(scope: AsyncCancellationScope):
                for i in range(1000):
                    scope.raise_if_cancelled()
                    await do_some_work()
        """
        if self._cancelled:
            raise CancellationError()

    async def wait_for_cancelled(self) -> None:
        """
        Wait until the scope is cancelled.

        This is useful for tasks that should run until cancellation.

        Example:
            async def monitor_task(scope: AsyncCancellationScope):
                while True:
                    await scope.wait_for_cancelled()
                    break
                # Cleanup logic here
        """
        while not self._cancelled:
            await asyncio.sleep(0.001)

    def __enter__(self) -> AsyncCancellationScope:
        """Enter the cancellation scope (sync context)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the cancellation scope (sync context)."""
        self._run_cleanup()

    async def __aenter__(self) -> AsyncCancellationScope:
        """Enter the cancellation scope (async context)."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the cancellation scope (async context)."""
        self._run_cleanup()

    def _run_cleanup(self) -> None:
        """Run all cleanup callbacks."""
        for callback in self._cleanup_callbacks:
            try:
                result = callback()
                # If callback returns a coroutine, schedule it
                if asyncio.iscoroutine(result):
                    task = asyncio.create_task(result)  # type: ignore[arg-type]
                    self._cleanup_tasks.append(task)
            except Exception:
                # Log but don't fail - cleanup should be best-effort
                pass


__all__ = [
    "AsyncCancellationScope",
    "CancellationError",
]
