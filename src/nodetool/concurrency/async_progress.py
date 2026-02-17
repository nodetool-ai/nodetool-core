import asyncio
from collections.abc import Callable
from typing import Any


class ProgressUpdate:
    """
    Represents a single progress update.

    Attributes:
        current: Current progress value (e.g., items processed)
        total: Total value for completion (e.g., total items)
        percentage: Completion percentage (0.0 to 1.0)
        message: Optional human-readable progress message
        metadata: Optional additional data about the progress
    """

    def __init__(
        self,
        current: float,
        total: float,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.current = current
        self.total = total
        self.message = message
        self.metadata = metadata or {}

    @property
    def percentage(self) -> float:
        """Return completion as a percentage (0.0 to 1.0)."""
        if self.total <= 0:
            return 1.0
        return min(max(self.current / self.total, 0.0), 1.0)

    @property
    def percentage_str(self) -> str:
        """Return completion as a formatted string (e.g., '75%')."""
        return f"{self.percentage * 100:.1f}%"

    def __repr__(self) -> str:
        current_str = f"{int(self.current)}" if self.current == int(self.current) else f"{self.current}"
        total_str = f"{int(self.total)}" if self.total == int(self.total) else f"{self.total}"
        return f"ProgressUpdate({self.percentage_str}, {current_str}/{total_str})"


ProgressCallback = Callable[[ProgressUpdate], None]


class AsyncProgressTracker:
    """
    Track progress of async operations with callbacks and percentage completion.

    This utility provides a way to monitor and report progress during long-running
    async operations. It supports callbacks for real-time updates and provides
    percentage completion tracking.

    Example:
        async def process_items(items: list[str]):
            tracker = AsyncProgressTracker(
                total=len(items),
                message="Processing items"
            )

            def on_progress(update: ProgressUpdate):
                print(f"Progress: {update.percentage_str}")

            tracker.add_callback(on_progress)

            for i, item in enumerate(items):
                await process(item)
                tracker.update(i + 1)

            tracker.complete()

        # Or use as context manager
        async with AsyncProgressTracker(total=100) as tracker:
            for i in range(100):
                await work()
                tracker.update(i + 1)
    """

    def __init__(
        self,
        total: float,
        *,
        message: str | None = None,
        initial: float = 0.0,
    ) -> None:
        """
        Initialize the progress tracker.

        Args:
            total: The total value representing completion (e.g., total items)
            message: Optional description of the work being tracked
            initial: Optional initial progress value (default: 0.0)
        """
        if total <= 0:
            raise ValueError(f"total must be positive, got {total}")

        if initial < 0:
            raise ValueError(f"initial must be non-negative, got {initial}")

        if initial > total:
            raise ValueError(f"initial ({initial}) cannot exceed total ({total})")

        self._total = total
        self._current = initial
        self._message = message
        self._callbacks: list[ProgressCallback] = []
        self._lock = asyncio.Lock()
        self._completed = False
        self._started = False
        self._metadata: dict[str, Any] = {}

    @property
    def total(self) -> float:
        """Return the total value for completion."""
        return self._total

    @property
    def current(self) -> float:
        """Return the current progress value."""
        return self._current

    @property
    def percentage(self) -> float:
        """Return completion as a percentage (0.0 to 1.0)."""
        if self._total <= 0:
            return 1.0
        return min(max(self._current / self._total, 0.0), 1.0)

    @property
    def is_completed(self) -> bool:
        """Return True if progress tracking is complete."""
        return self._completed

    @property
    def is_started(self) -> bool:
        """Return True if progress tracking has started."""
        return self._started

    @property
    def message(self) -> str | None:
        """Return the current progress message."""
        return self._message

    def add_callback(self, callback: ProgressCallback) -> None:
        """
        Add a callback to be invoked on progress updates.

        Args:
            callback: Function to call with ProgressUpdate on each update
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: ProgressCallback) -> None:
        """
        Remove a previously added callback.

        Args:
            callback: The callback function to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata for the current progress state.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value

    async def update(
        self,
        current: float,
        message: str | None = None,
    ) -> None:
        """
        Update the progress and notify callbacks.

        Args:
            current: New current progress value
            message: Optional message override for this update
        """
        if self._completed:
            raise RuntimeError("Cannot update a completed tracker")

        async with self._lock:
            self._started = True
            self._current = min(max(current, 0.0), self._total)

            if message:
                self._message = message

            await self._notify()

    async def increment(self, amount: float = 1.0) -> None:
        """
        Increment progress by the specified amount.

        Args:
            amount: Amount to increment by (default: 1.0)
        """
        await self.update(self._current + amount)

    def set_message(self, message: str) -> None:
        """
        Update the progress message.

        Args:
            message: New progress message
        """
        self._message = message

    async def complete(self, message: str | None = None) -> None:
        """
        Mark the tracker as complete and notify callbacks.

        Args:
            message: Optional completion message
        """
        async with self._lock:
            self._started = True
            self._current = self._total
            self._completed = True

            if message:
                self._message = message

            await self._notify()

    def get_progress(self) -> ProgressUpdate:
        """
        Get the current progress state without updating.

        Returns:
            ProgressUpdate with current state
        """
        return ProgressUpdate(
            current=self._current,
            total=self._total,
            message=self._message,
            metadata=self._metadata.copy(),
        )

    async def _notify(self) -> None:
        """Notify all registered callbacks with current progress."""
        update = self.get_progress()

        for callback in self._callbacks:
            try:
                result = callback(update)

                # If callback is async, await it
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # Don't let one callback failure break others
                pass

    async def __aenter__(self) -> "AsyncProgressTracker":
        """Enter context manager, starting progress tracking."""
        self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, marking as complete if not already."""
        if not self._completed and exc_type is None:
            await self.complete()

    def __repr__(self) -> str:
        status = "completed" if self._completed else "active" if self._started else "pending"
        return f"AsyncProgressTracker({status}, {self.percentage:.1%}, {self._current}/{self._total})"


__all__ = [
    "AsyncProgressTracker",
    "ProgressCallback",
    "ProgressUpdate",
]
