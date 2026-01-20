import asyncio
from collections.abc import Callable
from typing import Any


class AsyncEvent:
    """
    An async event primitive for signaling between tasks.

    Allows multiple consumers to wait for an event and be notified
    when it is set. Supports both one-shot and manual-reset modes.

    Example:
        event = AsyncEvent()

        async def producer():
            await asyncio.sleep(1)
            event.set("data_ready")

        async def consumer():
            value = await event.wait()
            print(f"Received: {value}")

        # Start both tasks
        group = AsyncTaskGroup()
        group.spawn("producer", producer())
        group.spawn("consumer", consumer())
        await group.run()
    """

    def __init__(self, *, auto_reset: bool = False) -> None:
        """
        Initialize the async event.

        Args:
            auto_reset: If True, the event resets automatically after waiters
                       are awakened. If False, the event stays set until
                       explicitly cleared.
        """
        self._auto_reset = auto_reset
        self._event = asyncio.Event()
        self._set_value: Any = None
        self._is_set = False

    def is_set(self) -> bool:
        """Return True if the event is set."""
        return self._is_set

    def clear(self) -> None:
        """Clear the event, resetting it to unset state."""
        self._event.clear()
        self._set_value = None
        self._is_set = False

    def set(self, value: Any = None) -> None:
        """
        Set the event and notify all waiting tasks.

        Args:
            value: Optional value to pass to waiters.
        """
        self._set_value = value
        self._is_set = True
        self._event.set()

    async def wait(self) -> Any:
        """
        Wait for the event to be set.

        Returns:
            The value passed to set(), or None if no value was provided.

        Raises:
            asyncio.CancelledError: If the wait is cancelled.
        """
        await self._event.wait()

        value = self._set_value

        if self._auto_reset:
            self._event.clear()
            self._is_set = False
            self._set_value = None

        return value

    async def wait_for(self, predicate: Callable[[Any], bool]) -> Any:
        """
        Wait for the event to be set and the predicate to be True.

        Args:
            predicate: A callable that takes the value from set() and
                       returns True when the condition is met.

        Returns:
            The value passed to set().

        Raises:
            asyncio.CancelledError: If the wait is cancelled.
        """
        while True:
            value = await self.wait()
            if predicate(value):
                return value
            self._event.clear()
            self._is_set = False

    @property
    def waiters(self) -> int:
        """Return the number of tasks waiting on this event."""
        return len(self._event._waiters)  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        state = "set" if self.is_set() else "unset"
        mode = "auto-reset" if self._auto_reset else "manual-reset"
        return f"AsyncEvent({state}, {mode})"


__all__ = ["AsyncEvent"]
