import asyncio


class AsyncCounter:
    """
    A thread-safe atomic counter for async operations.

    This provides a counter that can be safely incremented and decremented
    concurrently from multiple async tasks. It's useful for:
    - Tracking concurrent operation counts
    - Simple rate limiting
    - Statistics gathering

    Example:
        counter = AsyncCounter()

        # Increment and get value
        await counter.increment()
        value = counter.value  # 1

        # Decrement and get value
        await counter.decrement()
        value = counter.value  # 0

        # Update by a specific amount
        await counter.add(5)
        value = counter.value  # 5

        # Reset to a value
        counter.reset(0)
        value = counter.value  # 0
    """

    def __init__(self, initial_value: int = 0) -> None:
        self._value = initial_value
        self._lock = asyncio.Lock()

    @property
    def value(self) -> int:
        """Get the current counter value."""
        return self._value

    async def increment(self, amount: int = 1) -> int:
        """
        Increment the counter by the specified amount.

        Args:
            amount: The amount to increment by (default: 1).

        Returns:
            The new counter value after incrementing.
        """
        async with self._lock:
            self._value += amount
            return self._value

    async def decrement(self, amount: int = 1) -> int:
        """
        Decrement the counter by the specified amount.

        Args:
            amount: The amount to decrement by (default: 1).

        Returns:
            The new counter value after decrementing.
        """
        async with self._lock:
            self._value -= amount
            return self._value

    async def add(self, amount: int) -> int:
        """
        Add a signed amount to the counter.

        Args:
            amount: The amount to add (can be positive or negative).

        Returns:
            The new counter value after adding.
        """
        async with self._lock:
            self._value += amount
            return self._value

    def reset(self, value: int = 0) -> None:
        """
        Reset the counter to the specified value.

        Args:
            value: The value to reset to (default: 0).
        """
        self._value = value

    async def get_and_increment(self, amount: int = 1) -> int:
        """
        Atomically get the current value and increment by the specified amount.

        This is useful for generating unique IDs or sequence numbers.

        Args:
            amount: The amount to increment by (default: 1).

        Returns:
            The value before incrementing.
        """
        async with self._lock:
            current = self._value
            self._value += amount
            return current

    async def get_and_set(self, value: int) -> int:
        """
        Atomically get the current value and set to a new value.

        Args:
            value: The new value to set.

        Returns:
            The value before setting.
        """
        async with self._lock:
            current = self._value
            self._value = value
            return current

    def __repr__(self) -> str:
        return f"AsyncCounter(value={self._value})"
