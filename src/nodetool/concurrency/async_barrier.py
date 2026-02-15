import asyncio
from typing import Optional


class AsyncBarrier:
    """
    An async barrier for coordinating multiple coroutines.

    A barrier allows a set of coroutines to synchronize at a specific point.
    Each coroutine waits at the barrier until all participating coroutines have
    arrived, then they all proceed together.

    This is useful for phased concurrent operations where multiple tasks must
    complete one phase before any can start the next.

    Example:
        barrier = AsyncBarrier(parties=3)

        async def worker(task_id: int):
            for phase in range(3):
                # Do some work
                await asyncio.sleep(0.1)
                print(f"Task {task_id} phase {phase} done")

                # Wait for all tasks to complete this phase
                await barrier.wait()
                print(f"Task {task_id} starting phase {phase + 1}")

        # Launch 3 concurrent workers
        await asyncio.gather(*[worker(i) for i in range(3)])
    """

    def __init__(self, parties: int):
        """
        Initialize the barrier.

        Args:
            parties (int): Number of coroutines that must call wait() before
                           any of them proceed. Must be greater than 0.

        Raises:
            ValueError: If parties is not a positive integer.
        """
        if parties <= 0:
            raise ValueError("parties must be a positive integer")

        self._parties = parties
        self._count = 0
        self._condition = asyncio.Condition()

    @property
    def parties(self) -> int:
        """Return the number of coroutines required to pass the barrier."""
        return self._parties

    @property
    def waiting(self) -> int:
        """Return the number of coroutines currently waiting at the barrier."""
        return self._count

    @property
    def remaining(self) -> int:
        """Return the number of coroutines still needed to pass the barrier."""
        return self._parties - self._count

    async def wait(self) -> bool:
        """
        Wait at the barrier until all parties have arrived.

        When the specified number of coroutines have called wait(), they are
        all released simultaneously and this method returns True for all but
        one coroutine, which returns False. This allows one "leader" to be
        designated if needed.

        Returns:
            bool: True for all but one coroutine, False for one designated
                  "leader" coroutine. The leader is arbitrary but consistent.

        Example:
            is_leader = await barrier.wait()
            if is_leader:
                # Only one task does this cleanup
                await cleanup_shared_resources()
        """
        async with self._condition:
            self._count += 1

            if self._count == self._parties:
                # Last coroutine to arrive wakes everyone
                self._condition.notify_all()
                # Reset for next use
                self._count = 0
                return False  # The last one is "not leader" (arbitrary choice)

            # Wait for the last coroutine to arrive
            await self._condition.wait()
            return True

    async def reset(self) -> None:
        """
        Reset the barrier to its initial state.

        This method is useful if the barrier needs to be reused after a timeout
        or cancellation. Any coroutines currently waiting will raise
        asyncio.CancelledError.

        Warning:
            This should only be called in exceptional circumstances (e.g., when
            a task has been cancelled). Normal usage automatically resets the
            barrier when all parties pass through.
        """
        async with self._condition:
            self._count = 0
            self._condition.notify_all()

    def __repr__(self) -> str:
        return (
            f"AsyncBarrier(parties={self._parties}, "
            f"waiting={self._count}, remaining={self.remaining})"
        )


__all__ = ["AsyncBarrier"]
