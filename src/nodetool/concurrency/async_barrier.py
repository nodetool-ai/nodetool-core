import asyncio
from typing import Any


class BrokenBarrierError(RuntimeError):
    """Raised when a barrier is broken due to timeout or cancellation."""

    pass


class AsyncBarrier:
    """
    An async barrier primitive for synchronizing groups of tasks.

    A barrier allows multiple tasks to wait for each other at a checkpoint.
    All tasks must reach the barrier before any can proceed past it. This is
    useful for phased operations where all participants must complete one
    phase before starting the next.

    Example:
        barrier = AsyncBarrier(3)

        async def participant(participant_id):
            # Do some work in phase 1
            await asyncio.sleep(participant_id * 0.1)

            # Wait for all participants to reach the barrier
            await barrier.wait()

            # All participants can now proceed to phase 2
            print(f"Participant {participant_id} proceeding to phase 2")

        # Create tasks for all participants
        tasks = [participant(i) for i in range(3)]
        await asyncio.gather(*tasks)
    """

    def __init__(self, parties: int, *, timeout: float | None = None) -> None:
        """
        Initialize the barrier.

        Args:
            parties: The number of tasks that must reach the barrier before
                     any can proceed.
            timeout: Optional timeout in seconds. If a wait operation exceeds
                     this time, a BrokenBarrierError is raised.

        Raises:
            ValueError: If parties is less than 1.
        """
        if parties < 1:
            raise ValueError("parties must be at least 1")

        self._parties = parties
        self._timeout = timeout
        self._arrived = 0
        self._waiting: set[asyncio.Task[Any]] = set()
        self._event = asyncio.Event()
        self._broken = False

    @property
    def parties(self) -> int:
        """Return the number of parties required to pass the barrier."""
        return self._parties

    @property
    def n_waiting(self) -> int:
        """Return the number of tasks currently waiting at the barrier."""
        return len(self._waiting)

    @property
    def broken(self) -> bool:
        """Return True if the barrier is in a broken state."""
        return self._broken

    async def wait(self) -> int:
        """
        Wait for all parties to reach the barrier.

        This method blocks until all parties have called wait(), at which
        point all tasks proceed and the barrier is reset for reuse.

        Returns:
            The arrival index (0 to parties-1) of this party. This can be
            used to designate a leader or coordinator.

        Raises:
            BrokenBarrierError: If the barrier is broken or timeout expires.
            asyncio.CancelledError: If the wait is cancelled.
        """
        if self._broken:
            raise BrokenBarrierError("Barrier was broken")

        self._arrived += 1

        if self._arrived >= self._parties:
            self._arrived = 0
            self._event.set()
            self._event = asyncio.Event()
            return 0

        task = asyncio.current_task()
        if task is not None:
            self._waiting.add(task)

        try:
            if self._timeout is None:
                await self._event.wait()
            else:
                try:
                    await asyncio.wait_for(self._event.wait(), timeout=self._timeout)
                except TimeoutError:
                    self._broken = True
                    self._arrived = self._parties
                    self._event.set()
                    self._event = asyncio.Event()
                    raise BrokenBarrierError(f"Barrier wait timed out after {self._timeout} seconds") from None
        finally:
            if task is not None:
                self._waiting.discard(task)

        if self._broken:
            raise BrokenBarrierError("Barrier was broken")

        return self._arrived

    def reset(self) -> None:
        """
        Manually reset the barrier to its initial state.

        This is useful if the barrier is in a broken state or if you need
        to synchronize a new group of parties. Wakes up all currently
        waiting tasks.
        """
        self._arrived = 0
        self._event.set()
        self._event = asyncio.Event()

    def abort(self) -> None:
        """
        Abort the barrier, breaking all waiting parties.

        This causes all current and future waiters to receive a
        BrokenBarrierError.
        """
        self._broken = True
        self._arrived = self._parties
        self._event.set()
        self._event = asyncio.Event()

    def __repr__(self) -> str:
        state = "broken" if self.broken else "intact"
        return f"AsyncBarrier(parties={self._parties}, waiting={self.n_waiting}, {state})"


__all__ = ["AsyncBarrier", "BrokenBarrierError"]
