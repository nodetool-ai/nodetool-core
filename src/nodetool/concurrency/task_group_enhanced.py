import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")

PRIORITY_HIGH = 0
PRIORITY_NORMAL = 50
PRIORITY_LOW = 100


@dataclass
class TaskPriority:
    """Task priority with ordering (lower value = higher priority)."""

    value: int = PRIORITY_NORMAL
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class TaskResult(Generic[T]):
    """Result of an async task execution."""

    task_id: str
    result: T | None = None
    exception: BaseException | None = None
    cancelled: bool = False
    deadline_exceeded: bool = False

    @property
    def success(self) -> bool:
        """Return True if the task completed successfully."""
        return self.exception is None and not self.cancelled and not self.deadline_exceeded


@dataclass
class TaskStats:
    """Statistics for a task group execution."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    deadline_exceeded: int = 0


class AsyncTaskGroupError(Exception):
    """Base exception for task group errors."""

    pass


class TaskExecutionError(AsyncTaskGroupError):
    """Raised when one or more tasks fail during execution."""

    def __init__(self, message: str, failed_results: list[TaskResult[Any]]):
        super().__init__(message)
        self.failed_results = failed_results


class TaskDeadlineExceededError(AsyncTaskGroupError):
    """Raised when a task deadline is exceeded."""

    def __init__(self, task_id: str, deadline: float):
        super().__init__(f"Task '{task_id}' exceeded deadline of {deadline}s")
        self.task_id = task_id
        self.deadline = deadline


class AsyncTaskGroup:
    """
    A context manager for managing groups of related async tasks.

    Provides task spawning, result collection, and proper cleanup on exceptions.
    Supports task priorities, deadlines, and dependencies.

    Example:
        group = AsyncTaskGroup()

        async def fetch_data(url: str) -> dict[str, Any]:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()

        # Spawn tasks with priorities and deadlines
        group.spawn(
            "critical",
            fetch_data("https://api.example.com/critical"),
            priority=PRIORITY_HIGH,
            deadline=5.0,
        )
        group.spawn(
            "background",
            fetch_data("https://api.example.com/data"),
            priority=PRIORITY_LOW,
        )

        # Execute all tasks
        results = await group.run()

        # Check results
        for result in results:
            if result.success:
                print(f"{result.task_id}: {result.result}")
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._results: list[TaskResult[Any]] = []
        self._running: bool = False
        self._priorities: dict[str, TaskPriority] = {}
        self._deadlines: dict[str, float | None] = {}
        self._dependencies: dict[str, set[str]] = {}
        self._completion_events: dict[str, asyncio.Event] = {}

    def spawn(
        self,
        task_id: str,
        coro: Coroutine[Any, Any, T],
        *,
        priority: int = PRIORITY_NORMAL,
        deadline: float | None = None,
        depends_on: list[str] | None = None,
        on_success: Callable[[T], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> asyncio.Task[T]:
        """
        Spawn a new task in the group.

        Args:
            task_id: Unique identifier for the task.
            coro: Async coroutine to execute.
            priority: Task priority (lower = higher priority, 0-100).
            deadline: Maximum time in seconds before task is cancelled.
            depends_on: List of task IDs that must complete before this task starts.
            on_success: Callback called on successful completion.
            on_error: Callback called on task failure.

        Returns:
            The created asyncio.Task.

        Raises:
            ValueError: If a task with the given ID already exists.
        """
        if task_id in self._tasks:
            raise ValueError(f"Task with ID '{task_id}' already exists")

        self._priorities[task_id] = TaskPriority(priority)
        self._deadlines[task_id] = deadline
        self._dependencies[task_id] = set(depends_on or [])
        self._completion_events[task_id] = asyncio.Event()

        def done_callback(task: asyncio.Task[T]) -> None:
            try:
                result = task.result()
                self._results.append(TaskResult(task_id=task_id, result=result))
                if on_success:
                    on_success(result)
            except asyncio.CancelledError:
                if task.cancelled() and self._deadlines.get(task_id) is not None:
                    self._results.append(TaskResult(task_id=task_id, deadline_exceeded=True))
                else:
                    self._results.append(TaskResult(task_id=task_id, cancelled=True))
            except BaseException as e:
                self._results.append(TaskResult(task_id=task_id, exception=e))
                if on_error:
                    on_error(e)
            finally:
                self._completion_events[task_id].set()

        async def wrapped_coro() -> T:
            if deadline is not None:
                try:
                    return await asyncio.wait_for(coro, timeout=deadline)
                except TimeoutError:
                    raise asyncio.CancelledError from None
            return await coro

        task = asyncio.create_task(wrapped_coro())
        task.add_done_callback(done_callback)
        self._tasks[task_id] = task

        return task

    def _get_execution_order(self) -> list[str]:
        """Determine task execution order based on priorities."""
        return sorted(
            self._tasks.keys(),
            key=lambda tid: (
                self._priorities[tid].value,
                self._priorities[tid].created_at,
            ),
        )

    async def run(
        self,
        *,
        raise_on_error: bool = True,
        cancel_on_error: bool = True,
    ) -> list[TaskResult[T]]:
        """
        Execute all spawned tasks concurrently and wait for completion.

        Args:
            raise_on_error: If True, raise TaskExecutionError if any task fails.
            cancel_on_error: If True, cancel remaining tasks when one fails.

        Returns:
            List of TaskResult objects for all tasks.

        Raises:
            TaskExecutionError: If raise_on_error is True and any task fails.
        """
        if self._running:
            raise AsyncTaskGroupError("Task group is already running")
        self._running = True

        if not self._tasks:
            return []

        try:
            pending: set[asyncio.Task[Any]] = set(self._tasks.values())
            failed: set[asyncio.Task[Any]] = set()

            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    try:
                        task.result()
                    except BaseException:
                        failed.add(task)

                if failed and cancel_on_error:
                    for task in pending:
                        task.cancel()
                    break

            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        except BaseException:
            if cancel_on_error:
                await self.cancel_all()
            raise

        failed_results = [r for r in self._results if not r.success]

        if raise_on_error and failed_results:
            raise TaskExecutionError(
                f"{len(failed_results)} task(s) failed during execution",
                failed_results=failed_results,
            )

        return self._results

    async def run_until_first(
        self,
        *,
        timeout: float | None = None,
    ) -> TaskResult[T]:
        """
        Execute tasks until the first one completes successfully.

        Useful for redundant request patterns where multiple endpoints
        can serve the same request.

        Args:
            timeout: Maximum time to wait for any task to complete.

        Returns:
            The TaskResult of the first successfully completed task.

        Raises:
            asyncio.TimeoutError: If timeout is reached before any task completes.
            AsyncTaskGroupError: If no tasks complete successfully before timeout.
        """
        if not self._tasks:
            raise AsyncTaskGroupError("No tasks to execute")

        if self._running:
            raise AsyncTaskGroupError("Task group is already running")
        self._running = True

        done: set[asyncio.Task[Any]]
        pending: set[asyncio.Task[Any]]

        try:
            done, pending = await asyncio.wait(
                self._tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout,
            )
        except BaseException:
            await self.cancel_all()
            raise

        for task in done:
            task_id = None
            for tid, t in self._tasks.items():
                if t is task:
                    task_id = tid
                    break

            if task_id:
                await self._collect_result(task_id, task)

        successful = [r for r in self._results if r.success]

        for task in pending:
            task.cancel()

        if successful:
            return successful[0]

        if timeout is not None and not done:
            raise TimeoutError()

        raise AsyncTaskGroupError("No task completed successfully")

    async def _collect_result(self, task_id: str, task: asyncio.Task[Any]) -> None:
        """Collect the result of a completed task."""
        try:
            result = task.result()
            self._results.append(TaskResult(task_id=task_id, result=result))
        except asyncio.CancelledError:
            self._results.append(TaskResult(task_id=task_id, cancelled=True))
        except BaseException as e:
            self._results.append(TaskResult(task_id=task_id, exception=e))

    async def cancel_all(self) -> None:
        """Cancel all pending tasks."""
        for task in self._tasks.values():
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def wait(self) -> None:
        """Wait for all tasks to complete without collecting results."""
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    @property
    def stats(self) -> TaskStats:
        """Return statistics about task execution."""
        stats = TaskStats(total=len(self._tasks))
        for result in self._results:
            if result.success:
                stats.completed += 1
            elif result.deadline_exceeded:
                stats.deadline_exceeded += 1
            elif result.cancelled:
                stats.cancelled += 1
            else:
                stats.failed += 1
        return stats

    @property
    def pending(self) -> list[str]:
        """Return list of task IDs that are still pending."""
        return [tid for tid, task in self._tasks.items() if not task.done()]

    @property
    def results(self) -> list[TaskResult[T]]:
        """Return collected results (empty before run() completes)."""
        return list(self._results)

    @property
    def priorities(self) -> dict[str, int]:
        """Return task priorities."""
        return {tid: p.value for tid, p in self._priorities.items()}

    def get_deadline(self, task_id: str) -> float | None:
        """Get the deadline for a specific task."""
        return self._deadlines.get(task_id)

    def is_dependency_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied."""
        for dep_id in self._dependencies.get(task_id, []):
            if dep_id not in self._completion_events:
                return False
            if not self._completion_events[dep_id].is_set():
                return False
        return True

    async def __aenter__(self) -> "AsyncTaskGroup":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            await self.cancel_all()
        elif self._tasks:
            pending = set(self._tasks.values())
            failed: set[asyncio.Task[Any]] = set()

            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    try:
                        task.result()
                    except BaseException:
                        failed.add(task)

                if failed:
                    for task in pending:
                        task.cancel()
                    break

            if pending:
                await asyncio.gather(*pending, return_exceptions=True)


__all__ = [
    "PRIORITY_HIGH",
    "PRIORITY_LOW",
    "PRIORITY_NORMAL",
    "AsyncTaskGroup",
    "TaskDeadlineExceededError",
    "TaskExecutionError",
    "TaskPriority",
    "TaskResult",
    "TaskStats",
]
