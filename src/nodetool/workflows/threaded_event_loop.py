import asyncio
import contextvars
import threading
from asyncio import AbstractEventLoop
from concurrent.futures import Future
from typing import Any, Callable, Coroutine, Optional, TypeVar

from nodetool.config.logging_config import get_logger

T = TypeVar("T")
log = get_logger(__name__)  # Setup logger


class ThreadedEventLoop:
    """
    ## Overview

    The `ThreadedEventLoop` class provides a convenient way to run an asyncio event loop in a separate thread.
    This is particularly useful for integrating asynchronous operations with synchronous code or for isolating certain async operations.

    ## Usage Examples

    ### Basic Usage

    ```python
    tel = ThreadedEventLoop()
    tel.start()

    # Run a coroutine
    async def my_coroutine():
        await asyncio.sleep(1)
        return "Hello, World!"

    future = tel.run_coroutine(my_coroutine())
    result = future.result()  # Blocks until the coroutine completes
    print(result)  # Output: Hello, World!

    tel.stop()
    ```

    ### Using as a Context Manager

    ```python
    async def my_coroutine():
        await asyncio.sleep(1)
        return "Hello, World!"

    with ThreadedEventLoop() as tel:
        future = tel.run_coroutine(my_coroutine())
        result = future.result()
        print(result)  # Output: Hello, World!
    ```

    ### Running a Synchronous Function

    ```python
    import time

    def slow_function(duration):
        time.sleep(duration)
        return f"Slept for {duration} seconds"

    with ThreadedEventLoop() as tel:
        future = tel.run_in_executor(slow_function, 2)
        result = future.result()
        print(result)  # Output: Slept for 2 seconds
    ```

    ## Thread Safety and Best Practices

    1. The `run_coroutine` and `run_in_executor` methods are thread-safe and can be called from any thread.
    2. Avoid directly accessing or modifying the internal event loop (`self._loop`) from outside the class.
    3. Always ensure that `stop()` is called when you're done with the `ThreadedEventLoop`, either explicitly or by using it as a context manager.
    4. Remember that coroutines scheduled with `run_coroutine` run in the separate thread. Be cautious about shared state and race conditions.
    5. The `ThreadedEventLoop` is designed for long-running operations. For short-lived async operations, consider using `asyncio.run()` instead.

    ## Note on Error Handling

    Errors that occur within coroutines or functions scheduled on the `ThreadedEventLoop` are captured in the returned `Future` objects. Always check for exceptions when getting results from these futures:

    ```python
    future = tel.run_coroutine(some_coroutine())
    try:
        result = future.result()
    except Exception as e:
        print(f"An error occurred: {e}")
    ```

    By following these guidelines and using the provided methods, you can safely integrate asynchronous operations into synchronous code or isolate certain async operations in a separate thread.
    """

    def __init__(self):
        # Lazily create the loop on start to avoid leaking FDs
        # when instances are constructed but never started.
        self._loop: Optional[AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._stop_initiated: bool = False  # New flag

    def start(self) -> None:
        """Start the event loop in a separate thread."""
        if self._running:
            return
        self._running = True
        self._stop_initiated = False  # Reset flag on start
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

    async def _cancel_all_tasks_and_wait(self):
        """Internal coroutine to cancel all tasks and wait for them.
        To be run by _shutdown_loop.
        """
        all_current_tasks = asyncio.all_tasks(self._loop)
        # Exclude the task running this coroutine itself, if applicable, though ensure_future handles it well.
        tasks_to_cancel = [
            t for t in all_current_tasks if t is not asyncio.current_task()
        ]

        if not tasks_to_cancel:
            log.debug("No tasks to cancel.")
            return

        log.debug(f"Cancelling {len(tasks_to_cancel)} task(s).")
        for task in tasks_to_cancel:
            task.cancel()

        # Wait for all tasks to complete after cancellation. This allows them to handle CancelledError.
        # Using a timeout to prevent hanging indefinitely if a task misbehaves.
        try:
            # results will contain exceptions for cancelled tasks (CancelledError)
            await asyncio.wait_for(
                asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=5.0
            )
            log.debug("All cancellable tasks finished after cancellation signal.")
        except TimeoutError:
            log.warning(
                "Timeout waiting for tasks to finish during shutdown. Some tasks may not have exited cleanly."
            )
        except Exception as e:
            log.error(
                f"Error waiting for tasks during shutdown: {e}",
                exc_info=True,
            )

    def _shutdown_loop(self):
        """Helper to cancel tasks and then schedule loop stop.
        This runs IN the event loop's thread via call_soon_threadsafe.
        """
        if self._stop_initiated:
            # Avoid multiple shutdown attempts if called rapidly
            return
        self._stop_initiated = True
        log.debug("Starting shutdown sequence in event loop thread.")

        # Ensure the task cancellation logic runs to completion before stopping the loop.
        # We create a task for _cancel_all_tasks_and_wait, and then ensure_future
        # will schedule it. After it completes, we stop the loop.
        async def shutdown_sequence_coro():
            await self._cancel_all_tasks_and_wait()
            log.debug("Task cancellation complete, stopping loop.")
            if self._loop and self._loop.is_running():  # Check again before stopping
                self._loop.stop()
            else:
                log.debug(
                    "Loop was already stopped before explicit stop in shutdown_sequence."
                )

        self._shutdown_future = asyncio.ensure_future(
            shutdown_sequence_coro(), loop=self._loop
        )

    def stop(self) -> None:
        """Stop the event loop and wait for the thread to finish."""
        if not self._thread:
            log.debug("Stop called but no thread was started.")
            return

        if self._stop_initiated:
            log.debug("Stop already initiated.")
            return

        if not self._running:
            log.debug("Stop called but loop was not running. Continuing with cleanup.")

        log.debug(
            "Initiating stop. Current thread: %s, Loop thread: %s",
            threading.get_ident(),
            self._thread.ident if self._thread else "N/A",
        )

        self._running = False

        loop_running = bool(self._loop and self._loop.is_running())
        if loop_running:
            log.debug("Scheduling _shutdown_loop via call_soon_threadsafe.")
            assert self._loop is not None
            self._loop.call_soon_threadsafe(self._shutdown_loop)
        else:
            log.warning(
                "Stop called, but internal loop was not reported as running. Cleanup will proceed."
            )
            self._stop_initiated = True

        current_thread_id = threading.get_ident()
        target_thread_id = self._thread.ident if self._thread else None

        if self._thread and current_thread_id != target_thread_id:
            log.debug(f"Waiting for event loop thread {target_thread_id} to join.")
            import platform

            timeout = 3.0 if platform.system() == "Windows" else 10.0
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning(
                    f"Thread did not join in time ({timeout}s) after stop. Loop might be stuck or tasks are non-cooperative."
                )
        elif self._thread and current_thread_id == target_thread_id:
            log.warning(
                "Stop called from within the event loop thread. Join will be skipped. Loop should stop via _shutdown_loop."
            )

        thread_still_alive = bool(self._thread and self._thread.is_alive())
        if thread_still_alive:
            log.warning(
                "Event loop thread still alive after join attempt; skipping loop.close() here."
            )
            return

        self._thread = None

        if self._loop is not None:
            try:
                if self._loop.is_running():
                    log.warning(
                        "Internal loop reports running but thread is not alive; skipping close()."
                    )
                    return
                if not self._loop.is_closed():
                    self._loop.close()
            finally:
                self._loop = None

        self._stop_initiated = True
        log.debug("Stop method finished.")

    def _run_event_loop(self) -> None:
        """Set the event loop for this thread and run it."""
        log.debug(f"Event loop thread {threading.get_ident()} started.")
        if self._loop is None:
            # Should not happen; guard to avoid AttributeError in rare cases
            self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            # Clear per-thread caches now that the loop is stopping, to avoid cross-workflow leaks
            try:
                from nodetool.config.environment import Environment
                Environment.clear_thread_caches()
                log.debug("Cleared thread-local caches via Environment.")
            except Exception as e:
                log.warning(
                    f"Failed to clear thread-local caches: {e}",
                )

            if self._loop and not self._loop.is_closed():
                log.debug("Closing event loop.")
                self._loop.close()
                log.debug("Event loop closed.")
            else:
                log.debug(
                    "Event loop was already closed before explicit close call in _run_event_loop."
                )
            log.debug(f"Event loop thread {threading.get_ident()} finished.")

    def run_coroutine(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """Schedule a coroutine to run in this event loop.

        Propagates contextvars from the caller thread to the loop thread,
        ensuring that context (e.g., ResourceScope) is preserved across
        thread boundaries.
        """
        if self._loop is None:
            raise RuntimeError("Not started. Use start() or context manager.")

        # Capture the current context in the caller thread
        outer_ctx = contextvars.copy_context()

        # Create a Future to return to the caller
        result_future: Future[T] = Future()

        def run_with_context():
            """Wrapper to run the coroutine with the captured context."""
            # Create and schedule the task with the captured context
            task = self._loop.create_task(coro)

            def on_done(t):
                """Callback when the task completes."""
                try:
                    exc = t.exception()
                    if exc is not None:
                        result_future.set_exception(exc)
                    else:
                        result_future.set_result(t.result())
                except asyncio.CancelledError:
                    result_future.cancel()
                except Exception as e:
                    result_future.set_exception(e)

            task.add_done_callback(on_done)

        # Schedule the wrapper with context propagation
        self._loop.call_soon_threadsafe(outer_ctx.run, run_with_context)
        return result_future

    def run_in_executor(self, func: Callable[..., T], *args: Any) -> asyncio.Future[T]:
        """Run a synchronous function in the default executor of this event loop."""
        if self._loop is None:
            raise RuntimeError("Not started. Use start() or context manager.")
        return self._loop.run_in_executor(None, func, *args)

    @property
    def is_running(self) -> bool:
        """Check if the event loop is running."""
        return self._running

    def __enter__(self) -> "ThreadedEventLoop":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __del__(self) -> None:  # best-effort safety net
        try:
            if (
                self._loop is not None
                and not self._loop.is_closed()
                and not self._loop.is_running()
            ):
                # If user never started the loop, there is no thread; safe to close.
                # If it was started but not stopped, closing here is best-effort.
                self._loop.close()
        except Exception:
            # Avoid raising during GC
            pass
        finally:
            self._loop = None
