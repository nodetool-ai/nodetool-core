import asyncio
from asyncio import AbstractEventLoop
from concurrent.futures import Future
import threading
from typing import Callable, Coroutine, Any, Optional, TypeVar
import logging  # Use logging instead of print for library code


T = TypeVar("T")
log = logging.getLogger(__name__)  # Setup logger


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
            log.debug("ThreadedEventLoop: No tasks to cancel.")
            return

        log.debug(f"ThreadedEventLoop: Cancelling {len(tasks_to_cancel)} task(s).")
        for task in tasks_to_cancel:
            task.cancel()

        # Wait for all tasks to complete after cancellation. This allows them to handle CancelledError.
        # Using a timeout to prevent hanging indefinitely if a task misbehaves.
        try:
            # results will contain exceptions for cancelled tasks (CancelledError)
            await asyncio.wait_for(
                asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=5.0
            )
            log.debug(
                "ThreadedEventLoop: All cancellable tasks finished after cancellation signal."
            )
        except asyncio.TimeoutError:
            log.warning(
                "ThreadedEventLoop: Timeout waiting for tasks to finish during shutdown. Some tasks may not have exited cleanly."
            )
        except Exception as e:
            log.error(
                f"ThreadedEventLoop: Error waiting for tasks during shutdown: {e}",
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
        log.debug("ThreadedEventLoop: Starting shutdown sequence in event loop thread.")

        # Ensure the task cancellation logic runs to completion before stopping the loop.
        # We create a task for _cancel_all_tasks_and_wait, and then ensure_future
        # will schedule it. After it completes, we stop the loop.
        async def shutdown_sequence_coro():
            await self._cancel_all_tasks_and_wait()
            log.debug("ThreadedEventLoop: Task cancellation complete, stopping loop.")
            if self._loop.is_running():  # Check again before stopping
                self._loop.stop()
            else:
                log.debug(
                    "ThreadedEventLoop: Loop was already stopped before explicit stop in shutdown_sequence."
                )

        asyncio.ensure_future(shutdown_sequence_coro(), loop=self._loop)

    def stop(self) -> None:
        """Stop the event loop and wait for the thread to finish."""
        if not self._running or not self._thread or self._stop_initiated:
            if self._stop_initiated and self._thread and not self._thread.is_alive():
                # If stop was initiated and thread is dead, we might be in a re-entrant call after cleanup
                log.debug(
                    "ThreadedEventLoop: Stop called but shutdown already completed or thread is dead."
                )
                return
            if not self._running and not self._thread:
                log.debug(
                    "ThreadedEventLoop: Stop called but loop was not running or thread doesn't exist."
                )
                return
            # If only _stop_initiated is true, but thread is alive, let join handle it.
            # log.debug(f"ThreadedEventLoop: Stop called, _running={self._running}, _thread_exists={bool(self._thread)}, _stop_initiated={self._stop_initiated}")

        log.debug(
            f"ThreadedEventLoop: Initiating stop. Current thread: {threading.get_ident()}, Loop thread: {self._thread.ident if self._thread else 'N/A'}"
        )
        self._running = False  # Signal that no new work should be accepted / loop should stop running tasks

        if self._loop and self._loop.is_running():
            log.debug(
                "ThreadedEventLoop: Scheduling _shutdown_loop via call_soon_threadsafe."
            )
            self._loop.call_soon_threadsafe(self._shutdown_loop)
        else:
            # If loop is not running, but thread exists, it implies _run_event_loop might have exited prematurely
            # or was never properly started. Forcing a stop on a non-running loop is mostly a no-op for stop itself.
            log.warning(
                "ThreadedEventLoop: Stop called, but internal loop was not reported as running. Cleanup will proceed."
            )

        current_thread_id = threading.get_ident()
        target_thread_id = self._thread.ident if self._thread else None

        if self._thread and current_thread_id != target_thread_id:
            log.debug(
                f"ThreadedEventLoop: Waiting for event loop thread {target_thread_id} to join."
            )
            self._thread.join(timeout=10.0)  # Timeout for join
            if self._thread.is_alive():
                log.warning(
                    "ThreadedEventLoop: Thread did not join in time after stop. Loop might be stuck or tasks are non-cooperative."
                )
        elif self._thread and current_thread_id == target_thread_id:
            log.warning(
                "ThreadedEventLoop: Stop called from within the event loop thread. Join will be skipped. Loop should stop via _shutdown_loop."
            )

        self._thread = None  # Clear thread reference after join attempt or if called from same thread
        self._stop_initiated = True  # Ensure flag is set post-join attempt too
        # Close the loop if it still exists and isn't closed
        if self._loop is not None:
            try:
                if not self._loop.is_closed():
                    self._loop.close()
            finally:
                self._loop = None
        log.debug("ThreadedEventLoop: Stop method finished.")

    def _run_event_loop(self) -> None:
        """Set the event loop for this thread and run it."""
        log.debug(
            f"ThreadedEventLoop: Event loop thread {threading.get_ident()} started."
        )
        if self._loop is None:
            # Should not happen; guard to avoid AttributeError in rare cases
            self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            log.debug(
                "ThreadedEventLoop: run_forever completed. Starting final cleanup in event loop thread."
            )
            # This part runs after run_forever() returns (i.e., after loop.stop() is processed from _shutdown_loop)
            # _shutdown_loop should have handled cancellation of tasks. This is a final sweep.
            try:
                # Ensure all tasks are truly finished if possible.
                all_tasks_final_sweep = asyncio.all_tasks(self._loop)
                # Exclude current task if any (though at this point, run_forever is done)
                tasks_to_gather_final = [
                    t for t in all_tasks_final_sweep if t is not asyncio.current_task()
                ]

                if tasks_to_gather_final:
                    log.debug(
                        f"ThreadedEventLoop: _run_event_loop finally: {len(tasks_to_gather_final)} tasks found. Gathering before close."
                    )
                    # This gather is to ensure any final exceptions/cancellations are processed by tasks
                    # before the loop is hard closed.
                    self._loop.run_until_complete(
                        asyncio.gather(*tasks_to_gather_final, return_exceptions=True)
                    )
                    log.debug(
                        "ThreadedEventLoop: _run_event_loop finally: Final gather completed."
                    )
            except RuntimeError as e:
                if (
                    "cannot call run_until_complete() event loop is already running"
                    in str(e)
                ):
                    log.warning(
                        "ThreadedEventLoop: _run_event_loop finally: Loop was unexpectedly still running during final cleanup sweep."
                    )
                elif "Event loop is closed" in str(e):
                    log.debug(
                        "ThreadedEventLoop: _run_event_loop finally: Loop was already closed during final task cleanup."
                    )
                else:
                    log.error(
                        f"ThreadedEventLoop: _run_event_loop finally: RuntimeError during final task cleanup: {e}",
                        exc_info=True,
                    )
            except Exception as e:
                log.error(
                    f"ThreadedEventLoop: _run_event_loop finally: Exception during final task cleanup: {e}",
                    exc_info=True,
                )

            if self._loop and not self._loop.is_closed():
                log.debug("ThreadedEventLoop: Closing event loop.")
                self._loop.close()
                log.debug("ThreadedEventLoop: Event loop closed.")
            else:
                log.debug(
                    "ThreadedEventLoop: Event loop was already closed before explicit close call in _run_event_loop."
                )
            log.debug(
                f"ThreadedEventLoop: Event loop thread {threading.get_ident()} finished."
            )

    def run_coroutine(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """Schedule a coroutine to run in this event loop."""
        if self._loop is None:
            raise RuntimeError("ThreadedEventLoop not started. Use start() or context manager.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore

    def run_in_executor(self, func: Callable[..., T], *args: Any) -> asyncio.Future[T]:
        """Run a synchronous function in the default executor of this event loop."""
        if self._loop is None:
            raise RuntimeError("ThreadedEventLoop not started. Use start() or context manager.")
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
            if self._loop is not None and not self._loop.is_closed():
                # If user never started the loop, there is no thread; safe to close.
                # If it was started but not stopped, closing here is best-effort.
                self._loop.close()
        except Exception:
            # Avoid raising during GC
            pass
        finally:
            self._loop = None
