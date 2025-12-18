import asyncio
import traceback


async def wrap_generators_parallel(*generators):
    """
    The Orchestra Conductor 🎭 - Processes multiple async generators concurrently

    This clever utility function allows multiple async generators to run in parallel,
    yielding results as they become available rather than waiting for each to complete.
    Think of it as a concert where each musician plays at their own pace, and you hear
    each note as it's played, not waiting for one musician to finish before the next begins.

    Args:
        *generators: A variable number of async generators to process simultaneously

    Yields:
        Any: Items from any generator as soon as they're produced, preserving concurrency

    Example:
        ```python
        async def slow_gen():
            for i in range(3):
                await asyncio.sleep(1)  # Takes its time
                yield f"Slow: {i}"

        async def fast_gen():
            for i in range(3):
                await asyncio.sleep(0.3)  # Speedy!
                yield f"Fast: {i}"

        # This will yield "Fast" items before "Slow" ones
        async for item in wrap_generators_parallel(slow_gen(), fast_gen()):
            print(item)
        ```
    """
    queue = asyncio.Queue()
    active_generators = len(generators)
    exceptions = []

    async def producer(gen):
        nonlocal active_generators
        try:
            async for item in gen:
                await queue.put(item)
        except Exception as e:
            exceptions.append(
                {
                    "exception": e,
                    "stack_trace": traceback.format_exc(),
                }
            )
        finally:
            active_generators -= 1
            await queue.put(None)  # Signal completion

    # Start all producers
    tasks = [asyncio.create_task(producer(gen)) for gen in generators]

    # Yield items as they come
    while active_generators > 0 or not queue.empty():
        item = await queue.get()
        if item is not None:
            yield item
        queue.task_done()

    # Wait for all tasks to complete
    await asyncio.wait(tasks)

    # Raise the first exception (preserve its type/message) if any were caught
    if exceptions:
        raise exceptions[0]["exception"]
