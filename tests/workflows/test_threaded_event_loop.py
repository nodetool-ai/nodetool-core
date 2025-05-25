import asyncio
from nodetool.workflows.threaded_event_loop import ThreadedEventLoop


async def add_one(x: int) -> int:
    await asyncio.sleep(0)
    return x + 1


def multiply(x: int) -> int:
    return x * 2


def test_run_coroutine_and_executor():
    with ThreadedEventLoop() as tel:
        result = tel.run_coroutine(add_one(1)).result(timeout=2)
        assert result == 2
    assert not tel.is_running
