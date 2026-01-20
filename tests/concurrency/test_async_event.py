import asyncio

import pytest

from nodetool.concurrency import AsyncEvent, AsyncTaskGroup


class TestAsyncEvent:
    """Tests for AsyncEvent class."""

    def test_init_unset(self):
        """Test that event starts unset."""
        event = AsyncEvent()
        assert not event.is_set()
        assert event.waiters == 0

    def test_init_auto_reset(self):
        """Test auto-reset flag initialization."""
        event_auto = AsyncEvent(auto_reset=True)
        event_manual = AsyncEvent(auto_reset=False)

        assert event_auto._auto_reset is True
        assert event_manual._auto_reset is False

    def test_set_and_is_set(self):
        """Test setting the event."""
        event = AsyncEvent()
        assert not event.is_set()

        event.set()
        assert event.is_set()

        event.clear()
        assert not event.is_set()

    def test_set_with_value(self):
        """Test setting the event with a value."""
        event = AsyncEvent()

        event.set("test_value")
        assert event.is_set()

        async def get_value():
            return await event.wait()

        result = asyncio.run(get_value())
        assert result == "test_value"

    def test_clear(self):
        """Test clearing the event."""
        event = AsyncEvent()
        event.set("value")
        assert event.is_set()

        event.clear()
        assert not event.is_set()

    @pytest.mark.asyncio
    async def test_wait_blocks_until_set(self):
        """Test that wait blocks until event is set."""
        event = AsyncEvent()

        async def setter():
            await asyncio.sleep(0.1)
            event.set()

        task = asyncio.create_task(setter())
        await asyncio.sleep(0.05)
        assert not event.is_set()

        await event.wait()
        assert event.is_set()
        await task

    @pytest.mark.asyncio
    async def test_wait_returns_immediately_if_set(self):
        """Test that wait returns immediately if already set."""
        event = AsyncEvent()
        event.set()

        result = await event.wait()
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_with_value(self):
        """Test that wait returns the set value."""
        event = AsyncEvent()
        event.set("my_data")

        result = await event.wait()
        assert result == "my_data"

    @pytest.mark.asyncio
    async def test_multiple_waiters(self):
        """Test multiple tasks waiting on the same event."""
        event = AsyncEvent()
        results = []

        async def waiter(task_id):
            value = await event.wait()
            results.append((task_id, value))

        tasks = [
            asyncio.create_task(waiter(1)),
            asyncio.create_task(waiter(2)),
            asyncio.create_task(waiter(3)),
        ]

        await asyncio.sleep(0.05)
        assert event.waiters == 3

        event.set("done")

        await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 3
        assert all(r[1] == "done" for r in results)
        assert {r[0] for r in results} == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_wait_for_predicate(self):
        """Test wait_for with a predicate."""
        event = AsyncEvent()

        async def setter():
            event.set(1)
            await asyncio.sleep(0.05)
            event.set(2)
            await asyncio.sleep(0.05)
            event.set(3)

        setter_task = asyncio.create_task(setter())

        result = await event.wait_for(lambda v: v >= 2)
        assert result == 2

        await setter_task

    @pytest.mark.asyncio
    async def test_auto_reset(self):
        """Test auto-reset behavior."""
        event = AsyncEvent(auto_reset=True)
        results = []

        async def waiter(task_id):
            value = await event.wait()
            results.append((task_id, value))

        task1 = asyncio.create_task(waiter(1))
        await asyncio.sleep(0.01)

        event.set("first")
        await asyncio.sleep(0.05)

        assert len(results) == 1
        assert results[0] == (1, "first")
        assert not event.is_set()
        await task1

        task2 = asyncio.create_task(waiter(2))
        await asyncio.sleep(0.01)

        event.set("second")
        await asyncio.sleep(0.05)

        assert len(results) == 2
        assert results[1] == (2, "second")
        assert not event.is_set()
        await task2

    @pytest.mark.asyncio
    async def test_manual_reset_persists(self):
        """Test that manual reset keeps event set."""
        event = AsyncEvent(auto_reset=False)
        results = []

        async def waiter(task_id):
            value = await event.wait()
            results.append((task_id, value))

        task1 = asyncio.create_task(waiter(1))

        await asyncio.sleep(0.05)
        event.set("value")

        await task1

        assert results[0][1] == "value"
        assert event.is_set()

        task2 = asyncio.create_task(waiter(2))
        await asyncio.sleep(0.05)

        assert task2.done()
        assert results[1][1] == "value"

    @pytest.mark.asyncio
    async def test_set_after_waiters_arrive(self):
        """Test that set wakes up waiting tasks."""
        event = AsyncEvent()
        woke_up = False

        async def waiter():
            nonlocal woke_up
            await event.wait()
            woke_up = True

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        assert not woke_up

        event.set()
        await asyncio.sleep(0.05)

        assert woke_up
        assert task.done()

    @pytest.mark.asyncio
    async def test_clear_between_waiters(self):
        """Test that clear works correctly with multiple waiters."""
        event = AsyncEvent()
        event.set("first")

        async def waiter():
            return await event.wait()

        task = asyncio.create_task(waiter())

        await asyncio.sleep(0.05)
        assert task.done()

        event.clear()

        async def second_waiter():
            return await event.wait()

        task2 = asyncio.create_task(second_waiter())
        await asyncio.sleep(0.05)

        assert not task2.done()
        event.set("second")
        result = await task2
        assert result == "second"

    @pytest.mark.asyncio
    async def test_integration_with_task_group(self):
        """Test AsyncEvent works well with AsyncTaskGroup."""
        event = AsyncEvent()
        results = []

        async def producer():
            await asyncio.sleep(0.1)
            event.set("data_1")
            await asyncio.sleep(0.05)
            event.set("data_2")

        async def consumer(task_id):
            value = await event.wait()
            results.append((task_id, value))

        group = AsyncTaskGroup()
        group.spawn("producer", producer())
        group.spawn("c1", consumer("c1"))

        await group.run()

        assert len(results) == 1
        assert results[0][1] == "data_1"

    @pytest.mark.asyncio
    async def test_waiters_count(self):
        """Test waiters property returns correct count."""
        event = AsyncEvent()

        async def waiter():
            await event.wait()

        assert event.waiters == 0

        task1 = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        assert event.waiters == 1

        task2 = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        assert event.waiters == 2

        event.set()

        await asyncio.sleep(0.01)

        for task in [task1, task2]:
            task.cancel()
