import asyncio

import pytest

from nodetool.concurrency.async_condition import AsyncCondition


class TestAsyncCondition:
    """Tests for AsyncCondition class."""

    def test_init_default_lock(self):
        """Test that condition initializes with a default lock."""
        condition = AsyncCondition()
        assert not condition.locked()
        assert condition.waiters == 0
        assert isinstance(condition.lock, asyncio.Lock)

    def test_init_custom_lock(self):
        """Test that condition can be initialized with a custom lock."""
        custom_lock = asyncio.Lock()
        condition = AsyncCondition(lock=custom_lock)
        assert condition.lock is custom_lock

    def test_repr(self):
        """Test string representation."""
        condition = AsyncCondition()
        assert repr(condition) == "AsyncCondition(unlocked, waiters=0)"

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test basic lock acquire and release."""
        condition = AsyncCondition()

        assert not condition.locked()

        await condition.acquire()
        assert condition.locked()

        condition.release()
        assert not condition.locked()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using condition as a context manager."""
        condition = AsyncCondition()

        async with condition:
            assert condition.locked()

        assert not condition.locked()

    @pytest.mark.asyncio
    async def test_basic_notify(self):
        """Test basic notify wakes one waiter at a time."""
        condition = AsyncCondition()
        state = {"count": 0}
        woke_up = []

        async def waiter(task_id: int):
            async with condition:
                while state["count"] < 2:
                    await condition.wait()
                woke_up.append(task_id)

        async def notifier():
            await asyncio.sleep(0.05)
            async with condition:
                state["count"] += 1
                condition.notify()

        # Start two waiters and two notifiers
        task1 = asyncio.create_task(waiter(1))
        task2 = asyncio.create_task(waiter(2))
        notifier_task1 = asyncio.create_task(notifier())
        notifier_task2 = asyncio.create_task(notifier())

        await asyncio.gather(task1, task2, notifier_task1, notifier_task2)

        # Both should have woken up
        assert len(woke_up) == 2
        assert set(woke_up) == {1, 2}

    @pytest.mark.asyncio
    async def test_notify_all(self):
        """Test notify_all wakes all waiters."""
        condition = AsyncCondition()
        state = {"ready": False}
        woke_up = []

        async def waiter(task_id: int):
            async with condition:
                while not state["ready"]:
                    await condition.wait()
                woke_up.append(task_id)

        async def notifier():
            await asyncio.sleep(0.05)
            async with condition:
                state["ready"] = True
                condition.notify_all()

        # Start multiple waiters
        tasks = [asyncio.create_task(waiter(i)) for i in range(5)]
        notifier_task = asyncio.create_task(notifier())

        await asyncio.gather(*tasks, notifier_task)

        # All should have woken up
        assert len(woke_up) == 5
        assert set(woke_up) == {0, 1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_wait_for_predicate(self):
        """Test wait_for waits until predicate is true."""
        condition = AsyncCondition()
        state = {"data": 0}

        async def waiter():
            async with condition:
                await condition.wait_for(lambda: state["data"] >= 10)

        async def producer():
            await asyncio.sleep(0.05)
            async with condition:
                for _ in range(10):
                    state["data"] += 1
                    condition.notify_all()

        # Start tasks
        waiter_task = asyncio.create_task(waiter())
        producer_task = asyncio.create_task(producer())

        await waiter_task
        await producer_task

        assert state["data"] == 10

    @pytest.mark.asyncio
    async def test_wait_for_immediate_success(self):
        """Test wait_for returns immediately if predicate is already true."""
        condition = AsyncCondition()
        state = {"ready": True}

        async def waiter():
            async with condition:
                await condition.wait_for(lambda: state["ready"])

        await waiter()

    @pytest.mark.asyncio
    async def test_wait_for_timeout_success(self):
        """Test wait_for_timeout returns True when predicate becomes true."""
        condition = AsyncCondition()
        state = {"ready": False}

        async def waiter():
            async with condition:
                result = await condition.wait_for_timeout(lambda: state["ready"], 1.0)
                return result

        async def setter():
            await asyncio.sleep(0.1)
            async with condition:
                state["ready"] = True
                condition.notify_all()

        waiter_task = asyncio.create_task(waiter())
        setter_task = asyncio.create_task(setter())

        result = await waiter_task
        await setter_task
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_timeout_failure(self):
        """Test wait_for_timeout returns False on timeout."""
        condition = AsyncCondition()
        state = {"ready": False}

        async def waiter():
            async with condition:
                result = await condition.wait_for_timeout(lambda: state["ready"], 0.1)
                return result

        result = await waiter()
        assert result is False

    @pytest.mark.asyncio
    async def test_producer_consumer_pattern(self):
        """Test classic producer-consumer pattern with condition variable."""
        condition = AsyncCondition()
        buffer: list[int] = []
        max_size = 5
        state = {"produced_count": 0, "consumed_count": 0}

        async def producer(id: int, items: int):
            for i in range(items):
                async with condition:
                    await condition.wait_for(lambda: len(buffer) < max_size)
                    buffer.append(id * 100 + i)
                    state["produced_count"] += 1
                    condition.notify_all()

        async def consumer(id: int, items: int):
            for _ in range(items):
                async with condition:
                    await condition.wait_for(lambda: len(buffer) > 0)
                    buffer.pop(0)
                    state["consumed_count"] += 1
                    condition.notify_all()

        # Run producers and consumers
        tasks = [
            asyncio.create_task(producer(1, 10)),
            asyncio.create_task(producer(2, 10)),
            asyncio.create_task(consumer(1, 10)),
            asyncio.create_task(consumer(2, 10)),
        ]

        await asyncio.gather(*tasks)

        assert state["produced_count"] == 20
        assert state["consumed_count"] == 20
        assert len(buffer) == 0

    @pytest.mark.asyncio
    async def test_waiters_property(self):
        """Test waiters property returns accurate count."""
        condition = AsyncCondition()

        async def waiter():
            async with condition:
                await condition.wait()

        # Start waiters without notifying
        tasks = [asyncio.create_task(waiter()) for _ in range(3)]

        # Give them time to start waiting
        await asyncio.sleep(0.05)

        assert condition.waiters == 3

        # Notify all
        async with condition:
            condition.notify_all()

        await asyncio.gather(*tasks)

        assert condition.waiters == 0

    @pytest.mark.asyncio
    async def test_shared_lock_multiple_conditions(self):
        """Test multiple conditions sharing a lock."""
        shared_lock = asyncio.Lock()
        condition1 = AsyncCondition(lock=shared_lock)
        condition2 = AsyncCondition(lock=shared_lock)

        state = {"ready1": False, "ready2": False}
        results = []

        async def waiter1():
            async with condition1:
                await condition1.wait_for(lambda: state["ready1"])
                results.append("waiter1")

        async def waiter2():
            async with condition2:
                await condition2.wait_for(lambda: state["ready2"])
                results.append("waiter2")

        async def setter():
            await asyncio.sleep(0.05)
            async with condition1:
                state["ready1"] = True
                condition1.notify()
            await asyncio.sleep(0.05)
            async with condition2:
                state["ready2"] = True
                condition2.notify()

        await asyncio.gather(waiter1(), waiter2(), setter())

        assert results == ["waiter1", "waiter2"]

    @pytest.mark.asyncio
    async def test_notify_multiple_waiters(self):
        """Test notifying multiple waiters at once."""
        condition = AsyncCondition()
        state = {"ready": False}
        woke_up = []

        async def waiter(task_id: int):
            async with condition:
                while not state["ready"]:
                    await condition.wait()
                woke_up.append(task_id)

        # Start 5 waiters
        tasks = [asyncio.create_task(waiter(i)) for i in range(5)]
        await asyncio.sleep(0.05)

        # Notify 3 at a time
        async with condition:
            state["ready"] = True
            condition.notify(3)

        await asyncio.sleep(0.05)
        assert len(woke_up) == 3

        # Notify the remaining 2
        async with condition:
            condition.notify(2)

        await asyncio.gather(*tasks)

        assert len(woke_up) == 5
        assert set(woke_up) == {0, 1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_condition_with_cancellation(self):
        """Test that condition handles task cancellation gracefully."""
        condition = AsyncCondition()

        async def waiter():
            async with condition:
                await condition.wait()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)

        # Cancel the waiting task
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Condition should still be usable
        async with condition:
            condition.notify_all()

        assert condition.waiters == 0
