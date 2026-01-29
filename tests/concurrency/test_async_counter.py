import asyncio

import pytest

from nodetool.concurrency.async_counter import AsyncCounter


class TestAsyncCounter:
    """Tests for AsyncCounter class."""

    def test_init_default(self):
        """Test that counter starts at 0 by default."""
        counter = AsyncCounter()
        assert counter.value == 0

    def test_init_with_value(self):
        """Test that counter starts at specified value."""
        counter = AsyncCounter(initial_value=5)
        assert counter.value == 5

    def test_repr(self):
        """Test string representation."""
        counter = AsyncCounter(initial_value=10)
        assert repr(counter) == "AsyncCounter(value=10)"

    def test_value_property(self):
        """Test that value property returns current value."""
        counter = AsyncCounter(initial_value=42)
        assert counter.value == 42

    @pytest.mark.asyncio
    async def test_increment(self):
        """Test incrementing the counter."""
        counter = AsyncCounter()

        new_value = await counter.increment()
        assert new_value == 1
        assert counter.value == 1

    @pytest.mark.asyncio
    async def test_increment_by_amount(self):
        """Test incrementing the counter by a specific amount."""
        counter = AsyncCounter()

        new_value = await counter.increment(5)
        assert new_value == 5
        assert counter.value == 5

    @pytest.mark.asyncio
    async def test_decrement(self):
        """Test decrementing the counter."""
        counter = AsyncCounter(initial_value=5)

        new_value = await counter.decrement()
        assert new_value == 4
        assert counter.value == 4

    @pytest.mark.asyncio
    async def test_decrement_by_amount(self):
        """Test decrementing the counter by a specific amount."""
        counter = AsyncCounter(initial_value=10)

        new_value = await counter.decrement(3)
        assert new_value == 7
        assert counter.value == 7

    @pytest.mark.asyncio
    async def test_add_positive(self):
        """Test adding a positive amount."""
        counter = AsyncCounter(initial_value=5)

        new_value = await counter.add(10)
        assert new_value == 15
        assert counter.value == 15

    @pytest.mark.asyncio
    async def test_add_negative(self):
        """Test adding a negative amount (subtraction)."""
        counter = AsyncCounter(initial_value=10)

        new_value = await counter.add(-3)
        assert new_value == 7
        assert counter.value == 7

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting the counter."""
        counter = AsyncCounter(initial_value=100)

        counter.reset(0)
        assert counter.value == 0

        counter.reset(50)
        assert counter.value == 50

    @pytest.mark.asyncio
    async def test_reset_to_zero(self):
        """Test resetting the counter to zero."""
        counter = AsyncCounter(initial_value=42)
        counter.reset()
        assert counter.value == 0

    @pytest.mark.asyncio
    async def test_get_and_increment(self):
        """Test get_and_increment returns old value."""
        counter = AsyncCounter(initial_value=10)

        old_value = await counter.get_and_increment()
        assert old_value == 10
        assert counter.value == 11

    @pytest.mark.asyncio
    async def test_get_and_increment_multiple(self):
        """Test get_and_increment generates sequential values."""
        counter = AsyncCounter()

        values = [await counter.get_and_increment() for _ in range(5)]
        assert values == [0, 1, 2, 3, 4]
        assert counter.value == 5

    @pytest.mark.asyncio
    async def test_get_and_set(self):
        """Test get_and_set returns old value and sets new one."""
        counter = AsyncCounter(initial_value=10)

        old_value = await counter.get_and_set(42)
        assert old_value == 10
        assert counter.value == 42

    @pytest.mark.asyncio
    async def test_concurrent_increments(self):
        """Test that concurrent increments are thread-safe."""
        counter = AsyncCounter()

        async def increment_many():
            for _ in range(100):
                await counter.increment()

        await asyncio.gather(*[increment_many() for _ in range(10)])
        assert counter.value == 1000

    @pytest.mark.asyncio
    async def test_concurrent_decrements(self):
        """Test that concurrent decrements are thread-safe."""
        counter = AsyncCounter(initial_value=1000)

        async def decrement_many():
            for _ in range(100):
                await counter.decrement()

        await asyncio.gather(*[decrement_many() for _ in range(10)])
        assert counter.value == 0

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """Test that concurrent mixed operations are thread-safe."""
        counter = AsyncCounter()

        async def mixed_operations():
            for _ in range(50):
                await counter.increment()
                await counter.increment(2)
                await counter.decrement()

        await asyncio.gather(*[mixed_operations() for _ in range(4)])
        assert counter.value == 400

    @pytest.mark.asyncio
    async def test_tracking_concurrent_tasks(self):
        """Test using counter to track concurrent task count."""
        counter = AsyncCounter()
        max_concurrent = 0
        lock = asyncio.Lock()

        async def task():
            await counter.increment()
            try:
                async with lock:
                    nonlocal max_concurrent
                    max_concurrent = max(max_concurrent, counter.value)
                await asyncio.sleep(0.01)
            finally:
                await counter.decrement()

        await asyncio.gather(*[task() for _ in range(10)])
        assert max_concurrent == 10
        assert counter.value == 0

    @pytest.mark.asyncio
    async def test_sequence_number_generation(self):
        """Test using get_and_increment for sequence numbers."""
        counter = AsyncCounter()

        async def get_next_id():
            return await counter.get_and_increment()

        ids = await asyncio.gather(*[get_next_id() for _ in range(100)])
        assert ids == list(range(100))
        assert counter.value == 100
