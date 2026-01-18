import asyncio

import pytest

from nodetool.concurrency.object_pool import (
    AsyncObjectPool,
    ObjectPoolError,
    PoolAcquireTimeoutError,
    PoolClosedError,
    pooled,
)


async def simple_factory():
    """Simple async factory for testing."""
    return object()


class TestAsyncObjectPool:
    """Tests for AsyncObjectPool class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        pool = AsyncObjectPool(factory=simple_factory, max_size=5)
        assert pool.max_size == 5
        assert pool.available == 5

    def test_init_with_invalid_max_size(self):
        """Test that invalid max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncObjectPool(factory=simple_factory, max_size=0)

        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncObjectPool(factory=simple_factory, max_size=-1)

    def test_init_with_invalid_initial_size(self):
        """Test that invalid initial_size raises ValueError."""
        with pytest.raises(ValueError, match="initial_size must be non-negative"):
            AsyncObjectPool(factory=simple_factory, max_size=5, initial_size=-1)

        with pytest.raises(ValueError, match="initial_size cannot exceed max_size"):
            AsyncObjectPool(factory=simple_factory, max_size=5, initial_size=10)

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        """Test basic acquire and release functionality."""
        created = []

        async def factory():
            obj = object()
            created.append(obj)
            return obj

        pool = AsyncObjectPool(factory=factory, max_size=2)

        obj1 = await pool.acquire()
        assert len(created) == 1
        assert obj1 in created

        obj2 = await pool.acquire()
        assert len(created) == 2
        assert obj2 in created

        pool.release(obj1)
        assert pool.available == 1

        pool.release(obj2)
        assert pool.available == 2

    @pytest.mark.asyncio
    async def test_acquire_no_timeout(self):
        """Test acquire without timeout waits indefinitely."""
        created = []

        async def factory():
            await asyncio.sleep(0.1)
            obj = object()
            created.append(obj)
            return obj

        pool = AsyncObjectPool(factory=factory, max_size=1)

        obj1 = await pool.acquire()
        assert len(created) == 1

        acquire_task = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0.05)
        assert not acquire_task.done()

        pool.release(obj1)
        obj2 = await acquire_task
        assert obj2 in created

    @pytest.mark.asyncio
    async def test_acquire_with_timeout_success(self):
        """Test acquire with timeout succeeds when object available."""

        async def factory():
            await asyncio.sleep(0.05)
            return object()

        pool = AsyncObjectPool(factory=factory, max_size=1)

        obj = await pool.acquire(timeout=1.0)
        assert obj is not None

    @pytest.mark.asyncio
    async def test_acquire_with_timeout_expires(self):
        """Test acquire with timeout expires when pool is exhausted."""

        async def factory():
            await asyncio.sleep(0.1)
            return object()

        pool = AsyncObjectPool(factory=factory, max_size=1)

        obj1 = await pool.acquire()
        assert obj1 is not None

        with pytest.raises(PoolAcquireTimeoutError):
            await pool.acquire(timeout=0.2)

    @pytest.mark.asyncio
    async def test_acquire_with_zero_timeout(self):
        """Test acquire with zero timeout returns immediately if available."""

        async def factory():
            return object()

        pool = AsyncObjectPool(factory=factory, max_size=1)

        obj = await pool.acquire(timeout=0)
        assert obj is not None
        assert pool.available == 0

        result = await pool.acquire(timeout=0)
        assert result is None

    @pytest.mark.asyncio
    async def test_validator(self):
        """Test that validator is called on acquire."""
        validated = []

        async def factory():
            return object()

        async def validator(obj):
            validated.append(obj)
            return True

        pool = AsyncObjectPool(factory=factory, validator=validator, max_size=1)

        obj = await pool.acquire()
        assert obj in validated

    @pytest.mark.asyncio
    async def test_validator_replaces_invalid(self):
        """Test that invalid objects are replaced by factory."""
        created = []
        validated = []
        closed = []

        async def factory():
            obj = object()
            created.append(obj)
            return obj

        async def validator(obj):
            validated.append(obj)
            return obj not in created[1:]

        async def destructor(obj):
            closed.append(obj)

        pool = AsyncObjectPool(
            factory=factory,
            validator=validator,
            destructor=destructor,
            max_size=1,
        )

        obj1 = await pool.acquire()
        assert len(created) == 1

        obj2 = await pool.acquire()
        assert len(created) == 2
        assert obj2 in created
        assert obj1 in closed

    @pytest.mark.asyncio
    async def test_close(self):
        """Test pool close method."""
        closed = []

        async def factory():
            return object()

        async def destructor(obj):
            closed.append(obj)

        pool = AsyncObjectPool(
            factory=factory,
            destructor=destructor,
            max_size=2,
        )

        obj1 = await pool.acquire()
        obj2 = await pool.acquire()

        await pool.close()

        assert obj1 in closed
        assert obj2 in closed

        with pytest.raises(PoolClosedError):
            await pool.acquire()

    @pytest.mark.asyncio
    async def test_close_then_acquire(self):
        """Test that acquiring from closed pool raises error."""

        async def factory():
            return object()

        pool = AsyncObjectPool(factory=factory, max_size=1)
        await pool.close()

        with pytest.raises(PoolClosedError):
            await pool.acquire()

    @pytest.mark.asyncio
    async def test_release_to_closed_pool(self):
        """Test that releasing to closed pool closes the object."""
        closed = []

        async def factory():
            return object()

        async def destructor(obj):
            closed.append(obj)

        pool = AsyncObjectPool(
            factory=factory,
            destructor=destructor,
            max_size=1,
        )

        obj = await pool.acquire()
        await pool.close()

        pool.release(obj)
        assert obj in closed

    @pytest.mark.asyncio
    async def test_prewarm(self):
        """Test prewarm method."""
        created = []

        async def factory():
            await asyncio.sleep(0.01)
            obj = object()
            created.append(obj)
            return obj

        pool = AsyncObjectPool(factory=factory, max_size=5, initial_size=0)

        count = await pool.prewarm(3)
        assert count == 3
        assert len(created) == 3

    @pytest.mark.asyncio
    async def test_prewarm_partial(self):
        """Test prewarm with max_size limit."""
        created = []

        async def factory():
            await asyncio.sleep(0.01)
            obj = object()
            created.append(obj)
            return obj

        pool = AsyncObjectPool(factory=factory, max_size=3, initial_size=0)

        count = await pool.prewarm(10)
        assert count == 3
        assert len(created) == 3

    @pytest.mark.asyncio
    async def test_try_acquire(self):
        """Test try_acquire method."""

        async def factory():
            return object()

        pool = AsyncObjectPool(factory=factory, max_size=1)

        obj = pool.try_acquire()
        assert obj is not None
        assert pool.available == 0

        obj = pool.try_acquire()
        assert obj is None

    @pytest.mark.asyncio
    async def test_try_acquire_after_close(self):
        """Test try_acquire returns None after pool is closed."""

        async def factory():
            return object()

        pool = AsyncObjectPool(factory=factory, max_size=1)
        await pool.close()

        obj = pool.try_acquire()
        assert obj is None

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self):
        """Test concurrent acquire from pool."""
        acquired = []
        max_concurrent = 3

        async def factory():
            await asyncio.sleep(0.05)
            return object()

        pool = AsyncObjectPool(factory=factory, max_size=max_concurrent)

        async def acquire_and_release():
            obj = await pool.acquire()
            try:
                acquired.append(obj)
                await asyncio.sleep(0.1)
            finally:
                pool.release(obj)

        tasks = [acquire_and_release() for _ in range(6)]
        await asyncio.gather(*tasks)

        assert len(acquired) == 6

    @pytest.mark.asyncio
    async def test_context_manager_exit_pool(self):
        """Test pool context manager."""
        created = []

        async def factory():
            obj = object()
            created.append(obj)
            return obj

        pool = AsyncObjectPool(factory=factory, max_size=2)

        async with pool:
            assert pool.available == 2

        assert pool.available == 2

    @pytest.mark.asyncio
    async def test_destructor_called_on_close(self):
        """Test that destructor is called when pool is closed."""
        closed = []

        async def factory():
            return object()

        async def destructor(obj):
            closed.append(obj)

        pool = AsyncObjectPool(
            factory=factory,
            destructor=destructor,
            max_size=2,
        )

        obj1 = await pool.acquire()
        obj2 = await pool.acquire()

        await pool.close()

        assert obj1 in closed
        assert obj2 in closed


class TestPooledFunction:
    """Tests for the pooled function."""

    @pytest.mark.asyncio
    async def test_pooled_basic_usage(self):
        """Test basic pooled function usage."""
        created = []

        async def factory():
            await asyncio.sleep(0.01)
            obj = object()
            created.append(obj)
            return obj

        pool = AsyncObjectPool(factory=factory, max_size=1)

        async with pooled(pool) as obj:
            assert obj in created

        assert pool.available == 1

    @pytest.mark.asyncio
    async def test_pooled_with_timeout(self):
        """Test pooled function with timeout."""

        async def factory():
            await asyncio.sleep(0.05)
            return object()

        pool = AsyncObjectPool(factory=factory, max_size=1)

        obj1 = await pool.acquire()

        with pytest.raises(PoolAcquireTimeoutError):
            async with pooled(pool, timeout=0.1):
                pass

        pool.release(obj1)

    @pytest.mark.asyncio
    async def test_pooled_exception_releases(self):
        """Test that pooled function releases on exception."""
        created = []

        async def factory():
            obj = object()
            created.append(obj)
            return obj

        pool = AsyncObjectPool(factory=factory, max_size=1)

        with pytest.raises(ValueError):
            async with pooled(pool) as obj:
                assert obj in created
                raise ValueError("test error")

        assert pool.available == 1


class TestPoolExceptions:
    """Tests for pool exception classes."""

    def test_pool_closed_error(self):
        """Test PoolClosedError exception."""
        with pytest.raises(PoolClosedError):
            raise PoolClosedError("Pool is closed")

    def test_pool_acquire_timeout_error(self):
        """Test PoolAcquireTimeoutError exception."""
        with pytest.raises(PoolAcquireTimeoutError):
            raise PoolAcquireTimeoutError("Timeout")

    def test_object_pool_error(self):
        """Test ObjectPoolError as base class."""
        with pytest.raises(ObjectPoolError):
            raise ObjectPoolError("Base error")


class TestPoolRepr:
    """Tests for pool string representation."""

    def test_repr_open(self):
        """Test repr of open pool."""
        pool = AsyncObjectPool(factory=simple_factory, max_size=5)
        repr_str = repr(pool)
        assert "AsyncObjectPool" in repr_str
        assert "max_size=5" in repr_str
        assert "status=open" in repr_str

    @pytest.mark.asyncio
    async def test_repr_closed(self):
        """Test repr of closed pool."""
        pool = AsyncObjectPool(factory=simple_factory, max_size=5)
        await pool.close()
        repr_str = repr(pool)
        assert "status=closed" in repr_str
