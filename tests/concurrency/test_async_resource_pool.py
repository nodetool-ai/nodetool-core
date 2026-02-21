"""Tests for AsyncResourcePool."""

import asyncio
from datetime import timedelta

import pytest

from nodetool.concurrency import AsyncResourcePool
from nodetool.concurrency.async_resource_pool import (
    ResourcePoolClosedError,
    ResourcePoolError,
)


class MockResource:
    """A mock resource for testing."""

    def __init__(self, value: int = 0):
        self.value = value
        self.closed = False

    def close(self):
        """Close the resource."""
        self.closed = True

    async def aclose(self):
        """Async close the resource."""
        self.closed = True


@pytest.fixture
def sync_factory():
    """Factory that creates sync resources."""
    _counter = 0

    def factory():
        nonlocal _counter
        _counter += 1
        return MockResource(_counter)

    return factory


@pytest.fixture
def async_factory():
    """Factory that creates async resources."""
    _counter = 0

    async def factory():
        nonlocal _counter
        await asyncio.sleep(0.001)
        _counter += 1
        return MockResource(_counter)

    return factory


class TestAsyncResourcePoolBasics:
    """Tests for basic AsyncResourcePool functionality."""

    @pytest.mark.asyncio
    async def test_create_pool_defaults(self, async_factory):
        """Test creating a pool with default values."""
        pool = AsyncResourcePool(factory=async_factory)
        assert pool.max_size == 10
        assert pool.min_size == 0
        assert pool.size == 0
        assert pool.available_count == 0
        assert pool.in_use_count == 0
        assert not pool.closed

    @pytest.mark.asyncio
    async def test_create_pool_custom_sizes(self, async_factory):
        """Test creating a pool with custom sizes."""
        pool = AsyncResourcePool(factory=async_factory, max_size=5, min_size=2)
        assert pool.max_size == 5
        assert pool.min_size == 2

    @pytest.mark.asyncio
    async def test_invalid_max_size(self, async_factory):
        """Test that max_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            AsyncResourcePool(factory=async_factory, max_size=0)

    @pytest.mark.asyncio
    async def test_invalid_min_size_negative(self, async_factory):
        """Test that min_size < 0 raises ValueError."""
        with pytest.raises(ValueError, match="min_size must be non-negative"):
            AsyncResourcePool(factory=async_factory, min_size=-1)

    @pytest.mark.asyncio
    async def test_min_size_exceeds_max_size(self, async_factory):
        """Test that min_size > max_size raises ValueError."""
        with pytest.raises(ValueError, match="min_size cannot exceed max_size"):
            AsyncResourcePool(factory=async_factory, max_size=5, min_size=10)

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, async_factory):
        """Test basic acquire and release."""
        pool = AsyncResourcePool(factory=async_factory, max_size=5)
        resource = await pool.acquire()
        assert isinstance(resource, MockResource)
        assert pool.in_use_count == 1
        assert pool.available_count == 0

        await pool.release(resource)
        assert pool.in_use_count == 0
        assert pool.available_count == 1

    @pytest.mark.asyncio
    async def test_acquire_context_manager(self, async_factory):
        """Test using acquire_context as a context manager."""
        pool = AsyncResourcePool(factory=async_factory, max_size=5)

        async with pool.acquire_context() as resource:
            assert isinstance(resource, MockResource)
            assert pool.in_use_count == 1

        assert pool.in_use_count == 0
        assert pool.available_count == 1

    @pytest.mark.asyncio
    async def test_sync_factory(self, sync_factory):
        """Test pool with sync factory function."""
        pool = AsyncResourcePool(factory=sync_factory, max_size=5)

        resource = await pool.acquire()
        assert isinstance(resource, MockResource)
        assert resource.value == 1

        await pool.release(resource)

    @pytest.mark.asyncio
    async def test_resource_reuse(self, async_factory):
        """Test that resources are reused from the pool."""
        pool = AsyncResourcePool(factory=async_factory, max_size=5)

        resource1 = await pool.acquire()
        await pool.release(resource1)

        resource2 = await pool.acquire()
        assert resource2 is resource1  # Same resource instance
        await pool.release(resource2)

        assert pool.size == 1  # Only one resource created


class TestAsyncResourcePoolConcurrency:
    """Tests for concurrent access to the pool."""

    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self, async_factory):
        """Test multiple concurrent acquisitions."""
        pool = AsyncResourcePool(factory=async_factory, max_size=3)

        resources = await asyncio.gather(
            pool.acquire(), pool.acquire(), pool.acquire()
        )

        assert pool.size == 3
        assert pool.in_use_count == 3

        # Release all resources
        await asyncio.gather(
            pool.release(resources[0]),
            pool.release(resources[1]),
            pool.release(resources[2]),
        )

        assert pool.available_count == 3

    @pytest.mark.asyncio
    async def test_max_size_enforcement(self, async_factory):
        """Test that max_size limits concurrent acquisitions."""
        pool = AsyncResourcePool(
            factory=async_factory, max_size=2, acquisition_timeout=0.1
        )

        # Acquire max_size resources
        r1 = await pool.acquire()
        r2 = await pool.acquire()

        # Third acquisition should block and eventually timeout
        with pytest.raises(asyncio.TimeoutError):
            await pool.acquire(timeout=0.05)

        # Release and try again
        await pool.release(r1)
        r3 = await pool.acquire()

        assert pool.size == 2

        await pool.release(r2)
        await pool.release(r3)

    @pytest.mark.asyncio
    async def test_waiting_for_available_resource(self, async_factory):
        """Test that acquisitions wait when pool is at max_size."""
        pool = AsyncResourcePool(factory=async_factory, max_size=2)

        acquired_resources = []

        async def acquire_and_hold(duration: float):
            resource = await pool.acquire()
            acquired_resources.append(resource)
            await asyncio.sleep(duration)
            await pool.release(resource)

        # Start two tasks that hold resources
        task1 = asyncio.create_task(acquire_and_hold(0.1))
        task2 = asyncio.create_task(acquire_and_hold(0.1))

        await asyncio.sleep(0.01)  # Let both acquisitions happen

        # Third task should wait and eventually succeed
        await acquire_and_hold(0.01)

        await task1
        await task2

        assert pool.size == 2


class TestAsyncResourcePoolLifecycle:
    """Tests for resource lifecycle management."""

    @pytest.mark.asyncio
    async def test_closer_sync(self, sync_factory):
        """Test pool with sync closer function."""
        close_log = []

        def closer(resource: MockResource):
            close_log.append(resource)
            resource.close()

        pool = AsyncResourcePool(
            factory=sync_factory, closer=closer, max_size=5
        )

        resource = await pool.acquire()
        await pool.release(resource)

        # Prune to trigger closer
        await pool.prune()

        assert len(close_log) == 0  # Resource not pruned (no idle time)

    @pytest.mark.asyncio
    async def test_closer_async(self, async_factory):
        """Test pool with async closer function."""
        close_log = []

        async def closer(resource: MockResource):
            close_log.append(resource)
            await resource.aclose()

        pool = AsyncResourcePool(
            factory=async_factory, closer=closer, max_size=5
        )

        resource = await pool.acquire()
        await pool.release(resource)

        # Prune should not close immediately (no idle time set)
        pruned = await pool.prune()
        assert pruned == 0

    @pytest.mark.asyncio
    async def test_max_idle_time(self, async_factory):
        """Test max_idle_time prunes idle resources."""
        close_log = []

        async def closer(resource: MockResource):
            close_log.append(resource)

        pool = AsyncResourcePool(
            factory=async_factory,
            closer=closer,
            max_size=5,
            min_size=0,  # Allow pruning to 0
            max_idle_time=0.01,  # 10ms
        )

        resource = await pool.acquire()
        await pool.release(resource)

        assert pool.available_count == 1

        # Wait for resource to become idle
        await asyncio.sleep(0.02)

        # Prune to remove idle resources
        pruned = await pool.prune()
        assert pruned >= 1
        assert len(close_log) >= 1  # The old resource was closed

    @pytest.mark.asyncio
    async def test_max_age(self, async_factory):
        """Test max_age expires old resources."""
        close_log = []

        async def closer(resource: MockResource):
            close_log.append(resource)

        pool = AsyncResourcePool(
            factory=async_factory,
            closer=closer,
            max_size=5,
            max_age=0.01,  # 10ms
        )

        resource = await pool.acquire()
        await pool.release(resource)

        # Wait for resource to expire
        await asyncio.sleep(0.02)

        # Next acquire should find expired resource
        resource2 = await pool.acquire()
        await pool.release(resource2)

        # The old resource should have been expired
        assert len(close_log) >= 1

    @pytest.mark.asyncio
    async def test_prune(self, async_factory):
        """Test manual prune operation."""
        close_log = []

        async def closer(resource: MockResource):
            close_log.append(resource)

        pool = AsyncResourcePool(
            factory=async_factory,
            closer=closer,
            max_size=5,
            min_size=0,
            max_idle_time=0.01,
        )

        # Create some resources
        r1 = await pool.acquire()
        r2 = await pool.acquire()
        await pool.release(r1)
        await pool.release(r2)

        assert pool.available_count == 2

        # Wait for idle timeout
        await asyncio.sleep(0.02)

        # Prune should remove idle resources
        pruned = await pool.prune()
        assert pruned >= 1  # At least one should be pruned
        assert pool.available_count <= 1
        assert len(close_log) >= 1  # At least one was closed


class TestAsyncResourcePoolClose:
    """Tests for pool closing behavior."""

    @pytest.mark.asyncio
    async def test_close_pool(self, async_factory):
        """Test closing the pool."""
        close_log = []

        async def closer(resource: MockResource):
            close_log.append(resource)

        pool = AsyncResourcePool(
            factory=async_factory, closer=closer, max_size=5
        )

        # Create some resources
        r1 = await pool.acquire()
        r2 = await pool.acquire()
        await pool.release(r1)
        await pool.release(r2)

        await pool.close()

        assert pool.closed
        assert pool.in_use_count == 0
        assert len(close_log) >= 1  # r1 or r2 was closed

    @pytest.mark.asyncio
    async def test_acquire_from_closed_pool(self, async_factory):
        """Test that acquiring from closed pool waits for resources that won't come."""
        close_log = []

        async def closer(resource: MockResource):
            close_log.append(resource)

        pool = AsyncResourcePool(
            factory=async_factory, closer=closer, max_size=1
        )

        # Acquire the only resource
        r1 = await pool.acquire()

        # Close the pool (resources still in use won't be closed yet)
        close_task = asyncio.create_task(pool.close())

        # Acquiring from a closed pool with no resources will timeout
        # because close() waits for in-use resources to be returned first
        with pytest.raises(TimeoutError):
            await pool.acquire(timeout=0.1)

        # Release the resource so close() can complete
        await pool.release(r1)
        await close_task

        assert pool.closed

    @pytest.mark.asyncio
    async def test_context_manager_close(self, async_factory):
        """Test using pool as a context manager."""
        close_log = []

        async def closer(resource: MockResource):
            close_log.append(resource)

        async with AsyncResourcePool(
            factory=async_factory, closer=closer, max_size=5
        ) as pool:
            r1 = await pool.acquire()
            await pool.release(r1)

        assert pool.closed

    @pytest.mark.asyncio
    async def test_close_twice(self, async_factory):
        """Test closing pool twice."""
        pool = AsyncResourcePool(factory=async_factory, max_size=5)
        await pool.close()

        with pytest.raises(ResourcePoolClosedError):
            await pool.close()


class TestAsyncResourcePoolStats:
    """Tests for pool statistics."""

    @pytest.mark.asyncio
    async def test_stats_initial(self, async_factory):
        """Test initial stats."""
        pool = AsyncResourcePool(factory=async_factory, max_size=5)
        stats = pool.stats

        assert stats["size"] == 0
        assert stats["available"] == 0
        assert stats["in_use"] == 0
        assert stats["max_size"] == 5
        assert stats["min_size"] == 0
        assert stats["total_created"] == 0
        assert stats["total_acquired"] == 0
        assert stats["total_released"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_use(self, async_factory):
        """Test stats after acquiring and releasing."""
        pool = AsyncResourcePool(factory=async_factory, max_size=5)

        r1 = await pool.acquire()
        stats = pool.stats
        assert stats["size"] == 1
        assert stats["in_use"] == 1
        assert stats["total_created"] == 1
        assert stats["total_acquired"] == 1

        await pool.release(r1)
        stats = pool.stats
        assert stats["available"] == 1
        assert stats["total_released"] == 1

    @pytest.mark.asyncio
    async def test_stats_after_reuse(self, async_factory):
        """Test stats when reusing resources."""
        pool = AsyncResourcePool(factory=async_factory, max_size=5)

        r1 = await pool.acquire()
        await pool.release(r1)

        r2 = await pool.acquire()
        await pool.release(r2)

        stats = pool.stats
        assert stats["total_created"] == 1  # Only one resource created
        assert stats["total_acquired"] == 2  # Acquired twice
        assert stats["total_released"] == 2


class TestAsyncResourcePoolErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_release_untracked_resource(self, async_factory):
        """Test that releasing an untracked resource raises ValueError."""
        pool = AsyncResourcePool(factory=async_factory, max_size=5)

        fake_resource = MockResource()

        with pytest.raises(ValueError, match="not acquired from this pool"):
            await pool.release(fake_resource)

    @pytest.mark.asyncio
    async def test_release_twice(self, async_factory):
        """Test that releasing a resource twice raises ValueError."""
        pool = AsyncResourcePool(factory=async_factory, max_size=5)

        resource = await pool.acquire()
        await pool.release(resource)

        with pytest.raises(ValueError, match="not acquired from this pool"):
            await pool.release(resource)

    @pytest.mark.asyncio
    async def test_closer_exception_handling(self, async_factory):
        """Test that closer exceptions don't break the pool."""
        def bad_closer(resource):
            raise RuntimeError("Closer failed!")

        pool = AsyncResourcePool(
            factory=async_factory, closer=bad_closer, max_size=5
        )

        resource = await pool.acquire()

        # Should not raise even though closer fails
        await pool.release(resource)

        # Prune should also handle closer errors gracefully
        await pool.prune()

        assert pool.available_count == 1


class TestAsyncResourcePoolTimedelta:
    """Tests for timedelta parameter handling."""

    @pytest.mark.asyncio
    async def test_max_age_timedelta(self, async_factory):
        """Test max_age with timedelta parameter."""
        pool = AsyncResourcePool(
            factory=async_factory, max_size=5, max_age=timedelta(seconds=0.01)
        )

        resource = await pool.acquire()
        await pool.release(resource)

        await asyncio.sleep(0.02)

        resource2 = await pool.acquire()
        await pool.release(resource2)

        # Old resource should have been expired
        assert pool.stats["total_expired"] >= 1

    @pytest.mark.asyncio
    async def test_max_idle_time_timedelta(self, async_factory):
        """Test max_idle_time with timedelta parameter."""
        pool = AsyncResourcePool(
            factory=async_factory, max_size=5, max_idle_time=timedelta(milliseconds=10)
        )

        resource = await pool.acquire()
        await pool.release(resource)

        await asyncio.sleep(0.02)

        # Prune should remove the idle resource
        pruned = await pool.prune()
        assert pruned >= 1
