"""
Tests for AsyncPool - async resource pooling functionality.
"""

import asyncio

import pytest

from nodetool.concurrency import AsyncPool


class MockResource:
    """A simple mock resource for testing."""

    def __init__(self, id: int):
        self.id = id
        self.closed = False

    def close(self):
        """Mark the resource as closed."""
        self.closed = True

    async def aclose(self):
        """Async version of close."""
        self.closed = True


class TestAsyncPoolBasics:
    """Test basic pool creation and initialization."""

    @pytest.mark.asyncio
    async def test_pool_creation(self):
        """Test creating a pool with valid parameters."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=5,
            initial_size=0,
        )
        assert pool.max_size == 5
        assert pool.size == 0
        assert pool.available == 0
        assert pool.in_use == 0

    @pytest.mark.asyncio
    async def test_pool_invalid_max_size(self):
        """Test that max_size must be positive."""
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncPool(factory=lambda: MockResource(1), max_size=0)

        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            AsyncPool(factory=lambda: MockResource(1), max_size=-1)

    @pytest.mark.asyncio
    async def test_pool_invalid_initial_size(self):
        """Test that initial_size must be between 0 and max_size."""
        with pytest.raises(ValueError, match="initial_size must be between 0 and max_size"):
            AsyncPool(factory=lambda: MockResource(1), max_size=5, initial_size=10)

        with pytest.raises(ValueError, match="initial_size must be between 0 and max_size"):
            AsyncPool(factory=lambda: MockResource(1), max_size=5, initial_size=-1)

    @pytest.mark.asyncio
    async def test_pool_invalid_factory(self):
        """Test that factory must be callable."""
        with pytest.raises(TypeError, match="factory must be callable"):
            AsyncPool(factory=None, max_size=5)

    @pytest.mark.asyncio
    async def test_pool_initialization(self):
        """Test creating initial resources."""
        creation_count = [0]

        def create_resource():
            creation_count[0] += 1
            return MockResource(creation_count[0])

        pool = AsyncPool(
            factory=create_resource,
            max_size=5,
            initial_size=3,
        )

        await pool._initialize()

        assert pool.size == 3
        assert pool.available == 3
        assert pool.in_use == 0
        assert creation_count[0] == 3

    @pytest.mark.asyncio
    async def test_pool_context_manager_initialization(self):
        """Test using pool as context manager initializes resources."""
        creation_count = [0]

        def create_resource():
            creation_count[0] += 1
            return MockResource(creation_count[0])

        async with AsyncPool(
            factory=create_resource,
            max_size=5,
            initial_size=2,
        ) as pool:
            assert pool.size == 2
            assert pool.available == 2


class TestAsyncPoolAcquireRelease:
    """Test resource acquisition and release."""

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        """Test basic acquire and release cycle."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=3,
        )

        # Acquire a resource
        ctx = await pool.acquire()
        assert pool.size == 1
        assert pool.available == 0
        assert pool.in_use == 1

        resource = ctx.resource
        assert isinstance(resource, MockResource)

        # Release it back
        await ctx.release()
        assert pool.size == 1
        assert pool.available == 1
        assert pool.in_use == 0

    @pytest.mark.asyncio
    async def test_acquire_with_context_manager(self):
        """Test using acquire context manager."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=3,
        )

        async with await pool.acquire() as resource:
            assert pool.in_use == 1
            assert isinstance(resource, MockResource)

        assert pool.available == 1
        assert pool.in_use == 0

    @pytest.mark.asyncio
    async def test_multiple_acquisitions(self):
        """Test acquiring multiple resources."""
        counter = [0]

        def create_resource():
            counter[0] += 1
            return MockResource(counter[0])

        pool = AsyncPool(
            factory=create_resource,
            max_size=5,
        )

        resources = []
        for _ in range(3):
            ctx = await pool.acquire()
            resources.append(ctx)

        assert pool.size == 3
        assert pool.in_use == 3
        assert pool.available == 0
        assert counter[0] == 3

        # Release them all
        for ctx in resources:
            await ctx.release()

        assert pool.in_use == 0
        assert pool.available == 3

    @pytest.mark.asyncio
    async def test_resource_reuse(self):
        """Test that resources are reused when available."""
        resources_created = []

        def create_resource():
            resource = MockResource(len(resources_created) + 1)
            resources_created.append(resource)
            return resource

        pool = AsyncPool(
            factory=create_resource,
            max_size=3,
        )

        # Acquire and release
        ctx1 = await pool.acquire()
        resource1 = ctx1.resource
        await ctx1.release()

        # Acquire again - should get the same resource
        ctx2 = await pool.acquire()
        resource2 = ctx2.resource

        assert resource1 is resource2
        assert len(resources_created) == 1

    @pytest.mark.asyncio
    async def test_acquire_async_factory(self):
        """Test pool with async factory function."""

        async def create_resource():
            await asyncio.sleep(0.01)
            return MockResource(1)

        pool = AsyncPool(
            factory=create_resource,
            max_size=2,
        )

        ctx = await pool.acquire()
        assert isinstance(ctx.resource, MockResource)
        await ctx.release()

    @pytest.mark.asyncio
    async def test_acquire_async_closer(self):
        """Test pool with async closer function."""
        closed_resources = []

        async def close_resource(resource: MockResource):
            await asyncio.sleep(0.01)
            resource.closed = True
            closed_resources.append(resource)

        pool = AsyncPool(
            factory=lambda: MockResource(1),
            closer=close_resource,
            max_size=2,
        )

        # Acquire and release
        ctx = await pool.acquire()
        resource = ctx.resource
        await ctx.release()

        # Close pool
        await pool.close()

        assert resource in closed_resources


class TestAsyncPoolLimits:
    """Test pool size limits and blocking behavior."""

    @pytest.mark.asyncio
    async def test_max_size_limit(self):
        """Test that pool doesn't exceed max_size."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=3,
        )

        resources = []
        for _ in range(3):
            ctx = await pool.acquire()
            resources.append(ctx)

        assert pool.size == 3
        assert pool.in_use == 3

        # Release one and acquire again - should not create new resource
        await resources[0].release()
        ctx = await pool.acquire()

        assert pool.size == 3  # Still 3 total

        # Cleanup
        for ctx in resources[1:]:
            await ctx.release()
        await ctx.release()

    @pytest.mark.asyncio
    async def test_exhausted_pool_waits(self):
        """Test that acquisition waits when pool is exhausted."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=2,
        )

        # Exhaust the pool
        ctx1 = await pool.acquire()
        ctx2 = await pool.acquire()

        acquisition_started = False

        async def delayed_acquire():
            nonlocal acquisition_started
            acquisition_started = True
            ctx = await pool.acquire()
            await asyncio.sleep(0)
            await ctx.release()

        # Start a task that will wait
        task = asyncio.create_task(delayed_acquire())
        await asyncio.sleep(0)  # Let it start waiting

        assert acquisition_started
        assert pool.size == 2
        assert pool.in_use == 2

        # Release one resource to unblock the waiting task
        await ctx1.release()

        # Wait for the task to complete
        await asyncio.wait_for(task, timeout=2.0)

        await ctx2.release()

    @pytest.mark.asyncio
    async def test_acquire_from_closed_pool(self):
        """Test that acquiring from closed pool raises error."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=2,
        )

        await pool.close()

        with pytest.raises(RuntimeError, match="Cannot acquire from a closed pool"):
            await pool.acquire()


class TestAsyncPoolClose:
    """Test pool closing and cleanup."""

    @pytest.mark.asyncio
    async def test_close_pool(self):
        """Test closing a pool."""
        closed_resources = []

        def close_resource(resource: MockResource):
            resource.closed = True
            closed_resources.append(resource)

        pool = AsyncPool(
            factory=lambda: MockResource(1),
            closer=close_resource,
            max_size=5,
        )

        # Create some resources (acquire all at once to avoid reuse)
        contexts = []
        resources = []
        for _ in range(2):
            ctx = await pool.acquire()
            contexts.append(ctx)
            resources.append(ctx.resource)

        # Now release them all
        for ctx in contexts:
            await ctx.release()

        # Resources are in the pool, not closed yet
        assert pool.available == 2
        assert len(closed_resources) == 0

        # Close the pool - this should close all available resources
        await pool.close()

        assert pool.available == 0
        assert len(closed_resources) == 2
        assert all(r.closed for r in resources)

    @pytest.mark.asyncio
    async def test_close_pool_context_manager(self):
        """Test closing pool via context manager."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            closer=lambda r: r.close(),
            max_size=2,
        )

        async with pool:
            ctx = await pool.acquire()
            resource = ctx.resource
            await ctx.release()

        # Pool should be closed now
        assert pool.available == 0
        assert resource.closed

    @pytest.mark.asyncio
    async def test_close_with_resources_in_use(self):
        """Test that close only affects available resources."""
        closed_resources = []

        def close_resource(resource: MockResource):
            resource.closed = True
            closed_resources.append(resource)

        pool = AsyncPool(
            factory=lambda: MockResource(1),
            closer=close_resource,
            max_size=3,
        )

        # Create and hold one resource
        ctx1 = await pool.acquire()
        resource1 = ctx1.resource

        # Create and release another
        ctx2 = await pool.acquire()
        resource2 = ctx2.resource
        await ctx2.release()

        assert pool.available == 1
        assert pool.in_use == 1

        # Close pool - should only close available resources
        await pool.close()

        # Only the available resource should be closed
        assert resource2 in closed_resources
        assert resource1 not in closed_resources
        assert not resource1.closed

        # Note: Cannot release resource1 after pool is closed (by design)
        # Resources still in use when pool closes remain in that state
        assert pool.available == 0

    @pytest.mark.asyncio
    async def test_release_to_closed_pool(self):
        """Test that releasing to closed pool raises error."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=2,
        )

        ctx = await pool.acquire()
        await pool.close()

        with pytest.raises(RuntimeError, match="Cannot release to a closed pool"):
            await pool.release(ctx.resource)


class TestAsyncPoolStats:
    """Test pool statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test that pool statistics are tracked correctly."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            closer=lambda r: r.close(),
            max_size=3,
        )

        stats = pool.stats
        assert stats["created"] == 0
        assert stats["acquired"] == 0
        assert stats["released"] == 0
        assert stats["closed"] == 0

        # Acquire and release
        ctx = await pool.acquire()
        stats = pool.stats
        assert stats["created"] == 1
        assert stats["acquired"] == 1

        await ctx.release()
        stats = pool.stats
        assert stats["released"] == 1

        # Close pool
        await pool.close()
        stats = pool.stats
        assert stats["closed"] == 1

    @pytest.mark.asyncio
    async def test_stats_multiple_operations(self):
        """Test stats with multiple acquire/release cycles."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=3,
        )

        # Multiple operations
        for _ in range(5):
            ctx = await pool.acquire()
            await ctx.release()

        stats = pool.stats
        assert stats["created"] == 1  # Only one resource created
        assert stats["acquired"] == 5
        assert stats["released"] == 5


class TestAsyncPoolContext:
    """Test AsyncPoolContext behavior."""

    @pytest.mark.asyncio
    async def test_context_resource_access(self):
        """Test accessing resource through context."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=2,
        )

        ctx = await pool.acquire()
        resource = ctx.resource

        assert isinstance(resource, MockResource)

        await ctx.release()

    @pytest.mark.asyncio
    async def test_context_double_release(self):
        """Test that double release is safe (idempotent)."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=2,
        )

        ctx = await pool.acquire()
        await ctx.release()
        await ctx.release()  # Should not raise

        assert pool.available == 1

    @pytest.mark.asyncio
    async def test_context_release_after_use(self):
        """Test that resource is inaccessible after release."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=2,
        )

        ctx = await pool.acquire()
        await ctx.release()

        with pytest.raises(RuntimeError, match="Resource has been released"):
            _ = ctx.resource

    @pytest.mark.asyncio
    async def test_context_with_statement(self):
        """Test using context in async with statement."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            max_size=2,
        )

        async with await pool.acquire() as resource:
            assert isinstance(resource, MockResource)
            assert pool.in_use == 1

        assert pool.in_use == 0
        assert pool.available == 1


class TestAsyncPoolEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_no_closer(self):
        """Test pool without closer function."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            closer=None,  # No closer
            max_size=2,
        )

        ctx = await pool.acquire()
        resource = ctx.resource
        await ctx.release()

        await pool.close()

        # Resource should not be closed
        assert not resource.closed

    @pytest.mark.asyncio
    async def test_factory_returns_none(self):
        """Test that factory can return None."""
        pool = AsyncPool(
            factory=lambda: None,
            max_size=2,
        )

        ctx = await pool.acquire()
        assert ctx.resource is None
        await ctx.release()

    @pytest.mark.asyncio
    async def test_sync_factory_and_closer(self):
        """Test pool with entirely sync factory and closer."""
        pool = AsyncPool(
            factory=lambda: MockResource(1),
            closer=lambda r: r.close(),
            max_size=2,
        )

        async with await pool.acquire() as resource:
            assert not resource.closed

        # Resource is back in pool, not closed yet
        assert not resource.closed

        # Close the pool to trigger closer
        await pool.close()

        # Now resource should be closed
        assert resource.closed

    @pytest.mark.asyncio
    async def test_empty_pool_acquire(self):
        """Test acquiring from empty pool creates new resource."""
        creation_count = [0]

        def create_resource():
            creation_count[0] += 1
            return MockResource(creation_count[0])

        pool = AsyncPool(
            factory=create_resource,
            max_size=5,
        )

        assert creation_count[0] == 0

        ctx = await pool.acquire()
        assert creation_count[0] == 1

        await ctx.release()
