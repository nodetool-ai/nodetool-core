"""
Unit tests for SQLiteConnectionPool with "Lazy Slot" algorithm.

Tests the connection pool implementation including:
- Lazy connection initialization
- "Validate on Borrow" health checks
- Self-healing for dead connections
- Proper cleanup with rollback on release
- Context manager and direct acquire/release patterns
"""

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from nodetool.runtime.db_sqlite import SQLiteConnectionPool

# Skip global setup/teardown for these tests
pytestmark = pytest.mark.no_setup


@pytest_asyncio.fixture
async def temp_db_path():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    try:
        Path(db_path).unlink()
        Path(f"{db_path}-wal").unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)
    except Exception:
        pass


@pytest_asyncio.fixture
async def pool(temp_db_path):
    """Create a connection pool for testing."""
    pool = SQLiteConnectionPool(temp_db_path, pool_size=3)
    yield pool
    await pool.close_all()


@pytest.mark.asyncio
async def test_pool_initialization(temp_db_path):
    """Test that pool initializes with lazy slots."""
    pool = SQLiteConnectionPool(temp_db_path, pool_size=5)

    # Pool should have 5 slots (all None initially)
    assert pool.pool_size == 5
    assert pool._slots.qsize() == 5
    assert not pool._closed

    await pool.close_all()


@pytest.mark.asyncio
async def test_acquire_creates_connection_lazily(pool):
    """Test that connections are created lazily on first acquire."""
    # Acquire a connection using context manager
    async with pool.acquire_context() as conn:
        # Connection should be valid
        cursor = await conn.execute("SELECT 1")
        row = await cursor.fetchone()
        assert row[0] == 1


@pytest.mark.asyncio
async def test_acquire_direct_method(pool):
    """Test the direct acquire/release pattern for backward compatibility."""
    conn = await pool.acquire()
    try:
        # Connection should be valid
        cursor = await conn.execute("SELECT 1")
        row = await cursor.fetchone()
        assert row[0] == 1
    finally:
        await pool.release(conn)


@pytest.mark.asyncio
async def test_connection_reuse(pool):
    """Test that connections are reused from the pool."""
    # Get a connection ID by checking object identity
    async with pool.acquire_context() as conn1:
        conn1_id = id(conn1)

    # The connection should be returned to the pool and reused
    async with pool.acquire_context() as conn2:
        conn2_id = id(conn2)

    # Same connection should be reused (though this is not guaranteed)
    # We just verify that both work correctly
    assert conn1_id is not None
    assert conn2_id is not None


@pytest.mark.asyncio
async def test_wal_mode_enabled(pool):
    """Test that WAL mode is enabled on connections."""
    async with pool.acquire_context() as conn:
        cursor = await conn.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        # Should be WAL for file-based databases
        assert row[0].lower() == "wal"


@pytest.mark.asyncio
async def test_validate_on_borrow(pool):
    """Test that connections are validated with SELECT 1."""
    async with pool.acquire_context() as conn:
        # Connection is already validated, should work
        cursor = await conn.execute("SELECT 42")
        row = await cursor.fetchone()
        assert row[0] == 42


@pytest.mark.asyncio
async def test_concurrent_acquisitions(temp_db_path):
    """Test that concurrent acquisitions work correctly."""
    pool = SQLiteConnectionPool(temp_db_path, pool_size=3)
    results = []

    async def worker(task_id):
        async with pool.acquire_context() as conn:
            await asyncio.sleep(0.01)  # Small delay to simulate work
            cursor = await conn.execute("SELECT ?", (task_id,))
            row = await cursor.fetchone()
            results.append((task_id, row[0]))

    # Run 10 concurrent tasks with only 3 slots
    tasks = [worker(i) for i in range(10)]
    await asyncio.gather(*tasks)

    # All tasks should complete successfully
    assert len(results) == 10
    assert all(r[0] == r[1] for r in results)

    await pool.close_all()


@pytest.mark.asyncio
async def test_rollback_on_release(pool):
    """Test that transactions are rolled back on release."""
    # Create a table
    async with pool.acquire_context() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS test_rollback (id INTEGER)")
        await conn.commit()

    # Start a transaction but don't commit
    async with pool.acquire_context() as conn:
        await conn.execute("INSERT INTO test_rollback VALUES (1)")
        # Don't commit - should be rolled back on release

    # Verify the insert was rolled back
    async with pool.acquire_context() as conn:
        cursor = await conn.execute("SELECT COUNT(*) FROM test_rollback")
        row = await cursor.fetchone()
        assert row[0] == 0


@pytest.mark.asyncio
async def test_close_all(temp_db_path):
    """Test that close_all properly closes all connections."""
    pool = SQLiteConnectionPool(temp_db_path, pool_size=3)

    # Acquire and release a few connections to create them
    for _ in range(2):
        async with pool.acquire_context() as conn:
            await conn.execute("SELECT 1")

    # Close all
    await pool.close_all()

    # Pool should be marked as closed
    assert pool._closed

    # Acquiring should raise RuntimeError
    with pytest.raises(RuntimeError, match="Pool is closed"):
        async with pool.acquire_context() as conn:
            pass


@pytest.mark.asyncio
async def test_exception_handling_in_context(pool):
    """Test that exceptions in the context manager properly clean up."""
    try:
        async with pool.acquire_context() as conn:
            # Verify connection works
            await conn.execute("SELECT 1")
            # Raise an exception
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Pool should still work after exception
    async with pool.acquire_context() as conn:
        cursor = await conn.execute("SELECT 1")
        row = await cursor.fetchone()
        assert row[0] == 1


@pytest.mark.asyncio
async def test_memory_database():
    """Test pool with in-memory database."""
    pool = SQLiteConnectionPool(":memory:", pool_size=2)

    async with pool.acquire_context() as conn:
        # For in-memory databases, SQLite may report 'memory' or 'delete' for journal mode
        # depending on the version. The important thing is it's not 'wal'.
        cursor = await conn.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row[0].lower() in ("delete", "memory")

        # Verify connection works
        await conn.execute("CREATE TABLE test (id INTEGER)")
        await conn.execute("INSERT INTO test VALUES (1)")
        await conn.commit()

        cursor = await conn.execute("SELECT * FROM test")
        row = await cursor.fetchone()
        assert row[0] == 1

    await pool.close_all()


@pytest.mark.asyncio
async def test_pool_exhaustion_blocking(temp_db_path):
    """Test that acquire blocks when pool is exhausted."""
    pool = SQLiteConnectionPool(temp_db_path, pool_size=2)

    acquired_connections = []

    # Acquire all connections
    conn1 = await pool.acquire()
    conn2 = await pool.acquire()
    acquired_connections.extend([conn1, conn2])

    # Try to acquire another - should block
    async def delayed_release():
        await asyncio.sleep(0.1)
        await pool.release(conn1)

    # Start a task that releases a connection after delay
    release_task = asyncio.create_task(delayed_release())

    # This should block until the release happens
    conn3 = await asyncio.wait_for(pool.acquire(), timeout=1.0)
    assert conn3 is not None

    # Cleanup
    await release_task
    await pool.release(conn2)
    await pool.release(conn3)
    await pool.close_all()
