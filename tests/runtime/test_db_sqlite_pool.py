import pytest

from nodetool.runtime.db_sqlite import SQLiteConnectionPool, SQLitePoolAcquireTimeoutError


@pytest.mark.asyncio
async def test_sqlite_pool_acquire_timeout_does_not_hang():
    """
    Regression guard:
    If a connection is held and the pool is exhausted, acquisition must fail fast
    (timeout) instead of blocking forever.
    """
    pool = SQLiteConnectionPool(":memory:", pool_size=1, acquire_timeout_s=0.1)

    conn = await pool.acquire()
    try:
        with pytest.raises(SQLitePoolAcquireTimeoutError):
            async with pool.acquire_context():
                pass
    finally:
        await pool.release(conn)
        await pool.close_all()

