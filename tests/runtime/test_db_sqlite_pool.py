import asyncio
import tempfile
from pathlib import Path

import pytest

from nodetool.runtime.db_sqlite import SQLiteConnectionPool, SQLitePoolAcquireTimeoutError


@pytest.mark.asyncio
async def test_sqlite_pool_acquire_timeout_does_not_hang():
    pool = SQLiteConnectionPool(":memory:", pool_size=1, acquire_timeout_s=0.1)

    conn = await pool.acquire()
    try:
        with pytest.raises(SQLitePoolAcquireTimeoutError):
            async with pool.acquire_context():
                pass
    finally:
        await pool.release(conn)
        await pool.close_all()


@pytest.mark.asyncio
async def test_sqlite_write_lock_serializes_writes_across_connections():
    # Use a file DB so the global per-db write lock is used.
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "test.sqlite3")

        # Use 2 connections so we can attempt concurrent writes.
        pool = SQLiteConnectionPool(db_path, pool_size=2, acquire_timeout_s=2.0)

        async with pool.acquire_context() as conn1:
            await conn1.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
            await conn1.commit()

            # Start a transaction and do a write, but don't commit yet (holds the write lock).
            await conn1.execute("BEGIN IMMEDIATE")
            await conn1.execute("INSERT INTO t (v) VALUES ('a')")

            async def writer_2():
                async with pool.acquire_context() as conn2:
                    await conn2.execute("INSERT INTO t (v) VALUES ('b')")
                    await conn2.commit()

            task = asyncio.create_task(writer_2())

            # Give task a moment to start and attempt write (it should block on our write lock).
            await asyncio.sleep(0.1)
            assert not task.done()

            # Commit first writer; this should release the global write lock and let task proceed.
            await conn1.commit()

            await asyncio.wait_for(task, timeout=2.0)

        await pool.close_all()

