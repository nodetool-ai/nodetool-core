#!/usr/bin/env python
"""
Verification script for SimpleSQLitePool concurrency testing.

This script tests the SQLite connection pool with the "Lazy Slot" algorithm
by spawning multiple concurrent tasks that acquire connections, perform
simple operations, and release them back to the pool.

It validates that:
- No "database is locked" errors occur under concurrent load
- No "database is closed" errors occur
- All operations complete successfully
- Connection validation and self-healing work correctly
"""

import asyncio
import random
import sys
import tempfile
import time
from pathlib import Path

# Add the src directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nodetool.runtime.db_sqlite import SQLiteConnectionPool


async def worker(pool: SQLiteConnectionPool, task_id: int, results: list) -> None:
    """Worker task that acquires a connection and performs a simple operation.
    
    Args:
        pool: The connection pool to use
        task_id: Unique identifier for this task
        results: List to store results (success/failure)
    """
    try:
        # Acquire connection from pool
        async with pool.acquire_context() as conn:
            # Simulate some IO with random sleep (0.01s as specified)
            await asyncio.sleep(random.uniform(0.005, 0.015))
            
            # Execute a simple query
            cursor = await conn.execute("SELECT 1 as value")
            row = await cursor.fetchone()
            
            # Validate result
            if row[0] != 1:
                results.append(f"Task {task_id}: Unexpected result {row[0]}")
                return
            
            # Occasionally do a write operation to test WAL mode
            if task_id % 10 == 0:
                await conn.execute(
                    "CREATE TABLE IF NOT EXISTS test_concurrency (id INTEGER PRIMARY KEY, task_id INTEGER)"
                )
                await conn.execute(
                    "INSERT OR REPLACE INTO test_concurrency (id, task_id) VALUES (?, ?)",
                    (task_id % 100, task_id)
                )
                await conn.commit()
            
        results.append(f"Task {task_id}: Success")
        
    except Exception as e:
        error_msg = str(e).lower()
        if "locked" in error_msg:
            results.append(f"Task {task_id}: FAIL - Database locked: {e}")
        elif "closed" in error_msg:
            results.append(f"Task {task_id}: FAIL - Database closed: {e}")
        else:
            results.append(f"Task {task_id}: FAIL - Error: {e}")


async def test_direct_acquire_release(pool: SQLiteConnectionPool, task_id: int, results: list) -> None:
    """Test the direct acquire/release pattern (backward compatibility).
    
    Args:
        pool: The connection pool to use
        task_id: Unique identifier for this task
        results: List to store results (success/failure)
    """
    conn = None
    try:
        conn = await pool.acquire()
        
        # Simulate some IO
        await asyncio.sleep(random.uniform(0.005, 0.015))
        
        # Execute a simple query
        cursor = await conn.execute("SELECT 1 as value")
        row = await cursor.fetchone()
        
        if row[0] != 1:
            results.append(f"Direct Task {task_id}: Unexpected result {row[0]}")
            return
            
        results.append(f"Direct Task {task_id}: Success")
        
    except Exception as e:
        error_msg = str(e).lower()
        if "locked" in error_msg:
            results.append(f"Direct Task {task_id}: FAIL - Database locked: {e}")
        elif "closed" in error_msg:
            results.append(f"Direct Task {task_id}: FAIL - Database closed: {e}")
        else:
            results.append(f"Direct Task {task_id}: FAIL - Error: {e}")
    finally:
        if conn is not None:
            await pool.release(conn)


async def main():
    """Main test function."""
    print("=" * 60)
    print("SQLite Connection Pool Concurrency Test")
    print("=" * 60)
    
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        db_path = f.name
    
    print(f"\nUsing database: {db_path}")
    
    # Test parameters
    pool_size = 5
    num_tasks = 50
    
    print(f"Pool size: {pool_size}")
    print(f"Number of concurrent tasks: {num_tasks}")
    
    # Create the pool
    pool = SQLiteConnectionPool(db_path, pool_size)
    
    try:
        results: list = []
        
        # Test 1: Context manager pattern
        print("\n--- Test 1: Context Manager Pattern ---")
        start_time = time.time()
        
        tasks = [worker(pool, i, results) for i in range(num_tasks)]
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        # Analyze results
        success_count = sum(1 for r in results if "Success" in r)
        locked_errors = sum(1 for r in results if "locked" in r.lower())
        closed_errors = sum(1 for r in results if "closed" in r.lower())
        other_errors = sum(1 for r in results if "FAIL" in r and "locked" not in r.lower() and "closed" not in r.lower())
        
        print(f"\nResults:")
        print(f"  Successful tasks: {success_count}/{num_tasks}")
        print(f"  Database locked errors: {locked_errors}")
        print(f"  Database closed errors: {closed_errors}")
        print(f"  Other errors: {other_errors}")
        print(f"  Time elapsed: {elapsed:.3f}s")
        
        # Test 2: Direct acquire/release pattern
        results.clear()
        print("\n--- Test 2: Direct Acquire/Release Pattern ---")
        start_time = time.time()
        
        tasks = [test_direct_acquire_release(pool, i, results) for i in range(num_tasks)]
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        success_count2 = sum(1 for r in results if "Success" in r)
        locked_errors2 = sum(1 for r in results if "locked" in r.lower())
        closed_errors2 = sum(1 for r in results if "closed" in r.lower())
        other_errors2 = sum(1 for r in results if "FAIL" in r and "locked" not in r.lower() and "closed" not in r.lower())
        
        print(f"\nResults:")
        print(f"  Successful tasks: {success_count2}/{num_tasks}")
        print(f"  Database locked errors: {locked_errors2}")
        print(f"  Database closed errors: {closed_errors2}")
        print(f"  Other errors: {other_errors2}")
        print(f"  Time elapsed: {elapsed:.3f}s")
        
        # Summary and assertions
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        total_success = success_count + success_count2
        total_locked = locked_errors + locked_errors2
        total_closed = closed_errors + closed_errors2
        total_other = other_errors + other_errors2
        
        print(f"Total successful operations: {total_success}/{num_tasks * 2}")
        print(f"Total database locked errors: {total_locked}")
        print(f"Total database closed errors: {total_closed}")
        print(f"Total other errors: {total_other}")
        
        # Assertions
        all_passed = True
        
        if total_locked > 0:
            print("\n❌ ASSERTION FAILED: 'database is locked' errors occurred")
            all_passed = False
        else:
            print("\n✓ No 'database is locked' errors")
            
        if total_closed > 0:
            print("❌ ASSERTION FAILED: 'database is closed' errors occurred")
            all_passed = False
        else:
            print("✓ No 'database is closed' errors")
            
        if total_success != num_tasks * 2:
            print(f"❌ ASSERTION FAILED: Not all tasks succeeded ({total_success}/{num_tasks * 2})")
            all_passed = False
        else:
            print("✓ All tasks completed successfully")
            
        if all_passed:
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED! ✓")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("SOME TESTS FAILED! ✗")
            print("=" * 60)
            return 1
            
    finally:
        # Clean up
        await pool.close_all()
        
        # Remove temporary database file
        try:
            Path(db_path).unlink()
            # Also remove WAL and SHM files if they exist
            Path(f"{db_path}-wal").unlink(missing_ok=True)
            Path(f"{db_path}-shm").unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
