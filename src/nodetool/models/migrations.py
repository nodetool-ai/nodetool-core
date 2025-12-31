"""
Database migration system for NodeTool.

Provides startup migrations using the professional migration runner with
version tracking, database-level locking, and rollback support.

This module serves as the bridge between the application startup and the
migration system. It handles:
- Automatic database state detection (fresh, legacy, or tracked)
- Migration execution with transaction safety
- Baselining for legacy databases
"""

import asyncio
from typing import TYPE_CHECKING

from nodetool.config.logging_config import get_logger

if TYPE_CHECKING:
    from nodetool.runtime.db_sqlite import SQLiteConnectionPool

log = get_logger(__name__)

# Only one migration at a time (process-level lock, DB-level lock in runner)
_migration_lock = asyncio.Lock()


async def run_startup_migrations(pool: "SQLiteConnectionPool | None" = None) -> None:
    """Run all database migrations at application startup.

    This function uses the new migration runner which:
    - Detects database state (fresh install, legacy, or tracked)
    - Handles baselining for legacy databases automatically
    - Uses database-level locking for multi-instance safety
    - Validates checksums of previously applied migrations
    - Runs migrations in transactions with rollback on failure

    Args:
        pool: Optional SQLite connection pool. If None, creates one using
              the environment configuration.

    Raises:
        MigrationError: If any migration fails
        ChecksumError: If checksum validation fails
        LockError: If migration lock cannot be acquired
    """
    from nodetool.migrations.runner import MigrationRunner
    from nodetool.migrations.state import DatabaseState, detect_database_state_sqlite
    from nodetool.runtime.db_sqlite import SQLiteConnectionPool as PoolClass

    log.info("Starting database migrations...")

    async with _migration_lock:
        # Get or create pool
        if pool is None:
            from pathlib import Path

            from nodetool.config.environment import Environment

            db_path = Environment.get("DB_PATH", "~/.config/nodetool/nodetool.sqlite3")
            db_path = str(Path(db_path).expanduser())
            pool = await PoolClass.get_shared(db_path)

        # Acquire connection for migrations
        conn = await pool.acquire()

        try:
            # Detect initial state for logging
            db_state = await detect_database_state_sqlite(conn)
            log.info(f"Database state: {db_state.value}")

            # Create migration runner and execute migrations
            runner = MigrationRunner(conn)

            # Run migrations - the runner handles all three scenarios
            applied = await runner.migrate(
                dry_run=False,
                validate_checksums=True,
            )

            if applied:
                log.info(f"Applied {len(applied)} migration(s): {', '.join(applied)}")
            else:
                log.info("No pending migrations - database is up to date")

            # Log final state
            status = await runner.status()
            log.info(
                f"Database migrations completed. "
                f"State: {status['state']}, "
                f"Version: {status['current_version'] or 'None'}, "
                f"Applied: {len(status['applied'])}, "
                f"Pending: {len(status['pending'])}"
            )

        except Exception as e:
            log.error(f"Migration failed: {e}", exc_info=True)
            raise

        finally:
            await pool.release(conn)

