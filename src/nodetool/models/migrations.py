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
    from nodetool.runtime.db_postgres import PostgresConnectionPool
    from nodetool.runtime.db_sqlite import SQLiteConnectionPool

log = get_logger(__name__)

# Only one migration at a time (process-level lock, DB-level lock in runner)
_migration_lock = asyncio.Lock()


async def run_startup_migrations(pool: "SQLiteConnectionPool | PostgresConnectionPool | None" = None) -> None:
    """Run all database migrations at application startup.

    This function uses the new migration runner which:
    - Detects database state (fresh install, legacy, or tracked)
    - Handles baselining for legacy databases automatically
    - Uses database-level locking for multi-instance safety
    - Validates checksums of previously applied migrations
    - Runs migrations in transactions with rollback on failure

    For Supabase deployments, programmatic migrations are skipped.
    Use 'nodetool migrations export' to generate SQL files, then
    run 'supabase db push' to apply them.

    Args:
        pool: Optional connection pool. If None, creates one based on
              environment configuration (PostgreSQL or SQLite).

    Raises:
        MigrationError: If any migration fails
        ChecksumError: If checksum validation fails
        LockError: If migration lock cannot be acquired
    """
    from nodetool.config.environment import Environment
    from nodetool.migrations.runner import MigrationRunner
    from nodetool.migrations.state import detect_database_state_postgres

    log.info("Starting database migrations...")

    # Check if using Supabase (Supabase URL present but no direct PostgreSQL connection)
    supabase_url = Environment.get_supabase_url()
    postgres_db = Environment.get("POSTGRES_DB")

    if supabase_url and not postgres_db:
        log.info("Supabase detected - programmatic migrations skipped")
        log.info("To manage schema, run:")
        log.info("  1. nodetool migrations export  # Generate SQL files")
        log.info("  2. supabase db push            # Apply migrations via Supabase CLI")
        return

    async with _migration_lock:
        # Get or create pool based on database type
        if pool is None:
            if postgres_db:
                # Use PostgreSQL
                from nodetool.runtime.db_postgres import PostgresConnectionPool

                db_params = Environment.get_postgres_params()
                conninfo = (
                    f"dbname={db_params['database']} user={db_params['user']} "
                    f"password={db_params['password']} host={db_params['host']} port={db_params['port']}"
                )
                pool = await PostgresConnectionPool.get_shared(conninfo)
            else:
                # Fall back to SQLite
                from pathlib import Path

                db_path = Environment.get("DB_PATH", "~/.config/nodetool/nodetool.sqlite3")
                db_path = str(Path(db_path).expanduser())
                from nodetool.runtime.db_sqlite import SQLiteConnectionPool

                pool = await SQLiteConnectionPool.get_shared(db_path)

        # Acquire connection for migrations
        # For PostgreSQL, pass the underlying psycopg pool; for SQLite, pass the connection
        if postgres_db:
            # PostgreSQL: pass the underlying psycopg pool to the migration runner
            psycopg_pool = await pool.get_pool()
            conn_or_pool = psycopg_pool
        else:
            # SQLite: acquire a connection for the migration runner
            conn_or_pool = await pool.acquire()

        try:
            # Detect initial state for logging
            if postgres_db:
                db_state = await detect_database_state_postgres(pool)
            else:
                from nodetool.migrations.state import detect_database_state_sqlite

                db_state = await detect_database_state_sqlite(conn_or_pool)
            log.info(f"Database state: {db_state.value}")

            # Create migration runner and execute migrations
            runner = MigrationRunner(conn_or_pool)

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
            # For SQLite, release the connection back to the pool
            if not postgres_db:
                await pool.release(conn_or_pool)  # type: ignore[arg-type]
