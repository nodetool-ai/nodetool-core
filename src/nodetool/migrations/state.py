"""
Database state detection for the migration system.

Provides functionality to detect the current state of the database
to determine whether it's a fresh install, legacy database, or
already has migration tracking.
"""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


class DatabaseState(Enum):
    """Enum representing the possible states of the database."""

    FRESH_INSTALL = "fresh_install"  # No tables at all
    LEGACY_DATABASE = "legacy_database"  # App tables exist, no migration tracking
    MIGRATION_TRACKED = "migration_tracked"  # Migration tracking is present


# Known application tables that indicate an existing installation
APPLICATION_TABLES = [
    "nodetool_workflows",
    "nodetool_assets",
    "nodetool_threads",
    "nodetool_messages",
    "nodetool_jobs",
    "nodetool_predictions",
    "nodetool_secrets",
]

# Migration system tables
MIGRATION_TRACKING_TABLE = "_nodetool_migrations"
MIGRATION_LOCK_TABLE = "_nodetool_migration_lock"


async def table_exists_sqlite(conn: "aiosqlite.Connection", table_name: str) -> bool:
    """Check if a table exists in SQLite database.

    Args:
        conn: SQLite database connection
        table_name: Name of the table to check

    Returns:
        True if the table exists, False otherwise
    """
    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    result = await cursor.fetchone()
    return result is not None


async def table_exists_postgres(pool, table_name: str) -> bool:
    """Check if a table exists in PostgreSQL database.

    Args:
        pool: PostgreSQL connection pool
        table_name: Name of the table to check

    Returns:
        True if the table exists, False otherwise
    """
    from psycopg.rows import dict_row

    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,),
        )
        result = await cursor.fetchone()
        return result["exists"] if result else False


async def detect_database_state_sqlite(conn: "aiosqlite.Connection") -> DatabaseState:
    """Detect the current state of a SQLite database.

    Args:
        conn: SQLite database connection

    Returns:
        DatabaseState enum value indicating the database state
    """
    # First, check if migration tracking table exists
    if await table_exists_sqlite(conn, MIGRATION_TRACKING_TABLE):
        return DatabaseState.MIGRATION_TRACKED

    # Check if any application tables exist
    for table_name in APPLICATION_TABLES:
        if await table_exists_sqlite(conn, table_name):
            return DatabaseState.LEGACY_DATABASE

    # No tables exist - fresh install
    return DatabaseState.FRESH_INSTALL


async def detect_database_state_postgres(pool) -> DatabaseState:
    """Detect the current state of a PostgreSQL database.

    Args:
        pool: PostgreSQL connection pool

    Returns:
        DatabaseState enum value indicating the database state
    """
    # First, check if migration tracking table exists
    if await table_exists_postgres(pool, MIGRATION_TRACKING_TABLE):
        return DatabaseState.MIGRATION_TRACKED

    # Check if any application tables exist
    for table_name in APPLICATION_TABLES:
        if await table_exists_postgres(pool, table_name):
            return DatabaseState.LEGACY_DATABASE

    # No tables exist - fresh install
    return DatabaseState.FRESH_INSTALL


async def detect_database_state(
    conn: "aiosqlite.Connection | None" = None,
    pool=None,
) -> DatabaseState:
    """Detect the current state of the database.

    This function determines if the database is:
    - FRESH_INSTALL: No tables exist at all
    - LEGACY_DATABASE: Application tables exist but no migration tracking
    - MIGRATION_TRACKED: Migration tracking tables are present

    Args:
        conn: SQLite database connection (optional)
        pool: PostgreSQL connection pool (optional)

    Returns:
        DatabaseState enum value

    Raises:
        ValueError: If neither conn nor pool is provided
    """
    if conn is not None:
        return await detect_database_state_sqlite(conn)
    elif pool is not None:
        return await detect_database_state_postgres(pool)
    else:
        raise ValueError("Either conn or pool must be provided")
