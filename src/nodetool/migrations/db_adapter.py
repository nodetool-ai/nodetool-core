"""
Database adapter interface for migrations.

Provides an abstract interface for database operations needed by the migration
system. This allows migrations to work with different database backends
(SQLite, PostgreSQL, MySQL, etc.) without database-specific code in migrations.
"""

from abc import ABC, abstractmethod
from typing import Any


class MigrationDBAdapter(ABC):
    """Abstract database adapter interface for migrations.

    This interface provides database-agnostic methods for the operations
    needed by the migration system. Implementations exist for SQLite,
    PostgreSQL, and other databases.

    The adapter wraps database-specific connection objects and provides
    a unified interface for:
    - Executing SQL statements
    - Transaction management (commit/rollback)
    - Schema introspection (table/column existence)
    - Parameter binding
    """

    @abstractmethod
    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> Any:
        """Execute a SQL statement.

        Args:
            sql: SQL statement to execute. Use '?' for parameter placeholders.
            params: Optional tuple of parameters to bind.

        Returns:
            Database-specific cursor or result object.
        """
        pass

    @abstractmethod
    async def executemany(self, sql: str, params_list: list[tuple[Any, ...]]) -> None:
        """Execute a SQL statement multiple times with different parameters.

        Args:
            sql: SQL statement to execute.
            params_list: List of parameter tuples.
        """
        pass

    @abstractmethod
    async def fetchone(self, sql: str, params: tuple[Any, ...] | None = None) -> dict[str, Any] | None:
        """Execute a query and fetch one row.

        Args:
            sql: SQL query to execute.
            params: Optional tuple of parameters to bind.

        Returns:
            Dictionary of column name to value, or None if no row.
        """
        pass

    @abstractmethod
    async def fetchall(self, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        """Execute a query and fetch all rows.

        Args:
            sql: SQL query to execute.
            params: Optional tuple of parameters to bind.

        Returns:
            List of dictionaries (column name to value).
        """
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        pass

    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if the table exists, False otherwise.
        """
        pass

    @abstractmethod
    async def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table.

        Args:
            table_name: Name of the table.
            column_name: Name of the column to check.

        Returns:
            True if the column exists, False otherwise.
        """
        pass

    @abstractmethod
    async def get_columns(self, table_name: str) -> list[str]:
        """Get list of column names in a table.

        Args:
            table_name: Name of the table.

        Returns:
            List of column names.
        """
        pass

    @abstractmethod
    async def index_exists(self, index_name: str) -> bool:
        """Check if an index exists.

        Args:
            index_name: Name of the index.

        Returns:
            True if the index exists, False otherwise.
        """
        pass

    @abstractmethod
    def get_rowcount(self) -> int:
        """Get the number of rows affected by the last statement.

        Returns:
            Number of affected rows.
        """
        pass

    @property
    @abstractmethod
    def db_type(self) -> str:
        """Get the database type identifier.

        Returns:
            Database type string (e.g., 'sqlite', 'postgres', 'mysql').
        """
        pass


class SQLiteMigrationAdapter(MigrationDBAdapter):
    """SQLite implementation of the migration database adapter."""

    def __init__(self, connection: Any):
        """Initialize with an aiosqlite connection.

        Args:
            connection: aiosqlite.Connection object.
        """
        self._conn = connection
        self._last_cursor = None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> Any:
        """Execute a SQL statement."""
        if params:
            self._last_cursor = await self._conn.execute(sql, params)
        else:
            self._last_cursor = await self._conn.execute(sql)
        return self._last_cursor

    async def executemany(self, sql: str, params_list: list[tuple[Any, ...]]) -> None:
        """Execute a SQL statement multiple times."""
        await self._conn.executemany(sql, params_list)

    async def fetchone(self, sql: str, params: tuple[Any, ...] | None = None) -> dict[str, Any] | None:
        """Execute a query and fetch one row."""
        if params:
            cursor = await self._conn.execute(sql, params)
        else:
            cursor = await self._conn.execute(sql)
        row = await cursor.fetchone()
        if row is None:
            return None
        # Convert to dict using cursor.description
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    async def fetchall(self, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        """Execute a query and fetch all rows."""
        if params:
            cursor = await self._conn.execute(sql, params)
        else:
            cursor = await self._conn.execute(sql)
        rows = await cursor.fetchall()
        if not rows:
            return []
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self._conn.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self._conn.rollback()

    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in SQLite."""
        result = await self.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return result is not None

    async def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a SQLite table."""
        columns = await self.get_columns(table_name)
        return column_name in columns

    async def get_columns(self, table_name: str) -> list[str]:
        """Get list of column names in a SQLite table."""
        rows = await self.fetchall(f"PRAGMA table_info({table_name})")
        return [row["name"] for row in rows]

    async def index_exists(self, index_name: str) -> bool:
        """Check if an index exists in SQLite."""
        result = await self.fetchone(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (index_name,),
        )
        return result is not None

    def get_rowcount(self) -> int:
        """Get the number of rows affected."""
        if self._last_cursor is None:
            return 0
        return self._last_cursor.rowcount

    @property
    def db_type(self) -> str:
        """Return database type."""
        return "sqlite"


class PostgresMigrationAdapter(MigrationDBAdapter):
    """PostgreSQL implementation of the migration database adapter."""

    def __init__(self, pool: Any):
        """Initialize with a psycopg pool.

        Args:
            pool: AsyncConnectionPool from psycopg_pool.
        """
        self._pool = pool
        self._conn = None
        self._cursor = None
        self._rowcount = 0

    async def _ensure_connection(self):
        """Ensure we have an active connection."""
        if self._conn is None:
            self._conn = await self._pool.getconn()

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> Any:
        """Execute a SQL statement."""
        await self._ensure_connection()
        # Convert ? placeholders to PostgreSQL %s style
        sql = sql.replace("?", "%s")
        async with self._conn.cursor() as cursor:
            if params:
                await cursor.execute(sql, params)
            else:
                await cursor.execute(sql)
            self._rowcount = cursor.rowcount
            return cursor

    async def executemany(self, sql: str, params_list: list[tuple[Any, ...]]) -> None:
        """Execute a SQL statement multiple times."""
        await self._ensure_connection()
        sql = sql.replace("?", "%s")
        async with self._conn.cursor() as cursor:
            await cursor.executemany(sql, params_list)

    async def fetchone(self, sql: str, params: tuple[Any, ...] | None = None) -> dict[str, Any] | None:
        """Execute a query and fetch one row."""
        await self._ensure_connection()
        sql = sql.replace("?", "%s")
        from psycopg.rows import dict_row

        async with self._conn.cursor(row_factory=dict_row) as cursor:
            if params:
                await cursor.execute(sql, params)
            else:
                await cursor.execute(sql)
            return await cursor.fetchone()

    async def fetchall(self, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        """Execute a query and fetch all rows."""
        await self._ensure_connection()
        sql = sql.replace("?", "%s")
        from psycopg.rows import dict_row

        async with self._conn.cursor(row_factory=dict_row) as cursor:
            if params:
                await cursor.execute(sql, params)
            else:
                await cursor.execute(sql)
            return await cursor.fetchall()

    async def commit(self) -> None:
        """Commit the current transaction."""
        if self._conn:
            await self._conn.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        if self._conn:
            await self._conn.rollback()

    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in PostgreSQL."""
        result = await self.fetchone(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = ?)",
            (table_name,),
        )
        return result["exists"] if result else False

    async def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a PostgreSQL table."""
        result = await self.fetchone(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = ? AND column_name = ?
            )
            """,
            (table_name, column_name),
        )
        return result["exists"] if result else False

    async def get_columns(self, table_name: str) -> list[str]:
        """Get list of column names in a PostgreSQL table."""
        rows = await self.fetchall(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
            """,
            (table_name,),
        )
        return [row["column_name"] for row in rows]

    async def index_exists(self, index_name: str) -> bool:
        """Check if an index exists in PostgreSQL."""
        result = await self.fetchone(
            "SELECT EXISTS (SELECT FROM pg_indexes WHERE indexname = ?)",
            (index_name,),
        )
        return result["exists"] if result else False

    def get_rowcount(self) -> int:
        """Get the number of rows affected."""
        return self._rowcount

    @property
    def db_type(self) -> str:
        """Return database type."""
        return "postgres"

    async def close(self) -> None:
        """Return the connection to the pool."""
        if self._conn:
            await self._pool.putconn(self._conn)
            self._conn = None


def create_migration_adapter(connection: Any) -> MigrationDBAdapter:
    """Factory function to create the appropriate migration adapter.

    Automatically detects the connection type and returns the appropriate
    adapter implementation.

    Args:
        connection: Database connection object (aiosqlite.Connection,
                   psycopg pool, etc.)

    Returns:
        MigrationDBAdapter implementation.

    Raises:
        TypeError: If the connection type is not supported.
    """
    # Check for aiosqlite connection
    if hasattr(connection, "execute") and hasattr(connection, "commit"):
        # Check if it's SQLite by looking for specific attributes
        if hasattr(connection, "_conn") or "sqlite" in type(connection).__module__.lower():
            return SQLiteMigrationAdapter(connection)

    # Check for psycopg pool
    if hasattr(connection, "getconn") and hasattr(connection, "putconn"):
        return PostgresMigrationAdapter(connection)

    # Try to detect based on module name
    module_name = type(connection).__module__.lower()
    if "sqlite" in module_name:
        return SQLiteMigrationAdapter(connection)
    elif "psycopg" in module_name or "postgres" in module_name:
        return PostgresMigrationAdapter(connection)

    raise TypeError(
        f"Unsupported database connection type: {type(connection)}. "
        "Expected aiosqlite.Connection or psycopg_pool.AsyncConnectionPool."
    )
