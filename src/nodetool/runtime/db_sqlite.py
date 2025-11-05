"""
SQLite connection pool for ResourceScope.

Provides async connection pooling for SQLite with per-scope adapter memoization.
"""

import aiosqlite
import asyncio
import os
from pathlib import Path
from typing import Any, Type, Optional, Dict

from nodetool.config.logging_config import get_logger
from nodetool.config.environment import Environment
from nodetool.models.sqlite_adapter import SQLiteAdapter
from nodetool.runtime.resources import DBResources

log = get_logger(__name__)


class SQLiteConnectionPool:
    """Simple async connection pool for SQLite."""

    # Class-level pools per database path
    _pools: Dict[str, "SQLiteConnectionPool"] = {}
    _pools_lock = asyncio.Lock()

    def __init__(self, db_path: str, pool_size: int = 10):
        """Initialize the connection pool.

        Args:
            db_path: Path to SQLite database file
            pool_size: Maximum number of pooled connections
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.available: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self.active_count = 0
        self._lock = asyncio.Lock()

    @classmethod
    async def get_shared(cls, db_path: str, pool_size: int = 10) -> "SQLiteConnectionPool":
        """Get or create a shared connection pool for a database path.

        Args:
            db_path: Path to database (defaults to environment config)
            pool_size: Maximum connections in pool

        Returns:
            A SQLiteConnectionPool instance
        """
        # Return existing pool if available
        if db_path in cls._pools:
            return cls._pools[db_path]

        # Create new pool
        async with cls._pools_lock:
            # Check again in case another coroutine just created it
            if db_path not in cls._pools:
                pool = cls(db_path, pool_size)
                cls._pools[db_path] = pool
                log.info(f"Created SQLite connection pool for {db_path} with size {pool_size}")

            return cls._pools[db_path]

    async def acquire(self) -> aiosqlite.Connection:
        """Acquire a connection from the pool.

        Creates a new connection if the pool is empty and below max size.
        Otherwise waits for a connection to be returned.

        Returns:
            An aiosqlite connection
        """
        # Try to get an existing connection
        try:
            return self.available.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # Create a new connection if below pool size
        async with self._lock:
            if self.active_count < self.pool_size:
                log.debug("Create new sqlite connection")
                self.active_count += 1
                conn = await self._create_connection()
                return conn

        # Wait for a connection to become available
        return await self.available.get()

    async def release(self, connection: aiosqlite.Connection) -> None:
        """Release a connection back to the pool.

        Args:
            connection: The connection to release
        """
        try:
            self.available.put_nowait(connection)
        except asyncio.QueueFull:
            # Pool is full, close the connection with WAL checkpoint
            await self._close_connection_with_checkpoint(connection)
            async with self._lock:
                self.active_count -= 1

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new SQLite connection with proper configuration.

        Returns:
            A configured aiosqlite connection
        """
        # Determine connection settings
        connect_kwargs: Dict[str, Any] = {"timeout": 30}
        if ":memory:" in self.db_path:
            connect_kwargs["check_same_thread"] = False

        # Ensure the parent directory exists for file-based databases.
        resolved_path = self.db_path
        if not self.db_path.startswith("file:"):
            resolved_path = str(Path(self.db_path).expanduser())
            if not resolved_path.startswith(":memory:"):
                Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)

        # Open connection
        connection = await aiosqlite.connect(resolved_path, **connect_kwargs)
        connection.row_factory = aiosqlite.Row

        # Apply pragmas for concurrency and performance
        try:
            if ":memory:" in self.db_path:
                await connection.execute("PRAGMA journal_mode=DELETE")
            else:
                await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA busy_timeout=5000")  # 5 seconds
            await connection.execute("PRAGMA synchronous=NORMAL")
            await connection.execute("PRAGMA cache_size=-64000")  # 64MB
            await connection.commit()
            log.debug("SQLite connection configured with PRAGMA settings")
        except Exception as e:
            log.warning(f"Error applying SQLite pragmas: {e}")
            await connection.close()
            raise

        return connection

    async def _close_connection_with_checkpoint(self, connection: aiosqlite.Connection) -> None:
        """Close a connection and checkpoint the WAL file.

        This ensures WAL data is written back to the main database file
        and helps prevent orphaned WAL files.

        Args:
            connection: The connection to close
        """
        try:
            if ":memory:" not in self.db_path:
                # Checkpoint WAL to write data back to main database
                await connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                await connection.commit()
                log.debug(f"WAL checkpoint completed for {self.db_path}")
        except Exception as e:
            log.warning(f"Error during WAL checkpoint for {self.db_path}: {e}")
        finally:
            try:
                await connection.close()
            except Exception as e:
                log.warning(f"Error closing connection for {self.db_path}: {e}")

    @staticmethod
    def _cleanup_orphaned_wal_files(db_path: str) -> None:
        """Remove orphaned WAL files from previous crashed sessions.

        This should be called before opening a new connection if you suspect
        orphaned files. Only removes -wal and -shm files if the main database
        file is very small or appears corrupted.

        Args:
            db_path: Path to the database file
        """
        if ":memory:" in db_path:
            return

        resolved_path = Path(db_path).expanduser()
        wal_path = resolved_path.parent / f"{resolved_path.name}-wal"
        shm_path = resolved_path.parent / f"{resolved_path.name}-shm"

        # Check if main database exists and is suspiciously small
        if resolved_path.exists():
            db_size = resolved_path.stat().st_size

            # If database is very small (< 8KB) and WAL files exist, they might be orphaned
            if db_size < 8192:
                if wal_path.exists() or shm_path.exists():
                    log.warning(
                        f"Detected small database ({db_size} bytes) with WAL files. "
                        f"This may indicate corruption. Consider removing WAL files manually."
                    )

        # Only remove WAL files if main database doesn't exist
        # (safer than automatically removing when database exists)
        if not resolved_path.exists():
            removed = []
            if wal_path.exists():
                try:
                    wal_path.unlink()
                    removed.append(str(wal_path))
                except Exception as e:
                    log.warning(f"Failed to remove orphaned WAL file {wal_path}: {e}")

            if shm_path.exists():
                try:
                    shm_path.unlink()
                    removed.append(str(shm_path))
                except Exception as e:
                    log.warning(f"Failed to remove orphaned SHM file {shm_path}: {e}")

            if removed:
                log.info(f"Removed orphaned WAL files: {removed}")

    async def close_all(self) -> None:
        """Close all pooled connections with proper WAL checkpointing."""
        connections_closed = 0
        while not self.available.empty():
            try:
                conn = self.available.get_nowait()
                await self._close_connection_with_checkpoint(conn)
                connections_closed += 1
                log.debug(f"Closed SQLite connection from pool for {self.db_path}")
            except asyncio.QueueEmpty:
                break

        # Reset active count since we closed all connections
        async with self._lock:
            self.active_count = 0

        if connections_closed > 0:
            log.info(f"Closed {connections_closed} connection(s) for {self.db_path} with WAL checkpoint")


class SQLiteScopeResources(DBResources):
    """Per-scope SQLite resources (connection + adapters)."""

    def __init__(self, connection: aiosqlite.Connection, pool: SQLiteConnectionPool | None = None):
        """Initialize scope resources.

        Args:
            connection: The connection from the pool
            db_path: Path to database
            pool: The pool to return connection to on cleanup
        """
        self.connection = connection
        self.pool = pool
        self._adapters: Dict[str, Any] = {}

    async def adapter_for_model(self, model_cls: Type[Any]) -> SQLiteAdapter:
        """Get or create an adapter for the given model class.

        Memoizes adapters per table within this scope.

        Args:
            model_cls: The model class to get an adapter for

        Returns:
            A SQLiteAdapter instance
        """
        table_name = model_cls.get_table_schema().get("table_name", "unknown")

        # Return memoized adapter if available
        if table_name in self._adapters:
            log.debug(f"Using memoized SQLite adapter for table '{table_name}'")
            return self._adapters[table_name]

        # Create new adapter
        log.debug(f"Creating new SQLite adapter for table '{table_name}'")
        assert self.connection is not None
        adapter = SQLiteAdapter(
            connection=self.connection,
            fields=model_cls.db_fields(),
            table_schema=model_cls.get_table_schema(),
            indexes=model_cls.get_indexes(),
        )

        # Memoize
        self._adapters[table_name] = adapter
        return adapter

    async def close_all(self) -> None:
        """Clean up resources and close all connections in the pool.

        This performs a full shutdown with WAL checkpointing.
        """
        if self.pool is not None:
            await self.pool.close_all()

    async def cleanup(self) -> None:
        """Clean up scope resources and return connection to pool.

        For per-scope cleanup, we return the connection to the pool
        for reuse. WAL checkpointing happens when the pool is closed
        or connections are evicted.
        """
        if self.connection is not None:
            try:
                # Clear adapter cache
                self._adapters.clear()

                # Return connection to pool for reuse
                if self.pool is not None:
                    await self.pool.release(self.connection)
                self.connection = None  # type: ignore
            except Exception as e:
                log.warning(f"Error releasing SQLite connection: {e}")


async def shutdown_all_sqlite_pools() -> None:
    """Shutdown all SQLite connection pools with proper WAL checkpointing.

    This should be called during application shutdown to ensure all
    WAL files are properly checkpointed and connections are closed.
    """
    async with SQLiteConnectionPool._pools_lock:
        if not SQLiteConnectionPool._pools:
            return

        log.info(f"Shutting down {len(SQLiteConnectionPool._pools)} SQLite connection pool(s)")

        for db_path, pool in SQLiteConnectionPool._pools.items():
            try:
                await pool.close_all()
                log.info(f"Closed connection pool for {db_path}")
            except Exception as e:
                log.error(f"Error closing connection pool for {db_path}: {e}")

        # Clear the pools dictionary
        SQLiteConnectionPool._pools.clear()
        log.info("All SQLite connection pools shut down")
