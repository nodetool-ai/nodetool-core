"""
SQLite connection pool for ResourceScope.

Provides async connection pooling for SQLite with per-scope adapter memoization.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Type

import aiosqlite

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.models.sqlite_adapter import SQLiteAdapter
from nodetool.runtime.resources import DBResources

log = get_logger(__name__)


class SQLiteConnectionPool:
    """Simple async connection pool for SQLite."""

    # Class-level pools per database path and event loop
    _pools: ClassVar[Dict[tuple[int, str], "SQLiteConnectionPool"]] = {}
    _loop_locks: ClassVar[Dict[int, asyncio.Lock]] = {}

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
    def _get_loop_lock(cls, loop_id: int) -> asyncio.Lock:
        """Return an asyncio.Lock bound to the current loop."""
        if loop_id not in cls._loop_locks:
            cls._loop_locks[loop_id] = asyncio.Lock()
        return cls._loop_locks[loop_id]

    @classmethod
    async def get_shared(cls, db_path: str, pool_size: int = 10) -> "SQLiteConnectionPool":
        """Get or create a shared connection pool for a database path and loop.

        Pools are keyed by (event_loop_id, db_path) to avoid sharing asyncio
        primitives across event loops, which triggers RuntimeError on Windows.

        Args:
            db_path: Path to database (defaults to environment config)
            pool_size: Maximum connections in pool

        Returns:
            A SQLiteConnectionPool instance
        """
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        pool_key = (loop_id, db_path)
        loop_lock = cls._get_loop_lock(loop_id)

        # Return existing pool if available for this loop
        if pool_key in cls._pools:
            return cls._pools[pool_key]

        # Create new pool scoped to the current loop
        async with loop_lock:
            # Check again in case another coroutine just created it
            if pool_key not in cls._pools:
                pool = cls(db_path, pool_size)
                cls._pools[pool_key] = pool
                log.info(f"Created SQLite connection pool for {db_path} with size {pool_size} (loop_id={loop_id})")

            return cls._pools[pool_key]

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
            # Increased busy_timeout to 30 seconds for high-concurrency scenarios
            await connection.execute("PRAGMA busy_timeout=30000")  # 30 seconds
            await connection.execute("PRAGMA synchronous=NORMAL")
            await connection.execute("PRAGMA cache_size=-64000")  # 64MB
            # Enable memory-mapped I/O for better read performance
            await connection.execute("PRAGMA mmap_size=268435456")  # 256MB
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


async def shutdown_all_sqlite_pools() -> None:
    """Shutdown SQLite connection pools for the current event loop.

    This avoids awaiting locks created on different loops, which can
    raise RuntimeError. Pools from other loops (if any) must be closed
    from within their respective loops.
    """
    loop_id = id(asyncio.get_running_loop())
    loop_lock = SQLiteConnectionPool._get_loop_lock(loop_id)
    async with loop_lock:
        for (pool_loop_id, db_path), pool in list(SQLiteConnectionPool._pools.items()):
            if pool_loop_id != loop_id:
                continue
            log.info(f"Shutting down SQLite pool for {db_path} (loop_id={loop_id})")
            await pool.close_all()
            del SQLiteConnectionPool._pools[(pool_loop_id, db_path)]


class SQLiteScopeResources(DBResources):
    """Per-scope SQLite resources (connection + adapters)."""

    def __init__(self, pool: SQLiteConnectionPool | None = None):
        """Initialize scope resources.

        Args:
            connection: The connection from the pool
            db_path: Path to database
            pool: The pool to return connection to on cleanup
        """
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
            return self._adapters[table_name]

        # Create new adapter
        log.debug(f"Creating new SQLite adapter for table '{table_name}'")
        assert self.pool is not None
        connection = await self.pool.acquire()
        adapter = SQLiteAdapter(
            connection=connection,
            fields=model_cls.db_fields(),
            table_schema=model_cls.get_table_schema(),
            indexes=model_cls.get_indexes(),
        )

        # Memoize
        self._adapters[table_name] = adapter
        return adapter

    async def cleanup(self) -> None:
        """Clean up scope resources and return connection to pool.

        For per-scope cleanup, we return the connection to the pool
        for reuse. The pool itself is NOT closed here - that happens
        when the session-scoped pool fixture is torn down.
        """
        try:
            # Release all connections held by adapters back to the pool
            if self.pool is not None:
                for adapter in self._adapters.values():
                    if hasattr(adapter, "connection") and adapter.connection is not None:
                        await self.pool.release(adapter.connection)

            # Clear adapter cache
            self._adapters.clear()
        except Exception as e:
            log.warning(f"Error releasing SQLite connection: {e}")
