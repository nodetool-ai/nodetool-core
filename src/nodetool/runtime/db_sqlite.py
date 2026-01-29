"""
SQLite connection pool for ResourceScope.

Provides async connection pooling for SQLite with per-scope adapter memoization.
Uses the "Lazy Slot" algorithm for efficient connection management.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, ClassVar, Optional

import aiosqlite

from nodetool.config.logging_config import get_logger
from nodetool.models.sqlite_adapter import SQLiteAdapter
from nodetool.runtime.resources import DBResources

log = get_logger(__name__)


class SQLiteConnectionPool:
    """Async connection pool for SQLite using the "Lazy Slot" algorithm.

    This pool uses a queue-based slot system with lazy connection initialization.
    Each slot in the queue can be either None (empty) or an active connection.
    Connections are created lazily on first acquire and validated on each borrow.

    Features:
    - Lazy connection initialization (connections created only when needed)
    - "Validate on Borrow" health checks (SELECT 1)
    - Self-healing: dead connections are replaced automatically
    - WAL mode for better concurrency
    - Proper cleanup with rollback before release
    """

    # Class-level pools per database path and event loop
    _pools: ClassVar[dict[tuple[int, str], "SQLiteConnectionPool"]] = {}
    _loop_locks: ClassVar[dict[int, asyncio.Lock]] = {}

    def __init__(self, db_path: str, pool_size: int = 5):
        """Initialize the connection pool with lazy slots.

        Args:
            db_path: Path to SQLite database file
            pool_size: Maximum number of concurrent connections (slots)
        """
        self.db_path = db_path
        self.pool_size = pool_size
        # Initialize queue with None values (lazy slots)
        self._slots: asyncio.Queue[Optional[aiosqlite.Connection]] = asyncio.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            self._slots.put_nowait(None)
        self._closed = False

    @classmethod
    def _get_loop_lock(cls, loop_id: int) -> asyncio.Lock:
        """Return an asyncio.Lock bound to the current loop."""
        if loop_id not in cls._loop_locks:
            cls._loop_locks[loop_id] = asyncio.Lock()
        return cls._loop_locks[loop_id]

    @classmethod
    async def get_shared(cls, db_path: str, pool_size: int = 5) -> "SQLiteConnectionPool":
        """Get or create a shared connection pool for a database path and loop.

        Pools are keyed by (event_loop_id, db_path) to avoid sharing asyncio
        primitives across event loops, which triggers RuntimeError on Windows.

        Args:
            db_path: Path to database
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

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new SQLite connection with WAL mode enabled.

        Returns:
            A configured aiosqlite connection with WAL mode and performance pragmas.

        Raises:
            Exception: If connection creation or configuration fails after retries.
        """
        # Determine connection settings
        connect_kwargs: dict[str, Any] = {"timeout": 30}
        if ":memory:" in self.db_path:
            connect_kwargs["check_same_thread"] = False

        # Ensure the parent directory exists for file-based databases
        resolved_path = self.db_path
        if not self.db_path.startswith("file:"):
            resolved_path = str(Path(self.db_path).expanduser())
            if not resolved_path.startswith(":memory:"):
                Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)

        # Open connection
        connection = await aiosqlite.connect(resolved_path, **connect_kwargs)
        connection.row_factory = aiosqlite.Row

        # Apply pragmas for concurrency and performance with retry logic
        max_retries = 5
        last_error = None

        for attempt in range(max_retries):
            try:
                # Enable WAL mode immediately (required by spec)
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
                log.debug("SQLite connection configured with WAL mode and PRAGMA settings")
                return connection
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                if "locked" in error_msg or "busy" in error_msg:
                    if attempt < max_retries - 1:
                        # Wait with exponential backoff before retrying
                        delay = 0.05 * (2**attempt)  # 50ms, 100ms, 200ms, 400ms
                        log.debug(
                            f"PRAGMA setup locked, retrying in {delay:.3f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                # For non-lock errors or final attempt, close and raise
                log.warning(f"Error applying SQLite pragmas: {e}")
                try:
                    await connection.close()
                except Exception:
                    pass
                raise

        # Should not reach here, but if it does, close and raise last error
        try:
            await connection.close()
        except Exception:
            pass
        raise last_error or RuntimeError("Failed to configure connection after retries")

    async def _validate_connection(self, conn: aiosqlite.Connection) -> bool:
        """Validate a connection using "Validate on Borrow" health check.

        Args:
            conn: The connection to validate.

        Returns:
            True if the connection is healthy, False otherwise.
        """
        try:
            await conn.execute("SELECT 1")
            return True
        except Exception as e:
            log.debug(f"Connection validation failed: {e}")
            return False

    async def _close_connection_safely(self, conn: aiosqlite.Connection) -> None:
        """Close a connection safely, ignoring any errors.

        Args:
            conn: The connection to close.
        """
        try:
            await conn.close()
        except Exception as e:
            log.debug(f"Error closing connection: {e}")

    async def _acquire_connection(self) -> aiosqlite.Connection:
        """Internal method to acquire a connection from the pool.

        This implements the "Lazy Slot" algorithm:
        - Pop a slot from the queue
        - If slot is None: create a fresh connection with WAL mode
        - If slot is a connection: validate it, self-heal if invalid

        Returns:
            An aiosqlite connection that is guaranteed to be healthy.

        Raises:
            RuntimeError: If the pool is closed.
            Exception: If connection creation fails.
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        # Pop a slot from the queue (blocks if all slots are in use)
        slot: Optional[aiosqlite.Connection] = await self._slots.get()

        try:
            if slot is None:
                # Case A: Empty slot - create a fresh connection
                log.debug("Creating new connection for empty slot")
                return await self._create_connection()
            else:
                # Case B: Existing connection - validate with "Validate on Borrow"
                if await self._validate_connection(slot):
                    return slot
                else:
                    # Connection is dead - self-healing: close and create fresh
                    log.debug("Connection validation failed, self-healing by creating new connection")
                    await self._close_connection_safely(slot)
                    return await self._create_connection()
        except Exception:
            # On failure, return the None slot to the pool
            await self._slots.put(None)
            raise

    async def acquire(self) -> aiosqlite.Connection:
        """Acquire a connection from the pool (direct method for backward compatibility).

        The caller is responsible for calling `release()` when done with the connection.
        For automatic cleanup, use the context manager with `async with pool.acquire_context()`.

        Returns:
            An aiosqlite connection that is guaranteed to be healthy.

        Raises:
            RuntimeError: If the pool is closed.
            Exception: If connection creation fails.
        """
        return await self._acquire_connection()

    @asynccontextmanager
    async def acquire_context(self) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire a connection from the pool using the context manager pattern.

        This implements the "Lazy Slot" algorithm with automatic cleanup:
        - Pop a slot from the queue
        - If slot is None: create a fresh connection with WAL mode
        - If slot is a connection: validate it, self-heal if invalid
        - On exit: rollback and return to pool, or close and return None slot on error

        Yields:
            An aiosqlite connection that is guaranteed to be healthy.

        Raises:
            RuntimeError: If the pool is closed.
            Exception: If connection creation fails.
        """
        conn = await self._acquire_connection()
        try:
            yield conn

            # Success path: rollback to clean state and return connection to pool
            try:
                await conn.rollback()
                await self._slots.put(conn)
            except Exception as e:
                # Rollback failed - close connection and return None slot
                log.warning(f"Rollback failed during release: {e}")
                await self._close_connection_safely(conn)
                await self._slots.put(None)

        except Exception:
            # Failure/crash path: close connection and return None slot
            await self._close_connection_safely(conn)
            await self._slots.put(None)
            raise

    async def release(self, connection: aiosqlite.Connection) -> None:
        """Release a connection back to the pool (legacy API compatibility).

        This method is provided for backward compatibility with existing code.
        The preferred way to use the pool is via the `acquire()` context manager.

        Args:
            connection: The connection to release
        """
        try:
            await connection.rollback()
            await self._slots.put(connection)
        except Exception as e:
            log.warning(f"Error releasing connection: {e}")
            await self._close_connection_safely(connection)
            await self._slots.put(None)

    async def close_all(self) -> None:
        """Close all pooled connections and mark pool as closed.

        Drains the queue and closes all underlying connections.
        After calling this, the pool cannot be used.
        """
        self._closed = True
        connections_closed = 0

        # Drain all slots and close any connections
        while True:
            try:
                slot = self._slots.get_nowait()
                if slot is not None:
                    try:
                        # Checkpoint WAL before closing (for file-based databases)
                        if ":memory:" not in self.db_path:
                            try:
                                await slot.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                                await slot.commit()
                            except Exception:
                                pass
                        await slot.close()
                        connections_closed += 1
                        log.debug(f"Closed SQLite connection from pool for {self.db_path}")
                    except Exception as e:
                        log.warning(f"Error closing connection for {self.db_path}: {e}")
            except asyncio.QueueEmpty:
                break

        log.info(f"Closed {connections_closed} connections for pool {self.db_path}")

    @staticmethod
    def _cleanup_orphaned_wal_files(db_path: str) -> None:
        """Remove orphaned WAL files from previous crashed sessions.

        This should be called before opening a new connection if you suspect
        orphaned files. Only removes -wal and -shm files if the main database
        file doesn't exist.

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
        self._adapters: dict[str, Any] = {}

    async def adapter_for_model(self, model_cls: type[Any]) -> SQLiteAdapter:
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

        Rolls back any pending transactions before releasing connections
        to prevent database locks.
        """
        try:
            # Release all connections held by adapters back to the pool
            if self.pool is not None:
                for adapter in self._adapters.values():
                    if hasattr(adapter, "connection") and adapter.connection is not None:
                        conn = adapter.connection
                        # Rollback any pending transactions to prevent locks
                        try:
                            await conn.rollback()
                        except Exception:
                            pass
                        await self.pool.release(conn)

            # Clear adapter cache
            self._adapters.clear()
        except Exception as e:
            log.warning(f"Error releasing SQLite connection: {e}")
