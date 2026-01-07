"""
PostgreSQL connection pool for ResourceScope.

Provides async connection pooling for PostgreSQL with per-scope adapter memoization.
"""

import asyncio
from typing import Any, ClassVar, Dict, Optional, Type

from psycopg_pool import AsyncConnectionPool

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.models.postgres_adapter import PostgresAdapter
from nodetool.runtime.resources import DBResources

log = get_logger(__name__)


class _AsyncConnectionWrapper:
    """Wrapper to provide connection() context manager interface."""

    def __init__(self, pool: "PostgresConnectionPool"):
        self.pool = pool

    async def __aenter__(self):
        """Acquire connection when entering context."""
        psycopg_pool = await self.pool.get_pool()
        self._conn_ctx = psycopg_pool.connection()
        return await self._conn_ctx.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release connection when exiting context."""
        return await self._conn_ctx.__aexit__(exc_type, exc_val, exc_tb)


class PostgresConnectionPool:
    """Async connection pool for PostgreSQL."""

    _pools: ClassVar[Dict[str, "PostgresConnectionPool"]] = {}
    _pool_locks: ClassVar[Dict[str, asyncio.Lock]] = {}

    def __init__(
        self,
        conninfo: str,
        min_size: int = 1,
        max_size: int = 10,
    ):
        """Initialize the connection pool.

        Args:
            conninfo: PostgreSQL connection string
            min_size: Minimum number of connections
            max_size: Maximum number of connections
        """
        self.conninfo = conninfo
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Optional[AsyncConnectionPool] = None
        self._lock = asyncio.Lock()

    @classmethod
    def _get_pool_lock(cls, pool_key: str) -> asyncio.Lock:
        """Return an asyncio.Lock for the pool key."""
        if pool_key not in cls._pool_locks:
            cls._pool_locks[pool_key] = asyncio.Lock()
        return cls._pool_locks[pool_key]

    @classmethod
    async def get_shared(
        cls,
        conninfo: str,
        min_size: int = 1,
        max_size: int = 10,
    ) -> "PostgresConnectionPool":
        """Get or create a shared connection pool.

        Args:
            conninfo: PostgreSQL connection string
            min_size: Minimum connections in pool
            max_size: Maximum connections in pool

        Returns:
            A PostgresConnectionPool instance
        """
        pool_key = conninfo
        pool_lock = cls._get_pool_lock(pool_key)

        if pool_key in cls._pools:
            return cls._pools[pool_key]

        async with pool_lock:
            if pool_key not in cls._pools:
                pool = cls(conninfo, min_size, max_size)
                cls._pools[pool_key] = pool
                log.info(f"Created PostgreSQL connection pool for {conninfo[:50]}... with size {min_size}-{max_size}")

            return cls._pools[pool_key]

    async def get_pool(self) -> AsyncConnectionPool:
        """Get or create the underlying psycopg pool."""
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    self._pool = AsyncConnectionPool(
                        self.conninfo,
                        min_size=self.min_size,
                        max_size=self.max_size,
                    )
                    await self._pool.open()
                    log.debug("Opened PostgreSQL connection pool")
        return self._pool

    def connection(self):
        """Get a connection context manager from the pool.

        Returns a context manager that acquires and releases a connection.
        This is the same interface as psycopg_pool.AsyncConnectionPool.
        """

        async def _get_connection():
            pool = await self.get_pool()
            return pool.connection()

        # Return an awaitable that returns the context manager
        return _AsyncConnectionWrapper(self)

    async def acquire(self):
        """Acquire a connection from the pool.

        Returns:
            An async connection
        """
        pool = await self.get_pool()
        return await pool.connection().__aenter__()

    async def release(self, conn) -> None:
        """Release a connection back to the pool."""
        await conn.close()

    async def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool is not None:
            await self._pool.close()
            log.debug("Closed PostgreSQL connection pool")
            self._pool = None


class PostgresScopeResources(DBResources):
    """Per-scope PostgreSQL resources (pool + adapters)."""

    def __init__(self, pool: PostgresConnectionPool | None = None):
        """Initialize scope resources.

        Args:
            pool: The PostgreSQL connection pool
        """
        self.pool = pool
        self._adapters: Dict[str, Any] = {}

    async def adapter_for_model(self, model_cls: Type[Any]) -> PostgresAdapter:
        """Get or create an adapter for the given model class.

        Memoizes adapters per table within this scope.

        Args:
            model_cls: The model class to get an adapter for

        Returns:
            A PostgresAdapter instance
        """
        table_name = model_cls.get_table_schema().get("table_name", "unknown")

        if table_name in self._adapters:
            return self._adapters[table_name]

        log.debug(f"Creating new PostgreSQL adapter for table '{table_name}'")

        if self.pool is None:
            db_params = Environment.get_postgres_params()
            conninfo = (
                f"dbname={db_params['database']} user={db_params['user']} "
                f"password={db_params['password']} host={db_params['host']} port={db_params['port']}"
            )
            self.pool = await PostgresConnectionPool.get_shared(conninfo)

        # Get the underlying psycopg pool to pass to the adapter
        psycopg_pool = None
        if self.pool is not None:
            psycopg_pool = await self.pool.get_pool()

        adapter = PostgresAdapter(
            db_params=Environment.get_postgres_params(),
            fields=model_cls.db_fields(),
            table_schema=model_cls.get_table_schema(),
            indexes=model_cls.get_indexes(),
            pool=psycopg_pool,
        )
        await adapter.initialize()

        self._adapters[table_name] = adapter
        return adapter

    async def cleanup(self) -> None:
        """Clean up scope resources and close pool.

        Closes all adapters and the connection pool.
        """
        try:
            for adapter in self._adapters.values():
                if hasattr(adapter, "_pool") and adapter._pool is not None:
                    await adapter._pool.close()

            self._adapters.clear()

            if self.pool is not None:
                await self.pool.close()
        except Exception as e:
            log.warning(f"Error releasing PostgreSQL resources: {e}")
