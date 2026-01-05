"""
Resource scope management for per-execution isolation.

Provides ResourceScope for managing per-execution resources (database adapters)
with proper cleanup and connection pooling.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any, Optional, Protocol, Type

import httpx

from nodetool.config.env_guard import RUNNING_PYTEST
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

if TYPE_CHECKING:
    from supabase import AsyncClient

    from nodetool.models.database_adapter import DatabaseAdapter
    from nodetool.storage.abstract_storage import AbstractStorage
    from nodetool.storage.memcache_node_cache import AbstractNodeCache
    from nodetool.storage.memory_uri_cache import MemoryUriCache

log = get_logger(__name__)


class DBResources(Protocol):
    """Protocol for database resources (connection + adapters)."""

    async def adapter_for_model(self, model_cls: Type[Any]) -> DatabaseAdapter:
        """Get or create an adapter for the given model class.

        Args:
            model_cls: The model class to get an adapter for

        Returns:
            A DatabaseAdapter instance for the model
        """
        ...

    async def cleanup(self) -> None:
        """Clean up resources and return connections to pool."""
        ...


# ContextVar to store the current scope
_current_scope: contextvars.ContextVar[Optional[ResourceScope]] = contextvars.ContextVar("_current_scope", default=None)


def require_scope() -> ResourceScope:
    """Get the current resource scope or raise if none is bound.

    Returns:
        The current ResourceScope

    Raises:
        RuntimeError: If no scope is currently bound
    """
    scope = _current_scope.get()
    if scope is None:
        raise RuntimeError("No ResourceScope is currently bound")
    return scope


def maybe_scope() -> Optional[ResourceScope]:
    """Get the current resource scope or None if not bound.

    Returns:
        The current ResourceScope or None
    """
    return _current_scope.get()


def get_static_auth_provider() -> Any:
    """Get the static token authentication provider (global singleton).

    Can be called without a ResourceScope for server initialization.
    Creates the provider on first access and reuses it thereafter.

    Returns:
        StaticTokenAuthProvider instance

    Raises:
        ValueError: If WORKER_AUTH_TOKEN is not configured
    """
    from nodetool.deploy.auth import get_worker_auth_token
    from nodetool.security.providers.static_token import StaticTokenAuthProvider

    if ResourceScope._class_static_auth_provider is None:
        token = get_worker_auth_token()
        if not token:
            raise ValueError("WORKER_AUTH_TOKEN is required for static authentication.")
        ResourceScope._class_static_auth_provider = StaticTokenAuthProvider(static_token=token)
    return ResourceScope._class_static_auth_provider


def get_user_auth_provider() -> Any:
    """Get the configured user authentication provider (global singleton).

    Can be called without a ResourceScope for server initialization.
    Creates the provider on first access based on AUTH_PROVIDER setting.

    Returns the appropriate provider:
    - none: Returns None (no auth enforcement)
    - local: LocalAuthProvider (always returns user "1")
    - static: StaticTokenAuthProvider (shared token auth)
    - supabase: SupabaseAuthProvider (validates Supabase JWTs)

    Returns:
        AuthProvider instance or None if auth is disabled
    """
    if ResourceScope._class_user_auth_provider is None:
        kind = Environment.get_auth_provider_kind()
        if kind == "none":
            ResourceScope._class_user_auth_provider = None
        elif kind == "local":
            from nodetool.security.providers.local import LocalAuthProvider

            ResourceScope._class_user_auth_provider = LocalAuthProvider()
        elif kind == "static":
            # Reuse static token provider for user auth path
            ResourceScope._class_user_auth_provider = get_static_auth_provider()
        elif kind == "supabase":
            supabase_url = Environment.get("SUPABASE_URL")
            supabase_key = Environment.get("SUPABASE_KEY")
            if supabase_url and supabase_key:
                from nodetool.security.providers.supabase import SupabaseAuthProvider

                ResourceScope._class_user_auth_provider = SupabaseAuthProvider(
                    supabase_url=supabase_url,
                    supabase_key=supabase_key,
                    cache_ttl=Environment.get_auth_cache_ttl(),
                    cache_max=Environment.get_auth_cache_max(),
                )
            else:
                ResourceScope._class_user_auth_provider = None
        else:
            ResourceScope._class_user_auth_provider = None
    return ResourceScope._class_user_auth_provider


class ResourceScope:
    """Per-execution resource scope with connection pooling.

    Acquires database connections from shared pools and provides per-scope
    adapter memoization. Automatically releases connections on exit.
    """

    # Class-level auth provider singletons (shared across all scopes)
    _class_static_auth_provider: Any = None
    _class_user_auth_provider: Any = None

    def __init__(
        self,
        pool: Any | None = None,
    ) -> None:
        """Initialize a ResourceScope.

        Auto-detects database type from environment and uses appropriate pool.

        Args:
            pool: The SQLite connection pool to use
        """
        self._token: Optional[contextvars.Token] = None
        self.pool = pool
        self._owns_db = False  # Track if we own the db resources (vs borrowed from parent)
        self._owns_http_client = False  # Track if we own the HTTP client (vs borrowed from parent)
        scope = maybe_scope()
        if scope:
            self.db = scope.db
            self._asset_storage = scope.get_asset_storage()
            self._temp_storage = scope.get_temp_storage()
            self._node_cache = scope.get_node_cache()
            self._memory_uri_cache = scope.get_memory_uri_cache()
            self._http_client = scope.get_http_client()
            # Borrowed from parent, don't own
            self._owns_http_client = False
        else:
            self.db: Optional[DBResources] = None
            self._asset_storage: AbstractStorage | None = None
            self._temp_storage: AbstractStorage | None = None
            self._node_cache: AbstractNodeCache | None = None
            self._memory_uri_cache: MemoryUriCache | None = None
            self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> ResourceScope:
        """Enter the async context manager.

        Acquires resources from the appropriate pool and binds this scope.

        Returns:
            This ResourceScope instance
        """
        try:
            # Auto-detect and acquire database resources
            if self.db is None:
                self.db = await self._acquire_db_resources()
                self._owns_db = True  # Mark that we own these resources
                log.debug(f"Acquired database resources: {type(self.db).__name__}")

            # Bind this scope to the context variable
            self._token = _current_scope.set(self)

            return self
        except Exception as e:
            log.error(f"Error entering ResourceScope: {e}")
            raise

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager.

        Unbinds the scope and releases resources back to pools.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        try:
            # Unbind the scope from the context variable
            if self._token is not None:
                try:
                    _current_scope.reset(self._token)
                    log.debug("ResourceScope unbound from context")
                except ValueError:
                    # Token was created in a different context (e.g., nested task)
                    # This is expected in some test scenarios, log and continue
                    log.debug("ResourceScope token reset skipped (different context)")
                    pass

            # Only clean up database resources if we own them (not borrowed from parent)
            if self.db is not None and self._owns_db:
                try:
                    await self.db.cleanup()
                    log.debug("Cleaned up owned database resources")
                except Exception as e:
                    log.warning(f"Error cleaning up database resources: {e}")
        except Exception as e:
            log.error(f"Error exiting ResourceScope: {e}")
            raise
        finally:
            # Clean up HTTP client if we own it (not borrowed from parent)
            if self._http_client is not None and self._owns_http_client:
                try:
                    await self._http_client.aclose()
                    log.debug("Closed HTTP client")
                except Exception as e:
                    log.warning(f"Error closing HTTP client: {e}")

            # Ensure storage and cache references are released
            # Note: auth providers are class-level singletons, not cleaned up per-scope
            self._asset_storage = None
            self._temp_storage = None
            self._node_cache = None
            self._memory_uri_cache = None
            self._http_client = None

    async def _acquire_db_resources(self) -> DBResources:
        """Acquire database resources from the appropriate pool.

        Auto-detects database type from environment and acquires
        connection from shared pool.

        Returns:
            DBResources instance (SQLiteScopeResources, PostgresScopeResources, or SupabaseScopeResources)
        """
        supabase_url = Environment.get_supabase_url()
        supabase_key = Environment.get_supabase_key()
        postgres_db = Environment.get("POSTGRES_DB")

        if supabase_url and supabase_key:
            from nodetool.runtime.db_supabase import SupabaseScopeResources

            try:
                from supabase import AsyncClient
            except ImportError as exc:  # pragma: no cover - only hit when optional dep missing
                raise ImportError(
                    "Supabase support requires the 'supabase' package. Install optional "
                    "dependencies or unset SUPABASE_URL/SUPABASE_KEY."
                ) from exc

            client = AsyncClient(supabase_url, supabase_key)
            return SupabaseScopeResources(client)
        elif postgres_db:
            from nodetool.runtime.db_postgres import PostgresConnectionPool, PostgresScopeResources

            db_params = Environment.get_postgres_params()
            conninfo = (
                f"dbname={db_params['database']} user={db_params['user']} "
                f"password={db_params['password']} host={db_params['host']} port={db_params['port']}"
            )
            pool = await PostgresConnectionPool.get_shared(conninfo)
            return PostgresScopeResources(pool)
        else:
            from nodetool.runtime.db_sqlite import (
                SQLiteConnectionPool,
                SQLiteScopeResources,
            )

            if self.pool is None:
                pool = await SQLiteConnectionPool.get_shared(Environment.get_db_path())
                return SQLiteScopeResources(pool)
            else:
                assert isinstance(self.pool, SQLiteConnectionPool), "Pool must be a SQLiteConnectionPool"
                return SQLiteScopeResources(self.pool)

    def get_asset_storage(self, use_s3: bool = False) -> AbstractStorage:
        """Get or create the asset storage adapter for this scope."""
        if self._asset_storage is None:
            # Check environment dynamically to support pytest-xdist workers
            import os

            is_pytest = "PYTEST_CURRENT_TEST" in os.environ or RUNNING_PYTEST
            if is_pytest:
                from nodetool.storage.memory_storage import MemoryStorage

                log.info("Using memory storage for asset storage")
                self._asset_storage = MemoryStorage(base_url=Environment.get_storage_api_url())
            else:
                supabase_url = Environment.get_supabase_url()
                supabase_key = Environment.get_supabase_key()
                if supabase_url and supabase_key:
                    from supabase import AsyncClient as SupabaseAsyncClient  # type: ignore

                    from nodetool.storage.supabase_storage import SupabaseStorage

                    log.info("Using Supabase storage for asset storage")
                    client = SupabaseAsyncClient(supabase_url, supabase_key)
                    self._asset_storage = SupabaseStorage(
                        bucket_name=Environment.get_asset_bucket(),
                        supabase_url=supabase_url,
                        client=client,
                    )
                elif Environment.is_production() or Environment.get_s3_access_key_id() is not None or use_s3:
                    log.info("Using S3 storage for asset storage")
                    self._asset_storage = self.get_s3_storage(
                        Environment.get_asset_bucket(),
                        Environment.get_asset_domain(),
                    )
                else:
                    from nodetool.storage.file_storage import FileStorage

                    base_path = Environment.get_asset_folder()
                    base_url = Environment.get_storage_api_url()
                    log.info(
                        "Using folder %s for asset storage with base url %s",
                        base_path,
                        base_url,
                    )
                    self._asset_storage = FileStorage(
                        base_path=base_path,
                        base_url=base_url,
                    )

        return self._asset_storage

    @classmethod
    def get_s3_storage(cls, bucket: str, domain: str):
        """
        Get the S3 service.
        """
        import boto3

        from nodetool.storage.s3_storage import S3Storage

        endpoint_url = Environment.get_s3_endpoint_url()
        access_key_id = Environment.get_s3_access_key_id()
        secret_access_key = Environment.get_s3_secret_access_key()

        assert access_key_id is not None, "AWS access key ID is required"
        assert secret_access_key is not None, "AWS secret access key is required"
        assert endpoint_url is not None, "S3 endpoint URL is required"

        client = boto3.client(
            "s3",
            region_name=Environment.get_s3_region(),
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

        return S3Storage(
            bucket_name=bucket,
            domain=domain,
            endpoint_url=endpoint_url,
            client=client,
        )

    def get_temp_storage(self, use_s3: bool = False) -> Any:
        """Get or create the temporary asset storage adapter for this scope."""
        if self._temp_storage is None:
            if not Environment.is_production():
                supabase_url = Environment.get_supabase_url()
                supabase_key = Environment.get_supabase_key()
                if supabase_url and supabase_key and Environment.get_asset_temp_bucket():
                    try:
                        from supabase import AsyncClient as SupabaseAsyncClient  # type: ignore

                        from nodetool.storage.supabase_storage import SupabaseStorage

                        log.info("Using Supabase storage for temp storage")
                        client = SupabaseAsyncClient(supabase_url, supabase_key)
                        self._temp_storage = SupabaseStorage(
                            bucket_name=Environment.get_asset_temp_bucket(),
                            supabase_url=supabase_url,
                            client=client,
                        )
                    except Exception as e:
                        log.error(f"Failed to initialize Supabase temp storage, using memory. Error: {e}")
                        from nodetool.storage.memory_storage import MemoryStorage

                        log.info("Using memory storage for temp storage")
                        self._temp_storage = MemoryStorage(
                            base_url=Environment.get_temp_storage_api_url(),
                        )
                else:
                    from nodetool.storage.memory_storage import MemoryStorage

                    log.info("Using memory storage for temp storage")
                    self._temp_storage = MemoryStorage(
                        base_url=Environment.get_temp_storage_api_url(),
                    )
            else:
                supabase_url = Environment.get_supabase_url()
                supabase_key = Environment.get_supabase_key()
                if supabase_url and supabase_key and Environment.get_asset_temp_bucket():
                    try:
                        from supabase import AsyncClient as SupabaseAsyncClient  # type: ignore

                        from nodetool.storage.supabase_storage import SupabaseStorage

                        log.info("Using Supabase storage for temp asset storage")
                        client = SupabaseAsyncClient(supabase_url, supabase_key)
                        self._temp_storage = SupabaseStorage(
                            bucket_name=Environment.get_asset_temp_bucket(),
                            supabase_url=supabase_url,
                            client=client,
                        )
                    except Exception as e:
                        log.error(f"Failed to initialize Supabase temp storage, falling back to S3. Error: {e}")
                        assert Environment.get_s3_access_key_id() is not None or use_s3, "S3 access key ID is required"
                        assert Environment.get_asset_temp_bucket() is not None, "Asset temp bucket is required"
                        assert Environment.get_asset_temp_domain() is not None, "Asset temp domain is required"
                        log.info("Using S3 storage for temp asset storage")
                        self._temp_storage = self.get_s3_storage(
                            Environment.get_asset_temp_bucket(),
                            Environment.get_asset_temp_domain(),
                        )
                else:
                    assert Environment.get_s3_access_key_id() is not None or use_s3, "S3 access key ID is required"
                    assert Environment.get_asset_temp_bucket() is not None, "Asset temp bucket is required"
                    assert Environment.get_asset_temp_domain() is not None, "Asset temp domain is required"
                    log.info("Using S3 storage for temp asset storage")
                    self._temp_storage = self.get_s3_storage(
                        Environment.get_asset_temp_bucket(),
                        Environment.get_asset_temp_domain(),
                    )

        return self._temp_storage

    def get_node_cache(self) -> Any:
        """Get or create the node cache for this scope.

        The node cache stores node execution results for cacheable nodes.
        Uses memcache if configured, otherwise uses in-memory cache.

        Returns:
            AbstractNodeCache instance for caching node results
        """
        if self._node_cache is None:
            memcache_host = Environment.get_memcache_host()
            memcache_port = Environment.get_memcache_port()

            if memcache_host and memcache_port:
                from nodetool.storage.memcache_node_cache import MemcachedNodeCache

                log.info("Using memcache for node cache")
                self._node_cache = MemcachedNodeCache(host=memcache_host, port=int(memcache_port))
            else:
                from nodetool.storage.memory_node_cache import MemoryNodeCache

                log.info("Using memory for node cache")
                self._node_cache = MemoryNodeCache()

        return self._node_cache

    def get_memory_uri_cache(self) -> Any:
        """Get or create the memory URI cache for this scope.

        The memory URI cache stores objects addressed by URIs (memory://, http(s)://).
        Uses an in-memory TTL cache with 5-minute default expiry.

        Returns:
            MemoryUriCache instance for caching URI content
        """
        if self._memory_uri_cache is None:
            from nodetool.storage.memory_uri_cache import MemoryUriCache

            log.info("Creating memory URI cache for scope")
            self._memory_uri_cache = MemoryUriCache(default_ttl=300)

        return self._memory_uri_cache

    def get_http_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client for this scope.

        The HTTP client is created per-scope to ensure it's bound to the correct
        event loop. This prevents "Future attached to different loop" errors.

        Returns:
            httpx.AsyncClient instance for making HTTP requests
        """
        if self._http_client is None:
            HTTP_HEADERS = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
            log.info("Creating HTTP client for scope")
            self._http_client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=600,
                verify=False,
                headers=HTTP_HEADERS.copy(),
            )
            # Mark that we own this HTTP client
            self._owns_http_client = True
        return self._http_client

    def get_static_auth_provider(self) -> Any:
        """Get or create the static token authentication provider.

        The static auth provider validates worker tokens for internal services.
        Uses a class-level singleton shared across all scopes for performance.

        Returns:
            StaticTokenAuthProvider instance

        Raises:
            ValueError: If WORKER_AUTH_TOKEN is not configured
        """
        from nodetool.deploy.auth import get_worker_auth_token
        from nodetool.security.providers.static_token import StaticTokenAuthProvider

        if ResourceScope._class_static_auth_provider is None:
            token = get_worker_auth_token()
            if not token:
                raise ValueError("WORKER_AUTH_TOKEN is required for static authentication.")
            ResourceScope._class_static_auth_provider = StaticTokenAuthProvider(static_token=token)
        return ResourceScope._class_static_auth_provider

    def get_user_auth_provider(self) -> Any:
        """Get or create the configured user authentication provider.

        Returns the appropriate provider based on AUTH_PROVIDER environment setting:
        - none: Returns None (no auth enforcement)
        - local: LocalAuthProvider (always returns user "1")
        - static: StaticTokenAuthProvider (shared token auth)
        - supabase: SupabaseAuthProvider (validates Supabase JWTs)

        Uses a class-level singleton shared across all scopes for performance.

        Returns:
            AuthProvider instance or None if auth is disabled
        """
        if ResourceScope._class_user_auth_provider is None:
            kind = Environment.get_auth_provider_kind()
            if kind == "none":
                ResourceScope._class_user_auth_provider = None
            elif kind == "local":
                from nodetool.security.providers.local import LocalAuthProvider

                ResourceScope._class_user_auth_provider = LocalAuthProvider()
            elif kind == "static":
                # Reuse static token provider for user auth path
                ResourceScope._class_user_auth_provider = self.get_static_auth_provider()
            elif kind == "supabase":
                supabase_url = Environment.get("SUPABASE_URL")
                supabase_key = Environment.get("SUPABASE_KEY")
                if supabase_url and supabase_key:
                    from nodetool.security.providers.supabase import SupabaseAuthProvider

                    ResourceScope._class_user_auth_provider = SupabaseAuthProvider(
                        supabase_url=supabase_url,
                        supabase_key=supabase_key,
                        cache_ttl=Environment.get_auth_cache_ttl(),
                        cache_max=Environment.get_auth_cache_max(),
                    )
                else:
                    ResourceScope._class_user_auth_provider = None
            else:
                ResourceScope._class_user_auth_provider = None
        return ResourceScope._class_user_auth_provider

    def get_supabase_client(cls):
        """
        Get the supabase client.
        """
        from supabase import create_async_client

        supabase_url = Environment.get_supabase_url()
        supabase_key = Environment.get_supabase_key()

        if supabase_url is None or supabase_key is None:
            raise Exception("Supabase URL or key is not set")

        return create_async_client(supabase_url, supabase_key)
