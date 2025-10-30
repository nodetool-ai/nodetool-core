import os
import threading
import tempfile
from pathlib import Path
from nodetool.config.logging_config import get_logger
from typing import Any, Optional, Dict, TYPE_CHECKING

from nodetool.storage.abstract_node_cache import AbstractNodeCache
from nodetool.config.settings import (
    get_system_data_path,
    load_settings,
    get_value,
    get_system_file_path,
    SETTINGS_FILE,
)

# Global test storage instances to avoid thread-local issues in tests
_test_asset_storage = None
_test_temp_storage = None
_test_db_path = None
_test_db_file = None

if TYPE_CHECKING:
    from nodetool.security.providers.static_token import StaticTokenAuthProvider
    from nodetool.security.providers.supabase import SupabaseAuthProvider

DEFAULT_ENV = {
    "ASSET_BUCKET": "images",
    "ASSET_DOMAIN": None,
    "ASSET_TEMP_BUCKET": None,
    "ASSET_TEMP_DOMAIN": None,
    "CHROMA_URL": None,
    "CHROMA_PATH": str(get_system_data_path("chroma")),
    "COMFY_FOLDER": None,
    "JOB_EXECUTION_STRATEGY": "threaded",  # threaded, subprocess, docker
    "MEMCACHE_HOST": None,
    "MEMCACHE_PORT": None,
    "DB_PATH": str(get_system_file_path("nodetool.sqlite3")),
    "OLLAMA_API_URL": "http://127.0.0.1:11434",
    "ENV": "development",
    "LOG_LEVEL": "INFO",
    "REMOTE_AUTH": "0",
    "DEBUG": None,
    "AWS_REGION": "us-east-1",
    "NODETOOL_API_URL": None,
    "SENTRY_DSN": None,
    "SUPABASE_URL": None,
    "SUPABASE_KEY": None,
}

NOT_GIVEN = object()

"""
Environment Configuration Management Module

This module provides centralized configuration management for the Nodetool application through
the Environment class. It handles loading and accessing configuration from multiple sources:

- Environment variables
- Settings file (settings.yaml)
- Default values

Key Features:
- Configuration hierarchy with environment variables taking precedence
- Type-safe configuration access
- Environment-aware behavior (development/production/test)
- Service connection management (database, S3, memcache, etc.)
- Resource initialization (logging, Sentry, storage adapters)

The Environment class provides class methods to access all configuration values and
initialize required services. It supports both local development with file-based
storage and production deployment with cloud services.
"""


def load_dotenv_files():
    """Load environment variables from .env files based on current environment."""
    from dotenv import load_dotenv

    # Get the project root directory (where .env files should be located)
    current_file = Path(__file__)
    project_root = (
        current_file.parent.parent.parent.parent
    )  # Go up to workspace/nodetool-core

    # Determine environment - check ENV var first, then default to development
    env_name = os.environ.get("ENV", "development")

    # Load .env files in order of precedence (later files override earlier ones)
    env_files = [
        project_root / ".env",  # Base .env file
        project_root / f".env.{env_name}",  # Environment-specific file
        project_root / f".env.{env_name}.local",  # Local overrides (gitignored)
    ]

    for env_file in env_files:
        if env_file.exists():
            # Be resilient to stubbed/mocked load_dotenv in tests that may not
            # accept keyword arguments.
            try:
                load_dotenv(env_file, override=False)  # type: ignore[arg-type]
            except TypeError:
                try:
                    load_dotenv(env_file)  # type: ignore[call-arg]
                except TypeError:
                    load_dotenv()  # type: ignore[misc]


class Environment(object):
    """
    A class that manages environment variables and provides default values and type conversions.

    This class acts as a central place to manage environment variables and settings for the application.
    It provides methods to retrieve and set various configuration values, such as AWS credentials, API keys,
    database paths, and more.

    Settings and Secrets:
    The class supports loading and saving settings from/to YAML files.

    Local Mode:
    In local mode (non-production environment), the class uses default values or prompts the user for
    input during the setup process. It also supports local file storage and SQLite database for
    development purposes.
    """

    settings: Optional[Dict[str, Any]] = None
    _sqlite_connection: Any = None
    remote_auth: bool = True
    _thread_local: threading.local = threading.local()
    _static_auth_provider: "StaticTokenAuthProvider | None" = None
    _user_auth_provider: "SupabaseAuthProvider | None" = None

    @classmethod
    def _tls(cls) -> threading.local:
        return cls._thread_local

    @classmethod
    def load_settings(cls):
        # Load .env files first
        load_dotenv_files()
        cls.settings = load_settings()

    @classmethod
    def get_settings(cls):
        if cls.settings is None:
            cls.load_settings()
        assert cls.settings is not None
        return cls.settings

    @classmethod
    def get_environment(cls):
        settings = cls.get_settings()

        env = DEFAULT_ENV.copy()
        env.update(os.environ)

        for k, v in settings.items():
            if v is not None:
                env[k] = v

        return env

    @classmethod
    def get(cls, key: str, default: Any = None):
        if cls.settings is None:
            cls.load_settings()
        assert cls.settings is not None
        return get_value(key, cls.settings, DEFAULT_ENV, default)

    @classmethod
    def has_settings(cls):
        return get_system_file_path(SETTINGS_FILE).exists()

    @classmethod
    def get_aws_region(cls):
        """
        The AWS region is the region where we run AWS services.
        """
        return cls.get("AWS_REGION")

    @classmethod
    def get_asset_bucket(cls):
        """
        The asset bucket is the S3 bucket where we store asset files.
        """
        return cls.get("ASSET_BUCKET")

    @classmethod
    def get_env(cls):
        """
        The environment is either "development" or "production".
        """
        return cls.get("ENV")

    @classmethod
    def set_env(cls, env: str):
        """
        Set the environment.
        """
        os.environ["ENV"] = env

    @classmethod
    def is_production(cls):
        """
        Is the environment production?
        """
        return cls.get_env() == "production"

    @classmethod
    def is_test(cls):
        """
        Is the environment test?
        """
        return os.environ.get("PYTEST_CURRENT_TEST") is not None

    @classmethod
    def set_remote_auth(cls, remote_auth: bool):
        """
        Set the remote auth flag.
        """
        os.environ["REMOTE_AUTH"] = "1" if remote_auth else "0"

    @classmethod
    def use_remote_auth(cls):
        """
        A single local user with id 1 is used for authentication when this evaluates to False.
        """
        return cls.is_production() or cls.get("REMOTE_AUTH") == "1"

    @classmethod
    def get_static_auth_provider(cls):
        """
        Return the static token authentication provider.
        """
        if cls._static_auth_provider is None:
            from nodetool.deploy.auth import get_worker_auth_token
            from nodetool.security.providers.static_token import StaticTokenAuthProvider

            token = get_worker_auth_token()
            if not token:
                raise ValueError(
                    "WORKER_AUTH_TOKEN is required for static authentication."
                )
            cls._static_auth_provider = StaticTokenAuthProvider(static_token=token)
        return cls._static_auth_provider

    @classmethod
    def _get_int_setting(cls, key: str, default: int) -> int:
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return default
        if cls.settings is None:
            cls.load_settings()
        assert cls.settings is not None
        raw = cls.settings.get(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    @classmethod
    def get_remote_auth_cache_ttl(cls) -> int:
        return cls._get_int_setting("REMOTE_AUTH_CACHE_TTL", 60)

    @classmethod
    def get_remote_auth_cache_max(cls) -> int:
        return cls._get_int_setting("REMOTE_AUTH_CACHE_MAX", 2000)

    @classmethod
    def get_user_auth_provider(cls):
        """
        Return the Supabase authentication provider if configured.
        """
        if cls._user_auth_provider is None:
            supabase_url = cls.get("SUPABASE_URL")
            supabase_key = cls.get("SUPABASE_KEY")
            if supabase_url and supabase_key:
                from nodetool.security.providers.supabase import SupabaseAuthProvider

                cls._user_auth_provider = SupabaseAuthProvider(
                    supabase_url=supabase_url,
                    supabase_key=supabase_key,
                    cache_ttl=cls.get_remote_auth_cache_ttl(),
                    cache_max=cls.get_remote_auth_cache_max(),
                )
        return cls._user_auth_provider

    @classmethod
    def is_debug(cls):
        """
        Is debug flag on?
        """
        return cls.get("DEBUG")

    @classmethod
    def get_log_level(cls):
        """Return desired log level string.

        Priority:
        1) Explicit LOG_LEVEL from settings/env via get()
        2) If DEBUG env is truthy, return "DEBUG"
        3) NODETOOL_LOG_LEVEL env (default "INFO")
        """
        level = os.getenv("LOG_LEVEL")
        if level:
            try:
                return str(level).upper()
            except Exception:
                return "INFO"
        debug_env = os.getenv("DEBUG")
        if debug_env and debug_env.lower() not in ("0", "false", "no", "off", ""):
            return "DEBUG"
        return os.getenv("NODETOOL_LOG_LEVEL", "INFO").upper()

    @classmethod
    def get_memcache_host(cls):
        """
        The memcache host is the host of the memcache server.
        """
        return os.environ.get("MEMCACHE_HOST")

    @classmethod
    def get_memcache_port(cls):
        """
        The memcache port is the port of the memcache server.
        """
        return os.environ.get("MEMCACHE_PORT")

    @classmethod
    def set_node_cache(cls, node_cache: AbstractNodeCache):
        setattr(cls._tls(), "node_cache", node_cache)

    @classmethod
    def get_default_execution_strategy(cls):
        """
        The execution strategy is the strategy that we use to execute the workflow.
        """
        strategy = cls.get("DEFAULT_EXECUTION_STRATEGY", None)
        if strategy:
            return strategy
        return cls.get("JOB_EXECUTION_STRATEGY", "threaded")

    @classmethod
    def get_node_cache(cls) -> AbstractNodeCache:
        memcache_host = cls.get_memcache_host()
        memcache_port = cls.get_memcache_port()

        if not hasattr(cls._tls(), "node_cache"):
            if memcache_host and memcache_port:
                from nodetool.storage.memcache_node_cache import MemcachedNodeCache

                setattr(
                    cls._tls(),
                    "node_cache",
                    MemcachedNodeCache(host=memcache_host, port=int(memcache_port)),
                )
            else:
                from nodetool.storage.memory_node_cache import MemoryNodeCache

                setattr(cls._tls(), "node_cache", MemoryNodeCache())

        return getattr(cls._tls(), "node_cache")

    @classmethod
    def set_memory_uri_cache(cls, uri_cache: AbstractNodeCache):
        """Override the default in-process memory URI cache (mainly for testing)."""
        setattr(cls._tls(), "memory_uri_cache", uri_cache)

    @classmethod
    def set_thread_memory_cache(cls, cache: dict):
        """Set a specific dictionary to be used as the memory cache for the current thread."""
        if not hasattr(cls._tls(), "memory_uri_cache_override"):
            cls._tls().memory_uri_cache_override = {}
        cls._tls().memory_uri_cache_override = cache

    @classmethod
    def clear_thread_memory_cache(cls):
        """Clear the thread-specific memory cache override."""
        if hasattr(cls._tls(), "memory_uri_cache_override"):
            delattr(cls._tls(), "memory_uri_cache_override")

    @classmethod
    def get_memory_uri_cache(cls) -> AbstractNodeCache:
        """
        Global cache for objects addressed by URIs.

        - Used for memory:// objects and downloaded http(s) blobs
        - Defaults to a simple in-memory TTL cache (5 minutes)
        """
        if hasattr(cls._tls(), "memory_uri_cache_override"):
            # If an override is set for this thread, wrap it in a compatible AbstractNodeCache interface.
            # This allows ProcessingContext's shared dict to be used wherever
            # Environment.get_memory_uri_cache() is called.
            from nodetool.storage.memory_uri_cache import MemoryUriCache

            cache_dict = getattr(cls._tls(), "memory_uri_cache_override")
            return MemoryUriCache(initial_data=cache_dict, default_ttl=300)

        if not hasattr(cls._tls(), "memory_uri_cache"):
            # Lazy import to avoid import cycles
            from nodetool.storage.memory_uri_cache import MemoryUriCache

            setattr(cls._tls(), "memory_uri_cache", MemoryUriCache(default_ttl=300))
        return getattr(cls._tls(), "memory_uri_cache")

    @classmethod
    def get_db_path(cls):
        """
        The database url is the url of the database.
        """
        if cls.is_test():
            # Use a temporary file-based database for tests
            global _test_db_path, _test_db_file
            if _test_db_path is None:
                _test_db_file = tempfile.NamedTemporaryFile(
                    suffix=".db", prefix="nodetool_test_", delete=False
                )
                _test_db_path = _test_db_file.name
                _test_db_file.close()
            return _test_db_path
        else:
            return cls.get("DB_PATH")

    @classmethod
    def get_postgres_params(cls):
        """
        The postgres params are the parameters that we use to connect to the database.
        """
        return {
            "database": cls.get("POSTGRES_DB"),
            "user": cls.get("POSTGRES_USER"),
            "password": cls.get("POSTGRES_PASSWORD"),
            "host": cls.get("POSTGRES_HOST"),
            "port": cls.get("POSTGRES_PORT"),
        }

    @classmethod
    def get_supabase_client(cls):
        """
        Get the supabase client.
        """
        from supabase import create_async_client

        supabase_url = cls.get_supabase_url()
        supabase_key = cls.get_supabase_key()

        if supabase_url is None or supabase_key is None:
            raise Exception("Supabase URL or key is not set")

        return create_async_client(supabase_url, supabase_key)

    @classmethod
    def has_database(cls):
        """
        Check if the database is configured.
        """
        return (
            cls.get("POSTGRES_DB", None) is not None
            or cls.get("SUPABASE_URL", None) is not None
            or cls.get_db_path() is not None
        )

    @classmethod
    async def get_database_adapter(
        cls,
        fields: dict[str, Any],
        table_schema: dict[str, Any],
        indexes: list[dict[str, Any]],
    ):
        """
        The database adapter is the adapter that we use to connect to the database.
        """
        if cls.get("POSTGRES_DB", None) is not None:
            from nodetool.models.postgres_adapter import PostgresAdapter  # type: ignore

            return PostgresAdapter(
                db_params=cls.get_postgres_params(),
                fields=fields,
                table_schema=table_schema,
                indexes=indexes,
            )
        elif cls.get("SUPABASE_URL", None) is not None:
            from nodetool.models.supabase_adapter import SupabaseAdapter  # type: ignore

            return SupabaseAdapter(
                supabase_url=cls.get_supabase_url(),
                supabase_key=cls.get_supabase_key(),
                fields=fields,
                table_schema=table_schema,
            )
        elif cls.get_db_path() is not None:
            from nodetool.models.sqlite_adapter import SQLiteAdapter  # type: ignore
            import aiosqlite

            # Use thread-local storage for SQLite connections to avoid database locks
            tls = cls._tls()
            if not hasattr(tls, "sqlite_connection") or tls.sqlite_connection is None:
                import threading

                cls.get_logger().debug(
                    f"Creating new SQLite connection for thread {threading.get_ident()}"
                )
                tls.sqlite_connection = await aiosqlite.connect(
                    cls.get_db_path(), timeout=30
                )
                tls.sqlite_connection.row_factory = aiosqlite.Row
                # Configure SQLite for better concurrency and deadlock avoidance
                await tls.sqlite_connection.execute("PRAGMA journal_mode=WAL")
                await tls.sqlite_connection.execute(
                    "PRAGMA busy_timeout=5000"
                )  # 5 seconds
                await tls.sqlite_connection.execute("PRAGMA synchronous=NORMAL")
                # Increase cache size for better performance (negative means KB)
                await tls.sqlite_connection.execute("PRAGMA cache_size=-64000")  # 64MB
                await tls.sqlite_connection.commit()
                # await tls.sqlite_connection.set_trace_callback(log.debug)

            db_path = cls.get_db_path()
            # Only create directories for file paths, not URIs or :memory:
            if db_path != ":memory:" and not db_path.startswith("file:"):
                os.makedirs(os.path.dirname(db_path), exist_ok=True)

            adapter = SQLiteAdapter(
                db_path=cls.get_db_path(),
                connection=tls.sqlite_connection,
                fields=fields,
                table_schema=table_schema,
                indexes=indexes,
            )

            await adapter.auto_migrate()

            return adapter

        else:
            raise Exception("No database adapter configured")

    @classmethod
    def get_s3_endpoint_url(cls):
        """
        The endpoint url is the url of the S3 server.
        """
        return os.environ.get("S3_ENDPOINT_URL", None)

    @classmethod
    def get_s3_access_key_id(cls):
        """
        The access key id is the id of the AWS user.
        """
        # If we are in production, we don't need an access key id.
        # We use the IAM role instead.
        return os.environ.get("S3_ACCESS_KEY_ID", None)

    @classmethod
    def get_s3_secret_access_key(cls):
        """
        The secret access key is the secret of the AWS user.
        """
        # If we are in production, we don't need a secret access key.
        # We use the IAM role instead.
        return os.environ.get("S3_SECRET_ACCESS_KEY", None)

    @classmethod
    def get_s3_region(cls):
        """
        The region name is the region of the S3 server.
        """
        return os.environ.get("S3_REGION", cls.get_aws_region())

    @classmethod
    def get_asset_domain(cls):
        """
        The asset domain is the domain where assets are stored.
        """
        return cls.get("ASSET_DOMAIN")

    @classmethod
    def get_nodetool_api_url(cls):
        """
        The nodetool api url is the url of the nodetool api server.
        """
        return cls.get("NODETOOL_API_URL") or "http://localhost:8000"

    @classmethod
    def get_storage_api_url(cls):
        """
        The storage API endpoint.
        """
        return f"{cls.get_nodetool_api_url()}/api/storage"

    @classmethod
    def get_temp_storage_api_url(cls):
        """
        The temp storage API endpoint.
        """
        return f"{cls.get_nodetool_api_url()}/api/storage/temp"

    @classmethod
    def get_chroma_token(cls):
        """
        The chroma token is the token of the chroma server.
        """
        return cls.get("CHROMA_TOKEN")

    @classmethod
    def get_chroma_url(cls):
        """
        The chroma url is the url of the chroma server.
        """
        return cls.get("CHROMA_URL")

    @classmethod
    def get_chroma_path(cls):
        """
        The chroma path is the path of the chroma server.
        """
        return cls.get("CHROMA_PATH")

    # @classmethod
    # def get_chroma_settings(cls):
    #     from chromadb.config import Settings

    #     if cls.get_chroma_url() is not None:
    #         return Settings(
    #             chroma_api_impl="chromadb.api.fastapi.FastAPI",
    #             chroma_client_auth_provider="token",
    #             chroma_client_auth_credentials=cls.get_chroma_token(),
    #             # chroma_server_host=cls.get_chroma_url(),
    #         )
    #     else:
    #         return Settings(
    #             chroma_api_impl="chromadb.api.segment.SegmentAPI",
    #             is_persistent=True,
    #             persist_directory="multitenant",
    #         )

    @classmethod
    def get_comfy_folder(cls):
        """
        The comfy folder is the folder where ComfyUI is located.
        """
        return cls.get("COMFY_FOLDER")

    @classmethod
    def get_asset_folder(cls):
        """
        The asset folder is the folder where assets are located.
        Can be overridden with ASSET_FOLDER environment variable for containerized deployments.
        If ASSET_BUCKET looks like a filesystem path, it will be used instead of S3.
        """
        # Check for explicit override first (for Docker/containerized deployments)
        override = os.environ.get("ASSET_FOLDER")
        if override:
            return str(override)

        # Check if ASSET_BUCKET looks like a filesystem path (starts with / or .)
        # This allows using ASSET_BUCKET for both S3 bucket names and filesystem paths
        asset_bucket = cls.get("ASSET_BUCKET")
        if asset_bucket and (
            asset_bucket.startswith("/") or asset_bucket.startswith(".")
        ):
            return str(asset_bucket)

        # Default to system-specific path
        return str(get_system_file_path("assets"))

    @classmethod
    def get_s3_storage(cls, bucket: str, domain: str):
        """
        Get the S3 service.
        """
        from nodetool.storage.s3_storage import S3Storage
        import boto3

        endpoint_url = cls.get_s3_endpoint_url()
        access_key_id = cls.get_s3_access_key_id()
        secret_access_key = cls.get_s3_secret_access_key()

        assert access_key_id is not None, "AWS access key ID is required"
        assert secret_access_key is not None, "AWS secret access key is required"
        assert endpoint_url is not None, "S3 endpoint URL is required"

        client = boto3.client(
            "s3",
            region_name=cls.get_s3_region(),
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

    @classmethod
    def get_asset_storage(cls, use_s3: bool = False):
        """
        Get the storage adapter for assets.
        """
        global _test_asset_storage

        if cls.is_test() and _test_asset_storage is not None:
            return _test_asset_storage

        if not hasattr(cls._tls(), "asset_storage"):
            if cls.is_test():
                from nodetool.storage.memory_storage import MemoryStorage

                cls.get_logger().info("Using memory storage for asset storage")

                storage = MemoryStorage(base_url=cls.get_storage_api_url())
                _test_asset_storage = storage
                setattr(cls._tls(), "asset_storage", storage)
            elif (
                cls.is_production() or cls.get_s3_access_key_id() is not None or use_s3
            ):
                cls.get_logger().info("Using S3 storage for asset storage")
                setattr(
                    cls._tls(),
                    "asset_storage",
                    cls.get_s3_storage(cls.get_asset_bucket(), cls.get_asset_domain()),
                )
            else:
                from nodetool.storage.file_storage import FileStorage

                cls.get_logger().info(
                    f"Using folder {cls.get_asset_folder()} for asset storage with base url {cls.get_storage_api_url()}"
                )
                setattr(
                    cls._tls(),
                    "asset_storage",
                    FileStorage(
                        base_path=cls.get_asset_folder(),
                        base_url=cls.get_storage_api_url(),
                    ),
                )

        asset_storage = getattr(cls._tls(), "asset_storage")
        assert asset_storage is not None
        return asset_storage

    @classmethod
    def set_asset_storage(cls, asset_storage):
        """Override the default asset storage (mainly for testing)."""
        setattr(cls._tls(), "asset_storage", asset_storage)

    @classmethod
    def clear_test_storage(cls):
        """Clear global test storage instances and clean up test database."""
        global _test_asset_storage, _test_temp_storage, _test_db_path, _test_db_file
        _test_asset_storage = None
        _test_temp_storage = None

        # Clean up test database file
        if _test_db_path is not None:
            try:
                if os.path.exists(_test_db_path):
                    os.unlink(_test_db_path)
            except Exception as e:
                # Log but don't fail - file might be locked or already deleted
                logger = cls.get_logger()
                logger.debug(
                    f"Could not delete test database file {_test_db_path}: {e}"
                )
            _test_db_path = None
            _test_db_file = None

    @classmethod
    def get_logger(cls):
        """Return the shared nodetool logger using centralized config."""
        return get_logger("nodetool")

    @classmethod
    def get_torch_device(cls):
        """
        Get the torch device.
        Returns None if torch is not installed.
        """
        try:
            import torch
        except Exception:
            return None

        try:
            if torch.backends.mps.is_available():
                import torch.mps

                return torch.device("mps")
        except Exception:
            pass

        try:
            import torch.cuda

            if torch.cuda.device_count() > 0:
                return torch.device("cuda")
        except Exception:
            return torch.device("cpu")

    @classmethod
    def initialize_sentry(cls):
        """
        Initialize Sentry error tracking if SENTRY_DSN is configured.
        """
        sentry_dsn = cls.get("SENTRY_DSN", None)
        if sentry_dsn:
            import sentry_sdk  # type: ignore

            sentry_sdk.init(
                dsn=sentry_dsn,
                environment=cls.get_env(),
                # Set traces_sample_rate to 1.0 to capture 100%
                # of transactions for performance monitoring.
                traces_sample_rate=1.0,
                # Set profiles_sample_rate to 1.0 to profile 100%
                # of sampled transactions.
                profiles_sample_rate=1.0,
            )
            cls.get_logger().info("Sentry initialized")

    @classmethod
    def get_asset_temp_bucket(cls):
        """
        The temp asset bucket is the S3 bucket where temporary assets are stored.
        """
        return cls.get("ASSET_TEMP_BUCKET")

    @classmethod
    def get_asset_temp_domain(cls):
        """
        The temp asset domain is the domain where temporary assets are stored.
        """
        return cls.get("ASSET_TEMP_DOMAIN")

    @classmethod
    def get_temp_storage(cls, use_s3: bool = False):
        """
        Get the storage adapter for temporary assets.
        """
        if not hasattr(cls._tls(), "temp_storage"):
            if not cls.is_production():
                from nodetool.storage.memory_storage import MemoryStorage

                cls.get_logger().info("Using memory storage for temp storage")
                setattr(
                    cls._tls(),
                    "temp_storage",
                    MemoryStorage(base_url=cls.get_temp_storage_api_url()),
                )
            else:
                assert (
                    cls.get_s3_access_key_id() is not None or use_s3
                ), "S3 access key ID is required"
                assert (
                    cls.get_asset_temp_bucket() is not None
                ), "Asset temp bucket is required"
                assert (
                    cls.get_asset_temp_domain() is not None
                ), "Asset temp domain is required"
                cls.get_logger().info("Using S3 storage for temp asset storage")
                setattr(
                    cls._tls(),
                    "temp_storage",
                    cls.get_s3_storage(
                        cls.get_asset_temp_bucket(), cls.get_asset_temp_domain()
                    ),
                )

        temp_storage = getattr(cls._tls(), "temp_storage")
        assert temp_storage is not None
        return temp_storage

    @classmethod
    def clear_thread_caches(cls):
        """Clear per-thread caches to avoid cross-workflow leaks."""
        tls = cls._tls()

        # Close SQLite connection if it exists
        if hasattr(tls, "sqlite_connection") and tls.sqlite_connection is not None:
            try:
                import asyncio
                import threading

                cls.get_logger().debug(
                    f"Closing SQLite connection for thread {threading.get_ident()}"
                )
                # If we're in an async context, schedule the close
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(tls.sqlite_connection.close())
                except RuntimeError:
                    # No running loop, connection will be closed when GC'd
                    pass
            except Exception:
                pass
            finally:
                tls.sqlite_connection = None

        for attr in (
            "node_cache",
            "memory_uri_cache",
            "asset_storage",
            "temp_storage",
        ):
            if hasattr(tls, attr):
                try:
                    delattr(tls, attr)
                except Exception:
                    pass

        if cls._user_auth_provider is not None:
            cls._user_auth_provider.clear_caches()
            cls._user_auth_provider = None
        cls._static_auth_provider = None

    @classmethod
    def get_supabase_url(cls):
        """
        The Supabase URL.
        """
        return cls.get("SUPABASE_URL")

    @classmethod
    def get_supabase_key(cls):
        """
        The Supabase service key.
        """
        return cls.get("SUPABASE_KEY")
