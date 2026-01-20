import os
import socket
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from nodetool.config.env_guard import (
    RUNNING_PYTEST,
    get_system_env,
    get_system_env_value,
)
from nodetool.config.logging_config import get_logger
from nodetool.config.settings import (
    SETTINGS_FILE,
    get_system_data_path,
    get_system_file_path,
    get_value,
    load_settings,
)

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
    "DB_PATH": str(get_system_data_path("nodetool.sqlite3")),
    "OLLAMA_API_URL": None,  # Must be explicitly configured; defaults to None to avoid connection errors in containers
    "ENV": "development",
    "LOG_LEVEL": "INFO",
    "AUTH_PROVIDER": "local",  # valid: none, local, static, supabase
    "DEBUG": None,
    "AWS_REGION": "us-east-1",
    "NODETOOL_API_URL": None,
    "NODETOOL_ENABLE_TERMINAL_WS": "1",  # Enable terminal WebSocket in dev/test (blocked in production)
    "WORKER_ID": None,
    "SENTRY_DSN": None,
    "SUPABASE_URL": None,
    "SUPABASE_KEY": None,
    "NODE_SUPABASE_URL": None,
    "NODE_SUPABASE_KEY": None,
    "NODE_SUPABASE_SCHEMA": None,
    "NODE_SUPABASE_TABLE_PREFIX": None,
}

NOT_GIVEN = object()


if RUNNING_PYTEST:
    DEFAULT_ENV["ENV"] = "test"

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

    from nodetool.config.logging_config import get_logger

    logger = get_logger(__name__)

    if RUNNING_PYTEST:
        logger.info("Skipping .env loading when running under pytest")
        return

    # Get the project root directory (where .env files should be located)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent  # Go up to workspace/nodetool-core

    # Determine environment - check ENV var first, then default to development
    env_name = get_system_env_value("ENV", DEFAULT_ENV.get("ENV", "development"))
    if env_name == "test":
        logger.info("Not loading env file for tests")
        return

    logger.debug(f"Loading environment: {env_name}")

    # Load .env files in order of precedence (later files override earlier ones)
    env_files = [
        project_root / ".env",  # Base .env file
        project_root / f".env.{env_name}",  # Environment-specific file
        project_root / f".env.{env_name}.local",  # Local overrides (gitignored)
    ]

    loaded_files = []
    for env_file in env_files:
        if env_file.exists():
            logger.info(f"Loading environment file: {env_file}")
            loaded_files.append(str(env_file))
            load_dotenv(env_file, override=False)  # type: ignore[arg-type]
        else:
            logger.debug(f"Environment file not found: {env_file}")

    if loaded_files:
        logger.info(f"Loaded {len(loaded_files)} environment file(s): {loaded_files}")
    else:
        logger.warning("No environment files found - using defaults and system environment variables only")

    # Log key non-secret environment variables that were loaded
    safe_log_vars = [
        "ENV",
        "LOG_LEVEL",
        "DEBUG",
        "AUTH_PROVIDER",
        "ASSET_BUCKET",
        "ASSET_DOMAIN",
        "POSTGRES_DB",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "SUPABASE_URL",
        "NODE_SUPABASE_URL",
        "DB_PATH",
        "OLLAMA_API_URL",
        "CHROMA_PATH",
        "S3_REGION",
        "S3_ENDPOINT_URL",
        "JOB_EXECUTION_STRATEGY",
    ]

    loaded_vars = {}
    for var in safe_log_vars:
        value = get_system_env_value(var)
        if value:
            # Mask URLs for privacy
            if var.endswith("_URL") or var.endswith("_ENDPOINT_URL"):
                # Show protocol and domain only
                try:
                    from urllib.parse import urlparse

                    parsed = urlparse(value)
                    masked = f"{parsed.scheme}://{parsed.netloc}"
                    loaded_vars[var] = masked
                except Exception:
                    loaded_vars[var] = "***"
            else:
                loaded_vars[var] = value

    if loaded_vars:
        logger.debug(f"Environment variables loaded: {loaded_vars}")
    else:
        logger.debug("No notable environment variables found")


class Environment:
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

    settings: Optional[dict[str, Any]] = None
    _sqlite_connection: Any = None

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
        env.update(get_system_env())

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
    def get_auth_provider_kind(cls) -> str:
        """Return the configured auth provider: none, local, static, supabase.

        Defaults to "local" in development. If an unknown value is provided,
        falls back to "local".
        """
        kind = str(cls.get("AUTH_PROVIDER", "local")).lower()
        if kind not in ("none", "local", "static", "supabase"):
            kind = "local"
        return kind

    @classmethod
    def enforce_auth(cls) -> bool:
        """Whether HTTP/WebSocket endpoints should enforce authentication.

        - none/local: no enforcement (developer convenience)
        - static/supabase: enforce
        """
        return cls.get_auth_provider_kind() in ("static", "supabase")

    @classmethod
    def _get_int_setting(cls, key: str, default: int) -> int:
        value = get_system_env_value(key)
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
    def get_auth_cache_ttl(cls) -> int:
        # Backward compatible with REMOTE_AUTH_CACHE_TTL
        ttl = get_system_env_value("AUTH_CACHE_TTL")
        if ttl is not None:
            try:
                return int(ttl)
            except ValueError:
                pass
        return cls._get_int_setting("REMOTE_AUTH_CACHE_TTL", 60)

    @classmethod
    def get_auth_cache_max(cls) -> int:
        # Backward compatible with REMOTE_AUTH_CACHE_MAX
        mx = get_system_env_value("AUTH_CACHE_MAX")
        if mx is not None:
            try:
                return int(mx)
            except ValueError:
                pass
        return cls._get_int_setting("REMOTE_AUTH_CACHE_MAX", 2000)

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
        level = get_system_env_value("LOG_LEVEL")
        if level:
            try:
                return str(level).upper()
            except Exception:
                return "INFO"
        debug_env = get_system_env_value("DEBUG")
        if debug_env and debug_env.lower() not in ("0", "false", "no", "off", ""):
            return "DEBUG"
        return get_system_env_value("NODETOOL_LOG_LEVEL", "INFO").upper()

    @classmethod
    def get_memcache_host(cls):
        """
        The memcache host is the host of the memcache server.
        """
        return get_system_env_value("MEMCACHE_HOST")

    @classmethod
    def get_memcache_port(cls):
        """
        The memcache port is the port of the memcache server.
        """
        return get_system_env_value("MEMCACHE_PORT")

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
    def get_db_path(cls):
        """
        The database url is the url of the sqlite database.
        """
        if RUNNING_PYTEST:
            import tempfile
            from pathlib import Path

            return str(Path(tempfile.gettempdir()) / "nodetool_test_subprocess.db")
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
    def get_s3_endpoint_url(cls):
        """
        The endpoint url is the url of the S3 server.
        """
        return get_system_env_value("S3_ENDPOINT_URL", None)

    @classmethod
    def get_s3_access_key_id(cls):
        """
        The access key id is the id of the AWS user.
        """
        # If we are in production, we don't need an access key id.
        # We use the IAM role instead.
        return get_system_env_value("S3_ACCESS_KEY_ID", None)

    @classmethod
    def get_s3_secret_access_key(cls):
        """
        The secret access key is the secret of the AWS user.
        """
        # If we are in production, we don't need a secret access key.
        # We use the IAM role instead.
        return get_system_env_value("S3_SECRET_ACCESS_KEY", None)

    @classmethod
    def get_s3_region(cls):
        """
        The region name is the region of the S3 server.
        """
        return get_system_env_value("S3_REGION", cls.get_aws_region())

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
        return cls.get("NODETOOL_API_URL") or "http://localhost:7777"

    @classmethod
    def get_storage_api_url(cls):
        """
        The storage API endpoint path or full URL if domain is configured.
        """
        storage_domain = cls.get_asset_domain()
        if storage_domain:
            storage_domain = storage_domain.rstrip("/")
        if cls.is_production() and storage_domain:
            return storage_domain
        return "/api/storage"

    @classmethod
    def get_temp_storage_api_url(cls):
        """
        The temp storage API endpoint.
        """
        temp_domain = cls.get_asset_temp_domain()
        if temp_domain:
            temp_domain = temp_domain.rstrip("/")
        if cls.is_production() and temp_domain:
            return temp_domain
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
        override = get_system_env_value("ASSET_FOLDER")
        if override:
            return str(override)

        # Check if ASSET_BUCKET looks like a filesystem path (starts with / or .)
        # This allows using ASSET_BUCKET for both S3 bucket names and filesystem paths
        asset_bucket = cls.get("ASSET_BUCKET")
        if asset_bucket and (asset_bucket.startswith("/") or asset_bucket.startswith(".")):
            return str(asset_bucket)

        # Default to system-specific path
        return str(get_system_data_path("assets"))

    @classmethod
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
            # Check if CUDA is available - this can raise RuntimeError if CUDA is not compiled
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                if torch.cuda.device_count() > 0:
                    return torch.device("cuda")
        except (RuntimeError, AttributeError):
            # PyTorch not compiled with CUDA support or other CUDA-related error
            pass

        return torch.device("cpu")

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

    @classmethod
    def get_node_supabase_url(cls):
        """
        Supabase URL for user-provided nodes (kept separate from core SUPABASE_URL).
        """
        return cls.get("NODE_SUPABASE_URL")

    @classmethod
    def get_node_supabase_key(cls):
        """
        Supabase service key for user-provided nodes (kept separate from core SUPABASE_KEY).
        """
        return cls.get("NODE_SUPABASE_KEY")

    @classmethod
    def get_node_supabase_schema(cls):
        """
        Optional schema name for user-provided nodes.
        """
        return cls.get("NODE_SUPABASE_SCHEMA")

    @classmethod
    def get_node_supabase_table_prefix(cls):
        """
        Optional prefix applied to user-provided node tables to avoid clashes with core tables.
        """
        return cls.get("NODE_SUPABASE_TABLE_PREFIX")

    @classmethod
    def clear_thread_caches(cls):
        """
        Clear any thread-local caches.

        This method is called when cleaning up thread resources to prevent
        memory leaks and cross-workflow contamination.
        """
        # Currently no thread-local caches in Environment
        # This method exists to prevent errors when called from threaded_event_loop
        pass

    _worker_id: Optional[str] = None

    @classmethod
    def get_worker_id(cls) -> str:
        """
        Get the unique worker ID for this instance.
        If not configured, generates one based on hostname + pid + uuid.
        """
        if cls._worker_id is not None:
            return cls._worker_id

        # 1. Try environment/settings
        worker_id = cls.get("WORKER_ID")

        # 2. Auto-generate if missing
        if not worker_id:
            hostname = socket.gethostname()
            pid = os.getpid()
            unique_suffix = str(uuid.uuid4())[:8]
            worker_id = f"{hostname}-{pid}-{unique_suffix}"

        cls._worker_id = worker_id
        return worker_id
