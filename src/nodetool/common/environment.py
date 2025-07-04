import os
from typing import Any

from nodetool.common.nodetool_api_client import (
    NodetoolAPIClient,
    NODETOOL_INTERNAL_API,
)
from nodetool.storage.abstract_node_cache import AbstractNodeCache
from nodetool.common.settings import (
    get_system_data_path,
    load_settings,
    get_value,
    get_system_file_path,
    SETTINGS_FILE,
    SECRETS_FILE,
)

DEFAULT_ENV = {
    "ASSET_BUCKET": "images",
    "ASSET_DOMAIN": None,
    "ASSET_TEMP_BUCKET": None,
    "ASSET_TEMP_DOMAIN": None,
    "CHROMA_URL": None,
    "CHROMA_PATH": str(get_system_data_path("chroma")),
    "COMFY_FOLDER": None,
    "MEMCACHE_HOST": None,
    "MEMCACHE_PORT": None,
    "DB_PATH": str(get_system_file_path("nodetool.sqlite3")),
    "OLLAMA_API_URL": "http://localhost:11434",
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
- Secrets file (secrets.yaml)
- Default values

Key Features:
- Configuration hierarchy with environment variables taking precedence
- Secure secrets management
- Type-safe configuration access
- Environment-aware behavior (development/production/test)
- Service connection management (database, S3, memcache, etc.)
- Resource initialization (logging, Sentry, storage adapters)

The Environment class provides class methods to access all configuration values and
initialize required services. It supports both local development with file-based
storage and production deployment with cloud services.
"""


class Environment(object):
    """
    A class that manages environment variables and provides default values and type conversions.

    This class acts as a central place to manage environment variables and settings for the application.
    It provides methods to retrieve and set various configuration values, such as AWS credentials, API keys,
    database paths, and more.

    Settings and Secrets:
    The class supports loading and saving settings and secrets from/to YAML files. The settings file
    (`settings.yaml`) stores general configuration options, while the secrets file (`secrets.yaml`)
    stores sensitive information like API keys.

    Local Mode:
    In local mode (non-production environment), the class uses default values or prompts the user for
    input during the setup process. It also supports local file storage and SQLite database for
    development purposes.
    """

    settings: dict[str, Any] | None = None
    secrets: dict[str, Any] | None = None
    remote_auth: bool = True

    @classmethod
    def load_settings(cls):
        cls.settings, cls.secrets = load_settings()

    @classmethod
    def get_settings(cls):
        if cls.settings is None:
            cls.load_settings()
        assert cls.settings is not None
        return cls.settings

    @classmethod
    def get_secrets(cls):
        if cls.secrets is None:
            cls.load_settings()
        assert cls.secrets is not None
        return cls.secrets

    @classmethod
    def get_environment(cls):
        settings = cls.get_settings()
        secrets = cls.get_secrets()

        env = DEFAULT_ENV.copy()
        env.update(os.environ)

        for k, v in settings.items():
            if v is not None:
                env[k] = v
        for k, v in secrets.items():
            if v is not None:
                env[k] = v

        return env

    @classmethod
    def get(cls, key: str, default: Any = None):
        if cls.settings is None or cls.secrets is None:
            cls.load_settings()
        assert cls.settings is not None and cls.secrets is not None
        return get_value(key, cls.settings, cls.secrets, DEFAULT_ENV, default)

    @classmethod
    def has_settings(cls):
        return get_system_file_path(SETTINGS_FILE).exists()

    @classmethod
    def has_secrets(cls):
        return get_system_file_path(SECRETS_FILE).exists()

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
    def set_worker_url(cls, worker_url: str):
        """
        Set the worker url.
        """
        os.environ["WORKER_URL"] = worker_url

    @classmethod
    def set_nodetool_api_url(cls, nodetool_api_url: str):
        """
        Set the nodetool api url.
        """
        os.environ["NODETOOL_API_URL"] = nodetool_api_url

    @classmethod
    def use_remote_auth(cls):
        """
        A single local user with id 1 is used for authentication when this evaluates to False.
        """
        return cls.is_production() or cls.get("REMOTE_AUTH") == "1"

    @classmethod
    def is_debug(cls):
        """
        Is debug flag on?
        """
        return cls.get("DEBUG")

    @classmethod
    def get_log_level(cls):
        """
        The log level is the level of logging that we use.
        """
        return cls.get("LOG_LEVEL")

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
        cls.node_cache = node_cache

    @classmethod
    def get_node_cache(cls) -> AbstractNodeCache:
        memcache_host = cls.get_memcache_host()
        memcache_port = cls.get_memcache_port()

        if not hasattr(cls, "node_cache"):
            if memcache_host and memcache_port:
                from nodetool.storage.memcache_node_cache import MemcachedNodeCache

                cls.node_cache = MemcachedNodeCache(
                    host=memcache_host, port=int(memcache_port)
                )
            else:
                from nodetool.storage.memory_node_cache import MemoryNodeCache

                cls.node_cache = MemoryNodeCache()

        return cls.node_cache

    @classmethod
    def get_db_path(cls):
        """
        The database url is the url of the database.
        """
        if cls.is_test():
            return "/tmp/nodetool_test.db"
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
    def get_database_adapter(
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

            if cls.get_db_path() != ":memory:":
                os.makedirs(os.path.dirname(cls.get_db_path()), exist_ok=True)

            return SQLiteAdapter(
                db_path=cls.get_db_path(),
                fields=fields,
                table_schema=table_schema,
                indexes=indexes,
            )
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
    def get_worker_url(cls):
        """
        The worker url is the url of the worker server.
        """
        return os.environ.get("WORKER_URL")

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
    def get_worker_api_client(cls):
        from nodetool.common.worker_api_client import WorkerAPIClient

        worker_url = cls.get_worker_url()
        if worker_url:
            return WorkerAPIClient(base_url=worker_url)
        else:
            return None

    @classmethod
    def get_nodetool_api_client(
        cls, user_id: str, auth_token: str, api_url: str | None = None
    ) -> NodetoolAPIClient:
        """
        The nodetool api client is a wrapper around the nodetool api.
        """
        from httpx import AsyncClient, ASGITransport

        if api_url is None:
            api_url = cls.get_nodetool_api_url()

        if api_url is None or Environment.is_test():
            from nodetool.api.server import create_app  # type: ignore

            app = create_app()
            transport = ASGITransport(app=app)  # type: ignore
            return NodetoolAPIClient(
                user_id=user_id,
                auth_token=auth_token,
                base_url=NODETOOL_INTERNAL_API,
                client=AsyncClient(transport=transport),
            )
        else:
            return NodetoolAPIClient(
                user_id=user_id,
                auth_token=auth_token,
                base_url=api_url,
                client=AsyncClient(timeout=30, verify=False),
            )

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
        """
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
        if not hasattr(cls, "asset_storage"):
            if cls.is_test():
                from nodetool.storage.memory_storage import MemoryStorage

                cls.get_logger().info("Using memory storage for asset storage")

                cls.asset_storage = MemoryStorage(base_url=cls.get_storage_api_url())
            elif (
                cls.is_production() or cls.get_s3_access_key_id() is not None or use_s3
            ):
                cls.get_logger().info("Using S3 storage for asset storage")
                cls.asset_storage = cls.get_s3_storage(
                    cls.get_asset_bucket(), cls.get_asset_domain()
                )
            else:
                from nodetool.storage.file_storage import FileStorage

                cls.get_logger().info(
                    f"Using folder {cls.get_asset_folder()} for asset storage"
                )
                cls.asset_storage = FileStorage(
                    base_path=cls.get_asset_folder(),
                    base_url=cls.get_storage_api_url(),
                )

        assert cls.asset_storage is not None
        return cls.asset_storage

    @classmethod
    def get_logger(cls):
        """
        Get a logger.
        """
        import logging

        if not hasattr(cls, "logger"):
            cls.logger = logging.getLogger("nodetool")
            cls.logger.setLevel(cls.get_log_level())

            cls.logger.handlers.clear()

            handler = logging.StreamHandler()
            cls.logger.addHandler(handler)

            # Prevent propagation to parent loggers
            cls.logger.propagate = False

            # Disable httpx and httpcore logging
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)

        return cls.logger

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
        if not hasattr(cls, "temp_storage"):
            if not cls.is_production():
                # from nodetool.storage.memory_storage import MemoryStorage
                from nodetool.storage.file_storage import FileStorage

                cls.get_logger().info("Using memory storage for temp storage")
                # cls.temp_storage = MemoryStorage(
                #     base_url=cls.get_temp_storage_api_url()
                # )
                cls.temp_storage = FileStorage(
                    base_path="/tmp",
                    base_url=cls.get_temp_storage_api_url(),
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
                cls.temp_storage = cls.get_s3_storage(
                    cls.get_asset_temp_bucket(), cls.get_asset_temp_domain()
                )

        assert cls.temp_storage is not None
        return cls.temp_storage

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
