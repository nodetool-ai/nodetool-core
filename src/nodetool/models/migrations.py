"""
Database migration system for NodeTool.

Provides startup migrations, migration tracking, and coordination to prevent
race conditions during schema initialization.
"""

import asyncio
from typing import Set, Type

from nodetool.config.logging_config import get_logger
from nodetool.runtime.db_sqlite import SQLiteConnectionPool

log = get_logger(__name__)

# Only one migration at a time
_migration_lock = asyncio.Lock()


def get_all_models() -> list[Type]:
    """Get all DBModel classes that need migration.

    Returns:
        List of model classes in dependency order
    """
    from nodetool.models.asset import Asset
    from nodetool.models.job import Job
    from nodetool.models.message import Message
    from nodetool.models.oauth_credential import OAuthCredential
    from nodetool.models.prediction import Prediction
    from nodetool.models.secret import Secret
    from nodetool.models.thread import Thread
    from nodetool.models.workflow import Workflow
    from nodetool.models.workflow_version import WorkflowVersion

    # Order matters: migrations run in this order to handle foreign keys
    return [
        Workflow,
        WorkflowVersion,
        Asset,
        Thread,
        Message,
        Job,
        Prediction,
        Secret,
        OAuthCredential,
    ]


async def run_startup_migrations(pool: SQLiteConnectionPool | None = None) -> None:
    """Run all database migrations at application startup.

    This should be called once during server initialization before accepting
    requests. It ensures all tables are created and migrated before the
    application starts processing requests.

    Args:
        db_path: Optional database path. If None, uses environment config.

    Raises:
        Exception: If any migration fails
    """
    from nodetool.runtime.resources import ResourceScope

    log.info("Starting database migrations...")

    async with ResourceScope(pool=pool) as scope:
        models = get_all_models()
        assert scope.db

        async with _migration_lock:
            for model_cls in models:
                table_name = model_cls.get_table_name()

                try:
                    adapter = await scope.db.adapter_for_model(model_cls)
                    await adapter.auto_migrate()

                except Exception as e:
                    log.error(f"Failed to migrate table {table_name}: {e}", exc_info=True)
                    raise

    log.info(f"Database migrations completed successfully ({len(models)} tables)")
