"""
Database migration stub.

Migrations are now handled by the TypeScript server (Drizzle).
This module provides basic table creation for standalone Python usage and tests.
"""

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


async def run_startup_migrations(pool=None) -> None:
    """Create database tables if they don't exist.

    In production, the TypeScript server handles migrations via Drizzle.
    This function ensures tables exist for standalone Python usage and tests.
    """
    from nodetool.models.asset import Asset
    from nodetool.models.job import Job
    from nodetool.models.message import Message
    from nodetool.models.oauth_credential import OAuthCredential
    from nodetool.models.prediction import Prediction
    from nodetool.models.run_event import RunEvent
    from nodetool.models.run_inbox_message import RunInboxMessage
    from nodetool.models.run_lease import RunLease
    from nodetool.models.run_node_state import RunNodeState
    from nodetool.models.secret import Secret
    from nodetool.models.thread import Thread
    from nodetool.models.trigger_input import TriggerInput
    from nodetool.models.workflow import Workflow
    from nodetool.models.workspace import Workspace
    from nodetool.runtime.resources import ResourceScope, maybe_scope

    models = [Asset, Job, Message, OAuthCredential, Prediction, RunEvent, RunInboxMessage, RunLease, RunNodeState, Secret, Thread, TriggerInput, Workflow, Workspace]

    # If we're already in a scope, create tables directly
    scope = maybe_scope()
    if scope is not None:
        for model in models:
            try:
                await model.create_table()
            except Exception as e:
                log.debug(f"Table creation for {model.__name__}: {e}")
        log.info("Database tables ensured.")
        return

    # Otherwise create a temporary scope using the provided pool
    if pool is not None:
        async with ResourceScope(pool=pool):
            for model in models:
                try:
                    await model.create_table()
                except Exception as e:
                    log.debug(f"Table creation for {model.__name__}: {e}")
        log.info("Database tables ensured.")
        return

    log.info("No pool or scope available — skipping table creation.")
