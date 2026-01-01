"""
Migration: Add agent execution fields to messages
Version: 20251011_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20251011_000000"
name = "add_agent_execution_fields_to_messages"

creates_tables = []
modifies_tables = ["nodetool_messages"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add agent_execution_id and execution_event_type columns to messages table."""
    columns = await db.get_columns("nodetool_messages")

    if "agent_execution_id" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_messages ADD COLUMN agent_execution_id TEXT
        """)

    if "execution_event_type" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_messages ADD COLUMN execution_event_type TEXT
        """)


async def down(db: "MigrationDBAdapter") -> None:
    """Remove agent execution fields from messages table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
