"""
Migration: Add agent execution fields to messages
Version: 20251011_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20251011_000000"
name = "add_agent_execution_fields_to_messages"

# Tables this migration modifies
creates_tables = []
modifies_tables = ["nodetool_messages"]


async def up(db: "aiosqlite.Connection") -> None:
    """Add agent_execution_id and execution_event_type columns to messages table."""
    cursor = await db.execute("PRAGMA table_info(nodetool_messages)")
    columns = await cursor.fetchall()
    column_names = [col[1] for col in columns]

    if "agent_execution_id" not in column_names:
        await db.execute("""
            ALTER TABLE nodetool_messages ADD COLUMN agent_execution_id TEXT
        """)

    if "execution_event_type" not in column_names:
        await db.execute("""
            ALTER TABLE nodetool_messages ADD COLUMN execution_event_type TEXT
        """)


async def down(db: "aiosqlite.Connection") -> None:
    """Remove agent execution fields from messages table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
