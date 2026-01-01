"""
Migration: Create messages table
Version: 20250428_212009_004
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20250428_212009_004"
name = "create_messages"

creates_tables = ["nodetool_messages"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the messages table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_messages (
            id TEXT PRIMARY KEY,
            user_id TEXT DEFAULT '',
            workflow_id TEXT,
            graph TEXT,
            thread_id TEXT,
            tools TEXT,
            tool_call_id TEXT,
            role TEXT,
            name TEXT,
            content TEXT,
            tool_calls TEXT,
            collections TEXT,
            input_files TEXT,
            output_files TEXT,
            created_at TEXT,
            provider TEXT,
            model TEXT,
            cost REAL,
            agent_mode INTEGER,
            help_mode INTEGER,
            agent_execution_id TEXT,
            execution_event_type TEXT,
            workflow_target TEXT
        )
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the messages table."""
    await db.execute("DROP TABLE IF EXISTS nodetool_messages")
