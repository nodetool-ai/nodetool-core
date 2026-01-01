"""
Migration: Create workflows table
Version: 20250428_212009_001
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20250428_212009_001"
name = "create_workflows"

creates_tables = ["nodetool_workflows"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the workflows table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_workflows (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            access TEXT,
            created_at TEXT,
            updated_at TEXT,
            name TEXT,
            tags TEXT,
            description TEXT,
            thumbnail TEXT,
            graph TEXT,
            settings TEXT,
            receive_clipboard INTEGER
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodetool_workflows_user_id
        ON nodetool_workflows (user_id)
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the workflows table."""
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_workflows_user_id")
    await db.execute("DROP TABLE IF EXISTS nodetool_workflows")
