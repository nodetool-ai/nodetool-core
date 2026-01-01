"""
Migration: Create threads table
Version: 20250428_212009_003
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20250428_212009_003"
name = "create_threads"

creates_tables = ["nodetool_threads"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the threads table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_threads (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the threads table."""
    await db.execute("DROP TABLE IF EXISTS nodetool_threads")
