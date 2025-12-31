"""
Migration: Create threads table
Version: 20250428_212009_003
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20250428_212009_003"
name = "create_threads"

# Tables this migration creates
creates_tables = ["nodetool_threads"]
modifies_tables = []


async def up(db: "aiosqlite.Connection") -> None:
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


async def down(db: "aiosqlite.Connection") -> None:
    """Drop the threads table."""
    await db.execute("DROP TABLE IF EXISTS nodetool_threads")
