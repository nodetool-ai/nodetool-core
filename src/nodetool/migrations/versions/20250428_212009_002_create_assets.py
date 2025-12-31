"""
Migration: Create assets table
Version: 20250428_212009_002
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20250428_212009_002"
name = "create_assets"

# Tables this migration creates
creates_tables = ["nodetool_assets"]
modifies_tables = []


async def up(db: "aiosqlite.Connection") -> None:
    """Create the assets table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_assets (
            id TEXT PRIMARY KEY,
            type TEXT,
            user_id TEXT,
            workflow_id TEXT,
            parent_id TEXT,
            file_id TEXT,
            name TEXT,
            content_type TEXT,
            metadata TEXT,
            created_at TEXT,
            duration REAL
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodetool_assets_user_id_parent_id
        ON nodetool_assets (user_id, parent_id)
    """)


async def down(db: "aiosqlite.Connection") -> None:
    """Drop the assets table."""
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_assets_user_id_parent_id")
    await db.execute("DROP TABLE IF EXISTS nodetool_assets")
