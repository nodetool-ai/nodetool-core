"""
Migration: Add size column to assets
Version: 20250501_000002
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20250501_000002"
name = "add_size_to_assets"

# Tables this migration modifies
creates_tables = []
modifies_tables = ["nodetool_assets"]


async def up(db: "aiosqlite.Connection") -> None:
    """Add size column to assets table."""
    cursor = await db.execute("PRAGMA table_info(nodetool_assets)")
    columns = await cursor.fetchall()
    column_names = [col[1] for col in columns]

    if "size" not in column_names:
        await db.execute("""
            ALTER TABLE nodetool_assets ADD COLUMN size INTEGER
        """)


async def down(db: "aiosqlite.Connection") -> None:
    """Remove size column from assets table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
