"""
Migration: Add package_name and thumbnail_url to workflows
Version: 20250501_000001
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20250501_000001"
name = "add_package_name_and_thumbnail_url_to_workflows"

# Tables this migration modifies
creates_tables = []
modifies_tables = ["nodetool_workflows"]


async def up(db: "aiosqlite.Connection") -> None:
    """Add package_name and thumbnail_url columns to workflows table."""
    cursor = await db.execute("PRAGMA table_info(nodetool_workflows)")
    columns = await cursor.fetchall()
    column_names = [col[1] for col in columns]

    if "package_name" not in column_names:
        await db.execute("""
            ALTER TABLE nodetool_workflows ADD COLUMN package_name TEXT
        """)

    if "thumbnail_url" not in column_names:
        await db.execute("""
            ALTER TABLE nodetool_workflows ADD COLUMN thumbnail_url TEXT
        """)


async def down(db: "aiosqlite.Connection") -> None:
    """Remove package_name and thumbnail_url columns.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
