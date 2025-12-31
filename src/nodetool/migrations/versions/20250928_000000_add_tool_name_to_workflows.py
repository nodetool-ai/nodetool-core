"""
Migration: Add tool_name column to workflows
Version: 20250928_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20250928_000000"
name = "add_tool_name_to_workflows"

# Tables this migration modifies
creates_tables = []
modifies_tables = ["nodetool_workflows"]


async def up(db: "aiosqlite.Connection") -> None:
    """Add tool_name column to workflows table."""
    cursor = await db.execute("PRAGMA table_info(nodetool_workflows)")
    columns = await cursor.fetchall()
    column_names = [col[1] for col in columns]

    if "tool_name" not in column_names:
        await db.execute("""
            ALTER TABLE nodetool_workflows ADD COLUMN tool_name TEXT
        """)


async def down(db: "aiosqlite.Connection") -> None:
    """Remove tool_name column from workflows table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
