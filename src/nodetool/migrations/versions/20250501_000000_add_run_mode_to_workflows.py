"""
Migration: Add run_mode to workflows
Version: 20250501_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20250501_000000"
name = "add_run_mode_to_workflows"

# Tables this migration modifies
creates_tables = []
modifies_tables = ["nodetool_workflows"]


async def up(db: "aiosqlite.Connection") -> None:
    """Add run_mode column to workflows table."""
    # Check if column exists first
    cursor = await db.execute("PRAGMA table_info(nodetool_workflows)")
    columns = await cursor.fetchall()
    column_names = [col[1] for col in columns]

    if "run_mode" not in column_names:
        await db.execute("""
            ALTER TABLE nodetool_workflows ADD COLUMN run_mode TEXT
        """)


async def down(db: "aiosqlite.Connection") -> None:
    """Remove run_mode column from workflows table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety - the column will remain but be unused.
    """
    # SQLite 3.35.0+ supports DROP COLUMN, but for compatibility we leave it
    pass
