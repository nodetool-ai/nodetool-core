"""
Migration: Add logs column to jobs
Version: 20251010_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20251010_000000"
name = "add_logs_to_jobs"

# Tables this migration modifies
creates_tables = []
modifies_tables = ["nodetool_jobs"]


async def up(db: "aiosqlite.Connection") -> None:
    """Add logs column to jobs table."""
    cursor = await db.execute("PRAGMA table_info(nodetool_jobs)")
    columns = await cursor.fetchall()
    column_names = [col[1] for col in columns]

    if "logs" not in column_names:
        await db.execute("""
            ALTER TABLE nodetool_jobs ADD COLUMN logs TEXT
        """)


async def down(db: "aiosqlite.Connection") -> None:
    """Remove logs column from jobs table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
