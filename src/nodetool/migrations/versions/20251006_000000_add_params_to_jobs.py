"""
Migration: Add params column to jobs
Version: 20251006_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20251006_000000"
name = "add_params_to_jobs"

# Tables this migration modifies
creates_tables = []
modifies_tables = ["nodetool_jobs"]


async def up(db: "aiosqlite.Connection") -> None:
    """Add params column to jobs table."""
    cursor = await db.execute("PRAGMA table_info(nodetool_jobs)")
    columns = await cursor.fetchall()
    column_names = [col[1] for col in columns]

    if "params" not in column_names:
        await db.execute("""
            ALTER TABLE nodetool_jobs ADD COLUMN params TEXT
        """)


async def down(db: "aiosqlite.Connection") -> None:
    """Remove params column from jobs table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
