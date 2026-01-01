"""
Migration: Add logs column to jobs
Version: 20251010_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20251010_000000"
name = "add_logs_to_jobs"

creates_tables = []
modifies_tables = ["nodetool_jobs"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add logs column to jobs table."""
    columns = await db.get_columns("nodetool_jobs")

    if "logs" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_jobs ADD COLUMN logs TEXT
        """)


async def down(db: "MigrationDBAdapter") -> None:
    """Remove logs column from jobs table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
