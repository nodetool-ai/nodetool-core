"""
Migration: Add params column to jobs
Version: 20251006_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20251006_000000"
name = "add_params_to_jobs"

creates_tables = []
modifies_tables = ["nodetool_jobs"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add params column to jobs table."""
    columns = await db.get_columns("nodetool_jobs")

    if "params" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_jobs ADD COLUMN params TEXT
        """)


async def down(db: "MigrationDBAdapter") -> None:
    """Remove params column from jobs table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
