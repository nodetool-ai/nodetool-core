"""
Migration: Add run_mode to workflows
Version: 20250501_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20250501_000000"
name = "add_run_mode_to_workflows"

creates_tables = []
modifies_tables = ["nodetool_workflows"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add run_mode column to workflows table."""
    columns = await db.get_columns("nodetool_workflows")

    if "run_mode" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_workflows ADD COLUMN run_mode TEXT
        """)


async def down(db: "MigrationDBAdapter") -> None:
    """Remove run_mode column from workflows table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety - the column will remain but be unused.
    """
    pass
