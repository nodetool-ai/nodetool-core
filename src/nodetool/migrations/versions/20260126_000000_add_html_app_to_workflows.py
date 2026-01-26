"""
Migration: Add html_app column to workflows
Version: 20260126_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260126_000000"
name = "add_html_app_to_workflows"

creates_tables = []
modifies_tables = ["nodetool_workflows"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add html_app column to workflows table."""
    columns = await db.get_columns("nodetool_workflows")

    if "html_app" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_workflows ADD COLUMN html_app TEXT
        """)


async def down(db: "MigrationDBAdapter") -> None:
    """Remove html_app column from workflows table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
