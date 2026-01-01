"""
Migration: Add tool_name column to workflows
Version: 20250928_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20250928_000000"
name = "add_tool_name_to_workflows"

creates_tables = []
modifies_tables = ["nodetool_workflows"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add tool_name column to workflows table."""
    columns = await db.get_columns("nodetool_workflows")

    if "tool_name" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_workflows ADD COLUMN tool_name TEXT
        """)


async def down(db: "MigrationDBAdapter") -> None:
    """Remove tool_name column from workflows table.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
