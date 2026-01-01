"""
Migration: Add package_name and thumbnail_url to workflows
Version: 20250501_000001
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20250501_000001"
name = "add_package_name_and_thumbnail_url_to_workflows"

creates_tables = []
modifies_tables = ["nodetool_workflows"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add package_name and thumbnail_url columns to workflows table."""
    columns = await db.get_columns("nodetool_workflows")

    if "package_name" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_workflows ADD COLUMN package_name TEXT
        """)

    if "thumbnail_url" not in columns:
        await db.execute("""
            ALTER TABLE nodetool_workflows ADD COLUMN thumbnail_url TEXT
        """)


async def down(db: "MigrationDBAdapter") -> None:
    """Remove package_name and thumbnail_url columns.

    Note: SQLite doesn't support DROP COLUMN directly in older versions.
    This is a no-op for safety.
    """
    pass
