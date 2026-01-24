"""
Migration: Add node_id and job_id fields to assets table
Version: 20260124_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260124_000000"
name = "add_node_job_fields_to_assets"

creates_tables = []
modifies_tables = ["nodetool_assets"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add node_id and job_id columns to assets table."""
    try:
        await db.execute("""
            ALTER TABLE nodetool_assets
            ADD COLUMN node_id TEXT DEFAULT NULL
        """)
    except Exception:
        pass

    try:
        await db.execute("""
            ALTER TABLE nodetool_assets
            ADD COLUMN job_id TEXT DEFAULT NULL
        """)
    except Exception:
        pass

    await db.commit()


async def down(db: "MigrationDBAdapter") -> None:
    """Remove the added columns."""
    await db.execute("ALTER TABLE nodetool_assets DROP COLUMN job_id")
    await db.execute("ALTER TABLE nodetool_assets DROP COLUMN node_id")
    await db.commit()
