"""
Migration: Add autosave fields to workflow_versions table
Version: 20260104_000001
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260104_000001"
name = "add_autosave_fields_to_workflow_versions_v2"

creates_tables = []
modifies_tables = ["nodetool_workflow_versions"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add save_type and autosave_metadata columns to workflow_versions table."""
    try:
        await db.execute("""
            ALTER TABLE nodetool_workflow_versions
            ADD COLUMN save_type TEXT DEFAULT 'manual' CHECK(save_type IN ('autosave', 'manual', 'checkpoint', 'restore'))
        """)
    except Exception:
        pass

    try:
        await db.execute("""
            ALTER TABLE nodetool_workflow_versions
            ADD COLUMN autosave_metadata TEXT DEFAULT '{}'
        """)
    except Exception:
        pass

    try:
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodetool_workflow_versions_save_type
            ON nodetool_workflow_versions (workflow_id, save_type, created_at)
        """)
    except Exception:
        pass

    await db.commit()


async def down(db: "MigrationDBAdapter") -> None:
    """Remove the added columns and index."""
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_workflow_versions_save_type")
    await db.execute("ALTER TABLE nodetool_workflow_versions DROP COLUMN autosave_metadata")
    await db.execute("ALTER TABLE nodetool_workflow_versions DROP COLUMN save_type")
    await db.commit()
