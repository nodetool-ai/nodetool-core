"""
Migration: Create jobs table
Version: 20250428_212009_005
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20250428_212009_005"
name = "create_jobs"

creates_tables = ["nodetool_jobs"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the jobs table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_jobs (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            job_type TEXT,
            status TEXT,
            workflow_id TEXT,
            started_at TEXT,
            finished_at TEXT,
            graph TEXT,
            error TEXT,
            cost REAL
        )
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the jobs table."""
    await db.execute("DROP TABLE IF EXISTS nodetool_jobs")
