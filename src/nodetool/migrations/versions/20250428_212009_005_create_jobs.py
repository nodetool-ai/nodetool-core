"""
Migration: Create jobs table
Version: 20250428_212009_005
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20250428_212009_005"
name = "create_jobs"

# Tables this migration creates
creates_tables = ["nodetool_jobs"]
modifies_tables = []


async def up(db: "aiosqlite.Connection") -> None:
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


async def down(db: "aiosqlite.Connection") -> None:
    """Drop the jobs table."""
    await db.execute("DROP TABLE IF EXISTS nodetool_jobs")
