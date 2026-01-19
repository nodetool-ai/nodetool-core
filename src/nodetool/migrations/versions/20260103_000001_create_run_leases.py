"""
Migration: Create run_leases table
Version: 20260103_000001

Lease-based locking for distributed workflow execution.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260103_000001"
name = "create_run_leases"

creates_tables = ["run_leases"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the run_leases table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS run_leases (
            run_id TEXT PRIMARY KEY,
            worker_id TEXT NOT NULL,
            acquired_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_run_leases_expires
        ON run_leases(expires_at)
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the run_leases table."""
    await db.execute("DROP INDEX IF EXISTS idx_run_leases_expires")
    await db.execute("DROP TABLE IF EXISTS run_leases")
