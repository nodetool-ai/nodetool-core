"""
Migration: Add execution state fields to jobs table (unified Job/RunState model)
Version: 20260125_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260125_000000"
name = "add_execution_state_to_jobs"

creates_tables = []
modifies_tables = ["nodetool_jobs"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add execution state columns to nodetool_jobs table."""
    # These columns are from the RunState model that we're merging into Job
    columns = [
        ("status", "TEXT"),
        ("updated_at", "TEXT"),
        ("suspended_node_id", "TEXT"),
        ("suspension_reason", "TEXT"),
        ("suspension_state_json", "TEXT"),
        ("suspension_metadata_json", "TEXT"),
        ("completed_at", "TEXT"),
        ("failed_at", "TEXT"),
        ("error_message", "TEXT"),
        ("execution_strategy", "TEXT"),
        ("execution_id", "TEXT"),
        ("worker_id", "TEXT"),
        ("heartbeat_at", "TEXT"),
        ("retry_count", "INTEGER DEFAULT 0"),
        ("max_retries", "INTEGER DEFAULT 3"),
        ("metadata_json", "TEXT"),
        ("version", "INTEGER DEFAULT 0"),
    ]

    for col_name, col_type in columns:
        # Check if column exists to make it idempotent
        if not await db.column_exists("nodetool_jobs", col_name):
            await db.execute(f"ALTER TABLE nodetool_jobs ADD COLUMN {col_name} {col_type}")

    # Add indices for efficient queries
    await db.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON nodetool_jobs(status)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_job_updated ON nodetool_jobs(updated_at)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_job_worker ON nodetool_jobs(worker_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_job_heartbeat ON nodetool_jobs(heartbeat_at)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_job_recovery ON nodetool_jobs(status, heartbeat_at)")


async def down(db: "MigrationDBAdapter") -> None:
    """Revert changes."""
    # Drop indices
    await db.execute("DROP INDEX IF EXISTS idx_job_recovery")
    await db.execute("DROP INDEX IF EXISTS idx_job_heartbeat")
    await db.execute("DROP INDEX IF EXISTS idx_job_worker")
    await db.execute("DROP INDEX IF EXISTS idx_job_updated")
    await db.execute("DROP INDEX IF EXISTS idx_job_status")

    # Drop columns (if SQLite >= 3.35 supports it)
    columns = [
        "status",
        "version",
        "metadata_json",
        "max_retries",
        "retry_count",
        "heartbeat_at",
        "worker_id",
        "execution_id",
        "execution_strategy",
        "error_message",
        "failed_at",
        "completed_at",
        "suspension_metadata_json",
        "suspension_state_json",
        "suspension_reason",
        "suspended_node_id",
        "updated_at",
    ]

    for col in columns:
        try:
            if await db.column_exists("nodetool_jobs", col):
                await db.execute(f"ALTER TABLE nodetool_jobs DROP COLUMN {col}")
        except Exception:
            # SQLite < 3.35 doesn't support DROP COLUMN, ignore
            pass
