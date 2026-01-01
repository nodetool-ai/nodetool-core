"""
Migration: RunState extensions for job execution
Version: 20260101_000000
"""

from typing import TYPE_CHECKING

from nodetool.migrations.db_adapter import MigrationDBAdapter

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20260101_000000"
name = "add_job_execution_fields"

creates_tables = []
modifies_tables = ["run_state"]


async def up(db: "MigrationDBAdapter") -> None:
    """Add new columns for execution tracking."""
    # Add new columns manually using SQL
    # Using TEXT for datetime storage (ISO format) for compatibility
    columns = [
        ("execution_strategy", "TEXT"),
        ("execution_id", "TEXT"),
        ("worker_id", "TEXT"),
        ("heartbeat_at", "TEXT"),
        ("retry_count", "INTEGER DEFAULT 0"),
        ("max_retries", "INTEGER DEFAULT 3"),
        ("metadata_json", "TEXT"),
    ]

    for col_name, col_type in columns:
        # Check if column exists to make it idempotent
        if not await db.column_exists("run_state", col_name):
            await db.execute(f"ALTER TABLE run_state ADD COLUMN {col_name} {col_type}")

    # Add indices
    await db.execute("CREATE INDEX IF NOT EXISTS idx_run_state_worker ON run_state(worker_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_run_state_heartbeat ON run_state(heartbeat_at)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_run_state_recovery ON run_state(status, heartbeat_at)")


async def down(db: "MigrationDBAdapter") -> None:
    """Revert changes."""
    # Drop indices
    await db.execute("DROP INDEX IF EXISTS idx_run_state_recovery")
    await db.execute("DROP INDEX IF EXISTS idx_run_state_heartbeat")
    await db.execute("DROP INDEX IF EXISTS idx_run_state_worker")

    # Drop columns
    columns = [
        "metadata_json",
        "max_retries",
        "retry_count",
        "heartbeat_at",
        "worker_id",
        "execution_id",
        "execution_strategy",
    ]

    for col in columns:
        try:
            if await db.column_exists("run_state", col):
                await db.execute(f"ALTER TABLE run_state DROP COLUMN {col}")
        except Exception:
            # SQLite < 3.35 doesn't support DROP COLUMN, ignore
            pass
