"""
Migration: Create run_inbox_messages table
Version: 20251228_000002

Durable inbox for idempotent node message delivery.
Supports at-least-once and exactly-once semantics.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20251228_000002"
name = "create_run_inbox_messages"

creates_tables = ["run_inbox_messages"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the run_inbox_messages table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS run_inbox_messages (
            id TEXT PRIMARY KEY,

            message_id TEXT NOT NULL UNIQUE,
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            handle TEXT NOT NULL,

            msg_seq INTEGER NOT NULL,

            payload_json TEXT,
            payload_ref TEXT,

            status TEXT NOT NULL,
            claim_worker_id TEXT,
            claim_expires_at TEXT,
            consumed_at TEXT,

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_inbox_run_node_handle_seq
        ON run_inbox_messages(run_id, node_id, handle, msg_seq)
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_inbox_run_node_handle_status
        ON run_inbox_messages(run_id, node_id, handle, status)
    """)

    await db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_inbox_message_id
        ON run_inbox_messages(message_id)
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the run_inbox_messages table."""
    await db.execute("DROP INDEX IF EXISTS idx_inbox_run_node_handle_seq")
    await db.execute("DROP INDEX IF EXISTS idx_inbox_run_node_handle_status")
    await db.execute("DROP INDEX IF EXISTS idx_inbox_message_id")
    await db.execute("DROP TABLE IF EXISTS run_inbox_messages")
