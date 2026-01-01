"""
Migration: Create secrets table
Version: 20251019_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.migrations.db_adapter import MigrationDBAdapter

version = "20251019_000000"
name = "create_secrets"

creates_tables = ["nodetool_secrets"]
modifies_tables = []


async def up(db: "MigrationDBAdapter") -> None:
    """Create the secrets table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_secrets (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            encrypted_value TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    await db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nodetool_secrets_user_id_key
        ON nodetool_secrets (user_id, key)
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodetool_secrets_user_id
        ON nodetool_secrets (user_id)
    """)


async def down(db: "MigrationDBAdapter") -> None:
    """Drop the secrets table."""
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_secrets_user_id_key")
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_secrets_user_id")
    await db.execute("DROP TABLE IF EXISTS nodetool_secrets")
