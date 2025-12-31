"""
Migration: Create secrets table
Version: 20251019_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20251019_000000"
name = "create_secrets"

# Tables this migration creates
creates_tables = ["nodetool_secrets"]
modifies_tables = []


async def up(db: "aiosqlite.Connection") -> None:
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

    # Create index for user_id and key combination (unique constraint)
    await db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nodetool_secrets_user_id_key
        ON nodetool_secrets (user_id, key)
    """)

    # Create index for user_id for listing secrets
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodetool_secrets_user_id
        ON nodetool_secrets (user_id)
    """)


async def down(db: "aiosqlite.Connection") -> None:
    """Drop the secrets table."""
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_secrets_user_id_key")
    await db.execute("DROP INDEX IF EXISTS idx_nodetool_secrets_user_id")
    await db.execute("DROP TABLE IF EXISTS nodetool_secrets")
