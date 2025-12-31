"""
Migration: Create oauth_credentials table
Version: 20251225_000000
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

version = "20251225_000000"
name = "create_oauth_credentials"

# Tables this migration creates
creates_tables = ["nodetool_oauth_credentials"]
modifies_tables = []


async def up(db: "aiosqlite.Connection") -> None:
    """Create the oauth_credentials table."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS nodetool_oauth_credentials (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            account_id TEXT NOT NULL,
            username TEXT,
            encrypted_access_token TEXT NOT NULL,
            encrypted_refresh_token TEXT,
            token_type TEXT DEFAULT 'Bearer',
            scope TEXT,
            received_at TEXT NOT NULL,
            expires_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Create index for user_id and provider combination
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_oauth_credentials_user_provider
        ON nodetool_oauth_credentials (user_id, provider)
    """)

    # Create unique index for user_id, provider, and account_id
    await db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_oauth_credentials_user_provider_account
        ON nodetool_oauth_credentials (user_id, provider, account_id)
    """)


async def down(db: "aiosqlite.Connection") -> None:
    """Drop the oauth_credentials table."""
    await db.execute("DROP INDEX IF EXISTS idx_oauth_credentials_user_provider")
    await db.execute("DROP INDEX IF EXISTS idx_oauth_credentials_user_provider_account")
    await db.execute("DROP TABLE IF EXISTS nodetool_oauth_credentials")
