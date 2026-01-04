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
CREATE INDEX IF NOT EXISTS idx_oauth_credentials_user_provider
ON nodetool_oauth_credentials (user_id, provider)
CREATE UNIQUE INDEX IF NOT EXISTS idx_oauth_credentials_user_provider_account
ON nodetool_oauth_credentials (user_id, provider, account_id)