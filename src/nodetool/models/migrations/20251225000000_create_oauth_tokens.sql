-- Migration for OAuthToken model

CREATE TABLE IF NOT EXISTS nodetool_oauth_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    account_id TEXT NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    token_type TEXT NOT NULL DEFAULT 'bearer',
    scope TEXT NOT NULL DEFAULT '',
    received_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Create unique index for user_id, provider, and account_id combination
CREATE UNIQUE INDEX IF NOT EXISTS idx_nodetool_oauth_tokens_user_provider_account 
    ON nodetool_oauth_tokens (user_id, provider, account_id);

-- Create index for user_id and provider for listing tokens
CREATE INDEX IF NOT EXISTS idx_nodetool_oauth_tokens_user_provider 
    ON nodetool_oauth_tokens (user_id, provider);

-- Create index for user_id for all user tokens
CREATE INDEX IF NOT EXISTS idx_nodetool_oauth_tokens_user_id 
    ON nodetool_oauth_tokens (user_id);
