CREATE TABLE IF NOT EXISTS nodetool_secrets (
id TEXT PRIMARY KEY,
user_id TEXT NOT NULL,
key TEXT NOT NULL,
encrypted_value TEXT NOT NULL,
description TEXT,
created_at TEXT NOT NULL,
updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_nodetool_secrets_user_id_key
ON nodetool_secrets (user_id, key);
CREATE INDEX IF NOT EXISTS idx_nodetool_secrets_user_id
ON nodetool_secrets (user_id);