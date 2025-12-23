-- Create provider_calls table
CREATE TABLE IF NOT EXISTS nodetool_provider_calls (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    cost REAL NOT NULL DEFAULT 0.0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER,
    reasoning_tokens INTEGER,
    created_at TEXT NOT NULL,
    metadata TEXT
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_provider_call_user_id ON nodetool_provider_calls(user_id);
CREATE INDEX IF NOT EXISTS idx_provider_call_provider ON nodetool_provider_calls(provider);
CREATE INDEX IF NOT EXISTS idx_provider_call_model_id ON nodetool_provider_calls(model_id);
CREATE INDEX IF NOT EXISTS idx_provider_call_user_provider ON nodetool_provider_calls(user_id, provider);
CREATE INDEX IF NOT EXISTS idx_provider_call_user_model ON nodetool_provider_calls(user_id, model_id);
CREATE INDEX IF NOT EXISTS idx_provider_call_created_at ON nodetool_provider_calls(created_at);

