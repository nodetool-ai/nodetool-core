-- Migration for Asset model

CREATE TABLE IF NOT EXISTS nodetool_assets (
    id TEXT PRIMARY KEY,
    type TEXT,
    user_id TEXT,
    workflow_id TEXT,
    parent_id TEXT,
    file_id TEXT,
    name TEXT,
    content_type TEXT,
    metadata JSONB,
    created_at TIMESTAMP,
    duration REAL
);

CREATE INDEX IF NOT EXISTS idx_nodetool_assets_user_id_parent_id ON nodetool_assets (user_id, parent_id); 