-- Migration for Workflow model

CREATE TABLE IF NOT EXISTS nodetool_workflows (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    access TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    name TEXT,
    tags JSONB,
    description TEXT,
    thumbnail TEXT,
    graph JSONB,
    settings JSONB,
    receive_clipboard BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_nodetool_workflows_user_id ON nodetool_workflows (user_id);
