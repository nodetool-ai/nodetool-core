-- Migration to create workflow_versions table for storing workflow version history

CREATE TABLE IF NOT EXISTS nodetool_workflow_versions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    name TEXT DEFAULT '',
    description TEXT DEFAULT '',
    graph TEXT DEFAULT '{}'
);

-- Create index for efficient lookup by workflow_id
CREATE INDEX IF NOT EXISTS idx_nodetool_workflow_versions_workflow_id ON nodetool_workflow_versions(workflow_id);

-- Create unique constraint to ensure version numbers are unique per workflow
CREATE UNIQUE INDEX IF NOT EXISTS idx_nodetool_workflow_versions_workflow_version ON nodetool_workflow_versions(workflow_id, version);
