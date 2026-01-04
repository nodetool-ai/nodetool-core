CREATE TABLE IF NOT EXISTS nodetool_workflow_versions (
id TEXT PRIMARY KEY,
workflow_id TEXT NOT NULL,
user_id TEXT NOT NULL,
version INTEGER NOT NULL DEFAULT 1,
created_at TEXT NOT NULL,
name TEXT DEFAULT '',
description TEXT DEFAULT '',
graph TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_nodetool_workflow_versions_workflow_id
ON nodetool_workflow_versions (workflow_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_nodetool_workflow_versions_workflow_version
ON nodetool_workflow_versions (workflow_id, version);