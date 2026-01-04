ALTER TABLE nodetool_workflow_versions
ADD COLUMN save_type TEXT DEFAULT 'manual' CHECK(save_type IN ('autosave', 'manual', 'checkpoint', 'restore'));
ALTER TABLE nodetool_workflow_versions
ADD COLUMN autosave_metadata TEXT DEFAULT '{}';
CREATE INDEX IF NOT EXISTS idx_nodetool_workflow_versions_save_type
ON nodetool_workflow_versions (workflow_id, save_type, created_at);