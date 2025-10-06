-- Migration to add tool_name field to workflows table

ALTER TABLE nodetool_workflows
ADD COLUMN IF NOT EXISTS tool_name TEXT;

