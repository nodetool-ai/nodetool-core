-- Migration to add package_name and thumbnail_url fields to workflows table

ALTER TABLE nodetool_workflows
ADD COLUMN IF NOT EXISTS package_name TEXT;

ALTER TABLE nodetool_workflows
ADD COLUMN IF NOT EXISTS thumbnail_url TEXT;
