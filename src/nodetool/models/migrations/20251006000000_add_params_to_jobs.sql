-- Migration to add params column to jobs table

ALTER TABLE nodetool_jobs
ADD COLUMN IF NOT EXISTS params JSONB;
