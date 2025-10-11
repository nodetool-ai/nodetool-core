-- Migration to add logs column to jobs table

ALTER TABLE nodetool_jobs
ADD COLUMN IF NOT EXISTS logs JSONB;
