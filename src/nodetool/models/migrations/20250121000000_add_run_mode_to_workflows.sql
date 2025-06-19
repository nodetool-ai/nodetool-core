-- Migration to add run_mode field to workflows table

-- Add the run_mode column
ALTER TABLE nodetool_workflows ADD COLUMN IF NOT EXISTS run_mode TEXT;

-- Migrate existing data from settings to run_mode
UPDATE nodetool_workflows 
SET run_mode = settings->>'run_mode' 
WHERE settings IS NOT NULL 
  AND settings->>'run_mode' IS NOT NULL;

-- Remove run_mode from settings JSON
UPDATE nodetool_workflows 
SET settings = settings - 'run_mode'
WHERE settings IS NOT NULL 
  AND settings ? 'run_mode';