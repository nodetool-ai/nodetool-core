-- Migration to add size column to assets table

-- Add the size column for file size in bytes (NULL for folders)
ALTER TABLE nodetool_assets ADD COLUMN IF NOT EXISTS size INTEGER;
