-- Add new fields to predictions table for cost tracking
ALTER TABLE nodetool_predictions ADD COLUMN total_tokens INTEGER;
ALTER TABLE nodetool_predictions ADD COLUMN cached_tokens INTEGER;
ALTER TABLE nodetool_predictions ADD COLUMN reasoning_tokens INTEGER;
ALTER TABLE nodetool_predictions ADD COLUMN metadata TEXT;

-- Create indexes for cost aggregation queries
CREATE INDEX IF NOT EXISTS idx_prediction_user_provider ON nodetool_predictions(user_id, provider);
CREATE INDEX IF NOT EXISTS idx_prediction_user_model ON nodetool_predictions(user_id, model);
CREATE INDEX IF NOT EXISTS idx_prediction_created_at ON nodetool_predictions(created_at);
