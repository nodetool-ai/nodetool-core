CREATE INDEX IF NOT EXISTS idx_prediction_user_provider
ON nodetool_predictions(user_id, provider)
CREATE INDEX IF NOT EXISTS idx_prediction_user_model
ON nodetool_predictions(user_id, model)
CREATE INDEX IF NOT EXISTS idx_prediction_created_at
ON nodetool_predictions(created_at)