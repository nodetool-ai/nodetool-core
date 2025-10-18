-- Migration for Prediction model

CREATE TABLE IF NOT EXISTS nodetool_predictions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    node_id TEXT,
    provider TEXT,
    model TEXT,
    workflow_id TEXT,
    error TEXT,
    logs TEXT,
    status TEXT,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    cost REAL,
    duration REAL,
    hardware TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER
);
