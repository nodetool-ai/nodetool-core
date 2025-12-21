-- Migration: Create trigger_inputs table
-- Durable storage for trigger events that wake up workflows
-- Provides idempotent delivery and cross-process coordination

CREATE TABLE IF NOT EXISTS trigger_inputs (
    id TEXT PRIMARY KEY,
    
    -- Identification
    input_id TEXT NOT NULL UNIQUE,  -- Idempotency key
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    
    -- Payload
    payload_json TEXT,  -- JSON
    
    -- Processing state
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    processed_at TIMESTAMP,
    
    -- Optional cursor for ordered triggers
    cursor TEXT,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trigger_input_run_node_processed ON trigger_inputs(run_id, node_id, processed);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trigger_input_id ON trigger_inputs(input_id);
