-- Migration: Create run_node_state table
-- Authoritative source of truth for per-node execution state
-- Replaces event log projection as source of correctness

CREATE TABLE IF NOT EXISTS run_node_state (
    id TEXT PRIMARY KEY,  -- Format: "{run_id}::{node_id}"
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    
    -- Current state
    status TEXT NOT NULL,  -- idle | scheduled | running | completed | failed | suspended
    attempt INTEGER NOT NULL DEFAULT 1,
    
    -- Timestamps
    scheduled_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    failed_at TIMESTAMP,
    suspended_at TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Failure information
    last_error TEXT,
    retryable BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Suspension/resumption state
    suspension_reason TEXT,
    resume_state_json TEXT,  -- JSON
    
    -- Outputs (optional, may be large)
    outputs_json TEXT  -- JSON
);

CREATE INDEX IF NOT EXISTS idx_run_node_state_run_status ON run_node_state(run_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_run_node_state_run_node ON run_node_state(run_id, node_id);
