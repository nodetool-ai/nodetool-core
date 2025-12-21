-- Migration: Create run_state table
-- Authoritative source of truth for workflow run status
-- Replaces event log as source of correctness

CREATE TABLE IF NOT EXISTS run_state (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,  -- running | suspended | completed | failed | cancelled | recovering
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Suspension state (when status=suspended)
    suspended_node_id TEXT,
    suspension_reason TEXT,
    suspension_state_json TEXT,  -- JSON
    suspension_metadata_json TEXT,  -- JSON
    
    -- Completion/failure metadata
    completed_at TIMESTAMP,
    failed_at TIMESTAMP,
    error_message TEXT,
    
    -- Optimistic locking
    version INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_run_state_status ON run_state(status);
CREATE INDEX IF NOT EXISTS idx_run_state_updated ON run_state(updated_at);
