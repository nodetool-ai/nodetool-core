-- Migration for RunProjection model
-- Materialized view of workflow run state

CREATE TABLE IF NOT EXISTS run_projections (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    last_event_seq INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    node_states JSONB NOT NULL,
    trigger_cursors JSONB,
    pending_messages JSONB,
    metadata JSONB
);

-- Index for querying by status
CREATE INDEX IF NOT EXISTS idx_run_projections_status ON run_projections(status);
