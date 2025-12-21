-- Migration for RunEvent model
-- Append-only event log for workflow execution

CREATE TABLE IF NOT EXISTS run_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    node_id TEXT,
    payload JSONB NOT NULL,
    
    -- Ensure sequence uniqueness per run
    UNIQUE(run_id, seq)
);

-- Index for sequential event replay
CREATE INDEX IF NOT EXISTS idx_run_events_run_seq ON run_events(run_id, seq);

-- Index for node-specific queries
CREATE INDEX IF NOT EXISTS idx_run_events_run_node ON run_events(run_id, node_id);

-- Index for event type queries
CREATE INDEX IF NOT EXISTS idx_run_events_run_type ON run_events(run_id, event_type);
