-- Migration for run lease management
-- Ensures only one worker processes a run at a time

CREATE TABLE IF NOT EXISTS run_leases (
    run_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    acquired_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

-- Index for finding expired leases
CREATE INDEX IF NOT EXISTS idx_run_leases_expires ON run_leases(expires_at);
