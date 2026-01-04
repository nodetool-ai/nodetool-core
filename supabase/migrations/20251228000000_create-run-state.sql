CREATE TABLE IF NOT EXISTS run_state (
run_id TEXT PRIMARY KEY,
status TEXT NOT NULL,
created_at TEXT NOT NULL,
updated_at TEXT NOT NULL,
suspended_node_id TEXT,
suspension_reason TEXT,
suspension_state_json TEXT,
suspension_metadata_json TEXT,
completed_at TEXT,
failed_at TEXT,
error_message TEXT,
version INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_run_state_status
ON run_state(status);
CREATE INDEX IF NOT EXISTS idx_run_state_updated
ON run_state(updated_at);